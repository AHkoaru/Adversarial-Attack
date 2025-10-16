import os
import torch
from tqdm import tqdm
import datetime
import importlib
import numpy as np
from PIL import Image
import multiprocessing as mp
from functools import partial

# CUDA 멀티프로세싱을 위한 시작 방법 설정
mp.set_start_method('spawn', force=True)

from mmseg.apis import init_model, inference_model
from dataset import CitySet, ADESet, VOCSet 
from sparse_rs import RSAttack

from function import *
from evaluation import *
from utils import save_experiment_results

import argparse
import setproctitle


def load_config(config_path):
    """
    Load and return config dictionary from a python file at config_path.
    The config file should contain a dictionary named 'config'.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def init_model_for_process(model_configs, dataset, model_name, device):
    """각 프로세스에서 모델을 초기화하는 함수"""
    if model_name == "setr":
        model = init_model(model_configs["config"], None, 'cuda')
        checkpoint = torch.load(model_configs["checkpoint"], map_location='cuda', weights_only=False)
        # 모델의 projection 레이어에 bias 추가
        model.backbone.patch_embed.projection.bias = torch.nn.Parameter(
            torch.zeros(checkpoint["state_dict"]["backbone.patch_embed.projection.weight"].shape[0], device='cuda')
        )
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = init_model(model_configs["config"], None, device)
        # 2. 체크포인트 로드 (weights_only=False 직접 설정)
        checkpoint = torch.load(model_configs["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    del checkpoint  # 체크포인트 변수 삭제
    torch.cuda.empty_cache()  # GPU 캐시 정리
    
    return model


def process_single_image(args):
    """단일 이미지를 처리하는 함수 (멀티프로세싱용)"""
    (img_bgr, filename, gt, model_configs, config, base_dir, idx, total_images) = args
    
    # 프로세스별 모델 초기화
    model = init_model_for_process(model_configs, config["dataset"], config["model"], config["device"])
    
    setproctitle.setproctitle(f"({idx+1}/{total_images})_SparseRS_Attack_{config['dataset']}_{config['model']}_{config['iters']}_{config['attack_pixel']}_{config['loss']}")

    img_tensor_bgr = torch.from_numpy(img_bgr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(config["device"])
    gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0).long().to(config["device"])

    ori_result = inference_model(model, img_bgr.copy()) 
    ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy()

    # 공격 객체를 한 번만 생성하고 재사용
    attack = RSAttack(
        model=model,
        cfg=config, # Pass the simplified config for RSAttack internal use
        norm=config["norm"], # or 'patches'
        n_queries=config["n_queries"],
        eps=config["eps"], # For L0, this is number of pixels. For patches, it's area.
        p_init=config["p_init"],
        n_restarts=config["n_restarts"],
        seed=0,
        verbose=config.get("verbose", False),  # config에서 verbose 설정 가져오기
        targeted=False,
        loss=config["loss"], # As used in the class
        resc_schedule=True,
        device=config["device"],
        log_path=None, # Disable logging for this simple test or provide a path
        original_img=img_bgr,
        d=5,
        use_decision_loss=config["use_decision_loss"],
        is_mmseg_model=True,
        enable_success_reporting=False
    )

    adv_img_bgr_list = []
    total_queries = config["iters"] * config["n_queries"]
    save_steps = [0] + [int(total_queries * (i+1) / 5) for i in range(5)]  # Include 0 queries
    
    # Save original image as 0th result
    adv_img_bgr_list.append(img_tensor_bgr)
    for iter_idx in range(config["iters"]):
        current_query, adv_img_bgr = attack.perturb(img_tensor_bgr, gt_tensor)
        img_tensor_bgr = adv_img_bgr
        # 다음 iteration을 위해 업데이트
        if current_query in save_steps[1:]:  # Skip the 0 query check since it's already added
            adv_img_bgr_list.append(adv_img_bgr)
    
    # 모든 save_steps에 도달하지 못한 경우 마지막 결과로 채우기
    while len(adv_img_bgr_list) < 6:
        adv_img_bgr_list.append(adv_img_bgr)

    # 결과 저장
    current_img_save_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(current_img_save_dir, exist_ok=True)

    Image.fromarray(img_bgr[:, :, ::-1]).save(os.path.join(current_img_save_dir, "original.png"))

    print(f"file_name: {filename}")
    
    # 메트릭 계산을 위한 리스트
    l0_metrics = []
    ratio_metrics = []
    impact_metrics = []
    
    for i, adv_img_bgr in enumerate(adv_img_bgr_list):
        adv_img_bgr = adv_img_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # For original image (i==0), only calculate metrics without saving images
        if i == 0:
            l0_norm = 0
            pixel_ratio = 0.0
            impact = 0.0
        else:
            # Create query-specific directory for adversarial images only
            query_img_save_dir = os.path.join(current_img_save_dir, f"{i}000query")
            os.makedirs(query_img_save_dir, exist_ok=True)
            
            # 적대적 이미지에 대한 추론 (main.py 참조)
            adv_result = inference_model(model, adv_img_bgr)
            adv_pred = adv_result.pred_sem_seg.data.squeeze().cpu().numpy()
            delta_img = np.abs(img_bgr.astype(np.int16) - adv_img_bgr.astype(np.int16)).astype(np.uint8)
            
            l0_norm = calculate_l0_norm(img_bgr, adv_img_bgr)
            pixel_ratio = calculate_pixel_ratio(img_bgr, adv_img_bgr)
            impact = calculate_impact(img_bgr, adv_img_bgr, ori_pred, adv_pred)
        
            Image.fromarray(adv_img_bgr[:, :, ::-1]).save(os.path.join(query_img_save_dir, "adv.png"))
            Image.fromarray(delta_img).save(os.path.join(query_img_save_dir, "delta.png"))
            # 시각화된 분할 마스크 저장 (main.py의 visualize_segmentation 사용)

            visualize_segmentation(img_bgr, ori_pred,
                                save_path=os.path.join(query_img_save_dir, "ori_seg.png"),
                                alpha=0.5, dataset=config["dataset"]) # 데이터셋에 맞는 팔레트 사용
            
            visualize_segmentation(img_bgr, ori_pred,
                                save_path=os.path.join(query_img_save_dir, "ori_seg_only.png"),
                                alpha=1, dataset=config["dataset"])
            
            visualize_segmentation(adv_img_bgr, adv_pred,
                                save_path=os.path.join(query_img_save_dir, "adv_seg.png"),
                                alpha=0.5, dataset=config["dataset"])
            
            visualize_segmentation(adv_img_bgr, adv_pred,
                                save_path=os.path.join(query_img_save_dir, "adv_seg_only.png"),
                                alpha=1, dataset=config["dataset"])
        
    
        print(f"L0 norm: {l0_norm}, Pixel ratio: {pixel_ratio}, Impact: {impact}")

        l0_metrics.append(l0_norm)
        ratio_metrics.append(pixel_ratio)
        impact_metrics.append(impact)

    # 모델 메모리 정리
    del model
    del attack  # 공격 객체도 삭제
    torch.cuda.empty_cache()
    
    return {
        'img_bgr': img_bgr,
        'gt': gt,
        'filename': filename,
        'adv_img_bgr_list': adv_img_bgr_list,
        'l0_metrics': l0_metrics,
        'ratio_metrics': ratio_metrics,
        'impact_metrics': impact_metrics
    }


def main(config):
    # 1. 공격 설정 로드 (main.py의 방식과 유사하게 argparse 또는 기본값 사용)
    # 예시: 공격 관련 설정은 main.py의 config 객체나 별도의 argparse로 관리
    # 여기서는 main.py의 config 로딩 방식을 차용하되, rs_eval 특화 설정을 추가할 수 있습니다.

    # main.py의 model_configs 와 유사한 방식으로 모델 정보 관리
    model_configs = {
        "cityscapes": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221203_045030-9a86a225.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
                "checkpoint": 'ckpt/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'
            },
            "setr": {
                "config": 'configs/setr/setr_vit-l_pup_8xb1-80k_cityscapes-768x768.py',
                "checkpoint": 'ckpt/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth'
            }
        },
        "ade20k": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py',
                "checkpoint": 'ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py',
                "checkpoint": 'ckpt/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-80k_ade20k-512x512.py',
                "checkpoint": 'ckpt/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py',
                "checkpoint": 'ckpt/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth'
            },
            "setr": {
                "config": 'configs/setr/setr_vit-l_pup_8xb2-160k_ade20k-512x512.py',
                "checkpoint": 'ckpt/setr_pup_512x512_160k_b16_ade20k_20210619_191343-7e0ce826.pth'
            }
        },
        "VOC2012": {
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-20k_voc12aug-512x512.py',
                "checkpoint": 'ckpt/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-40k_voc12aug-512x512.py',
                "checkpoint": 'ckpt/pspnet_r101-d8_512x512_20k_voc12aug_20200617_102003-4aef3c9a.pth'
            }
        }
    }

    device = config["device"]

    # Initialize model
    if config["dataset"] not in model_configs:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    if config["model"] not in model_configs[config["dataset"]]:
        raise ValueError(f"Unsupported model: {config['model']} for dataset {config['dataset']}")

    model_cfg = model_configs[config["dataset"]][config["model"]]

    # Load dataset
    if config["dataset"] == "cityscapes":
        dataset = CitySet(dataset_dir=config["data_dir"])
    elif config["dataset"] == "ade20k":
        dataset = ADESet(dataset_dir=config["data_dir"])
    elif config["dataset"] == "VOC2012":
        dataset = VOCSet(dataset_dir=config["data_dir"])
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    

    num_images = config["num_images"]

    # 데이터셋 전체를 랜덤하게 섞기 위한 인덱스 생성 및 셔플
    n_total = len(dataset.images)
    indices = np.arange(n_total)
    # np.random.shuffle(indices)
    # 섞인 인덱스를 사용하여 데이터셋 리스트 재정렬
    dataset.images = [dataset.images[i] for i in indices]
    dataset.filenames = [dataset.filenames[i] for i in indices]
    dataset.gt_images = [dataset.gt_images[i] for i in indices]

    # 이후 코드가 num_images 만큼 앞에서부터 선택하므로, 결과적으로 랜덤 샘플링됨

    dataset.images = dataset.images[:min(len(dataset.images), num_images)]
    dataset.filenames = dataset.filenames[:min(len(dataset.filenames), num_images)]
    dataset.gt_images = dataset.gt_images[:min(len(dataset.gt_images), num_images)]

    # 결과 저장을 위한 디렉토리 설정
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['dataset']}_{config['model']}_sparse-rs_{current_time}"
    base_dir = os.path.join(config["base_dir"], current_time)
    os.makedirs(base_dir, exist_ok=True, mode=0o777)
    
    # 멀티프로세싱을 위한 데이터 준비
    process_args = []
    for idx, (img_bgr, filename, gt) in enumerate(zip(dataset.images, dataset.filenames, dataset.gt_images)):
        process_args.append((img_bgr, filename, gt, model_cfg, config, base_dir, idx, len(dataset.images)))

    # 멀티프로세싱 대신 순차적으로 실행
    print(f"Sequential processing for {len(process_args)} images...")
    results = []
    for args in tqdm(process_args, total=len(process_args), desc="Running Sparse-RS Attack"):
        result = process_single_image(args)
        results.append(result)
    
    # 결과 수집 및 정리
    img_list = []
    gt_list = []
    filename_list = []
    adv_img_lists = [[] for _ in range(6)]
    all_l0_metrics = [[] for _ in range(6)] 
    all_ratio_metrics = [[] for _ in range(6)] 
    all_impact_metrics = [[] for _ in range(6)] 

    for result in results:
        img_list.append(result['img_bgr'])
        gt_list.append(result['gt'])
        filename_list.append(result['filename'])
        
        for i, adv_img_bgr in enumerate(result['adv_img_bgr_list']):
            adv_img_lists[i].append(adv_img_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            all_l0_metrics[i].append(result['l0_metrics'][i])
            all_ratio_metrics[i].append(result['ratio_metrics'][i])
            all_impact_metrics[i].append(result['impact_metrics'][i])

    # 평가를 위한 모델 초기화 (한 번만)
    if config["model"] == "setr":
        model = init_model(model_cfg["config"], None, 'cuda')
        checkpoint = torch.load(model_cfg["checkpoint"], map_location='cuda', weights_only=False)
        model.backbone.patch_embed.projection.bias = torch.nn.Parameter(
            torch.zeros(checkpoint["state_dict"]["backbone.patch_embed.projection.weight"].shape[0], device='cuda')
        )
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = init_model(model_cfg["config"], None, device)
        checkpoint = torch.load(model_cfg["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    del checkpoint
    torch.cuda.empty_cache()

    _, init_mious = eval_miou(model, img_list, img_list, gt_list, config)
    
    benign_to_adv_mious = []
    gt_to_adv_mious = []
    gt_mean_accuracy  = []
    gt_overall_accuracy = []
    benign_mean_accuracy  = []
    benign_overall_accuracy = []
    mean_l0 = []
    mean_ratio = []
    mean_impact = []
    
    # 클래스별 IoU 값들을 저장할 리스트 (VOC2012일 때만 label 0 제외)
    benign_to_adv_per_ious_excluding_label0 = []
    gt_to_adv_per_ious_excluding_label0 = []
    
    # Per-category IoU (full)
    benign_to_adv_per_ious = []
    gt_to_adv_per_ious = []
    
    for i in range(6):
        benign_to_adv_miou, gt_to_adv_miou = eval_miou(model, img_list, adv_img_lists[i], gt_list, config)
        
        # 기존 메트릭들
        benign_to_adv_mious.append(benign_to_adv_miou['mean_iou'].item())
        gt_to_adv_mious.append(gt_to_adv_miou['mean_iou'].item())
        gt_mean_accuracy.append(gt_to_adv_miou['mean_accuracy'].item())
        gt_overall_accuracy.append(gt_to_adv_miou['overall_accuracy'].item())
        benign_mean_accuracy.append(benign_to_adv_miou['mean_accuracy'].item())
        benign_overall_accuracy.append(benign_to_adv_miou['overall_accuracy'].item())

        # Save per-category IoU
        if 'per_category_iou' in benign_to_adv_miou:
            benign_to_adv_per_ious.append(benign_to_adv_miou['per_category_iou'].tolist())
        else:
            benign_to_adv_per_ious.append(None)
            
        if 'per_category_iou' in gt_to_adv_miou:
            gt_to_adv_per_ious.append(gt_to_adv_miou['per_category_iou'].tolist())
        else:
            gt_to_adv_per_ious.append(None)

        # VOC2012 데이터셋일 때만 per_category_iou에서 label 0을 제외한 평균 계산
        if config["dataset"] == "VOC2012":
            if 'per_category_iou' in benign_to_adv_miou:
                benign_per_iou_values = benign_to_adv_miou['per_category_iou']
                # label 0 (첫 번째 클래스)을 제외하고 nan이 아닌 값들만으로 평균 계산
                benign_per_iou_excluding_label0 = benign_per_iou_values[1:]  # 인덱스 1부터 (label 1~)
                # nan 값들을 제외하고 평균 계산
                benign_mean_iou_excluding_label0 = np.nanmean(benign_per_iou_excluding_label0).item()
                benign_to_adv_per_ious_excluding_label0.append(benign_mean_iou_excluding_label0)
            else:
                benign_to_adv_per_ious_excluding_label0.append(None)
                
            if 'per_category_iou' in gt_to_adv_miou:
                gt_per_iou_values = gt_to_adv_miou['per_category_iou']
                # label 0 (첫 번째 클래스)을 제외하고 nan이 아닌 값들만으로 평균 계산  
                gt_per_iou_excluding_label0 = gt_per_iou_values[1:]  # 인덱스 1부터 (label 1~)
                # nan 값들을 제외하고 평균 계산
                gt_mean_iou_excluding_label0 = np.nanmean(gt_per_iou_excluding_label0).item()
                gt_to_adv_per_ious_excluding_label0.append(gt_mean_iou_excluding_label0)
            else:
                gt_to_adv_per_ious_excluding_label0.append(None)
        else:
            # VOC2012가 아닌 경우에는 None으로 설정
            benign_to_adv_per_ious_excluding_label0.append(None)
            gt_to_adv_per_ious_excluding_label0.append(None)

        mean_l0.append(np.mean(all_l0_metrics[i]).item())
        mean_ratio.append(np.mean(all_ratio_metrics[i]).item())
        mean_impact.append(np.mean(all_impact_metrics[i]).item())

    final_results = {
        # Main metrics in the specified order
        "Init mIoU" : init_mious['mean_iou'],
        "Adversarial mIoU(benign)" : benign_to_adv_mious,
        "Adversarial mIoU(gt)" : gt_to_adv_mious,
        "Accuracy(benign)": benign_mean_accuracy,
        "Overall Accuracy(benign)": benign_overall_accuracy,
        "Accuracy(gt)": gt_mean_accuracy,
        "Overall Accuracy(gt)": gt_overall_accuracy,
        "L0": mean_l0,
        "Ratio": mean_ratio,
        "Impact": mean_impact,
        "Per-category IoU(benign)": benign_to_adv_per_ious,
        "Per-category IoU(gt)": gt_to_adv_per_ious,
    }
    
    # VOC2012 데이터셋일 때만 label 0을 제외한 mIoU 메트릭 추가
    if config["dataset"] == "VOC2012":
        final_results["Average mIoU excluding label 0 (benign)"] = benign_to_adv_per_ious_excluding_label0
        final_results["Average mIoU excluding label 0 (gt)"] = gt_to_adv_per_ious_excluding_label0

    print("\n--- Experiment Summary ---")
    print(final_results)
    
    save_experiment_results(final_results,
                            config,
                            sweep_config=None, # Pass if needed
                            timestamp=current_time,
                            save_dir=base_dir # Save summary in the main run folder
                            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Sparse-RS attack evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    parser.add_argument('--n_queries', type=int, default=10, help='Max number of queries for RSAttack.')
    parser.add_argument('--eps', type=float, default=0.0001, help='Epsilon for L0 norm in RSAttack (perturbation budget, e.g., percentage of pixels).')
    parser.add_argument('--p_init', type=float, default=0.8, help='Initial probability p_init for RSAttack.')
    parser.add_argument('--n_restarts', type=int, default=1, help='Number of restarts for RSAttack.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to evaluate from the dataset.')
    parser.add_argument('--iters', type=int, default=500, help='Number of iterations for RSAttack.')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes for parallel processing.')
    parser.add_argument('--use_decision_loss', type=str, default='False', choices=['True', 'False'], help='Whether to use decision loss.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--norm', type=str, default='L0', choices=['L0', 'patches'], help='Norm for RSAttack.')
    parser.add_argument('--loss', type=str, default='prob', choices=['margin', 'prob', 'decision', 'decision_change'], help='Loss function for RSAttack.')
    args = parser.parse_args()

    config = load_config(args.config)
    config["attack_method"] = "Sparse-RS"
    config["device"] = args.device
    config["n_queries"] = args.n_queries
    config["eps"] = args.eps
    config["attack_pixel"] = args.eps
    config["p_init"] = args.p_init
    config["n_restarts"] = args.n_restarts
    config["num_images"] = args.num_images
    config["iters"] = args.iters
    config["num_processes"] = args.num_processes
    config["base_dir"] = f"./data/{config['attack_method']}/results/{config['dataset']}/{config['model']}"
    # decision/decision_change 선택 시 decision loss 자동 활성화
    if args.loss in ['decision', 'decision_change']:
        config["use_decision_loss"] = True
    else:
        config["use_decision_loss"] = args.use_decision_loss.lower() == 'true'  # 문자열을 boolean으로 변환
    config["verbose"] = args.verbose
    config["norm"] = args.norm
    config["loss"] = args.loss
    main(config)
