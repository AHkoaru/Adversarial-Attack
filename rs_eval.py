import os
import torch
from tqdm import tqdm
import datetime
import importlib
import numpy as np
from PIL import Image

from mmseg.apis import init_model, inference_model
from dataset import CitySet, ADESet # main.py에서 사용된 데이터셋 클래스
from sparse_rs import RSAttack # sparse-rs.py의 공격 클래스

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

    if config["model"] == "setr":
        model = init_model(model_cfg["config"], None, 'cuda')
        checkpoint = torch.load(model_cfg["checkpoint"], map_location='cuda', weights_only=False)
        # 모델의 projection 레이어에 bias 추가
        model.backbone.patch_embed.projection.bias = torch.nn.Parameter(
            torch.zeros(checkpoint["state_dict"]["backbone.patch_embed.projection.weight"].shape[0], device='cuda')
        )
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = init_model(model_cfg["config"], None, device)
        # 2. 체크포인트 로드 (weights_only=False 직접 설정)
        checkpoint = torch.load(model_cfg["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    del checkpoint  # 체크포인트 변수 삭제
    torch.cuda.empty_cache()  # GPU 캐시 정리
    # Initialize RSAttack

    # 결과 저장을 위한 디렉토리 설정
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['dataset']}_{config['model']}_sparse-rs_{current_time}"
    base_dir = os.path.join(config["base_dir"], current_time)
    os.makedirs(base_dir, exist_ok=True)
    
    img_list = []
    gt_list = []
    filename_list = []
    adv_img_lists = [[] for _ in range(5)]
    adv_query_lists = [[] for _ in range(5)]
    all_l0_metrics = [[] for _ in range(5)] 
    all_ratio_metrics = [[] for _ in range(5)] 
    all_impact_metrics = [[] for _ in range(5)] 

    for i, (img_bgr, filename, gt) in tqdm(enumerate(dataset), desc="Running Sparse-RS Attack", total=len(dataset.images)):
        setproctitle.setproctitle(f"SparseRS_Attack_{config['dataset']}_{config['model']}_{config['attack_pixel']}({i}/{len(dataset.images)})")

        img_tensor_bgr = torch.from_numpy(img_bgr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(config["device"])
        gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0).long().to(config["device"])

        ori_result = inference_model(model, img_bgr.copy()) 
        ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy()

        adv_img_bgr_list = []
        attack = RSAttack(
            model=model,
            cfg=config, # Pass the simplified config for RSAttack internal use
            norm='L0', # or 'patches'
            n_queries=config["n_queries"],
            eps=config["eps"], # For L0, this is number of pixels. For patches, it's area.
            p_init=config["p_init"],
            n_restarts=config["n_restarts"],
            seed=0,
            verbose=True,
            targeted=False,
            loss='segmentation_prob', # As used in the class
            resc_schedule=True,
            device=config["device"],
            log_path=None # Disable logging for this simple test or provide a path
        )
            
        for i in range(5):
            query, adv_img_bgr = attack.perturb(img_tensor_bgr, gt_tensor)
            adv_img_bgr_list.append(adv_img_bgr)
            img_tensor_bgr = adv_img_bgr

            l0_norm = calculate_l0_norm(img_bgr, adv_img_bgr)
            pixel_ratio = calculate_pixel_ratio(img_bgr, adv_img_bgr)

            print(f"L0 norm: {l0_norm}, Pixel ratio: {pixel_ratio}")

        img_list.append(img_bgr)
        gt_list.append(gt)
        for query_idx, adv_img_bgr in enumerate(adv_img_bgr_list):
            adv_img_lists[query_idx].append(adv_img_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))

        current_img_save_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(filename))[0])
        os.makedirs(current_img_save_dir, exist_ok=True)

        Image.fromarray(img_bgr[:, :, ::-1]).save(os.path.join(current_img_save_dir, "original.png"))

        for i, adv_img_bgr in enumerate(adv_img_bgr_list):
            query_img_save_dir = os.path.join(current_img_save_dir, f"{i+1}000query")
            os.makedirs(query_img_save_dir, exist_ok=True)

            adv_img_bgr = adv_img_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # 적대적 이미지에 대한 추론 (main.py 참조)
            adv_result = inference_model(model, adv_img_bgr)
            adv_pred = adv_result.pred_sem_seg.data.squeeze().cpu().numpy()
            delta_img = np.abs(img_bgr.astype(np.int16) - adv_img_bgr.astype(np.int16)).astype(np.uint8)
        
            Image.fromarray(adv_img_bgr[:, :, ::-1]).save(os.path.join(query_img_save_dir, "adv.png"))
            Image.fromarray(delta_img).save(os.path.join(query_img_save_dir, "delta.png"))
            # 시각화된 분할 마스크 저장 (main.py의 visualize_segmentation 사용)

            visualize_segmentation(img_bgr, ori_pred,
                                save_path=os.path.join(query_img_save_dir, "ori_seg.png"),
                                alpha=0.5, dataset=config["dataset"]) # 데이터셋에 맞는 팔레트 사용
            
            visualize_segmentation(adv_img_bgr, adv_pred,
                                save_path=os.path.join(query_img_save_dir, "adv_seg.png"),
                                alpha=0.5, dataset=config["dataset"])
        
            l0_norm = calculate_l0_norm(img_bgr, adv_img_bgr)
            pixel_ratio = calculate_pixel_ratio(img_bgr, adv_img_bgr)
            impact = calculate_impact(img_bgr, adv_img_bgr, ori_pred, adv_pred)

            all_l0_metrics[i].append(l0_norm)
            all_ratio_metrics[i].append(pixel_ratio)
            all_impact_metrics[i].append(impact)

    _, init_mious = eval_miou(model, img_list, img_list, gt_list, config)
    benign_to_adv_mious = []
    gt_to_adv_mious = []
    mean_l0 = []
    mean_ratio = []
    mean_impact = []
    for i in range(5):
        benign_to_adv_miou, gt_to_adv_miou = eval_miou(model, img_list, adv_img_lists[i], gt_list, config)
        benign_to_adv_mious.append(benign_to_adv_miou['mean_iou'].item())
        gt_to_adv_mious.append(gt_to_adv_miou['mean_iou'].item())

        mean_l0.append(np.mean(all_l0_metrics[i]).item())
        mean_ratio.append(np.mean(all_ratio_metrics[i]).item())
        mean_impact.append(np.mean(all_impact_metrics[i]).item())

    final_results = {
        "Init mIoU" : init_mious['mean_iou'],
        "Average Adversarial mIoU(benign)" : benign_to_adv_mious,
        "Average Adversarial mIoU(gt)" : gt_to_adv_mious,
        "Average L0": mean_l0,
        "Average Ratio": mean_ratio,
        "Average Impact": mean_impact
    }
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
    parser.add_argument('--n_queries', type=int, default=5000, help='Max number of queries for RSAttack.')
    parser.add_argument('--eps', type=float, default=0.05, help='Epsilon for L0 norm in RSAttack (perturbation budget, e.g., percentage of pixels).')
    parser.add_argument('--p_init', type=float, default=0.8, help='Initial probability p_init for RSAttack.')
    parser.add_argument('--n_restarts', type=int, default=1, help='Number of restarts for RSAttack.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to evaluate from the dataset.')
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
    config["base_dir"] = f"./data/{config['attack_method']}/results/{config['dataset']}/{config['model']}"
    main(config)
