"""
PointWise Attack Evaluation for MMSegmentation
Based on the original PointWise attack for image classification,
adapted for semantic segmentation models.
"""

import os
import sys
import torch
from tqdm import tqdm
import datetime
import importlib
import numpy as np
from PIL import Image
import multiprocessing as mp

# 상위 디렉토리와 현재 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from mmseg.apis import init_model, inference_model
from dataset import CitySet, ADESet, VOCSet
from pointwise_attack import PointWiseAttack, l0, gen_starting_point

from function import *
from evaluation import *
from utils import save_experiment_results

import argparse
import setproctitle


def load_config(config_path):
    """Load and return config dictionary from a python file at config_path."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def init_model_for_process(model_configs, dataset, model_name, device):
    """각 프로세스에서 모델을 초기화하는 함수"""
    if model_name == "setr":
        model = init_model(model_configs["config"], None, 'cuda')
        checkpoint = torch.load(model_configs["checkpoint"], map_location='cuda', weights_only=False)
        model.backbone.patch_embed.projection.bias = torch.nn.Parameter(
            torch.zeros(checkpoint["state_dict"]["backbone.patch_embed.projection.weight"].shape[0], device='cuda')
        )
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = init_model(model_configs["config"], None, device)
        checkpoint = torch.load(model_configs["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    del checkpoint
    torch.cuda.empty_cache()
    
    return model


def gen_starting_point_seg(attack, oimg, original_pred_labels, seed=None, init_mode='salt_pepper', max_attempts=100):
    """
    세그멘테이션용 Starting Point 생성 함수.
    원본 이미지와 다른 예측을 만드는 adversarial 이미지를 찾습니다.
    """
    if len(oimg.shape) == 3:
        oimg = oimg.unsqueeze(0)
    
    nquery = 0
    scales = [1, 2, 4, 8, 16, 32]
    
    for scale_idx, scale in enumerate(scales):
        for attempt in range(max_attempts // len(scales)):
            if seed is not None:
                torch.manual_seed(seed + attempt + scale_idx * 100)
            
            # Salt & Pepper 노이즈 생성
            c, h, w = oimg.shape[1], oimg.shape[2], oimg.shape[3]
            rnd = torch.rand(h // scale, w // scale).cuda()
            binary_pattern = (rnd > 0.5).float()
            
            # Upscale to original resolution
            noise_img = torch.zeros(h, w).cuda()
            for i in range(h // scale):
                for j in range(w // scale):
                    noise_img[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = binary_pattern[i, j]
            
            # Expand to all channels and scale to 0-255
            timg = noise_img.unsqueeze(0).repeat(c, 1, 1) * 255.0
            timg = timg.unsqueeze(0).float()
            
            nquery += 1
            is_adv, changed_ratio = attack.check_adv_status(timg, original_pred_labels)
            
            if is_adv:
                d = l0(oimg, timg)
                D = torch.ones(nquery, dtype=int).cuda() * d
                print(f'Starting point found: scale={scale}, changed_ratio={changed_ratio:.4f}, L0={d}')
                return timg, nquery, D
    
    # Fallback: 완전 랜덤 이미지
    timg = torch.rand_like(oimg).cuda() * 255.0
    d = l0(oimg, timg)
    D = torch.ones(nquery + 1, dtype=int).cuda() * d
    print(f'Fallback: using full random image, L0={d}')
    return timg, nquery + 1, D


def process_single_image(args):
    """단일 이미지를 처리하는 함수"""
    (img_bgr, filename, gt, model_configs, config, base_dir, idx, total_images, save_steps) = args
    
    # 프로세스별 모델 초기화
    model = init_model_for_process(model_configs, config["dataset"], config["model"], config["device"])
    
    setproctitle.setproctitle(f"({idx+1}/{total_images})_PointWise_Attack_{config['dataset']}_{config['model']}")

    img_tensor_bgr = torch.from_numpy(img_bgr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(config["device"])
    gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0).long().to(config["device"])

    ori_result = inference_model(model, img_bgr.copy()) 
    ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy()
    original_pred_labels = ori_result.pred_sem_seg.data.squeeze().cuda()

    # PointWise Attack 객체 생성
    attack = PointWiseAttack(
        model=model,
        cfg=config,
        is_mmseg=True,
        is_detectron2=False,
        success_threshold=config.get("success_threshold", 0.01),
        verbose=config.get("verbose", False)
    )

    # Starting Point 생성
    print(f"\n[{idx+1}/{total_images}] {filename}: Generating starting point...")
    timg, init_nqry, _ = gen_starting_point_seg(
        attack, img_tensor_bgr, original_pred_labels, 
        seed=config.get("seed", 0), 
        init_mode=config.get("init_mode", "salt_pepper")
    )

    levels = len(save_steps)
    adv_img_bgr_list = []
    adv_query_list = []
    total_nquery = init_nqry

    # Save original image as 0th result
    adv_img_bgr_list.append(img_tensor_bgr)
    adv_query_list.append(0)

    # PointWise Attack 실행
    print(f"[{idx+1}/{total_images}] {filename}: Running PointWise attack (mode={config['attack_mode']})...")
    
    if config["attack_mode"] == "single":
        x, nquery, D = attack.pw_perturb(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            max_query=config["max_query"]
        )
    elif config["attack_mode"] == "multiple":
        x, nquery, D = attack.pw_perturb_multiple(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            npix=config.get("npix", 196),
            max_query=config["max_query"]
        )
    elif config["attack_mode"] == "scheduling":
        x, nquery, D = attack.pw_perturb_multiple_scheduling(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            npix=config.get("npix", 196),
            max_query=config["max_query"]
        )
    else:
        raise ValueError(f"Unknown attack mode: {config['attack_mode']}")

    total_nquery += nquery
    
    # Reshape result to tensor
    adv_img_bgr = torch.from_numpy(x.reshape(img_tensor_bgr.squeeze(0).shape)).unsqueeze(0).float().cuda()
    
    # 최종 결과 저장
    adv_img_bgr_list.append(adv_img_bgr)
    adv_query_list.append(total_nquery)

    # 결과 저장
    current_img_save_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(current_img_save_dir, exist_ok=True)

    Image.fromarray(img_bgr[:, :, ::-1]).save(os.path.join(current_img_save_dir, "original.png"))
    Image.fromarray(gt).save(os.path.join(current_img_save_dir, "gt.png"))

    print(f"[{idx+1}/{total_images}] {filename}: Completed with {total_nquery} queries")
    
    # 메트릭 계산
    l0_metrics = []
    ratio_metrics = []
    impact_metrics = []
    
    for i, adv_img in enumerate(adv_img_bgr_list):
        query_val = adv_query_list[i]
        adv_img_np = adv_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        if i == 0:
            l0_norm = 0
            pixel_ratio = 0.0
            impact = 0.0
        else:
            query_img_save_dir = os.path.join(current_img_save_dir, f"{query_val}query")
            os.makedirs(query_img_save_dir, exist_ok=True)

            adv_result = inference_model(model, adv_img_np)
            adv_pred = adv_result.pred_sem_seg.data.squeeze().cpu().numpy()
            delta_img = np.abs(img_bgr.astype(np.int16) - adv_img_np.astype(np.int16)).astype(np.uint8)

            l0_norm = calculate_l0_norm(img_bgr, adv_img_np)
            pixel_ratio = calculate_pixel_ratio(img_bgr, adv_img_np)
            impact = calculate_impact(img_bgr, adv_img_np, ori_pred, adv_pred)

            Image.fromarray(adv_img_np[:, :, ::-1]).save(os.path.join(query_img_save_dir, "adv.png"))
            Image.fromarray(delta_img).save(os.path.join(query_img_save_dir, "delta.png"))

            visualize_segmentation(img_bgr, ori_pred,
                                save_path=os.path.join(query_img_save_dir, "ori_seg.png"),
                                alpha=0.5, dataset=config["dataset"])

            visualize_segmentation(adv_img_np, adv_pred,
                                save_path=os.path.join(query_img_save_dir, "adv_seg.png"),
                                alpha=0.5, dataset=config["dataset"])

        print(f"  L0 norm: {l0_norm}, Pixel ratio: {pixel_ratio:.4f}, Impact: {impact:.4f}")

        l0_metrics.append(l0_norm)
        ratio_metrics.append(pixel_ratio)
        impact_metrics.append(impact)

    # 모델 메모리 정리
    del model
    del attack
    torch.cuda.empty_cache()
    
    return {
        'img_bgr': img_bgr,
        'gt': gt,
        'filename': filename,
        'adv_img_bgr_list': adv_img_bgr_list,
        'adv_query_list': adv_query_list,
        'total_query': total_nquery,
        'l0_metrics': l0_metrics,
        'ratio_metrics': ratio_metrics,
        'impact_metrics': impact_metrics,
        'distance_history': D
    }


def main(config):
    # Model configs (rs_eval.py와 동일)
    model_configs = {
        "cityscapes": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
                "checkpoint": '../ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221203_045030-9a86a225.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
                "checkpoint": '../ckpt/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": '../ckpt/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": '../ckpt/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'
            }
        },
        "ade20k": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py',
                "checkpoint": '../ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py',
                "checkpoint": '../ckpt/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-80k_ade20k-512x512.py',
                "checkpoint": '../ckpt/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py',
                "checkpoint": '../ckpt/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth'
            }
        },
        "VOC2012": {
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-20k_voc12aug-512x512.py',
                "checkpoint": '../ckpt/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-40k_voc12aug-512x512.py',
                "checkpoint": '../ckpt/pspnet_r101-d8_512x512_20k_voc12aug_20200617_102003-4aef3c9a.pth'
            }
        }
    }

    device = config["device"]

    if config["dataset"] not in model_configs:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    if config["model"] not in model_configs[config["dataset"]]:
        raise ValueError(f"Unsupported model: {config['model']} for dataset {config['dataset']}")

    model_cfg = model_configs[config["dataset"]][config["model"]]

    # Load dataset
    data_dir = config["data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(os.path.join(current_dir, data_dir))

    if config["dataset"] == "cityscapes":
        dataset = CitySet(dataset_dir=data_dir)
    elif config["dataset"] == "ade20k":
        dataset = ADESet(dataset_dir=data_dir)
    elif config["dataset"] == "VOC2012":
        dataset = VOCSet(dataset_dir=data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    num_images = config["num_images"]
    dataset.images = dataset.images[:min(len(dataset.images), num_images)]
    dataset.filenames = dataset.filenames[:min(len(dataset.filenames), num_images)]
    dataset.gt_images = dataset.gt_images[:min(len(dataset.gt_images), num_images)]

    # 결과 저장 디렉토리
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['dataset']}_{config['model']}_pointwise_{current_time}"
    base_dir = os.path.join(config["base_dir"], current_time)
    os.makedirs(base_dir, exist_ok=True)

    save_steps = [0, config["max_query"]]  # 시작과 끝만 저장
    levels = len(save_steps)

    # 처리 데이터 준비
    process_args = []
    for idx, (img_bgr, filename, gt) in enumerate(zip(dataset.images, dataset.filenames, dataset.gt_images)):
        process_args.append((img_bgr, filename, gt, model_cfg, config, base_dir, idx, len(dataset.images), save_steps))

    # 모델 초기화 (메트릭 계산용)
    model = init_model_for_process(model_cfg, config["dataset"], config["model"], device)

    # 순차 처리
    print(f"\nProcessing {len(process_args)} images with PointWise Attack...")
    results = []
    
    img_list = []
    gt_list = []
    filename_list = []
    adv_img_lists = [[] for _ in range(levels)]
    all_l0_metrics = [[] for _ in range(levels)]
    all_ratio_metrics = [[] for _ in range(levels)]
    all_impact_metrics = [[] for _ in range(levels)]
    all_queries = []

    for args in tqdm(process_args, desc="PointWise Attack"):
        result = process_single_image(args)
        results.append(result)
        
        img_list.append(result['img_bgr'])
        gt_list.append(result['gt'])
        filename_list.append(result['filename'])
        all_queries.append(result['total_query'])

        for i, adv_img in enumerate(result['adv_img_bgr_list']):
            if i < levels:
                adv_img_lists[i].append(adv_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                all_l0_metrics[i].append(result['l0_metrics'][i])
                all_ratio_metrics[i].append(result['ratio_metrics'][i])
                all_impact_metrics[i].append(result['impact_metrics'][i])

    # mIoU 계산
    _, init_mious = eval_miou(model, img_list, img_list, gt_list, config)
    
    benign_to_adv_mious = []
    gt_to_adv_mious = []
    mean_l0 = []
    mean_ratio = []
    mean_impact = []
    
    for i in range(levels):
        if adv_img_lists[i]:
            benign_to_adv_miou, gt_to_adv_miou = eval_miou(model, img_list, adv_img_lists[i], gt_list, config)
            benign_to_adv_mious.append(benign_to_adv_miou['mean_iou'].item())
            gt_to_adv_mious.append(gt_to_adv_miou['mean_iou'].item())
        mean_l0.append(np.mean(all_l0_metrics[i]).item() if all_l0_metrics[i] else 0)
        mean_ratio.append(np.mean(all_ratio_metrics[i]).item() if all_ratio_metrics[i] else 0)
        mean_impact.append(np.mean(all_impact_metrics[i]).item() if all_impact_metrics[i] else 0)

    final_results = {
        "Attack Method": "PointWise",
        "Attack Mode": config["attack_mode"],
        "Init mIoU": init_mious['mean_iou'],
        "Adversarial mIoU(benign)": benign_to_adv_mious,
        "Adversarial mIoU(gt)": gt_to_adv_mious,
        "L0": mean_l0,
        "Ratio": mean_ratio,
        "Impact": mean_impact,
        "Average Queries": np.mean(all_queries).item(),
        "Max Query Limit": config["max_query"],
        "NPix": config.get("npix", 196)
    }

    print("\n--- Experiment Summary ---")
    for key, value in final_results.items():
        print(f"{key}: {value}")

    save_experiment_results(final_results,
                            config,
                            sweep_config=None,
                            timestamp=current_time,
                            save_dir=base_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PointWise attack evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use.')
    parser.add_argument('--max_query', type=int, default=1000, help='Maximum queries for attack.')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to evaluate.')
    parser.add_argument('--attack_mode', type=str, default='scheduling', 
                        choices=['single', 'multiple', 'scheduling'],
                        help='Attack mode: single, multiple, or scheduling.')
    parser.add_argument('--npix', type=int, default=196, help='Pixels per group for multiple mode.')
    parser.add_argument('--success_threshold', type=float, default=0.01, 
                        help='Threshold for attack success (ratio of changed pixels).')
    parser.add_argument('--init_mode', type=str, default='salt_pepper',
                        choices=['salt_pepper', 'random'],
                        help='Starting point initialization mode.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    args = parser.parse_args()

    config = load_config(args.config)
    config["attack_method"] = "PointWise"
    config["device"] = args.device
    config["max_query"] = args.max_query
    config["num_images"] = args.num_images
    config["attack_mode"] = args.attack_mode
    config["npix"] = args.npix
    config["success_threshold"] = args.success_threshold
    config["init_mode"] = args.init_mode
    config["seed"] = args.seed
    config["verbose"] = args.verbose
    config["base_dir"] = f"./data/PointWise/results/{config['dataset']}/{config['model']}"

    main(config)
