import torch
from tqdm import tqdm
import datetime
import os
import importlib
import numpy as np # Import numpy
from PIL import Image # Import Image

from mmseg.apis import init_model, inference_model

from function import *
from evaluation import *
from dataset import CitySet, ADESet

from pixle import Pixle
from utils import save_experiment_results

import argparse
import setproctitle
setproctitle.setproctitle("Pixle_Attack_Cityscapes_Process")
# 필요한 글로벌 객체 허용
# torch.serialization.add_safe_globals([HistoryBuffer])
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
# 1. Config & Model 불러오기
# config_path = './configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
# checkpoint_path = 'ckpt/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'


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
            }
        }
    }

    device = config["device"]

    #configs_attack/ade20k/config_mask2_swin_B.py
    #configs_attack/ade20k/config_seg.py

    #configs_attack/cityscapes/config_mask2_swin_B.py
    #configs_attack/cityscapes/config_seg.py

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

    # num_images = 5

    # # 데이터셋 전체를 랜덤하게 섞기 위한 인덱스 생성 및 셔플
    # n_total = len(dataset.images)
    # indices = np.arange(n_total)
    # np.random.shuffle(indices) # 인덱스를 무작위로 섞음

    # # 섞인 인덱스를 사용하여 데이터셋 리스트 재정렬
    # dataset.images = [dataset.images[i] for i in indices]
    # dataset.filenames = [dataset.filenames[i] for i in indices]
    # dataset.gt_images = [dataset.gt_images[i] for i in indices]

    # # 이후 코드가 num_images 만큼 앞에서부터 선택하므로, 결과적으로 랜덤 샘플링됨

    # dataset.images = dataset.images[:min(len(dataset.images), num_images)]
    # dataset.filenames = dataset.filenames[:min(len(dataset.filenames), num_images)]
    # dataset.gt_images = dataset.gt_images[:min(len(dataset.gt_images), num_images)]

    model = init_model(model_cfg["config"], None, device)
    # 2. 체크포인트 로드 (weights_only=False 직접 설정)
    checkpoint = torch.load(model_cfg["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    del checkpoint  # 체크포인트 변수 삭제
    torch.cuda.empty_cache()  # GPU 캐시 정리

    # Lists to store results across all images
    img_list = []
    gt_list = []
    filename_list = []
    adv_img_lists = [[] for _ in range(5)] # Store adv images for each iteration level
    adv_query_lists = [[] for _ in range(5)] # Store adv query for each iteration level
    all_l0_metrics = [[] for _ in range(5)] # List of lists for L0 for each iteration level
    all_ratio_metrics = [[] for _ in range(5)] # List of lists for ratio
    all_impact_metrics = [[] for _ in range(5)] # List of lists for impact

    #Record start time
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    print("Start time:", start_timestamp)

    # Setup base directory for this run
    base_dir = os.path.join(config["base_dir"], start_timestamp)
    os.makedirs(base_dir, exist_ok=True)

    for img, filename, gt in tqdm(dataset, desc="Generating adversarial examples"):
        # Convert to BGR for model inference if needed, keep original RGB for saving/display
        img_bgr = img[:, :, ::-1].copy()

        img_tensor = torch.from_numpy(img_bgr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) # Ensure float and on GPU
        gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0).long().to(device) # Ensure long and on GPU
        # print(img_tensor.shape)


        # --- Separate directory and filename from the original filename ---
        original_dir = os.path.dirname(filename) # e.g., "frankfurt" or ""
        original_basename = os.path.basename(filename) # e.g., "frankfurt_000000_005543_leftImg8bit.png"
        # ----------------------------------------------------------------

        # --- Calculate and save original segmentation ONCE per image ---
        ori_result = inference_model(model, img_bgr) # Use BGR image

        
        # Create the full directory path including the original subdirectory
        ori_seg_dir = os.path.join(base_dir, "ori_seg", original_dir)
        os.makedirs(ori_seg_dir, exist_ok=True) # Create the directory if it doesn't exist
        visualize_segmentation(img, ori_result.pred_sem_seg.data.squeeze().cpu().numpy(),
                            # Save using the original basename in the created directory
                            save_path=os.path.join(ori_seg_dir, original_basename))
        # --------------------------------------------------------------

        # Calculate the number of pixels per patch
        _, _, H, W = img_tensor.shape
        total_target_pixels_overall = H * W * config["attack_pixel"]
        pixels_per_single_patch_target = total_target_pixels_overall / 250

        # === 패치 크기 계산 로직 ===
        target_area_int = int(round(pixels_per_single_patch_target))

        h_found = 1 # 기본값 (target_area_int가 소수이거나 1인 경우)
        for h_candidate in range(int(np.sqrt(target_area_int)), 0, -1): 
            if target_area_int % h_candidate == 0:
                h_found = h_candidate
                break
        patch_h_pixels = h_found
        patch_w_pixels = target_area_int // patch_h_pixels
        # === 패치 크기 계산 로직 끝 ===
        
        pixle = Pixle( 
            model,
            x_dimensions=(patch_w_pixels, patch_w_pixels), 
            y_dimensions=(patch_h_pixels, patch_h_pixels), 
            restarts=250,
            max_iterations=20,
            threshold=21000,
            device=device,
            cfg = config
        )

        # Ensure input tensor is on the correct device and potentially float
        results = pixle.forward(img_tensor, gt_tensor) # Pass tensor to Pixle

        # Process results (assuming results['adv_images'] are BGR tensors)
        adv_examples_bgr_numpy = [(x.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype(np.uint8) for x in results['adv_images']]
        adv_examples_rgb_numpy = [x[:, :, ::-1] for x in adv_examples_bgr_numpy] # Convert to RGB for saving and metrics
        example_query = results['query']

        # Store necessary data for final evaluation
        img_list.append(img) # Store original RGB image
        gt_list.append(gt)
        filename_list.append(filename) # Store the original full filename maybe for reference
        for i in range(5):
            adv_img_lists[i].append(adv_examples_rgb_numpy[i]) # Store RGB adv image
            adv_query_lists[i].append(example_query[i])
        # --- Loop for saving iteration-specific results and calculating metrics ---
        for i in range(5):
            # Define base paths for this iteration
            adv_base_path = os.path.join(base_dir, f"adv/{i+1}000query")
            delta_base_path = os.path.join(base_dir, f"delta/{i+1}000query")
            adv_seg_base_path = os.path.join(base_dir, f"adv_seg/{i+1}000query")

            # Create the full directory paths including the original subdirectory
            full_adv_dir = os.path.join(adv_base_path, original_dir)
            full_delta_dir = os.path.join(delta_base_path, original_dir)
            full_adv_seg_dir = os.path.join(adv_seg_base_path, original_dir)

            os.makedirs(full_adv_dir, exist_ok=True)
            os.makedirs(full_delta_dir, exist_ok=True)
            os.makedirs(full_adv_seg_dir, exist_ok=True)

            current_adv_img_rgb = adv_examples_rgb_numpy[i]
            current_adv_img_bgr = adv_examples_bgr_numpy[i] # Use BGR for inference if model expects it

            # Save adversarial example (RGB)
            adv_img_pil = Image.fromarray(current_adv_img_rgb)
            # Save using the original basename in the created directory
            adv_img_pil.save(os.path.join(full_adv_dir, original_basename))

            # Calculate and save delta image (against original RGB)
            delta_img_np = np.abs(img.astype(np.int16) - current_adv_img_rgb.astype(np.int16)).astype(np.uint8)
            delta_img_pil = Image.fromarray(delta_img_np)
            # Save using the original basename in the created directory
            delta_img_pil.save(os.path.join(full_delta_dir, original_basename))

            # Calculate and save adversarial segmentation
            adv_result = inference_model(model, current_adv_img_bgr)
            # Save using the original basename in the created directory
            visualize_segmentation(current_adv_img_rgb, adv_result.pred_sem_seg.data.squeeze().cpu().numpy(), # Visualize with RGB
                                save_path=os.path.join(full_adv_seg_dir, original_basename))

            # Calculate metrics using RGB uint8 images
            l0 = calculate_l0_norm(img, current_adv_img_rgb)
            ratio = calculate_pixel_ratio(img, current_adv_img_rgb)
            impact = calculate_impact(img, current_adv_img_rgb,
                                    ori_result.pred_sem_seg.data.squeeze().cpu().numpy(),
                                    adv_result.pred_sem_seg.data.squeeze().cpu().numpy())

            # Append metrics for this image and iteration level
            all_l0_metrics[i].append(l0)
            all_ratio_metrics[i].append(ratio)
            all_impact_metrics[i].append(impact)
        # --- End of inner loop (i=0 to 4) ---
    # --- End of dataset loop ---

    # --- Perform mIoU evaluation AFTER processing all images ---
    init_benign_to_adv_miou, init_gt_to_adv_miou = eval_miou(model, img_list, img_list, gt_list, config) # Evaluate benign using original images
    benign_to_adv_mious = []
    gt_to_adv_mious = []

    for i in range(5):
        benign_to_adv_miou, gt_to_adv_miou = eval_miou(model, img_list, adv_img_lists[i], gt_list, config)
        benign_to_adv_mious.append(benign_to_adv_miou)
        gt_to_adv_mious.append(gt_to_adv_miou)



    print("-" * 20)
    print(f"Benign_to_adv mIoU: {init_benign_to_adv_miou['mean_iou']}")
    print(f"GT_to_adv mIoU: {init_gt_to_adv_miou['mean_iou']}")

    results_list = []
    for i in range(5):
        # Calculate mean metrics for iteration level i across all images
        mean_query = np.mean(adv_query_lists[i]) if adv_query_lists[i] else 0
        mean_l0 = np.mean(all_l0_metrics[i]) if all_l0_metrics[i] else 0
        mean_ratio = np.mean(all_ratio_metrics[i]) if all_ratio_metrics[i] else 0
        mean_impact = np.mean(all_impact_metrics[i]) if all_impact_metrics[i] else 0
        benign_to_adv_miou = benign_to_adv_mious[i]
        gt_to_adv_miou = gt_to_adv_mious[i] # You might want this too

        iteration_results = {
            "benign_to_adv_miou": benign_to_adv_miou['mean_iou'],
            "gt_to_adv_miou": gt_to_adv_miou['mean_iou'],
            "mean_query": mean_query,
            "mean_l0": mean_l0,
            "mean_ratio": mean_ratio,
            "mean_impact": mean_impact,
            "individual_query": adv_query_lists[i],
            "individual_l0": all_l0_metrics[i],
            "individual_ratio": all_ratio_metrics[i],
            "individual_impact": all_impact_metrics[i],
        }
        results_list.append(iteration_results)
        # Save the consolidated results file in the base directory
        
        print(f"--- Iteration {(i+1)*1000} ---")
        print(f"    Benign_to_adv mIoU: {benign_to_adv_miou['mean_iou']}")
        print(f"    GT_to_adv mIoU: {gt_to_adv_miou['mean_iou']}")
        print(f"    Mean L0: {mean_l0:.2f}")
        print(f"    Mean Ratio: {mean_ratio}")
        print(f"    Mean Impact: {mean_impact}")
        print(f"    Mean Query: {mean_query}")

    # --- Calculate final average metrics and save results ONCE ---
    final_results = {
        "experiment_config": config,
        "model_config": model_cfg,
        "num_images": len(img_list),
        "start_time": start_timestamp,
        "end_time": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "elapsed_time": (datetime.datetime.now() - start_time).total_seconds(),
        "init_benign_to_adv_miou": init_benign_to_adv_miou['mean_iou'],
        "init_gt_to_adv_miou": init_gt_to_adv_miou['mean_iou'],
        "benign_to_adv_miou_results": benign_to_adv_miou['mean_iou'],
        "gt_to_adv_miou_results": gt_to_adv_miou['mean_iou'],
        "mean_query_results": mean_query,
        "mean_l0_results": mean_l0,
        "mean_ratio_results": mean_ratio,
        "mean_impact_results": mean_impact,
        "query_1000": results_list[0],
        "query_2000": results_list[1],
        "query_3000": results_list[2],
        "query_4000": results_list[3],
        "query_5000": results_list[4]
    }

    save_experiment_results(final_results,
                            config,
                            sweep_config=None, # Pass if needed
                            timestamp=start_timestamp,
                            save_dir=base_dir # Save summary in the main run folder
                            )

    print("-" * 20)
    print(f"Experiment results saved in: {base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attack_pixel", type=float, required=True, help="Ratio of adversarial pixels to total image pixels.") # 새 pixel_ratio 인자 추가
    args = parser.parse_args()

    config = load_config(args.config)
    
    config["device"] = args.device
    config["attack_pixel"] = args.attack_pixel

    main(config)