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
from dataset import CitySet, ADESet, VOCSet

from pixle import Pixle
from utils import save_experiment_results

import argparse
import setproctitle

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
    elif config["dataset"] == "VOC2012":
        dataset = VOCSet(dataset_dir=config["data_dir"])
    else:   
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    
    num_images = config["num_images"]

    # 데이터셋 전체를 랜덤하게 섞기 위한 인덱스 생성 및 셔플
    n_total = len(dataset.images)
    indices = np.arange(n_total)
    # np.random.shuffle(indices) # 인덱스를 무작위로 섞음

    # 섞인 인덱스를 사용하여 데이터셋 리스트 재정렬
    dataset.images = [dataset.images[i] for i in indices]
    dataset.filenames = [dataset.filenames[i] for i in indices]
    dataset.gt_images = [dataset.gt_images[i] for i in indices]

    # 이후 코드가 num_images 만큼 앞에서부터 선택하므로, 결과적으로 랜덤 샘플링됨

    dataset.images = dataset.images[:min(len(dataset.images), num_images)]
    dataset.filenames = dataset.filenames[:min(len(dataset.filenames), num_images)]
    dataset.gt_images = dataset.gt_images[:min(len(dataset.gt_images), num_images)]
    
    print(f"Total dataset size: {n_total}")
    print(f"Requested num_images: {num_images}")
    print(f"Actual images to process: {len(dataset.images)}")
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

    # Lists to store results across all images
    img_list = []
    gt_list = []
    filename_list = []
    adv_img_lists = [[] for _ in range(6)] # Store adv images for each iteration level (0-5)
    adv_query_lists = [[] for _ in range(6)] # Store adv query for each iteration level (0-5)
    # Change to 2D lists to match rs_eval.py style
    all_l0_metrics = [[] for _ in range(6)]
    all_ratio_metrics = [[] for _ in range(6)]
    all_impact_metrics = [[] for _ in range(6)]

    #Record start time
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    print("Start time:", start_timestamp)

    # Setup base directory for this run
    base_dir = os.path.join(config["base_dir"], start_timestamp)
    os.makedirs(base_dir, exist_ok=True)

    for i, (img_bgr, filename, gt) in tqdm(enumerate(dataset), desc="Generating adversarial examples"):
        setproctitle.setproctitle(f"({i+1}/{len(dataset.images)})_Pixle_{config['dataset']}_{config['model']}_{config['attack_pixel']}")

        img_tensor = torch.from_numpy(img_bgr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) # Ensure float and on GPU
        gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0).long().to(device) # Ensure long and on GPU
        # print(img_tensor.shape)


        # --- Separate directory and filename from the original filename ---
        original_basename = os.path.basename(filename) # e.g., "frankfurt_000000_005543_leftImg8bit.png"
        image_name = os.path.splitext(original_basename)[0]  # Remove extension for directory name
        # ----------------------------------------------------------------

        # --- Calculate and save original segmentation ONCE per image ---
        ori_result = inference_model(model, img_bgr) # Use BGR image

        # Create individual directory for this image (rs_eval.py style)
        current_img_save_dir = os.path.join(base_dir, image_name)
        os.makedirs(current_img_save_dir, exist_ok=True)
        
        # Save original image
        img = img_bgr[:, :, ::-1]  # Convert BGR to RGB
        Image.fromarray(img).save(os.path.join(current_img_save_dir, "original.png"))
        # --------------------------------------------------------------

        # Calculate the number of pixels per patch
        _, _, H, W = img_tensor.shape
        total_target_pixels_overall = H * W * config["attack_pixel"]
        pixels_per_single_patch_target = total_target_pixels_overall / config["restarts"]

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
            restarts=config["restarts"],
            max_iterations=config["max_iterations"],
            threshold=21000,
            device=device,
            cfg = config,
            is_mmseg_model=True
        )

        # Ensure input tensor is on the correct device and potentially float
        results = pixle.forward(img_tensor, gt_tensor) # Pass tensor to Pixle

        # Process results (assuming results['adv_images'] are BGR tensors)
        adv_examples_bgr_numpy = [(x.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype(np.uint8) for x in results['adv_images']]
        adv_examples_rgb_numpy = [x[:, :, ::-1] for x in adv_examples_bgr_numpy] # Convert to RGB for saving and metrics
        example_query = results['query']

        # Store necessary data for final evaluation
        img_list.append(img_bgr) # Store original BGR image
        gt_list.append(gt)
        filename_list.append(filename) # Store the original full filename maybe for reference
        
        # Store original image as 0th query result
        adv_img_lists[0].append(img_bgr) # Store original BGR image as 0th result
        adv_query_lists[0].append(0) # 0 queries for original
        
        for i in range(5):
            adv_img_lists[i+1].append(adv_examples_bgr_numpy[i]) # Store BGR adv image
            adv_query_lists[i+1].append(example_query[i])
        # --- Loop for saving iteration-specific results and calculating metrics (rs_eval.py style) ---
        # Store metrics for original image (all should be 0) without saving images
        l0 = 0  # No perturbation
        ratio = 0.0  # No changed pixels
        impact = 0.0  # No segmentation change
        
        # Append metrics for original image
        all_l0_metrics[0].append(l0)
        all_ratio_metrics[0].append(ratio)
        all_impact_metrics[0].append(impact)
        
        for i in range(5):
            # Create query-specific directory for this image
            query_img_save_dir = os.path.join(current_img_save_dir, f"{i+1}000query")
            os.makedirs(query_img_save_dir, exist_ok=True)

            current_adv_img_rgb = adv_examples_rgb_numpy[i]
            current_adv_img_bgr = adv_examples_bgr_numpy[i] # Use BGR for inference if model expects it

            # Save adversarial image (RGB)
            Image.fromarray(current_adv_img_rgb).save(os.path.join(query_img_save_dir, "adv.png"))

            # Calculate and save delta image (against original RGB)
            delta_img_np = np.abs(img.astype(np.int16) - current_adv_img_rgb.astype(np.int16)).astype(np.uint8)
            Image.fromarray(delta_img_np).save(os.path.join(query_img_save_dir, "delta.png"))

            # Calculate and save adversarial segmentation
            adv_result = inference_model(model, current_adv_img_bgr)
            adv_pred = adv_result.pred_sem_seg.data.squeeze().cpu().numpy()
            ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy()

            # Save segmentation visualizations (rs_eval.py style)
            visualize_segmentation(img, ori_pred,
                                save_path=os.path.join(query_img_save_dir, "ori_seg.png"),
                                alpha=0.5, dataset=config["dataset"])
            
            visualize_segmentation(img, ori_pred,
                                save_path=os.path.join(query_img_save_dir, "ori_seg_only.png"),
                                alpha=1, dataset=config["dataset"])
            
            visualize_segmentation(current_adv_img_rgb, adv_pred,
                                save_path=os.path.join(query_img_save_dir, "adv_seg.png"),
                                alpha=0.5, dataset=config["dataset"])
            
            visualize_segmentation(current_adv_img_rgb, adv_pred,
                                save_path=os.path.join(query_img_save_dir, "adv_seg_only.png"),
                                alpha=1, dataset=config["dataset"])

            # Calculate metrics using RGB uint8 images
            l0 = calculate_l0_norm(img, current_adv_img_rgb)
            ratio = calculate_pixel_ratio(img, current_adv_img_rgb)
            impact = calculate_impact(img, current_adv_img_rgb, ori_pred, adv_pred)

            # Append metrics for this image and iteration level (rs_eval.py style)
            all_l0_metrics[i+1].append(l0)
            all_ratio_metrics[i+1].append(ratio)
            all_impact_metrics[i+1].append(impact)
        # --- End of inner loop (i=0 to 4) ---
    # --- End of dataset loop ---

    # --- Perform mIoU evaluation AFTER processing all images (rs_eval.py style) ---
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
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to process.")
    parser.add_argument("--restarts", type=int, default=250, help="Number of restarts.")
    parser.add_argument("--max_iterations", type=int, default=20, help="Number of max iterations.")
    args = parser.parse_args()

    config = load_config(args.config)
    
    config["attack_method"] = "Pixle"
    config["device"] = args.device
    config["attack_pixel"] = args.attack_pixel
    config["num_images"] = args.num_images
    config["base_dir"] = f"./data/{config['attack_method']}/results/{config['dataset']}/{config['model']}"
    config["restarts"] = args.restarts
    config["max_iterations"] = args.max_iterations
    main(config)