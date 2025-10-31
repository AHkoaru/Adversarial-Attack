import os
import torch
from tqdm import tqdm
import datetime
import numpy as np
from PIL import Image
import sys
import argparse
import setproctitle

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from adv_setting.py
from adv_setting import model_predict, load_model, set_seed

# Import from parent directory
from dataset import CitySet, ADESet, VOCSet
from pixle import Pixle
from function import *
from evaluation import *
from utils import save_experiment_results
from config import voc, config_city


def process_single_image_robust(args):
    """Process single image using Robust models instead of mmseg"""
    (img_bgr, filename, gt, model_config, config, base_dir, idx, total_images) = args
    
    setproctitle.setproctitle(f"({idx+1}/{total_images})_Pixle_Attack_{config['dataset']}_{config['model']}_{config['attack_pixel']}")
    
    # Load model using adv_setting.py
    model = load_model(model_config)
    checkpoint = torch.load(model_config["model_path"], map_location=config["device"], weights_only=False)
    model = model.to(config["device"])
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Convert tensor to numpy for tensor operations
    if isinstance(img_bgr, torch.Tensor):
        img_bgr_np = img_bgr.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        img_bgr_np = img_bgr.copy()
    
    img_tensor_bgr = torch.from_numpy(img_bgr_np.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(config["device"])
    gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0).long().to(config["device"])

    # Get original prediction using model_predict from adv_setting.py
    ori_confidence, ori_pred = model_predict(model, img_bgr_np, model_config)
    ori_pred = ori_pred.cpu().numpy()

    # Save results
    current_img_save_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(current_img_save_dir, exist_ok=True)

    Image.fromarray(img_bgr_np[:, :, ::-1]).save(os.path.join(current_img_save_dir, "original.png"))

    print(f"file_name: {filename}")
    
    # Calculate metrics for query=0 (original image)
    l0_metrics = [0]  # L0 norm is 0 for original image
    ratio_metrics = [0]  # Pixel ratio is 0 for original image
    impact_metrics = [0]  # Impact is 0 for original image

    # Calculate the number of pixels per patch
    _, _, H, W = img_tensor_bgr.shape
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
        restarts=500,
        max_iterations=20,
        threshold=21000,
        device=config["device"],
        cfg=config,
        is_mmseg_model=False,
        model_config=model_config
    )

    # Ensure input tensor is on the correct device and potentially float
    results = pixle.forward(img_tensor_bgr, gt_tensor)

    # Process results (assuming results['adv_images'] are BGR tensors)
    adv_examples_bgr_numpy = [(x.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype(np.uint8) for x in results['adv_images']]
    adv_examples_rgb_numpy = [x[:, :, ::-1] for x in adv_examples_bgr_numpy] # Convert to RGB for saving and metrics
    
    for i in range(5):
        query_img_save_dir = os.path.join(current_img_save_dir, f"{i+1}000query")
        os.makedirs(query_img_save_dir, exist_ok=True)

        current_adv_img_rgb = adv_examples_rgb_numpy[i]
        current_adv_img_bgr = adv_examples_bgr_numpy[i]
        
        # Get adversarial prediction using model_predict from adv_setting.py
        adv_confidence, adv_pred = model_predict(model, current_adv_img_bgr, model_config)
        adv_pred = adv_pred.cpu().numpy()
        
        # Save adversarial image (RGB)
        Image.fromarray(current_adv_img_rgb).save(os.path.join(query_img_save_dir, "adv.png"))

        # Calculate and save delta image
        delta_img = np.abs(img_bgr_np[:, :, ::-1].astype(np.int16) - current_adv_img_rgb.astype(np.int16)).astype(np.uint8)
        Image.fromarray(delta_img).save(os.path.join(query_img_save_dir, "delta.png"))
        
        # Visualize segmentation results
        visualize_segmentation(img_bgr_np[:, :, ::-1], ori_pred,
                            save_path=os.path.join(query_img_save_dir, "ori_seg.png"),
                            alpha=0.5, dataset=config["dataset"])
        
        visualize_segmentation(img_bgr_np[:, :, ::-1], ori_pred,
                            save_path=os.path.join(query_img_save_dir, "ori_seg_only.png"),
                            alpha=1, dataset=config["dataset"])
        
        visualize_segmentation(current_adv_img_rgb, adv_pred,
                            save_path=os.path.join(query_img_save_dir, "adv_seg.png"),
                            alpha=0.5, dataset=config["dataset"])
        
        visualize_segmentation(current_adv_img_rgb, adv_pred,
                            save_path=os.path.join(query_img_save_dir, "adv_seg_only.png"),
                            alpha=1, dataset=config["dataset"])
        
        l0_norm = calculate_l0_norm(img_bgr_np[:, :, ::-1], current_adv_img_rgb)
        pixel_ratio = calculate_pixel_ratio(img_bgr_np[:, :, ::-1], current_adv_img_rgb)
        impact = calculate_impact(img_bgr_np[:, :, ::-1], current_adv_img_rgb, ori_pred, adv_pred)
        
        print(f"L0 norm: {l0_norm}, Pixel ratio: {pixel_ratio}, Impact: {impact}")

        l0_metrics.append(l0_norm)
        ratio_metrics.append(pixel_ratio)
        impact_metrics.append(impact)

    # Clean up memory
    del model
    del pixle
    torch.cuda.empty_cache()
    
    return {
        'img_bgr': img_bgr_np,
        'gt': gt,
        'filename': filename,
        'adv_img_bgr_list': adv_examples_bgr_numpy,
        'l0_metrics': l0_metrics,
        'ratio_metrics': ratio_metrics,
        'impact_metrics': impact_metrics
    }


def main(config, model_config):
    # Set random seed
    set_seed(42)
    
    # Model configurations for Robust models (using adv_setting.py approach)
    device = config["device"]
    
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

    # Randomly shuffle dataset indices
    n_total = len(dataset.images)
    indices = np.arange(n_total)
    
    # Reorder dataset lists using shuffled indices
    dataset.images = [dataset.images[i] for i in indices]
    dataset.filenames = [dataset.filenames[i] for i in indices]
    dataset.gt_images = [dataset.gt_images[i] for i in indices]

    # Select subset of images
    dataset.images = dataset.images[:min(len(dataset.images), num_images)]
    dataset.filenames = dataset.filenames[:min(len(dataset.filenames), num_images)]
    dataset.gt_images = dataset.gt_images[:min(len(dataset.gt_images), num_images)]

    # Setup result directories
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['dataset']}_{config['model']}_pixle_{current_time}"
    base_dir = os.path.join(config["base_dir"], current_time)
    os.makedirs(base_dir, exist_ok=True, mode=0o777)
    
    # Prepare data for processing
    process_args = []
    for idx, (img_bgr, filename, gt) in enumerate(zip(dataset.images, dataset.filenames, dataset.gt_images)):
        process_args.append((img_bgr, filename, gt, model_config, config, base_dir, idx, len(dataset.images)))

    # Sequential processing
    print(f"Sequential processing for {len(process_args)} images...")
    results = []
    for args in tqdm(process_args, total=len(process_args), desc="Running Pixle Attack"):
        result = process_single_image_robust(args)
        results.append(result)
    
    # Collect results (including query=0)
    img_list = []
    gt_list = []
    filename_list = []
    adv_img_lists = [[] for _ in range(6)]  # 0query + 5 attack queries
    all_l0_metrics = [[] for _ in range(6)] 
    all_ratio_metrics = [[] for _ in range(6)] 
    all_impact_metrics = [[] for _ in range(6)] 

    for result in results:
        img_list.append(result['img_bgr'])
        gt_list.append(result['gt'])
        filename_list.append(result['filename'])
        
        # Add query=0 results (original image)
        adv_img_lists[0].append(result['img_bgr'])
        all_l0_metrics[0].append(result['l0_metrics'][0])  # 0
        all_ratio_metrics[0].append(result['ratio_metrics'][0])  # 0
        all_impact_metrics[0].append(result['impact_metrics'][0])  # 0
        
        # Add attack results (query 1-5)
        for i, adv_img_bgr in enumerate(result['adv_img_bgr_list']):
            adv_img_lists[i+1].append(adv_img_bgr)
            all_l0_metrics[i+1].append(result['l0_metrics'][i+1])
            all_ratio_metrics[i+1].append(result['ratio_metrics'][i+1])
            all_impact_metrics[i+1].append(result['impact_metrics'][i+1])

    # Initialize model for evaluation (only once)
    model = load_model(model_config)
    checkpoint = torch.load(model_config["model_path"], map_location=device, weights_only=False)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Use eval_miou from evaluation.py
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
    
    # Per-category IoU excluding label 0 for VOC2012
    benign_to_adv_per_ious_excluding_label0 = []
    gt_to_adv_per_ious_excluding_label0 = []
    
    # Per-category IoU (full)
    benign_to_adv_per_ious = []
    gt_to_adv_per_ious = []
    
    for i in range(6):
        # Use eval_miou from evaluation.py
        benign_to_adv_miou, gt_to_adv_miou = eval_miou(model, img_list, adv_img_lists[i], gt_list, config)
        
        # Existing metrics
        benign_to_adv_mious.append(benign_to_adv_miou['mean_iou'].item())
        gt_to_adv_mious.append(gt_to_adv_miou['mean_iou'].item())
        gt_mean_accuracy.append(gt_to_adv_miou.get('mean_accuracy', 0).item() if 'mean_accuracy' in gt_to_adv_miou else 0)
        gt_overall_accuracy.append(gt_to_adv_miou.get('overall_accuracy', 0).item() if 'overall_accuracy' in gt_to_adv_miou else 0)
        benign_mean_accuracy.append(benign_to_adv_miou.get('mean_accuracy', 0).item() if 'mean_accuracy' in benign_to_adv_miou else 0)
        benign_overall_accuracy.append(benign_to_adv_miou.get('overall_accuracy', 0).item() if 'overall_accuracy' in benign_to_adv_miou else 0)

        # Save per-category IoU
        if 'per_category_iou' in benign_to_adv_miou:
            benign_to_adv_per_ious.append(benign_to_adv_miou['per_category_iou'].tolist())
        else:
            benign_to_adv_per_ious.append(None)
            
        if 'per_category_iou' in gt_to_adv_miou:
            gt_to_adv_per_ious.append(gt_to_adv_miou['per_category_iou'].tolist())
        else:
            gt_to_adv_per_ious.append(None)

        # VOC2012 dataset specific: exclude label 0 from per_category_iou
        if config["dataset"] == "VOC2012":
            if 'per_category_iou' in benign_to_adv_miou:
                benign_per_iou_values = benign_to_adv_miou['per_category_iou']
                benign_per_iou_excluding_label0 = benign_per_iou_values[1:]  # Exclude index 0 (label 1~)
                benign_mean_iou_excluding_label0 = np.nanmean(benign_per_iou_excluding_label0).item()
                benign_to_adv_per_ious_excluding_label0.append(benign_mean_iou_excluding_label0)
            else:
                benign_to_adv_per_ious_excluding_label0.append(None)
                
            if 'per_category_iou' in gt_to_adv_miou:
                gt_per_iou_values = gt_to_adv_miou['per_category_iou']
                gt_per_iou_excluding_label0 = gt_per_iou_values[1:]  # Exclude index 0 (label 1~)
                gt_mean_iou_excluding_label0 = np.nanmean(gt_per_iou_excluding_label0).item()
                gt_to_adv_per_ious_excluding_label0.append(gt_mean_iou_excluding_label0)
            else:
                gt_to_adv_per_ious_excluding_label0.append(None)
        else:
            # For non-VOC2012 datasets, set to None
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
    
    # Add VOC2012 specific metrics if applicable
    if config["dataset"] == "VOC2012":
        final_results["Average mIoU excluding label 0 (benign)"] = benign_to_adv_per_ious_excluding_label0
        final_results["Average mIoU excluding label 0 (gt)"] = gt_to_adv_per_ious_excluding_label0

    print("\n--- Experiment Summary ---")
    print(final_results)
    
    save_experiment_results(final_results,
                            config,
                            sweep_config=None,
                            timestamp=current_time,
                            save_dir=base_dir
                            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Pixle attack evaluation using Robust models.")
    parser.add_argument("--config", type=str, required=True, 
                        choices=["pspnet_sat_voc", "pspnet_sat_city", "pspnet_vanilla_voc", "pspnet_vanilla_city",
                               "pspnet_ddcat_voc", "pspnet_ddcat_city",
                               "deeplabv3_sat_voc", "deeplabv3_sat_city", "deeplabv3_vanilla_voc", "deeplabv3_vanilla_city",
                               "deeplabv3_ddcat_voc", "deeplabv3_ddcat_city"],
                        help="Config file to use (without .py extension).")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    parser.add_argument('--attack_pixel', type=float, default=0.01, help='Ratio of adversarial pixels to total image pixels.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to evaluate from the dataset.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset directory (overrides config file if specified).')
    args = parser.parse_args()

    # Import config from the specified config file
    import importlib.util
    config_path = f"configs/{args.config}.py"
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Use the imported config as model_config
    model_config = config_module.config.copy()
    
    # Create runtime config with attack parameters
    config = {
        "dataset": model_config["dataset"],
        "num_class": model_config["num_class"],
        "device": args.device,
        "data_dir": args.data_dir if args.data_dir is not None else model_config["data_dir"],
        "attack_method": "Pixle",
        "attack_pixel": args.attack_pixel,
        "num_images": args.num_images,
        "verbose": args.verbose,
    }
    
    # Copy model-specific parameters
    config.update(model_config)
    config["base_dir"] = f"../data/{config['attack_method']}/results/{config['dataset']}/{config['model']}"
    
    main(config, model_config)