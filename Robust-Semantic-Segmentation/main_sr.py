import os
import torch
from tqdm import tqdm
import datetime
import numpy as np
from PIL import Image
import sys
import argparse

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from adv_setting.py
from adv_setting import model_predict, load_model, set_seed

# Import from parent directory
from dataset import CitySet, ADESet, VOCSet
from sparse_rs import RSAttack
from function import *
from evaluation import *
from utils import save_experiment_results
from config import voc, config_city


def load_config_from_yaml(config_path):
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_single_image_robust(args):
    """Process single image using Robust models instead of mmseg"""
    (img_bgr, filename, gt, model_config, config, base_dir, idx, total_images) = args
    
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

    # Create attack object
    attack = RSAttack(
        model=model,
        cfg=config,
        norm=config["norm"],
        n_queries=config["n_queries"],
        eps=config["eps"],
        p_init=config["p_init"],
        n_restarts=config["n_restarts"],
        seed=0,
        verbose=config.get("verbose", False),
        targeted=False,
        loss=config["loss"],
        resc_schedule=True,
        device=config["device"],
        log_path=None,
        original_img=img_bgr_np,
        d=5,
        use_decision_loss=config["use_decision_loss"]
    )

    adv_img_bgr_list = []
    total_queries = config["iters"] * config["n_queries"]
    save_steps = [int(total_queries * (i+1) / 5) for i in range(5)]
    
    for iter_idx in range(config["iters"]):
        current_query, adv_img_bgr = attack.perturb(img_tensor_bgr, gt_tensor)
        img_tensor_bgr = adv_img_bgr
        
        if current_query in save_steps:
            adv_img_bgr_list.append(adv_img_bgr)
    
    # Fill remaining save_steps if not reached
    while len(adv_img_bgr_list) < 5:
        adv_img_bgr_list.append(adv_img_bgr)

    # Save results
    current_img_save_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(current_img_save_dir, exist_ok=True)

    Image.fromarray(img_bgr_np[:, :, ::-1]).save(os.path.join(current_img_save_dir, "original.png"))

    print(f"file_name: {filename}")
    
    # Calculate metrics
    l0_metrics = []
    ratio_metrics = []
    impact_metrics = []
    
    for i, adv_img_bgr in enumerate(adv_img_bgr_list):
        query_img_save_dir = os.path.join(current_img_save_dir, f"{i+1}000query")
        os.makedirs(query_img_save_dir, exist_ok=True)

        adv_img_bgr_np = adv_img_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Get adversarial prediction using model_predict from adv_setting.py
        adv_confidence, adv_pred = model_predict(model, adv_img_bgr_np, model_config)
        adv_pred = adv_pred.cpu().numpy()
        
        delta_img = np.abs(img_bgr_np.astype(np.int16) - adv_img_bgr_np.astype(np.int16)).astype(np.uint8)
    
        Image.fromarray(adv_img_bgr_np[:, :, ::-1]).save(os.path.join(query_img_save_dir, "adv.png"))
        Image.fromarray(delta_img).save(os.path.join(query_img_save_dir, "delta.png"))
        
        # Visualize segmentation results
        visualize_segmentation(img_bgr_np, ori_pred,
                            save_path=os.path.join(query_img_save_dir, "ori_seg.png"),
                            alpha=0.5, dataset=config["dataset"])
        
        visualize_segmentation(img_bgr_np, ori_pred,
                            save_path=os.path.join(query_img_save_dir, "ori_seg_only.png"),
                            alpha=1, dataset=config["dataset"])
        
        visualize_segmentation(adv_img_bgr_np, adv_pred,
                            save_path=os.path.join(query_img_save_dir, "adv_seg.png"),
                            alpha=0.5, dataset=config["dataset"])
        
        visualize_segmentation(adv_img_bgr_np, adv_pred,
                            save_path=os.path.join(query_img_save_dir, "adv_seg_only.png"),
                            alpha=1, dataset=config["dataset"])
        
        l0_norm = calculate_l0_norm(img_bgr_np, adv_img_bgr_np)
        pixel_ratio = calculate_pixel_ratio(img_bgr_np, adv_img_bgr_np)
        impact = calculate_impact(img_bgr_np, adv_img_bgr_np, ori_pred, adv_pred)
        
        print(f"L0 norm: {l0_norm}, Pixel ratio: {pixel_ratio}, Impact: {impact}")

        l0_metrics.append(l0_norm)
        ratio_metrics.append(pixel_ratio)
        impact_metrics.append(impact)

    # Clean up memory
    del model
    del attack
    torch.cuda.empty_cache()
    
    return {
        'img_bgr': img_bgr_np,
        'gt': gt,
        'filename': filename,
        'adv_img_bgr_list': adv_img_bgr_list,
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
    experiment_name = f"{config['dataset']}_{config['model']}_sparse-rs_{current_time}"
    base_dir = os.path.join(config["base_dir"], current_time)
    os.makedirs(base_dir, exist_ok=True, mode=0o777)
    
    # Prepare data for processing
    process_args = []
    for idx, (img_bgr, filename, gt) in enumerate(zip(dataset.images, dataset.filenames, dataset.gt_images)):
        process_args.append((img_bgr, filename, gt, model_config, config, base_dir, idx, len(dataset.images)))

    # Sequential processing
    print(f"Sequential processing for {len(process_args)} images...")
    results = []
    for args in tqdm(process_args, total=len(process_args), desc="Running Sparse-RS Attack"):
        result = process_single_image_robust(args)
        results.append(result)
    
    # Collect results
    img_list = []
    gt_list = []
    filename_list = []
    adv_img_lists = [[] for _ in range(5)]
    all_l0_metrics = [[] for _ in range(5)] 
    all_ratio_metrics = [[] for _ in range(5)] 
    all_impact_metrics = [[] for _ in range(5)] 

    for result in results:
        img_list.append(result['img_bgr'])
        gt_list.append(result['gt'])
        filename_list.append(result['filename'])
        
        for i, adv_img_bgr in enumerate(result['adv_img_bgr_list']):
            adv_img_lists[i].append(adv_img_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            all_l0_metrics[i].append(result['l0_metrics'][i])
            all_ratio_metrics[i].append(result['ratio_metrics'][i])
            all_impact_metrics[i].append(result['impact_metrics'][i])

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
    
    for i in range(5):
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
    parser = argparse.ArgumentParser(description="Run Sparse-RS attack evaluation using Robust models.")
    parser.add_argument("--config", type=str, required=True, 
                        choices=["pspnet_sat_voc", "pspnet_sat_city", "pspnet_vanilla_voc", "pspnet_vanilla_city",
                               "deeplabv3_sat_voc", "deeplabv3_sat_city", "deeplabv3_vanilla_voc", "deeplabv3_vanilla_city"],
                        help="Config file to use (without .py extension).")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    parser.add_argument('--n_queries', type=int, default=10, help='Max number of queries for RSAttack.')
    parser.add_argument('--eps', type=float, default=0.0001, help='Epsilon for L0 norm in RSAttack.')
    parser.add_argument('--p_init', type=float, default=0.8, help='Initial probability p_init for RSAttack.')
    parser.add_argument('--n_restarts', type=int, default=1, help='Number of restarts for RSAttack.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to evaluate from the dataset.')
    parser.add_argument('--iters', type=int, default=500, help='Number of iterations for RSAttack.')
    parser.add_argument('--use_decision_loss', type=str, default='False', choices=['True', 'False'], help='Whether to use decision loss.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--norm', type=str, default='L0', choices=['L0', 'patches'], help='Norm for RSAttack.')
    parser.add_argument('--loss', type=str, default='prob', choices=['margin', 'prob'], help='Loss function for RSAttack.')
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
        "attack_method": "Sparse-RS",
        "n_queries": args.n_queries,
        "eps": args.eps,
        "attack_pixel": args.eps,
        "p_init": args.p_init,
        "n_restarts": args.n_restarts,
        "num_images": args.num_images,
        "iters": args.iters,
        "use_decision_loss": args.use_decision_loss.lower() == 'true',
        "verbose": args.verbose,
        "norm": args.norm,
        "loss": args.loss,
    }
    
    # Copy model-specific parameters
    config.update(model_config)
    config["base_dir"] = f"../data/{config['attack_method']}/results/{config['dataset']}/{config['model']}"
    
    main(config, model_config)