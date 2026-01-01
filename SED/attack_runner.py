import sys
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import datetime
import json
from PIL import Image

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the directory containing the current file (SED directory)
sed_dir = os.path.dirname(current_file_path)
# Get the parent directory (workspace directory)
workspace_dir = os.path.dirname(sed_dir)
# Get the mmsegmentation directory where dataset.py is located
mmseg_dir = os.path.join(workspace_dir, 'mmsegmentation')

# Insert workspace directory to sys.path at the beginning to prioritize local modules
# This ensures 'dataset.py' in workspace is found before any installed 'dataset' package
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

# Insert mmsegmentation directory to sys.path to find dataset.py, function.py, evaluation.py
if mmseg_dir not in sys.path:
    sys.path.insert(0, mmseg_dir)

# Insert SED directory to sys.path
if sed_dir not in sys.path:
    sys.path.insert(0, sed_dir)

from pixle import Pixle
from sparse_rs import RSAttack
from SED.train_net import setup, Trainer
from function import visualize_segmentation
from evaluation import calculate_l0_norm, calculate_pixel_ratio, calculate_impact, eval_miou
from utils import save_experiment_results

# Import datasets
from dataset import ADESet, CitySet, VOCSet

class AttackRunner:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = cfg.MODEL.DEVICE
        
        # Build Model using SED Trainer logic
        print("Building SED model...")
        self.model = Trainer.build_model(cfg)
        self.model.eval()
        
        # Load Checkpoint
        from detectron2.checkpoint import DetectionCheckpointer
        print(f"Loading checkpoint from {cfg.MODEL.WEIGHTS}...")
        DetectionCheckpointer(self.model).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        
        # Build Dataset
        self.dataset_name = args.dataset_name
        
        # Resolve data_dir path
        data_dir = args.data_dir
        if not os.path.exists(data_dir):
            # Try relative to workspace root
            potential_path = os.path.join(workspace_dir, data_dir)
            if os.path.exists(potential_path):
                print(f"Data directory not found at {data_dir}. Resolved to: {potential_path}")
                data_dir = potential_path
        
        self.dataset = self.build_dataset(self.dataset_name, data_dir)
        
        self.attack_type = args.attack_type
        
        # Setup Base Directory for Results (Matching px_eval/rs_eval structure)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(args.config_file))[0]
        # data/{AttackMethod}/results/{Dataset}/{ModelName}/{Timestamp}
        self.base_dir = os.path.join(workspace_dir, "data", self.attack_type, "results", self.dataset_name, model_name, timestamp)
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"Results will be saved to {self.base_dir}")
        
        # Save Config
        with open(os.path.join(self.base_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        print(f"Initialized AttackRunner for {self.attack_type} on {self.dataset_name}")

    def build_dataset(self, name, data_dir):
        print(f"Loading dataset {name} from {data_dir}...")
        if name == 'ade20k':
            return ADESet(data_dir)
        elif name == 'cityscapes':
            return CitySet(data_dir)
        elif name == 'VOC2012':
            return VOCSet(data_dir)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def run_attack(self, image_tensor, label_tensor, attack_cfg):
        """
        Run the selected attack.
        image_tensor: (1, C, H, W) tensor
        label_tensor: (1, H, W) tensor
        Returns: List of (query_label, adv_tensor) tuples
        """
        if self.attack_type == 'pixle':
            attacker = Pixle(
                model=self.model,
                restarts=attack_cfg.get('restarts', 20),
                max_iterations=attack_cfg.get('max_iterations', 10),
                threshold=attack_cfg.get('threshold', 2250),
                device=self.device,
                cfg=attack_cfg,
                is_mmseg_model=False,
                is_sed_model=True, # Flag for SED (Detectron2) model
                loss=attack_cfg.get('loss', 'prob')
            )
            
            # Pixle handles loops internally
            res = attacker.forward(image_tensor, label_tensor)
            
            # Pixle returns dict with 'adv_images' (list) and 'query' (list)
            # We map these snapshots to milestone query counts (e.g. 1000, 2000...) 
            # to ensure consistent folder naming like px_eval.py
            snapshots = res['adv_images']
            total_queries = attack_cfg['restarts'] * attack_cfg['max_iterations']
            num_snapshots = len(snapshots)
            
            results = []
            for i, adv_img in enumerate(snapshots):
                # Calculate milestone query count (e.g. 1000, 2000...)
                q_label = int(total_queries * (i + 1) / num_snapshots)
                results.append((q_label, adv_img))
            
            return results
            
        elif self.attack_type == 'sparse_rs':
            attacker = RSAttack(
                model=self.model,
                cfg=attack_cfg,
                eps=attack_cfg.get('eps', 0.05),
                n_queries=attack_cfg.get('n_queries', 5000),
                loss=attack_cfg.get('loss', 'decision_change'),
                is_mmseg_model=False,
                is_sed_model=True, # Flag for SED (Detectron2) model
                enable_success_reporting=True,
                device=self.device
            )
            
            # Sparse-RS loop structure matching rs_eval.py
            results = []
            curr_img = image_tensor
            
            n_iters = attack_cfg.get('iters', 500)
            n_queries_per_iter = attack_cfg.get('n_queries', 10)
            total_queries = n_iters * n_queries_per_iter
            
            # Define save milestones (5 snapshots)
            milestones = [int(total_queries * (i+1) / 5) for i in range(5)]
            next_milestone_idx = 0
            
            for i in range(n_iters):
                # Run one iteration of perturbation
                queries, adv_img, success = attacker.perturb(curr_img, label_tensor)
                curr_img = adv_img # Update image for next iteration
                
                # Check if we reached a milestone
                # attacker.current_query accumulates across calls
                current_total_q = attacker.current_query
                
                if next_milestone_idx < len(milestones) and current_total_q >= milestones[next_milestone_idx]:
                    results.append((milestones[next_milestone_idx], adv_img.clone()))
                    next_milestone_idx += 1
            
            return results
        
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def run(self):
        num_images = self.args.num_images
        print(f"Starting attack on {num_images} images...")
        
        # Initialize lists for metrics
        img_list = []
        gt_list = []
        adv_img_lists = [[] for _ in range(6)] # 0: Original, 1-5: Snapshots
        
        all_l0_metrics = [[] for _ in range(6)]
        all_ratio_metrics = [[] for _ in range(6)]
        all_impact_metrics = [[] for _ in range(6)]
        
        for i, sample in tqdm(enumerate(self.dataset), total=min(len(self.dataset), num_images)):
            if i >= num_images:
                break
                
            img_np, filename, gt_np = sample
            
            # Preprocess: numpy (H, W, C) -> tensor (1, C, H, W)
            img_tensor = torch.from_numpy(img_np.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device)
            gt_tensor = torch.from_numpy(gt_np.copy()).unsqueeze(0).long().to(self.device)
            
            try:
                # Prepare Image Directory
                image_name = os.path.splitext(os.path.basename(filename))[0]
                curr_img_dir = os.path.join(self.base_dir, image_name)
                os.makedirs(curr_img_dir, exist_ok=True)

                # Save Original Image (RGB)
                # Assuming img_np is BGR (standard for these datasets in this codebase)
                img_rgb = img_np[:, :, ::-1]
                Image.fromarray(img_rgb).save(os.path.join(curr_img_dir, "original.png"))
                
                # Save GT
                Image.fromarray(gt_np).save(os.path.join(curr_img_dir, "gt.png"))

                # Store Original Data for Evaluation
                img_list.append(img_np)
                gt_list.append(gt_np)
                
                # 0-th snapshot (Original)
                adv_img_lists[0].append(img_np)
                all_l0_metrics[0].append(0)
                all_ratio_metrics[0].append(0.0)
                all_impact_metrics[0].append(0.0)

                # Get Original Prediction for Visualization and Impact
                with torch.no_grad():
                    inputs = [{"image": img_tensor.squeeze(0)}]
                    outputs = self.model(inputs)
                    ori_pred = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()

                # Attack config
                attack_cfg = {
                    'restarts': self.args.restarts,
                    'max_iterations': self.args.max_iterations,
                    'threshold': self.args.threshold,
                    'eps': self.args.eps,
                    'n_queries': self.args.n_queries,
                    'loss': self.args.loss,
                    'dataset': self.dataset_name,
                    'iters': self.args.iters # Added iters for Sparse-RS
                }
                
                # Run Attack
                results = self.run_attack(img_tensor, gt_tensor, attack_cfg)
                
                # Save Results
                for i_snap, (query_label, adv_tensor) in enumerate(results):
                    query_dir = os.path.join(curr_img_dir, f"{query_label}query")
                    os.makedirs(query_dir, exist_ok=True)
                    
                    # Postprocess Adv Image
                    adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    adv_rgb = adv_np[:, :, ::-1] # BGR to RGB
                    
                    # Save Adv Image
                    Image.fromarray(adv_rgb).save(os.path.join(query_dir, "adv.png"))
                    
                    # Save Delta
                    delta = np.abs(img_rgb.astype(int) - adv_rgb.astype(int)).astype(np.uint8)
                    Image.fromarray(delta).save(os.path.join(query_dir, "delta.png"))
                    
                    # Get Adv Prediction
                    with torch.no_grad():
                        inputs = [{"image": adv_tensor.squeeze(0)}]
                        outputs = self.model(inputs)
                        adv_pred = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()
                    
                    # Save Visualizations
                    visualize_segmentation(img_rgb, ori_pred, os.path.join(query_dir, "ori_seg.png"), alpha=0.5, dataset=self.dataset_name)
                    visualize_segmentation(img_rgb, ori_pred, os.path.join(query_dir, "ori_seg_only.png"), alpha=1.0, dataset=self.dataset_name)
                    visualize_segmentation(adv_rgb, adv_pred, os.path.join(query_dir, "adv_seg.png"), alpha=0.5, dataset=self.dataset_name)
                    visualize_segmentation(adv_rgb, adv_pred, os.path.join(query_dir, "adv_seg_only.png"), alpha=1.0, dataset=self.dataset_name)

                    # Calculate Metrics
                    l0 = calculate_l0_norm(img_np, adv_np)
                    ratio = calculate_pixel_ratio(img_np, adv_np)
                    impact = calculate_impact(img_np, adv_np, ori_pred, adv_pred)
                    
                    # Store (Map to 1-5)
                    if i_snap < 5:
                        adv_img_lists[i_snap+1].append(adv_np)
                        all_l0_metrics[i_snap+1].append(l0)
                        all_ratio_metrics[i_snap+1].append(ratio)
                        all_impact_metrics[i_snap+1].append(impact)
                
            except Exception as e:
                print(f"Error attacking image {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
        print(f"Attack finished. Results saved to {self.base_dir}")
        
        # --- Evaluation Phase ---
        print("Calculating aggregate metrics...")
        
        # Prepare config for eval_miou
        eval_config = {
            "num_class": self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "dataset": self.dataset_name,
            "device": self.device,
            "is_sed_model": True # Flag for eval_miou to handle SED model
        }

        # Init mIoU (Original vs GT)
        print("Evaluating Initial State...")
        _, init_miou_dict = eval_miou(self.model, img_list, img_list, gt_list, eval_config)
        init_miou = init_miou_dict['mean_iou'].item() if hasattr(init_miou_dict['mean_iou'], 'item') else init_miou_dict['mean_iou']

        benign_to_adv_mious = []
        gt_to_adv_mious = []
        gt_mean_accuracy = []
        gt_overall_accuracy = []
        benign_mean_accuracy = []
        benign_overall_accuracy = []
        benign_to_adv_per_ious = []
        gt_to_adv_per_ious = []
        benign_to_adv_per_ious_excluding_label0 = []
        gt_to_adv_per_ious_excluding_label0 = []
        mean_l0 = []
        mean_ratio = []
        mean_impact = []

        for i in range(6):
            print(f"Evaluating snapshot {i}...")
            benign_res, gt_res = eval_miou(self.model, img_list, adv_img_lists[i], gt_list, eval_config)
            
            gt_to_adv_mious.append(gt_res['mean_iou'].item() if hasattr(gt_res['mean_iou'], 'item') else gt_res['mean_iou'])
            gt_mean_accuracy.append(gt_res['mean_accuracy'].item() if hasattr(gt_res['mean_accuracy'], 'item') else gt_res['mean_accuracy'])
            gt_overall_accuracy.append(gt_res['overall_accuracy'].item() if hasattr(gt_res['overall_accuracy'], 'item') else gt_res['overall_accuracy'])
            
            benign_to_adv_mious.append(benign_res['mean_iou'].item() if hasattr(benign_res['mean_iou'], 'item') else benign_res['mean_iou'])
            benign_mean_accuracy.append(benign_res['mean_accuracy'].item() if hasattr(benign_res['mean_accuracy'], 'item') else benign_res['mean_accuracy'])
            benign_overall_accuracy.append(benign_res['overall_accuracy'].item() if hasattr(benign_res['overall_accuracy'], 'item') else benign_res['overall_accuracy'])
            
            if 'per_category_iou' in gt_res:
                gt_to_adv_per_ious.append(gt_res['per_category_iou'].tolist())
            else:
                gt_to_adv_per_ious.append(None)
                
            if 'per_category_iou' in benign_res:
                benign_to_adv_per_ious.append(benign_res['per_category_iou'].tolist())
            else:
                benign_to_adv_per_ious.append(None)

            if self.dataset_name == "VOC2012":
                if 'per_category_iou' in benign_res:
                    benign_to_adv_per_ious_excluding_label0.append(np.nanmean(benign_res['per_category_iou'][1:]).item())
                else:
                    benign_to_adv_per_ious_excluding_label0.append(None)
                if 'per_category_iou' in gt_res:
                    gt_to_adv_per_ious_excluding_label0.append(np.nanmean(gt_res['per_category_iou'][1:]).item())
                else:
                    gt_to_adv_per_ious_excluding_label0.append(None)
            else:
                benign_to_adv_per_ious_excluding_label0.append(None)
                gt_to_adv_per_ious_excluding_label0.append(None)
            
            mean_l0.append(np.mean(all_l0_metrics[i]).item())
            mean_ratio.append(np.mean(all_ratio_metrics[i]).item())
            mean_impact.append(np.mean(all_impact_metrics[i]).item())

        final_results = {
            "Init mIoU" : init_miou,
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
        
        if self.dataset_name == "VOC2012":
            final_results["Average mIoU excluding label 0 (benign)"] = benign_to_adv_per_ious_excluding_label0
            final_results["Average mIoU excluding label 0 (gt)"] = gt_to_adv_per_ious_excluding_label0
            
        print("\n--- Experiment Summary ---")
        print(final_results)
        
        config_dict = vars(self.args)
        config_dict.update({
            "dataset": self.dataset_name,
            "model": os.path.splitext(os.path.basename(self.args.config_file))[0],
            "base_dir": self.base_dir,
            "num_class": self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        })
        
        save_experiment_results(
            final_results,
            config_dict,
            sweep_config=None,
            timestamp=os.path.basename(self.base_dir),
            save_dir=self.base_dir
        )
        print(f"Experiment results saved in: {self.base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SED Attack Runner")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true", help="whether to attempt to resume from the checkpoint directory")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    parser.add_argument("--dist-url", default="auto", help="dist url for init process group")
    
    # Attack specific args
    parser.add_argument("--attack-type", type=str, default="pixle", choices=["pixle", "sparse_rs"])
    parser.add_argument("--dataset-name", type=str, default="cityscapes")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=10)
    
    # Pixle args
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=2250)
    
    # Sparse-RS args
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--n-queries", type=int, default=10) # Default per iter
    parser.add_argument("--iters", type=int, default=500) # Added iters argument
    parser.add_argument("--loss", type=str, default="decision_change")

    # Opts must be last to capture config overrides
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    # Resolve config file path if it doesn't exist
    if args.config_file and not os.path.exists(args.config_file):
        # Try relative to workspace root
        potential_path = os.path.join(workspace_dir, args.config_file)
        if os.path.exists(potential_path) and os.path.isfile(potential_path):
            print(f"Config file not found at {args.config_file}. Resolved to: {potential_path}")
            args.config_file = potential_path

    # Ensure args.opts is a list
    if args.opts is None:
        args.opts = []

    # Automatically set NUM_CLASSES based on dataset to avoid mismatch
    # The config file might have 171 classes (COCO-Stuff), but we are evaluating on ADE20K (150 classes).
    if args.dataset_name == 'ade20k':
        print("Forcing NUM_CLASSES to 150 for ADE20K")
        args.opts.extend(['MODEL.SEM_SEG_HEAD.NUM_CLASSES', '150'])
    elif args.dataset_name == 'cityscapes':
        print("Forcing NUM_CLASSES to 19 for Cityscapes")
        args.opts.extend(['MODEL.SEM_SEG_HEAD.NUM_CLASSES', '19'])
    elif args.dataset_name == 'VOC2012':
        print("Forcing NUM_CLASSES to 21 for VOC2012")
        args.opts.extend(['MODEL.SEM_SEG_HEAD.NUM_CLASSES', '21'])

    cfg = setup(args)
    
    runner = AttackRunner(cfg, args)
    runner.run()
