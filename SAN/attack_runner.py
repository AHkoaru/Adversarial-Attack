import sys
import os
import argparse
import torch
import torch.nn.functional as F # Added for resizing
import numpy as np
from tqdm import tqdm
from pathlib import Path
import datetime
import json
from PIL import Image
from detectron2.data import MetadataCatalog # Added for vocabulary retrieval
from detectron2.evaluation import SemSegEvaluator, DatasetEvaluators # Added for evaluation
import warnings

# Suppress harmless warnings during evaluation
warnings.filterwarnings("ignore", category=UserWarning, message="Downcasting array dtype")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

# Force wandb to offline mode
os.environ["WANDB_MODE"] = "offline"

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the directory containing the current file (SAN directory)
san_dir = os.path.dirname(current_file_path)
# Get the parent directory (workspace directory)
workspace_dir = os.path.dirname(san_dir)
# Get the mmsegmentation directory where dataset.py is located
mmseg_dir = os.path.join(workspace_dir, 'mmsegmentation')

# Insert workspace directory to sys.path at the beginning to prioritize local modules
# This ensures 'dataset.py' in workspace is found before any installed 'dataset' package
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

# Insert mmsegmentation directory to sys.path to find dataset.py, function.py, evaluation.py
if mmseg_dir not in sys.path:
    sys.path.insert(0, mmseg_dir)

# Insert SAN directory to sys.path
if san_dir not in sys.path:
    sys.path.insert(0, san_dir)

# Add workspace directory to sys.path for pixle.py
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

# --- Monkey Patch for CLIP VisionTransformer ---
# SAN expects a modified CLIP with 'output_tokens' attribute.
# If using standard CLIP, this attribute is missing. We patch it here.
try:
    import clip
    from clip.model import VisionTransformer
    if not hasattr(VisionTransformer, 'output_tokens'):
        print("Warning: 'output_tokens' attribute missing in VisionTransformer. Monkey-patching it to True for SAN compatibility.")
        setattr(VisionTransformer, 'output_tokens', True)
except ImportError:
    print("Warning: Could not import clip to patch VisionTransformer.")
# -----------------------------------------------

# Temporarily disable mmseg import in pixle.py by setting a flag
import sys
sys.modules['mmseg'] = None  # Prevent mmseg from being imported

from pixle import Pixle
from sparse_rs import RSAttack

# Remove the block to restore normal behavior
if 'mmseg' in sys.modules and sys.modules['mmseg'] is None:
    del sys.modules['mmseg']

from SAN.train_net import setup, Trainer
from function import visualize_segmentation
from evaluation import calculate_l0_norm, calculate_pixel_ratio, calculate_impact # eval_miou removed
from utils import save_experiment_results

# Import datasets
from dataset import ADESet, CitySet, VOCSet

class SANModelWrapper(torch.nn.Module):
    """
    Wrapper for SAN model to handle input format mismatch and preprocessing.
    """
    def __init__(self, model, vocabulary, target_short_edge=640):
        super().__init__()
        self.model = model
        self.vocabulary = vocabulary
        self.target_short_edge = target_short_edge
        # Expose attributes needed by attacks
        if hasattr(model, 'device'):
            self.device = model.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess(self, x):
        # x: (B, C, H, W) float tensor
        # Resize short edge to self.target_short_edge
        h, w = x.shape[2], x.shape[3]
        if h < w:
            new_h = self.target_short_edge
            new_w = int(w * self.target_short_edge / h)
        else:
            new_w = self.target_short_edge
            new_h = int(h * self.target_short_edge / w)
        
        # Use bilinear interpolation to match predict.py/train_net.py behavior
        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return x_resized

    def forward(self, x):
        inputs = []
        
        if isinstance(x, list):
            # Handle list input (e.g. from eval_miou)
            for item in x:
                if isinstance(item, dict):
                    # Already a dict (likely pre-processed by eval_miou for SED models)
                    # Ensure vocabulary is present
                    if "vocabulary" not in item:
                        item["vocabulary"] = self.vocabulary
                    inputs.append(item)
                elif isinstance(item, np.ndarray):
                    # item is (H, W, C) numpy array
                    h, w = item.shape[0], item.shape[1]
                    # Convert to (C, H, W) float tensor
                    img_t = torch.from_numpy(item.copy()).permute(2, 0, 1).float().to(self.device)
                    
                    # Resize for model input
                    img_t = img_t.unsqueeze(0)
                    img_t_resized = self.preprocess(img_t).squeeze(0)
                    
                    inputs.append({
                        "image": img_t_resized,
                        "height": h, # Pass original height
                        "width": w,  # Pass original width
                        "vocabulary": self.vocabulary
                    })
                elif isinstance(item, torch.Tensor):
                    # item is (C, H, W) tensor
                    h, w = item.shape[1], item.shape[2]
                    
                    # Resize
                    item = item.unsqueeze(0)
                    item_resized = self.preprocess(item).squeeze(0)
                    
                    inputs.append({
                        "image": item_resized.to(self.device),
                        "height": h,
                        "width": w,
                        "vocabulary": self.vocabulary
                    })
                else:
                    print(f"Warning: Unsupported item type in SANModelWrapper: {type(item)}")
        elif isinstance(x, torch.Tensor):
            # Handle tensor input (e.g. from Attack) -> (B, C, H, W)
            # x is expected to be RGB
            
            # 1. Resize for model input (Differentiable resizing for attack)
            x_resized = self.preprocess(x)
            
            for i in range(x.shape[0]):
                # Original size for output interpolation
                h_orig, w_orig = x.shape[2], x.shape[3]
                
                inputs.append({
                    "image": x_resized[i], # Resized image
                    "height": h_orig,      # Original height
                    "width": w_orig,       # Original width
                    "vocabulary": self.vocabulary
                })
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
            
        if not inputs:
            # Avoid crashing inside model if inputs is empty
            return []

        return self.model(inputs)
    
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

class AttackRunner:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = cfg.MODEL.DEVICE
        
        # Build Model using SAN Trainer logic
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
        
        # Setup Vocabulary and Wrapper
        self.vocabulary = self.get_vocabulary(self.dataset_name)
        print(f"Loaded vocabulary for {self.dataset_name}: {len(self.vocabulary)} classes")
        
        # Initialize Wrapper with target short edge (default 640 for SAN)
        self.wrapped_model = SANModelWrapper(self.model, self.vocabulary, target_short_edge=640)

        self.attack_type = args.attack_type
        
        # Setup Base Directory for Results (Matching px_eval/rs_eval structure)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(args.config_file))[0]
        # data/{AttackMethod}/results/{Dataset}/{ModelName}/{Timestamp}
        self.base_dir = os.path.join(workspace_dir, "data", self.attack_type, "results", self.dataset_name, model_name, timestamp)
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"Results will be saved to {self.base_dir}")
        
        # Redirect evaluator output to base_dir
        self.cfg.defrost()
        self.cfg.OUTPUT_DIR = self.base_dir
        self.cfg.freeze()
        
        # Save Config
        with open(os.path.join(self.base_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        print(f"Initialized AttackRunner for {self.attack_type} on {self.dataset_name}")

    def get_vocabulary(self, dataset_name):
        if dataset_name == 'cityscapes':
            return [
                "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                "truck", "bus", "train", "motorcycle", "bicycle"
            ]
        elif dataset_name == 'VOC2012':
            # Try to get from MetadataCatalog if registered
            try:
                # Use the correct key for VOC. 'ade20k_sem_seg_val' is for ADE20K.
                return MetadataCatalog.get("voc_sem_seg_val").stuff_classes
            except:
                # Fallback: VOC2012 has 20 classes. Background (0) is usually ignored in evaluation.
                # We align with Detectron2's standard VOC evaluation (0=aeroplane).
                return [
                    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
                ]
        elif dataset_name == 'ade20k':
            # Try to get from MetadataCatalog if registered, otherwise use hardcoded list
            try:
                return MetadataCatalog.get("ade20k_sem_seg_val").stuff_classes
            except:
                # Fallback to standard ADE20K classes (truncated for brevity, ensure full list is used in prod)
                # For now, assuming the environment registers it or we need to add the full list.
                # Using a placeholder warning if not found.
                print("Warning: ADE20K vocabulary not found in MetadataCatalog. Using generic placeholder.")
                return ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", 
                        "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", 
                        "door", "table", "mountain", "plant", "curtain", "chair", "car", 
                        "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", 
                        "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", 
                        "lamp", "bathtub", "railing", "cushion", "base", "box", "column", 
                        "signboard", "chest", "counter", "sand", "sink", "skyscraper", 
                        "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", 
                        "case", "pool", "pillow", "screen", "stairway", "river", "bridge", 
                        "bookcase", "blind", "coffee", "toilet", "flower", "book", "hill", 
                        "bench", "countertop", "stove", "palm", "kitchen", "computer", 
                        "swivel", "boat", "bar", "arcade", "hovel", "bus", "towel", "light", 
                        "truck", "tower", "chandelier", "awning", "streetlight", "booth", 
                        "television", "airplane", "dirt", "apparel", "pole", "land", "bannister", 
                        "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", 
                        "ship", "fountain", "conveyer", "canopy", "washer", "plaything", 
                        "swimming", "stool", "barrel", "basket", "waterfall", "tent", "bag", 
                        "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade", 
                        "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", 
                        "blanket", "sculpture", "hood", "sconce", "vase", "traffic", "tray", 
                        "ashcan", "fan", "pier", "crt", "plate", "monitor", "bulletin", "shower", 
                        "radiator", "glass", "clock", "flag"]
        else:
            raise ValueError(f"Unknown dataset for vocabulary: {dataset_name}")

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
            # Calculate patch size based on attack_pixel ratio (from px_eval.py)
            _, _, H, W = image_tensor.shape
            attack_pixel_ratio = attack_cfg.get('attack_pixel', 0.05)
            total_target_pixels_overall = H * W * attack_pixel_ratio
            pixels_per_single_patch_target = total_target_pixels_overall / attack_cfg.get('restarts', 20)
            
            # Patch size calculation logic (from px_eval.py)
            target_area_int = int(round(pixels_per_single_patch_target))
            h_found = 1  # Default value
            for h_candidate in range(int(np.sqrt(target_area_int)), 0, -1):
                if target_area_int % h_candidate == 0:
                    h_found = h_candidate
                    break
            patch_h_pixels = h_found
            patch_w_pixels = target_area_int // patch_h_pixels if patch_h_pixels > 0 else 1
            
            # Ensure minimum patch size
            patch_h_pixels = max(1, patch_h_pixels)
            patch_w_pixels = max(1, patch_w_pixels)
            
            # print(f"Pixle Attack: Image size ({H}x{W}), attack_pixel={attack_pixel_ratio}, "
            #       f"patch_size=({patch_h_pixels}x{patch_w_pixels})")
            
            attacker = Pixle(
                model=self.wrapped_model, # Use wrapped model
                x_dimensions=(patch_w_pixels, patch_w_pixels),
                y_dimensions=(patch_h_pixels, patch_h_pixels),
                restarts=attack_cfg.get('restarts', 20),
                max_iterations=attack_cfg.get('max_iterations', 10),
                threshold=attack_cfg.get('threshold', 21000),
                device=self.device,
                cfg=attack_cfg,
                is_mmseg_model=False,
                is_sed_model=True, # Flag for SAN (Detectron2) model
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
                model=self.wrapped_model, # Use wrapped model
                cfg=attack_cfg,
                eps=attack_cfg.get('eps', 0.05),
                n_queries=attack_cfg.get('n_queries', 5000),
                loss=attack_cfg.get('loss', 'decision_change'),
                is_mmseg_model=False,
                is_sed_model=True, # Flag for SAN (Detectron2) model
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
        filename_list = [] # Store filenames for evaluator
        adv_img_lists = [[] for _ in range(6)] # 0: Original, 1-5: Snapshots
        
        all_l0_metrics = [[] for _ in range(6)]
        all_ratio_metrics = [[] for _ in range(6)]
        all_impact_metrics = [[] for _ in range(6)]
        
        # Directory for saving benign predictions as GT for evaluation
        benign_gt_dir = os.path.join(self.base_dir, "benign_gt")
        os.makedirs(benign_gt_dir, exist_ok=True)

        for i, sample in tqdm(enumerate(self.dataset), total=min(len(self.dataset), num_images)):
            if i >= num_images:
                break
                
            img_np, filename, gt_np = sample
            filename_list.append(filename)
            
            # Preprocess: numpy (H, W, C) -> tensor (1, C, H, W)
            # IMPORTANT: Convert BGR to RGB for SAN model input
            img_rgb_np = img_np[:, :, ::-1].copy()
            img_tensor = torch.from_numpy(img_rgb_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device)
            gt_tensor = torch.from_numpy(gt_np.copy()).unsqueeze(0).long().to(self.device)
            
            # Fix for VOC2012: Shift labels to match 20-class model (0=aeroplane)
            # Original GT: 0=background, 1=aeroplane, ...
            # Target GT: 255=background, 0=aeroplane, ...
            if self.dataset_name == 'VOC2012':
                gt_tensor = gt_tensor - 1
                gt_tensor[gt_tensor == -1] = 255
            
            try:
                # Prepare Image Directory
                image_name = os.path.splitext(os.path.basename(filename))[0]
                curr_img_dir = os.path.join(self.base_dir, image_name)
                os.makedirs(curr_img_dir, exist_ok=True)

                # Save Original Image (RGB)
                # img_np is BGR, so convert to RGB for saving
                img_rgb = img_np[:, :, ::-1].copy() # Ensure positive strides
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
                    # Use wrapped model for prediction to handle input format
                    outputs = self.wrapped_model(img_tensor)
                    ori_pred = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()

                # Save Benign Prediction as GT for Benign Evaluation
                # Force PNG extension for masks
                benign_save_path = os.path.join(benign_gt_dir, image_name + ".png")
                Image.fromarray(ori_pred.astype(np.uint8)).save(benign_save_path)

                # Attack config
                attack_cfg = {
                    'restarts': self.args.restarts,
                    'max_iterations': self.args.max_iterations,
                    'threshold': self.args.threshold,
                    'attack_pixel': self.args.attack_pixel,
                    'eps': self.args.eps,
                    'n_queries': self.args.n_queries,
                    'loss': self.args.loss,
                    'dataset': self.dataset_name,
                    'iters': self.args.iters 
                }
                
                # Run Attack
                results = self.run_attack(img_tensor, gt_tensor, attack_cfg)
                
                # Save Results
                for i_snap, (query_label, adv_tensor) in enumerate(results):
                    query_dir = os.path.join(curr_img_dir, f"{query_label}query")
                    os.makedirs(query_dir, exist_ok=True)
                    
                    # Postprocess Adv Image
                    adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    # adv_np is RGB because img_tensor was RGB
                    
                    # Save Adv Image
                    Image.fromarray(adv_np).save(os.path.join(query_dir, "adv.png"))
                    
                    # Save Delta
                    delta = np.abs(img_rgb.astype(int) - adv_np.astype(int)).astype(np.uint8)
                    Image.fromarray(delta).save(os.path.join(query_dir, "delta.png"))
                    
                    # Get Adv Prediction
                    with torch.no_grad():
                        # Use wrapped model
                        outputs = self.wrapped_model(adv_tensor)
                        adv_pred = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()
                    
                    # Save Visualizations
                    visualize_segmentation(img_rgb, ori_pred, os.path.join(query_dir, "ori_seg.png"), alpha=0.5, dataset=self.dataset_name)
                    visualize_segmentation(img_rgb, ori_pred, os.path.join(query_dir, "ori_seg_only.png"), alpha=1.0, dataset=self.dataset_name)
                    visualize_segmentation(adv_np, adv_pred, os.path.join(query_dir, "adv_seg.png"), alpha=0.5, dataset=self.dataset_name)
                    visualize_segmentation(adv_np, adv_pred, os.path.join(query_dir, "adv_seg_only.png"), alpha=1.0, dataset=self.dataset_name)

                    # Calculate Metrics
                    # Note: calculate_l0_norm etc might expect BGR or RGB. Assuming they handle numpy arrays.
                    # img_np is BGR. adv_np is RGB.
                    # We should convert adv_np to BGR to match img_np for metrics if needed, or convert img_np to RGB.
                    # Let's use RGB for both.
                    l0 = calculate_l0_norm(img_rgb, adv_np)
                    ratio = calculate_pixel_ratio(img_rgb, adv_np)
                    impact = calculate_impact(img_rgb, adv_np, ori_pred, adv_pred)
                    
                    # Store (Map to 1-5)
                    if i_snap < 5:
                        # Store BGR for consistency with img_list if needed by evaluator?
                        # run_eval takes img_list which is BGR.
                        # So we should store BGR in adv_img_lists.
                        adv_bgr = adv_np[:, :, ::-1].copy() # Ensure positive strides
                        adv_img_lists[i_snap+1].append(adv_bgr)
                        
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
        print("Calculating aggregate metrics using Detectron2 Evaluator...")
        
        # Helper to extract metrics from Detectron2 results
        def extract_metrics(results, vocabulary):
            metrics = {}
            if "sem_seg" in results:
                res = results["sem_seg"]
                metrics['mIoU'] = float(res.get("mIoU", 0.0))
                metrics['mACC'] = float(res.get("mACC", 0.0))
                metrics['pACC'] = float(res.get("pACC", 0.0))
                
                iou_list = []
                for name in vocabulary:
                    key = f"IoU-{name}"
                    val = res.get(key, float('nan'))
                    iou_list.append(float(val) if isinstance(val, (np.floating, float)) else val)
                metrics['per_category_iou'] = iou_list
            else:
                metrics['mIoU'] = 0.0
                metrics['mACC'] = 0.0
                metrics['pACC'] = 0.0
                metrics['per_category_iou'] = []
            return metrics

        # Helper to run evaluation
        def run_eval(images, filenames, mode='gt'):
            dataset_name_cfg = self.cfg.DATASETS.TEST[0]
            
            if mode == 'gt':
                # Use Trainer's builder for GT evaluation (handles dataset specifics)
                evaluator = Trainer.build_evaluator(self.cfg, dataset_name_cfg)
            else:
                # For Benign evaluation, use generic SemSegEvaluator
                # We use the dataset name to get metadata (classes, ignore_label)
                evaluator = SemSegEvaluator(dataset_name_cfg, distributed=False, output_dir=None)

            # Unwrap DatasetEvaluators if needed to access SemSegEvaluator internals
            target_evaluator = evaluator
            if isinstance(evaluator, DatasetEvaluators):
                for e in evaluator._evaluators:
                    if isinstance(e, SemSegEvaluator):
                        target_evaluator = e
                        break
            
            target_evaluator.reset()
            
            # Create mapping from basename to key
            basename_to_key = {}
            if hasattr(target_evaluator, 'input_file_to_gt_file'):
                basename_to_key = {os.path.basename(k): k for k in target_evaluator.input_file_to_gt_file.keys()}

            # If Benign mode, patch the GT paths
            if mode == 'benign':
                if hasattr(target_evaluator, 'input_file_to_gt_file'):
                    for fname in filenames:
                        basename = os.path.basename(fname)
                        if basename in basename_to_key:
                            key = basename_to_key[basename]
                            # Point to saved benign prediction
                            image_name = os.path.splitext(basename)[0]
                            benign_path = os.path.join(self.base_dir, "benign_gt", image_name + ".png")
                            target_evaluator.input_file_to_gt_file[key] = benign_path

            with torch.no_grad():
                for img_np, fname in zip(images, filenames):
                    h, w = img_np.shape[:2]
                    # img_np is BGR (from img_list/adv_img_lists)
                    # Convert to RGB for model input
                    img_rgb_np = img_np[:, :, ::-1].copy()
                    img_tensor = torch.from_numpy(img_rgb_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device)
                    
                    # Resize happens inside SANModelWrapper now
                    
                    fname_key = fname
                    # Fix filename mismatch
                    if hasattr(target_evaluator, 'input_file_to_gt_file'):
                        if fname not in target_evaluator.input_file_to_gt_file:
                            basename = os.path.basename(fname)
                            if basename in basename_to_key:
                                fname_key = basename_to_key[basename]
                            else:
                                # Skip if not found in registry (should not happen if dataset matches)
                                continue

                    inputs = [{
                        "image": img_tensor.squeeze(0), # Wrapper expects tensor, but here we pass to model directly? 
                        # Wait, self.model is Trainer.build_model(cfg). It expects list of dicts.
                        # self.wrapped_model is SANModelWrapper.
                        # We should use self.wrapped_model to handle resizing!
                        # But wrapped_model.forward expects tensor or list.
                        # Let's pass tensor to wrapped_model.
                    }]
                    
                    # Re-construct inputs for wrapped_model
                    # wrapped_model handles resizing internally if we pass tensor
                    outputs = self.wrapped_model(img_tensor)
                    
                    # Inputs for evaluator need to be what evaluator expects.
                    # Usually original image size.
                    # Fix ValueError: At least one stride in the given numpy array is negative
                    eval_inputs = [{
                        "image": torch.as_tensor(np.ascontiguousarray(img_np.transpose(2, 0, 1))), # Evaluator might not use image content but metadata
                        "height": h,
                        "width": w,
                        "file_name": fname_key
                    }]
                    
                    target_evaluator.process(eval_inputs, outputs)
            
            return target_evaluator.evaluate()

        # Calculate query labels
        total_queries = 0
        if self.args.attack_type == 'pixle':
             total_queries = self.args.restarts * self.args.max_iterations
        elif self.args.attack_type == 'sparse_rs':
             total_queries = self.args.iters * self.args.n_queries
        
        query_labels = []
        query_labels.append("0query")
        for i in range(1, 6):
            q = int(total_queries * i / 5)
            query_labels.append(f"{q}query")

        # Init Evaluation
        print("Evaluating Initial State (GT)...")
        init_res_gt = run_eval(img_list, filename_list, mode='gt')
        init_metrics_gt = extract_metrics(init_res_gt, self.vocabulary)
        
        print("Evaluating Initial State (Benign)...")
        # For initial state, Benign vs Benign should be perfect (mIoU 1.0)
        init_res_benign = run_eval(img_list, filename_list, mode='benign')
        init_metrics_benign = extract_metrics(init_res_benign, self.vocabulary)

        # Initialize result lists
        gt_to_adv_mious = [init_metrics_gt['mIoU']]
        gt_mean_accuracy = [init_metrics_gt['mACC']]
        gt_overall_accuracy = [init_metrics_gt['pACC']]
        
        benign_to_adv_mious = [init_metrics_benign['mIoU']]
        benign_mean_accuracy = [init_metrics_benign['mACC']]
        benign_overall_accuracy = [init_metrics_benign['pACC']]
        
        per_category_iou_gt = {query_labels[0]: init_metrics_gt['per_category_iou']}
        per_category_iou_benign = {query_labels[0]: init_metrics_benign['per_category_iou']}

        mean_l0 = [0.0]
        mean_ratio = [0.0]
        mean_impact = [0.0]

        for i in range(1, 6):
            print(f"Evaluating snapshot {i}...")
            
            # GT Evaluation
            res_gt = run_eval(adv_img_lists[i], filename_list, mode='gt')
            metrics_gt = extract_metrics(res_gt, self.vocabulary)
            
            gt_to_adv_mious.append(metrics_gt['mIoU'])
            gt_mean_accuracy.append(metrics_gt['mACC'])
            gt_overall_accuracy.append(metrics_gt['pACC'])
            per_category_iou_gt[query_labels[i]] = metrics_gt['per_category_iou']
            
            # Benign Evaluation
            res_benign = run_eval(adv_img_lists[i], filename_list, mode='benign')
            metrics_benign = extract_metrics(res_benign, self.vocabulary)
            
            benign_to_adv_mious.append(metrics_benign['mIoU'])
            benign_mean_accuracy.append(metrics_benign['mACC'])
            benign_overall_accuracy.append(metrics_benign['pACC'])
            per_category_iou_benign[query_labels[i]] = metrics_benign['per_category_iou']
            
            mean_l0.append(np.mean(all_l0_metrics[i]).item())
            mean_ratio.append(np.mean(all_ratio_metrics[i]).item())
            mean_impact.append(np.mean(all_impact_metrics[i]).item())

        # Helper to format dictionary with newlines for each query
        def format_iou_dict(d):
            lines = []
            for k, v in d.items():
                lines.append(f'  {k}: {v}')
            return "\n" + "\n".join(lines)

        final_results = {
            "Init mIoU": init_metrics_gt['mIoU'],
            "Adversarial mIoU(benign)": benign_to_adv_mious,
            "Adversarial mIoU(gt)": gt_to_adv_mious,
            "Accuracy(benign)": benign_mean_accuracy,
            "Overall Accuracy(benign)": benign_overall_accuracy,
            "Accuracy(gt)": gt_mean_accuracy,
            "Overall Accuracy(gt)": gt_overall_accuracy,
            "L0": mean_l0,
            "Ratio": mean_ratio,
            "Impact": mean_impact,
            "Per-category IoU(benign)": format_iou_dict(per_category_iou_benign),
            "Per-category IoU(gt)": format_iou_dict(per_category_iou_gt),
        }
        
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
        
        # Post-processing to fix formatting of Per-category IoU in the saved file
        try:
            results_file = os.path.join(self.base_dir, f"experiment_results_{os.path.basename(self.base_dir)}.txt")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    content = f.read()
                
                formatted_benign = format_iou_dict(per_category_iou_benign)
                formatted_gt = format_iou_dict(per_category_iou_gt)
                
                # json.dumps escapes newlines and adds quotes, which save_experiment_results likely did
                dumped_benign = json.dumps(formatted_benign)
                dumped_gt = json.dumps(formatted_gt)
                
                # Replace the dumped string with the raw formatted string
                if dumped_benign in content:
                    content = content.replace(dumped_benign, formatted_benign)
                if dumped_gt in content:
                    content = content.replace(dumped_gt, formatted_gt)
                    
                with open(results_file, 'w') as f:
                    f.write(content)
                print(f"Reformatted results file for readability: {results_file}")
        except Exception as e:
            print(f"Warning: Failed to reformat results file: {e}")

        print(f"Experiment results saved in: {self.base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAN Attack Runner")
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
    parser.add_argument("--restarts", type=int, default=500)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=21000)
    parser.add_argument("--attack-pixel", type=float, default=0.05, help="Ratio of adversarial pixels to total image pixels.")
    
    # Sparse-RS args
    parser.add_argument("--eps", type=float, default=0.0001)
    parser.add_argument("--n-queries", type=int, default=10) # Default per iter
    parser.add_argument("--iters", type=int, default=500) # Added iters argument
    parser.add_argument("--loss", type=str, default="prob", choices=["prob", "decision_change", "decision"])

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
        args.opts.extend(['DATASETS.TEST', "('ade20k_sem_seg_val',)"])
    elif args.dataset_name == 'cityscapes':
        print("Forcing NUM_CLASSES to 19 for Cityscapes")
        args.opts.extend(['MODEL.SEM_SEG_HEAD.NUM_CLASSES', '19'])
        args.opts.extend(['DATASETS.TEST', "('cityscapes_fine_sem_seg_val',)"])
    elif args.dataset_name == 'VOC2012':
        print("Forcing NUM_CLASSES to 20 for VOC2012")
        args.opts.extend(['MODEL.SEM_SEG_HEAD.NUM_CLASSES', '20'])
        args.opts.extend(['DATASETS.TEST', "('voc_sem_seg_val',)"])

    cfg = setup(args)
    
    runner = AttackRunner(cfg, args)
    runner.run()
