import torch
import numpy as np
import os
import PIL.Image as Image
from typing import List
from torchvision import transforms
import os.path as osp
import cv2

def data_preprocessing(
    np_image_list: List[np.ndarray],
    transform: transforms.Compose
) -> torch.Tensor:
    """
    Apply torchvision transform to a list of numpy arrays and return as a tensor batch.

    Args:
        np_image_list: List of RGB numpy arrays (each image: (H, W, 3), dtype=np.uint8)
        transform: torchvision.transforms.Compose object

    Returns:
        torch.Tensor: Transformed image tensor batch (shape: [N, C, H, W])
    """
    tensor_list = []
    for np_img in np_image_list:
        if not isinstance(np_img, np.ndarray):
            raise TypeError("The input list should only contain numpy arrays.")
        if np_img.ndim != 3 or np_img.shape[2] != 3:
            raise ValueError("Each image array should be in (H, W, 3) format of RGB image.")
        if np_img.dtype != np.uint8:
            raise ValueError("Image dtype should be np.uint8.")

        pil_img = Image.fromarray(np_img)
        tensor_img = transform(pil_img)
        tensor_list.append(tensor_img)

    return torch.stack(tensor_list)

# Cityscapes original label to trainId mapping dictionary
CITYSCAPES_LABEL_MAPPING = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18   # bicycle
}

def convert_gt_labels(gt_array):
    """
    Convert GT label array (gtFine_labelIds) to trainId used for Cityscapes evaluation.
    Unlabeled labels are processed as ignore index 255.
    """
    ignore_val = 255
    converted = np.full_like(gt_array, ignore_val)
    for orig_label, train_label in CITYSCAPES_LABEL_MAPPING.items():
        converted[gt_array == orig_label] = train_label
    return converted

# Color mapping for Cityscapes classes
CITYSCAPES_COLORMAP = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32)     # bicycle
}

ADE20K_COLORMAP = {
    0: (0, 0, 0),            # background / unlabeled
    1: (120, 120, 120),      # wall
    2: (180, 120, 120),      # building
    3: (6, 230, 230),        # sky
    4: (80, 50, 50),         # floor
    5: (4, 200, 3),          # tree
    6: (120, 120, 80),       # ceiling
    7: (140, 140, 140),      # road
    8: (204, 5, 255),        # bed
    9: (230, 230, 230),      # windowpane
    10: (4, 250, 7),
    11: (224, 5, 255),
    12: (235, 255, 7),
    13: (150, 5, 61),
    14: (120, 120, 70),
    15: (8, 255, 51),
    16: (255, 6, 82),
    17: (143, 255, 140),
    18: (204, 255, 4),
    19: (255, 51, 7),
    20: (204, 70, 3),
    21: (0, 102, 200),
    22: (61, 230, 250),
    23: (255, 6, 51),
    24: (11, 102, 255),
    25: (255, 7, 71),
    26: (255, 9, 224),
    27: (9, 7, 230),
    28: (220, 220, 220),
    29: (255, 9, 92),
    30: (112, 9, 255),
    31: (8, 255, 214),
    32: (7, 255, 224),
    33: (255, 184, 6),
    34: (10, 255, 71),
    35: (255, 41, 10),
    36: (7, 255, 255),
    37: (224, 255, 8),
    38: (102, 8, 255),
    39: (255, 61, 6),
    40: (255, 194, 7),
    41: (255, 122, 8),
    42: (0, 255, 20),
    43: (255, 8, 41),
    44: (255, 5, 153),
    45: (6, 51, 255),
    46: (235, 12, 255),
    47: (160, 150, 20),
    48: (0, 163, 255),
    49: (140, 140, 140),
    50: (250, 10, 15),
    51: (20, 255, 0),
    52: (31, 255, 0),
    53: (255, 31, 0),
    54: (255, 224, 0),
    55: (153, 255, 0),
    56: (0, 0, 255),
    57: (255, 71, 0),
    58: (0, 235, 255),
    59: (0, 173, 255),
    60: (31, 0, 255),
    61: (11, 200, 200),
    62: (255, 82, 0),
    63: (0, 255, 245),
    64: (0, 61, 255),
    65: (0, 255, 112),
    66: (0, 255, 133),
    67: (255, 0, 0),
    68: (255, 163, 0),
    69: (255, 102, 0),
    70: (194, 255, 0),
    71: (0, 143, 255),
    72: (51, 255, 0),
    73: (0, 82, 255),
    74: (0, 255, 41),
    75: (0, 255, 173),
    76: (10, 0, 255),
    77: (173, 255, 0),
    78: (0, 255, 153),
    79: (255, 92, 0),
    80: (255, 0, 255),
    81: (255, 0, 245),
    82: (255, 0, 102),
    83: (255, 173, 0),
    84: (255, 0, 20),
    85: (255, 184, 184),
    86: (0, 31, 255),
    87: (0, 255, 61),
    88: (0, 71, 255),
    89: (255, 0, 204),
    90: (0, 255, 194),
    91: (0, 255, 82),
    92: (0, 10, 255),
    93: (0, 112, 255),
    94: (51, 0, 255),
    95: (0, 194, 255),
    96: (0, 122, 255),
    97: (0, 255, 163),
    98: (255, 153, 0),
    99: (0, 255, 10),
    100: (255, 112, 0),
    101: (143, 255, 0),
    102: (82, 0, 255),
    103: (163, 255, 0),
    104: (255, 235, 0),
    105: (8, 184, 170),
    106: (133, 0, 255),
    107: (0, 255, 92),
    108: (184, 0, 255),
    109: (255, 0, 31),
    110: (0, 184, 255),
    111: (0, 214, 255),
    112: (255, 0, 112),
    113: (92, 255, 0),
    114: (0, 224, 255),
    115: (112, 224, 255),
    116: (70, 184, 160),
    117: (163, 0, 255),
    118: (153, 0, 255),
    119: (71, 255, 0),
    120: (255, 0, 163),
    121: (255, 204, 0),
    122: (255, 0, 143),
    123: (0, 255, 235),
    124: (133, 255, 0),
    125: (255, 0, 235),
    126: (245, 0, 255),
    127: (255, 0, 122),
    128: (255, 245, 0),
    129: (10, 190, 212),
    130: (214, 255, 0),
    131: (0, 204, 255),
    132: (20, 0, 255),
    133: (255, 255, 0),
    134: (0, 153, 255),
    135: (0, 41, 255),
    136: (0, 255, 204),
    137: (41, 0, 255),
    138: (41, 255, 0),
    139: (173, 0, 255),
    140: (0, 245, 255),
    141: (71, 0, 255),
    142: (122, 0, 255),
    143: (0, 255, 184),
    144: (0, 92, 255),
    145: (184, 255, 0),
    146: (0, 133, 255),
    147: (255, 214, 0),
    148: (25, 194, 194),
    149: (102, 255, 0),
    150: (92, 0, 255)
}

def visualize_segmentation(image: np.ndarray, pred_mask: np.ndarray, save_path: str = None, alpha: float = 0.5, dataset: str = "cityscapes"):
    """
    Overlay and save segmentation mask on input image in PNG format.

    Args:
        image (np.ndarray): Input image in BGR format (H, W, 3)
        pred_mask (np.ndarray): Predicted segmentation mask (H, W), where pixel values represent class indices
        save_path (str): Path to save the output image. '.png' extension recommended
        alpha (float): Transparency for the segmentation overlay (0~1)

    Returns:
        None
    """
    # Convert BGR to RGB
    image_rgb = image[..., ::-1]
    
    # Convert segmentation mask to color map
    height, width = pred_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    if dataset == "cityscapes":
        for class_idx, color in CITYSCAPES_COLORMAP.items():
            colored_mask[pred_mask == class_idx] = color
    elif dataset == "ade20k":
        for class_idx, color in ADE20K_COLORMAP.items():
            colored_mask[pred_mask == class_idx] = color

    # Create overlay image (RGB format)
    overlay = cv2.addWeighted(image_rgb, 1-alpha, colored_mask, alpha, 0)

    # Create output directory if it doesn't exist
    save_dir = osp.dirname(save_path)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
        
    # Add .png extension if not provided
    if not save_path.lower().endswith('.png'):
        save_path += '.png'
        
    # Save as PNG using PIL to preserve RGB format
    Image.fromarray(overlay).save(save_path, format='PNG', optimize=True)