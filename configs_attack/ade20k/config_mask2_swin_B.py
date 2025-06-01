# config_mask_swin_B.py

config = {
    "attack_method": "Pixel",
    "task": "segmentation",
    "dataset": "ade20k",
    "data_dir": "datasets/ade20k",         # Directory path where the dataset is located
    "model": "mask2former",
    "RGB": 3,                                       # Input dimension
    "attack_pixel": 0.01,                              # Attack dimension for the Remember process (recalculated later)
    "num_class": 150,
}