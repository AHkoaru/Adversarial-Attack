# config_mask_swin_T.py

config = {
    "attack_method": "Pixel",
    "task": "segmentation",
    "dataset": "cityscapes",
    "data_dir": "datasets/cityscapes",         # Directory path where the dataset is located
    "model": "mask2former",
    "RGB": 3,                                       # Input dimension
    "attack_pixel": 0.01,                              # Attack dimension for the Remember process (recalculated later) previow 5
    "num_class": 19,
}