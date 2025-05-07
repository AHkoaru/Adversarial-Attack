# config.py

config = {
    "task": "segmentation",
    "dataset": "cityscapes",
    "data_dir": "datasets/cityscapes",         # Directory path where the dataset is located
    "base_dir": "./data/PixelAttack/results/cityscapes/segformer",  # Base directory for saving results
    "model": "segformer",
    "RGB": 3,                                       # Input dimension
    "attack_pixel": 0.01,                              # Attack dimension for the Remember process (recalculated later)
    "num_class": 19,
}

