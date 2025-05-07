# config.py

config = {
    "task": "segmentation",
    "dataset": "ade20k",
    "data_dir": "datasets/ade20k",         # Directory path where the dataset is located
    "base_dir": "./data/PixelAttack/results/ade20k/segformer",  # Base directory for saving results
    "model": "segformer",
    "RGB": 3,                                       # Input dimension
    "attack_pixel": 0.01,                              # Attack dimension for the Remember process (recalculated later)
    "num_class": 150,
}
