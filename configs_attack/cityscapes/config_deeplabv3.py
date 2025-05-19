config = {
    "task": "segmentation",
    "dataset": "cityscapes",
    "data_dir": "datasets/cityscapes",         # Directory path where the dataset is located
    "base_dir": "./data/PixelAttack/results/cityscapes/deeplabv3",  # Base directory for saving results
    "model": "deeplabv3",
    "RGB": 3,                                       # Input dimension
    "attack_pixel": 0.01,                              # Attack dimension for the Remember process (recalculated later) previow 5
    "num_class": 19,
}
