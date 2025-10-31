config = {
    "attack_method": "Pixel",
    "task": "segmentation",
    "dataset": "VOC2012",
    "data_dir": "../datasets/VOC2012",         # Directory path where the dataset is located
    "model": "pspnet_ddcat",
    "RGB": 3,                                       # Input dimension
    "attack_pixel": 0.01,                              # Attack dimension for the Remember process (recalculated later) previow 5
    "num_class": 21,
}