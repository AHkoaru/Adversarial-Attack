# DeepLabV3 Vanilla model configuration for Cityscapes
config = {
    "model": "deeplabv3",
    "model_path": "/workspace/ckpt/pretrained_model/pretrain/cityscapes/deeplabv3/vanilla/train_epoch_400.pth",
    "data_dir": "/workspace/datasets/cityscapes",
    "device": "cuda",
    "dataset": "cityscapes",
    "layers": 50,
    "num_class": 19,
    "zoom_factor": 8,
    "scales": [1.0],
    "base_size": 1024,
    "crop_h": 449,
    "crop_w": 449,
    "stride_rate": 2/3,
    "process_name": "deeplabv3_vanilla_attack",
    "use_gt": False,
    "mean": [255*0.485, 255*0.456, 255*0.406],  # [R, G, B]
    "std": [255*0.229, 255*0.224, 255*0.225]    # [R, G, B]
}