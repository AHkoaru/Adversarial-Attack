# PSPNet SAT model configuration for VOC2012
config = {
    "model": "pspnet_sat",
    "model_path": "/workspace/ckpt/pretrained_model/pretrain/voc2012/pspnet/sat/train_epoch_50.pth",
    "data_dir": "/workspace/datasets/VOC2012",
    "device": "cuda",
    "dataset": "VOC2012",
    "layers": 50,
    "num_class": 21,
    "zoom_factor": 8,
    "scales": [1.0],
    "base_size": 512,
    "crop_h": 473,
    "crop_w": 473,
    "stride_rate": 2/3,
    "process_name": "pspnet_sat_attack",
    "use_gt": False,
    "mean": [255*0.485, 255*0.456, 255*0.406],  # [R, G, B]
    "std": [255*0.229, 255*0.224, 255*0.225]    # [R, G, B]
}