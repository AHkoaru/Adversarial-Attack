# citysetting
config_city = {
    "model": "pspnet_sat",
    "model_path": "adv_models/pretrain/cityscapes/pspnet/sat/train_epoch_400.pth",
    "layers": 50,
    "zoom_factor": 8,
    "scales": [1.0],
    "base_size": 1024,
    "crop_h": 449,
    "crop_w": 449,
    "stride_rate": 2/3,
    "process_name": "pspnet_sat_attack",
    "use_gt": False,
    "mean": [255*0.485, 255*0.456, 255*0.406],  # [R, G, B]
    "std": [255*0.229, 255*0.224, 255*0.225]    # [R, G, B]
}
## VOC setting
######### adv model setting #########
voc = {
    "model": "pspnet_sat",
    "model_path": "/workspace/ckpt/pretrained_model/pretrain/voc2012/pspnet/sat/train_epoch_50.pth",
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