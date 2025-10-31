import torch
from tqdm import tqdm
import datetime
import os
import importlib
import numpy as np # Import numpy
from PIL import Image # Import Image

from mmseg.apis import init_model, inference_model

from function import *
from evaluation import *
from dataset import CitySet, ADESet

from pixle import Pixle
from utils import save_experiment_results

import argparse
import setproctitle

# 필요한 글로벌 객체 허용
# torch.serialization.add_safe_globals([HistoryBuffer])
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
# 1. Config & Model 불러오기
# config_path = './configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
# checkpoint_path = 'ckpt/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'


def load_config(config_path):
    """
    Load and return config dictionary from a python file at config_path.
    The config file should contain a dictionary named 'config'.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def main(config):
    model_configs = {
        "cityscapes": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221203_045030-9a86a225.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
                "checkpoint": 'ckpt/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'
            },
            "setr": {
                "config": 'configs/setr/setr_vit-l_pup_8xb1-80k_cityscapes-768x768.py',
                "checkpoint": 'ckpt/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth'
            }
        },
        "ade20k": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py',
                "checkpoint": 'ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py',
                "checkpoint": 'ckpt/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-80k_ade20k-512x512.py',
                "checkpoint": 'ckpt/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py',
                "checkpoint": 'ckpt/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth'
            },
            "setr": {
                "config": 'configs/setr/setr_vit-l_pup_8xb2-160k_ade20k-512x512.py',
                "checkpoint": 'ckpt/setr_pup_512x512_160k_b16_ade20k_20210619_191343-7e0ce826.pth'
            }
        }
    }

    device = config["device"]

    #configs_attack/ade20k/config_mask2_swin_B.py
    #configs_attack/ade20k/config_seg.py

    #configs_attack/cityscapes/config_mask2_swin_B.py
    #configs_attack/cityscapes/config_seg.py

    # Initialize model
    if config["dataset"] not in model_configs:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    if config["model"] not in model_configs[config["dataset"]]:
        raise ValueError(f"Unsupported model: {config['model']} for dataset {config['dataset']}")

    model_cfg = model_configs[config["dataset"]][config["model"]]

    # Load dataset
    if config["dataset"] == "cityscapes":
        dataset = CitySet(dataset_dir=config["data_dir"])
    elif config["dataset"] == "ade20k":
        dataset = ADESet(dataset_dir=config["data_dir"])
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    setproctitle.setproctitle(f"Pixle_Attack_{config['dataset']}_{config['model']}_{config['attack_pixel']}_Process")
    # num_images = 5

    # # 데이터셋 전체를 랜덤하게 섞기 위한 인덱스 생성 및 셔플
    # n_total = len(dataset.images)
    # indices = np.arange(n_total)
    # np.random.shuffle(indices) # 인덱스를 무작위로 섞음

    # # 섞인 인덱스를 사용하여 데이터셋 리스트 재정렬
    # dataset.images = [dataset.images[i] for i in indices]
    # dataset.filenames = [dataset.filenames[i] for i in indices]
    # dataset.gt_images = [dataset.gt_images[i] for i in indices]

    # # 이후 코드가 num_images 만큼 앞에서부터 선택하므로, 결과적으로 랜덤 샘플링됨

    # dataset.images = dataset.images[:min(len(dataset.images), num_images)]
    # dataset.filenames = dataset.filenames[:min(len(dataset.filenames), num_images)]
    # dataset.gt_images = dataset.gt_images[:min(len(dataset.gt_images), num_images)]

    if config["model"] == "setr":
        model = init_model(model_cfg["config"], None, 'cuda')
        checkpoint = torch.load(model_cfg["checkpoint"], map_location='cuda', weights_only=False)
        # 모델의 projection 레이어에 bias 추가
        model.backbone.patch_embed.projection.bias = torch.nn.Parameter(
            torch.zeros(checkpoint["state_dict"]["backbone.patch_embed.projection.weight"].shape[0], device='cuda')
        )
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint  # 체크포인트 변수 삭제

    # del checkpoint  # 체크포인트 변수 삭제
    torch.cuda.empty_cache()  # GPU 캐시 정리

    #convert to BGR
    # dataset.images = [img[:, :, ::-1] for img in dataset.images]
    
    adv_result, gt_result = eval_miou(model, dataset.images, dataset.images, dataset.gt_images, config)
    print(gt_result['mean_iou'])
    print(adv_result['mean_iou'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attack_pixel", type=float, required=True, help="Ratio of adversarial pixels to total image pixels.") # 새 pixel_ratio 인자 추가
    args = parser.parse_args()

    config = load_config(args.config)
    
    config["device"] = args.device
    config["attack_pixel"] = args.attack_pixel

    main(config)