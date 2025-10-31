import torch
import numpy as np
from PIL import Image
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmengine.runner import load_checkpoint
from mmseg.datasets import CityscapesDataset
from torch.utils.data import DataLoader, Dataset
from mmcv.transforms import Compose
from tqdm import tqdm
from mmengine.registry import init_default_scope
from mmengine.dataset import default_collate
from mmseg.evaluation import IoUMetric
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

from mmseg.apis import init_model, inference_model
from torchvision.datasets import Cityscapes
import evaluate
from mmengine.dataset import default_collate
from mmengine.logging import HistoryBuffer

from pixle_x import generate_adv_examples
from function import *
from evaluation import eval_miou


# 필요한 글로벌 객체 허용
# torch.serialization.add_safe_globals([HistoryBuffer])
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
# 1. Config & Model 불러오기
# config_path = './configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
# checkpoint_path = 'ckpt/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

cf_path = 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
ckpt_path = 'checkpoint/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

# 2. Default scope 초기화
model = init_model(cf_path, None, 'cuda')
# 2. 체크포인트 로드 (weights_only=False 직접 설정)
checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

cfg = {
    "num_class": 19
}

#Dataset 불러오기
dataset = Cityscapes('./data/cityscapes', split='val', mode='fine', target_type='semantic')
# 랜덤 100개 샘플 선택
import random
random_indices = random.sample(range(len(dataset)), 3)
dataset.images = [dataset.images[i] for i in random_indices]
dataset.targets = [dataset.targets[i] for i in random_indices]
print(len(dataset))

# test_dataset = CityscapesDataset(
#     data_root='../../Dataset/cityscapes',
#     data_prefix=dict(
#         img_path='leftImg8bit/val',  # validation set
#         seg_map_path='gtFine/val'
#     ),
#     pipeline=test_pipeline,)
    


# dataloader = DataLoader(
#     test_dataset,
#     batch_size=1,
#     shuffle=False,
#     collate_fn=default_collate  # 함수 그대로 전달
# )


# 방법 2: for문
# for batch in dataloader:
#     print(len(batch))
    # break  # 하나만 보고 싶으면 break!
# Cityscapes 원본 라벨을 trainId로 변환하는 매핑 딕셔너리
row_list = []
adv_list = []
gt_list = []

with torch.no_grad():
    for img_pil,gt_pil in tqdm(dataset, desc="image"):
        # RGB를 BGR로 변환
        
        img_pil_rgb = img_pil.copy()
        r, g, b = img_pil_rgb.split()

        img_pil = Image.merge("RGB", (b, g, r))

        img_np = np.array(img_pil)
        gt_np = convert_gt_labels(np.array(gt_pil))

        adv_img = generate_adv_examples(
            model=model,
            init_img=img_np,
            num_iterations=50
        )
        
        row_list.append(img_np)
        adv_list.append(adv_img)
        gt_list.append(gt_np)

    benign_miou_score, adv_miou_score = eval_miou(model, row_list, adv_list, gt_list, cfg)
    print(benign_miou_score)
    print(adv_miou_score)


# print(np.mean(iou_list))