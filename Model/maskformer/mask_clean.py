import torch
import numpy as np
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from torchvision.datasets import Cityscapes
from tqdm import tqdm
import random
import os
# utils.py 파일 import (더 깔끔한 방식)

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from utils import label_to_train_id, save_results

# 설정값을 클래스로 정의하여 관리
class AttackConfig:
    model_name = "facebook/maskformer-resnet101-cityscapes"
    data = "cityscapes"
    DataSize = 500
    batch_size = 10
    Dataset = "val"

def infer_full_image(image, processor, model, device):
    """
    전체 이미지에 대한 추론을 수행합니다.
    
    Args:
        image: 입력 이미지 (PIL Image)
        processor: 이미지 전처리기
        model: 세그멘테이션 모델
        device: 연산 장치 (CPU/GPU)
        
    Returns:
        numpy array: 세그멘테이션 결과 (height, width)
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mask2Former 모델의 후처리
    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    # 결과가 Tensor인 경우 numpy 배열로 변환
    result = result.cpu().numpy().astype(np.int64)
    return result

def calculate_iou(confusion_matrix, num_classes=19):
    """
    혼동 행렬에서 IoU를 계산합니다.
    
    Args:
        confusion_matrix: 클래스별 혼동 행렬
        num_classes: 클래스 수
        
    Returns:
        tuple: (mIoU, 클래스별 IoU 리스트)
    """
    iou_per_class = []
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        union = tp + fp + fn
        iou = tp / union if union != 0 else float("nan")
        iou_per_class.append(iou)

    mIoU = np.nanmean(iou_per_class)
    return mIoU, iou_per_class

def main():
    """
    메인 함수: 모델 로드, 데이터셋 처리, 평가 수행
    """
    # 설정 로드
    config = AttackConfig()
    
    # Cityscapes 데이터셋 (fine annotation) 사용
    dataset = Cityscapes(root=f"./DataSet/{config.data}/", split=config.Dataset, mode="fine", target_type="semantic")
    selected_indices = list(random.sample(range(len(dataset)), config.DataSize))
    
    # 모델 로드
    processor = MaskFormerFeatureExtractor.from_pretrained(config.model_name)
    model = MaskFormerForInstanceSegmentation.from_pretrained(config.model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"사용 중인 디바이스: {device}")
    
    # 평가 준비
    num_classes = 19
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # 배치 처리 및 평가
    for batch_start in tqdm(range(0, len(selected_indices), config.batch_size), desc="batch"):
        batch_indices = selected_indices[batch_start:batch_start+config.batch_size]
        for idx in tqdm(batch_indices, desc="image", leave=False):
            image, gt_mask = dataset[idx]
            pred = infer_full_image(image, processor, model, device)
            
            # GT 마스크 전처리
            gt = label_to_train_id(gt_mask)
            
            # 명시적으로 int64 데이터 타입으로 변환
            pred_flat = pred.flatten().astype(np.int64)
            gt_flat = gt.flatten().astype(np.int64)
            mask = gt_flat != 255
            pred_flat = pred_flat[mask]
            gt_flat = gt_flat[mask]

            combined = num_classes * gt_flat + pred_flat
            cm = np.bincount(combined, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
            confusion_matrix += cm

    # IoU 계산
    mIoU, iou_per_class = calculate_iou(confusion_matrix)
    print("mIoU:", mIoU)
    print("클래스별 IoU:", iou_per_class)

    # 결과 저장
    results = {
        "model_name": config.model_name,
        "mIoU": float(mIoU),  # numpy float를 일반 float로 변환
        "Dataset": config.Dataset,
        "dataSize": config.DataSize
    }
    save_results(results, "mask_clean.json")

if __name__ == "__main__":
    main()