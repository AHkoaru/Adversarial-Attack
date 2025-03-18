import torch
import numpy as np
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from torchvision.datasets import Cityscapes
from tqdm import tqdm
import random
import os
import evaluate

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from utils import label_to_train_id, save_results, compute_metrics
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

def compute_metrics(eval_pred, metric, num_labels):
    """
    평가 메트릭을 계산합니다.
    
    Args:
        eval_pred: (예측, 라벨) 튜플
        metric: 사용할 메트릭 객체
        num_labels: 클래스 수
        
    Returns:
        dict: 계산된 메트릭
    """
    pred, labels = eval_pred
    
    # 메트릭 계산
    metrics = metric.compute(
        predictions=pred,
        references=labels,
        num_labels=num_labels,
        ignore_index=255,
        reduce_labels=False,
    )
    return metrics

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
    matrix = evaluate.load("mean_iou")
    miou_metrics_list = []

    # 배치 처리 및 평가
    for batch_start in tqdm(range(0, len(selected_indices), config.batch_size), desc="batch"):
        batch_indices = selected_indices[batch_start:batch_start+config.batch_size]
        for idx in tqdm(batch_indices, desc="image", leave=False):
            image, gt_mask = dataset[idx]
            pred = infer_full_image(image, processor, model, device)
            
            # GT 마스크 전처리
            gt = label_to_train_id(gt_mask)
            
            # evaluate 라이브러리를 사용하여 메트릭 계산
            metrics = compute_metrics([pred, gt], matrix, num_classes)
            miou_metrics_list.append(metrics["mean_iou"])
    
    # 평균 mIoU 계산
    miou_metrics = np.array(miou_metrics_list)
    mean_miou = np.mean(miou_metrics)
    print("model_name:", config.model_name)
    print(f"평균 mIoU: {mean_miou:.4f}")

    # 결과 저장
    results = {
        "model_name": config.model_name,
        "mIoU": float(mean_miou),
        "Dataset": config.Dataset,
        "dataSize": config.DataSize
    }
    save_results(results, "mask_clean.json")

if __name__ == "__main__":
    main()