import torch
import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from torchvision.datasets import Cityscapes
from tqdm import tqdm
import random
import os
import evaluate
# utils.py 파일 import (더 깔끔한 방식)

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from utils import label_to_train_id, save_results

# 설정값을 클래스로 정의하여 관리
class AttackConfig:
    model_name = "shi-labs/oneformer_cityscapes_swin_large"
    data = "cityscapes"
    DataSize = 500
    batch_size = 10
    Dataset = "val"

def compute_metrics(eval_pred, metric, num_labels):
    pred, labels = eval_pred
    
    # pred가 이미 최종 예측 결과인 경우 (클래스 인덱스)
    # 메트릭 계산
    metrics = metric.compute(
        predictions=pred,
        references=labels,
        num_labels=num_labels,
        ignore_index=255,
        reduce_labels=False,
    )
    return metrics

def infer_split_image(image, processor, model, device, split_size=(512, 1024)):
    """
    큰 이미지를 여러 조각으로 나누어 추론한 후 결과를 합칩니다.
    
    Args:
        image: 입력 이미지 (PIL Image)
        processor: 이미지 전처리기
        model: 세그멘테이션 모델
        device: 연산 장치 (CPU/GPU)
        split_size: 분할할 이미지 크기 (height, width)
        
    Returns:
        numpy array: 합쳐진 세그멘테이션 결과 (height, width)
    """
    from PIL import Image
    import numpy as np
    import torch
    
    # 원본 이미지 크기 확인
    width, height = image.size
    split_height, split_width = split_size
    
    # 결과를 저장할 배열 초기화
    result = np.zeros((height, width), dtype=np.int64)
    
    # 이미지를 4등분하여 처리
    for y in range(0, height, split_height):
        for x in range(0, width, split_width):
            # 이미지 조각 추출
            x_end = min(x + split_width, width)
            y_end = min(y + split_height, height)
            
            # 이미지 조각 생성
            tile = image.crop((x, y, x_end, y_end))
            
            # 모델 추론
            inputs = processor(images=tile, task_inputs=["semantic"], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                
            # 후처리
            tile_result = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(tile.size[1], tile.size[0])])[0]
            
            # 결과를 numpy 배열로 변환
            tile_result = tile_result.cpu().numpy().astype(np.int64)
            
            # 결과 배열에 조각 결과 삽입
            result[y:y_end, x:x_end] = tile_result
    
    return result

def main_with_split_inference():
    """
    이미지를 분할하여 추론하는 메인 함수
    """
    config = AttackConfig()
    
    # Cityscapes 데이터셋 (fine annotation) 사용
    dataset = Cityscapes(root=f"./DataSet/{config.data}/", split=config.Dataset, mode="fine", target_type="semantic")
    selected_indices = list(random.sample(range(len(dataset)), config.DataSize))
    
    # 모델 로드
    processor = OneFormerProcessor.from_pretrained(config.model_name)
    model = OneFormerForUniversalSegmentation.from_pretrained(config.model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"사용 중인 디바이스: {device}")
    
    # 평가 준비
    num_classes = 19
    matrix = evaluate.load("mean_iou")
    miou_metrics_list = []
    
    # 배치 처리 및 평가
    for batch_start in tqdm(range(0, len(selected_indices), config.batch_size), desc="배치 처리"):
        batch_indices = selected_indices[batch_start:batch_start+config.batch_size]
        for idx in tqdm(batch_indices, desc="이미지 처리", leave=False):
            image, gt_mask = dataset[idx]
            
            # 이미지 크기 확인 (Cityscapes는 일반적으로 1024x2048)
            width, height = image.size
            
            # 이미지가 충분히 큰 경우에만 분할 추론 적용
            pred = infer_split_image(image, processor, model, device)
            
            # GT 마스크 전처리
            gt = label_to_train_id(gt_mask)
            
            # 메트릭 계산
            metrics = compute_metrics([pred, gt], matrix, num_classes)
            miou_metrics_list.append(metrics["mean_iou"])
            # print(metrics["mean_iou"])
    
    # 평균 mIoU 계산
    miou_metrics = np.array(miou_metrics_list)
    mean_miou = np.mean(miou_metrics)
    print(f"평균 mIoU: {mean_miou:.4f}")
    
    # 결과 저장
    results = {
        "model_name": config.model_name,
        "mIoU": float(mean_miou),
        "Dataset": config.Dataset,
        "dataSize": config.DataSize,
        "split_inference": True
    }
    save_results(results, "one_split_inference.json")

# 분할 추론 실행
if __name__ == "__main__":
    main_with_split_inference()