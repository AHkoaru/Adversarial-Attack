import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torchvision.datasets import Cityscapes
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F 
import random
import evaluate
import json
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from utils import label_to_train_id, save_results, compute_metrics

# Attacker 모듈 경로 추가
sys.path.append('/workspace')
from Attacker.clean import Clean

def convert_to_train_id(label_array):
    """
    Cityscapes의 원본 레이블을 학습에 사용되는 레이블로 변환합니다.
    """
    mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18
    }
    return np.vectorize(lambda x: mapping.get(x, 255))(label_array)


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

def sliding_window_inference(image, feature_extractor, model, mode, device, tile_size=(1024, 1024), stride=(512, 512)):
    """
    슬라이딩 윈도우 방식으로 전체 이미지에 대해 단일 추론을 수행합니다.
    타일별 예측 결과를 누적하고, 겹치는 영역은 평균을 계산합니다.
    Clean 클래스의 attack 메서드를 사용하여 이미지에 clean 작업을 수행합니다.
    """
    width, height = image.size
    tile_width, tile_height = tile_size

    attaker = Clean()  # Clean 클래스 인스턴스 생성
    image = attaker.attack(image)

    # 이미지가 타일 크기보다 작으면 전체 이미지에 대해 모델 추론 진행 전에 clean 적용
    if width < tile_width or height < tile_height:
        image = attaker.attack(image)
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
        return logits.squeeze(0).cpu().numpy()

    num_classes = model.config.num_labels if hasattr(model.config, "num_labels") else 19
    logits_sum = torch.zeros((num_classes, height, width), dtype=torch.float32, device=device)
    count_map = torch.zeros((height, width), dtype=torch.float32, device=device)

    # 슬라이딩 윈도우 방식으로 타일 단위 추론 수행
    for y in range(0, height, stride[1]):
        for x in range(0, width, stride[0]):
            if x + tile_width > width:
                x_start = width - tile_width
            else:
                x_start = x
            if y + tile_height > height:
                y_start = height - tile_height
            else:
                y_start = y
            x_end = x_start + tile_width
            y_end = y_start + tile_height

            # 타일 추출 후 clean 적용
            tile = image[y_start:y_end, x_start:x_end]
            # tile = attaker.attack(tile)

            inputs = feature_extractor(images=tile, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            logits = F.interpolate(logits, size=(tile_height, tile_width), mode=mode, align_corners=True)
            logits = logits.squeeze(0)

            logits_sum[:, y_start:y_end, x_start:x_end] += logits
            count_map[y_start:y_end, x_start:x_end] += 1

    logits_avg = logits_sum / count_map.unsqueeze(0)
    return logits_avg.cpu().numpy()

if __name__ == "__main__":
    # Cityscapes 데이터셋 (fine annotation) 사용
    dataset = Cityscapes(root="./DataSet/cityscapes/", split="val", mode="fine", target_type="semantic")
    DataSize = 500
    selected_indices = list(random.sample(range(len(dataset)), DataSize))
    batch_size = 10

    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name, do_rescale=False)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"사용 중인 디바이스: {device}")
    
    num_classes = 19
    mode = "bilinear"
    matrix = evaluate.load("mean_iou")
    miou_metrics_list = []
    # 배치 처리 및 개별 이미지 처리에 tqdm 적용
    for batch_start in tqdm(range(0, len(selected_indices), batch_size), desc="Batch"):
        batch_indices = selected_indices[batch_start:batch_start + batch_size]  
        for idx in tqdm(batch_indices, desc="Image", leave=False):
            image, gt_mask = dataset[idx]
            # 슬라이딩 윈도우 추론 적용
            logits = sliding_window_inference(
                image, feature_extractor, model, mode, device,
                tile_size=(1024, 1024), stride=(512, 512)
            )
            pred = np.argmax(logits, axis=0)
            
            # GT 마스크 전처리
            gt = np.array(gt_mask) if isinstance(gt_mask, Image.Image) else np.array(gt_mask)
            gt = convert_to_train_id(gt)

            metrics = compute_metrics([pred, gt], matrix, num_classes)
            miou_metrics_list.append(metrics["mean_iou"])
    # IoU 계산


    mIoU = np.nanmean(miou_metrics_list)
    print("mIoU:", mIoU)

        # 모델 이름, 모드, mIoU 값 저장
    results = {
        "model_name": model_name,
        "mode": mode,
        "mIoU": mIoU,
        "dataSize": DataSize,
    }
    
    results_file = "seg_clean.json"
    save_results(results, results_file)
    print("결과가 'seg_clean.json' 파일에 추가 저장되었습니다.")
