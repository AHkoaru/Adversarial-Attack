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

Config = {
    "model_name": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "mode": "bilinear",
    "DataSize": 500,
    "batch_size": 10
}

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

    # attaker = Clean()  # Clean 클래스 인스턴스 생성
    # image = attaker.attack(image)
    # 전처리가 제대로 적용되지 않거나 변형이 있을 수 있습니다.

    # 이미지가 타일 크기보다 작으면 전체 이미지에 대해 모델 추론 진행 전에 clean 적용
    if width < tile_width or height < tile_height:
        # image = attaker.attack(image)
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

            # PIL Image에서는 crop 메서드를 사용하여 이미지 영역 추출
            # crop 메서드는 (left, upper, right, lower) 좌표를 사용
            tile = image.crop((x_start, y_start, x_end, y_end))
            
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
    DataSize = Config["DataSize"]
    selected_indices = list(random.sample(range(len(dataset)), DataSize))
    batch_size = Config["batch_size"]

    model_name = Config["model_name"]
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name, do_rescale=False)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"사용 중인 디바이스: {device}")
    
    num_classes = 19
    mode = Config["mode"]
    matrix = evaluate.load("mean_iou")
    miou_metrics_list = []
    class_ious_list = []  # 각 이미지의 클래스별 IoU 저장 리스트
    
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
            miou_metrics_list.append(metrics)
            
            # 클래스별 IoU 저장
            class_ious = metrics.get("per_category_iou", {})
            
            # 각 이미지의 클래스별 IoU를 저장할 딕셔너리 생성
            image_class_ious = {}
            
            # numpy.ndarray인 경우
            if isinstance(class_ious, np.ndarray):
                for class_idx, iou in enumerate(class_ious):
                    image_class_ious[str(class_idx)] = float(iou)
                    print(f"  클래스 {class_idx}: {iou:.4f}")
            # 딕셔너리인 경우
            elif isinstance(class_ious, dict):
                for class_idx, iou in class_ious.items():
                    image_class_ious[str(class_idx)] = float(iou)
                    print(f"  클래스 {class_idx}: {iou:.4f}")
            
            # 이미지 인덱스와 함께 저장
            class_ious_list.append({
                "image_idx": idx,
                "class_ious": image_class_ious,
                "mean_iou": float(metrics["mean_iou"])
            })
            
            print(f"  mIoU: {metrics['mean_iou']:.4f}")
            print("-" * 50)

    # 각 이미지의 mean_iou 값을 추출하여 평균 계산
    mean_iou_values = [metrics["mean_iou"] for metrics in miou_metrics_list]
    accuracy_values = [metrics["mean_accuracy"] for metrics in miou_metrics_list]
    mIoU = np.nanmean(mean_iou_values)
    print("전체 mIoU:", mIoU)
    
    # 전체 데이터셋의 클래스별 평균 IoU 계산 및 출력
    avg_class_ious = {}
    for class_idx in range(num_classes):
        # 수정된 class_ious_list 구조에 맞게 변경
        class_ious = [img_data["class_ious"].get(str(class_idx), 0) 
                     for img_data in class_ious_list 
                     if str(class_idx) in img_data["class_ious"]]
        if class_ious:
            avg_class_ious[str(class_idx)] = float(np.nanmean(class_ious))
    
    print("\n전체 데이터셋의 클래스별 평균 IoU:")
    for class_idx, avg_iou in avg_class_ious.items():
        print(f"  클래스 {class_idx}: {avg_iou:.4f}")

    # 모델 이름, 모드, mIoU 값 저장
    results = {
        "model_name": model_name,
        "mode": mode,
        "mIoU": float(mIoU),  # JSON 직렬화를 위해 numpy 타입을 파이썬 float로 변환
        "dataSize": DataSize,
        "miou_list": [float(m) for m in mean_iou_values],
        "mean_accuracy": float(np.nanmean(accuracy_values)),
        "class_ious": avg_class_ious,  # 전체 클래스별 평균 IoU 저장
        "per_image_class_ious": class_ious_list  # 각 이미지별 클래스별 IoU 저장
    }
    
    results_file = "class_miou.json"
    save_results(results, results_file)
    print("결과가 'class_miou.json' 파일에 추가 저장되었습니다.")
    
    # 각 이미지별 클래스별 IoU를 별도 파일로도 저장
    with open("per_image_class_ious.json", "w") as f:
        json.dump(class_ious_list, f, indent=2)
    print("각 이미지별 클래스별 IoU가 'per_image_class_ious.json' 파일에 저장되었습니다.")
