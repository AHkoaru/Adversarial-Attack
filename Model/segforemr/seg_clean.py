import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torchvision.datasets import Cityscapes
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F 
import random

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

def save_results(results, filename="mask2_clean.json"):
    """
    결과를 JSON 파일에 저장합니다.
    
    Args:
        results: 저장할 결과 딕셔너리
        filename: 저장할 파일 이름
    """
    # 현재 스크립트 파일이 있는 디렉토리에 JSON 파일 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            # 기존 파일이 리스트 형식이 아닐 경우 리스트로 감싸줍니다.
            existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
    else:
        existing_results = []
    
    existing_results.append(results)
    
    with open(file_path, "w") as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=4)
    print(f"결과가 '{file_path}' 파일에 추가 저장되었습니다.")

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
    dataset = Cityscapes(root="./DataSet/", split="val", mode="fine", target_type="semantic")
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
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

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
            
            pred_flat = pred.flatten()
            gt_flat = gt.flatten()
            mask = gt_flat != 255
            pred_flat = pred_flat[mask]
            gt_flat = gt_flat[mask]
            
            combined = num_classes * gt_flat + pred_flat
            cm = np.bincount(combined, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
            confusion_matrix += cm

    # IoU 계산
    iou_per_class = []
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        union = tp + fp + fn
        iou = tp / union if union != 0 else float("nan")
        iou_per_class.append(iou)

    mIoU = np.nanmean(iou_per_class)
    print("mIoU:", mIoU)
    print("클래스별 IoU:", iou_per_class)

        # 모델 이름, 모드, mIoU 값 저장
    results = {
        "model_name": model_name,
        "mode": mode,
        "mIoU": mIoU,
        "dataSize": DataSize,
    }
    
    import json, os
    results_file = "seg_clean.json"
        
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            # 기존 파일이 리스트 형식이 아닐 경우 리스트로 감싸줍니다.
            existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
    else:
        existing_results = []

    existing_results.append(results)

    with open(results_file, "w") as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=4)
    print("결과가 'seg_clean.json' 파일에 추가 저장되었습니다.")