import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torchvision.datasets import Cityscapes
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import random

# 직접 import
from Attacker.nes import NES
from utils import label_to_train_id, save_results

class AttackConfig:
    epsilon: float = 0.05
    learning_rate: float = 0.01
    samples_per_draw: int = 50
    attack_batch_size: int = 8
    query: int = 50
    dataSize: int = 1
    batch_size: int = 10
    Data = "cityscapes"
    model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

def sliding_window_inference(image, gt_mask, feature_extractor, model, mode, device, tile_size=(1024, 1024), stride=(512, 512)):
    """
    슬라이딩 윈도우 방식으로 전체 이미지에 대해 단일 추론을 수행합니다.
    메모리 효율적인 방식으로 구현되었습니다.
    """
    width, height = image.size
    tile_width, tile_height = tile_size

    gt = label_to_train_id(gt_mask)

    # 메모리 효율적인 NES 생성기 (샘플 수와 스텝 수를 줄임)
    attack_config = AttackConfig()
    attacker = NES(feature_extractor, model, 
                   epsilon=attack_config.epsilon,
                   learning_rate=attack_config.learning_rate,
                   samples_per_draw=attack_config.samples_per_draw,
                   batch_size=attack_config.attack_batch_size)

    # 메모리 절약을 위해 단계별로 처리
    # 먼저 적대적 이미지 생성
    adv_img = attacker.attack(image, gt, query=attack_config.query)
    adv_img_pil = Image.fromarray((adv_img * 255).astype(np.uint8))
    
    # 메모리 해제
    torch.cuda.empty_cache()
    
    # 이미지가 타일 크기보다 작으면 전체 이미지에 대해 모델 추론 진행
    if width < tile_width or height < tile_height:
        inputs = feature_extractor(images=adv_img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
        return logits.squeeze(0).cpu().numpy()

    num_classes = model.config.num_labels if hasattr(model.config, "num_labels") else 19
    logits_sum = torch.zeros((num_classes, height, width), dtype=torch.float32, device=device)
    count_map = torch.zeros((height, width), dtype=torch.float32, device=device)

    # 슬라이딩 윈도우 방식으로 타일 단위 추론 수행
    for y_pos in range(0, height, stride[1]):
        for x_pos in range(0, width, stride[0]):
            if x_pos + tile_width > width:
                x_start = width - tile_width
            else:
                x_start = x_pos
            if y_pos + tile_height > height:
                y_start = height - tile_height
            else:
                y_start = y_pos
            x_end = x_start + tile_width
            y_end = y_start + tile_height

            # PIL 이미지에서 타일 추출
            tile = adv_img_pil.crop((x_start, y_start, x_end, y_end))

            inputs = feature_extractor(images=tile, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs).logits
                
            # 메모리 효율성을 위해 즉시 크기 조정 및 누적
            outputs = F.interpolate(outputs, size=(tile_height, tile_width), mode=mode, align_corners=True)
            logits = outputs.squeeze(0)
            logits_sum[:, y_start:y_end, x_start:x_end] += logits
            count_map[y_start:y_end, x_start:x_end] += 1
            
            # 명시적 메모리 해제
            del outputs, logits, inputs
            torch.cuda.empty_cache()

    # 최종 결과 계산
    logits_avg = logits_sum / count_map.unsqueeze(0)
    return logits_avg.cpu().numpy()

if __name__ == "__main__":
    # Cityscapes 데이터셋 (fine annotation) 사용
    attack_config = AttackConfig()
    dataset = Cityscapes(root=f"./DataSet/{attack_config.Data}/", split="val", mode="fine", target_type="semantic")
    DataSize = attack_config.dataSize
    selected_indices = list(random.sample(range(len(dataset)), DataSize))
    batch_size = attack_config.batch_size

    model_name = attack_config.model_name
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name, do_rescale=False)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda")
    model.to(device)
    
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
                image, gt_mask, feature_extractor, model, mode, device,
                tile_size=(1024, 1024), stride=(512, 512)
            )
            pred = np.argmax(logits, axis=0)
            
            # GT 마스크 전처리
            gt = label_to_train_id(gt_mask)
            
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
    attack_config = AttackConfig()
    results = {
        "model_name": model_name,
        "mode": mode,
        "mIoU": mIoU,
        "dataSize": DataSize,
        "attack": "NES",
        "query": attack_config.query,
        "epsilon": attack_config.epsilon,
        "learning_rate": attack_config.learning_rate,
        "samples_per_draw": attack_config.samples_per_draw,
        "batch_size": attack_config.batch_size
    }
    
    results_file = "seg_nes.json"
    save_results(results, results_file)
    print("결과가 'seg_nes.json' 파일에 추가 저장되었습니다.")