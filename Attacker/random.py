import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

class Random:
    def __init__(self, model, feature_extractor, gt, device, epsilon=8/255, T=200, num_classes=19, ignore_label=255):
        """
        :param model: 세그먼테이션 모델 (예: SegformerForSemanticSegmentation)
        :param feature_extractor: 모델의 feature extractor (예: SegformerFeatureExtractor)
        :param gt: 해당 이미지의 ground truth segmentation (학습에 사용되는 train id, numpy array, shape: (H, W))
        :param device: 모델이 할당된 device (예: torch.device("cuda") or "cpu")
        :param epsilon: 최대 허용 perturbation (예: 8/255)
        :param T: 총 반복 횟수 (쿼리 수)
        :param num_classes: 클래스 수 (예: 19)
        :param ignore_label: 무시할 라벨 값 (보통 255)
        """
        self.epsilon = epsilon
        self.T = T
        self.model = model
        self.feature_extractor = feature_extractor
        self.gt = gt
        self.device = device
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def attack(self, image):
        """
        입력 PIL 이미지에 대해 Random Attack을 적용하여 adversarial example을 생성합니다.
        
        입력:
            image: PIL Image (RGB, 픽셀 값 범위 0~255)
        출력:
            adversarial 예시 (PIL Image)
        """
        # PIL 이미지를 numpy 배열로 변환하고 [0,1] 범위로 정규화합니다.
        x = np.array(image).astype(np.float32) / 255.0
        
        # 원본 이미지 기준 ±epsilon 범위 (단, 값은 [0,1] 내에서 클리핑)
        x_min = np.clip(x - self.epsilon, 0.0, 1.0)
        x_max = np.clip(x + self.epsilon, 0.0, 1.0)
        
        # 초기 adversarial 예시는 원본 이미지로 설정합니다.
        x_adv = x.copy()
        best_miou = self.proxy_index(x_adv)
        print(f"Initial mIoU: {best_miou:.4f}")
        
        for t in range(self.T):
            # [-epsilon/16, epsilon/16] 범위의 랜덤 노이즈 생성
            noise = np.random.uniform(low=-self.epsilon/16, high=self.epsilon/16, size=x.shape)
            candidate = x_adv + noise
            # 후보 이미지가 [x - epsilon, x + epsilon] 범위를 벗어나지 않도록 클리핑
            candidate = np.clip(candidate, x_min, x_max)
            candidate_miou = self.proxy_index(candidate)
            
            # mIoU가 낮을수록 공격 효과가 좋으므로, 값이 감소할 때만 업데이트합니다.
            if candidate_miou < best_miou:
                best_miou = candidate_miou
                x_adv = candidate
                print(f"Iteration {t+1}: mIoU improved to {best_miou:.4f}")
        
        # 최종 adversarial 예시를 [0,255] 범위로 변환 후 PIL 이미지로 반환합니다.
        x_adv = np.clip(x_adv * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(x_adv)

    def proxy_index(self, x):
        """
        mIoU를 proxy index로 사용합니다.
        입력 x는 [0,1] 범위의 numpy 배열 (H, W, 3)입니다.
        """
        # 후보 이미지를 PIL 이미지로 변환
        candidate_image = Image.fromarray((x * 255).astype(np.uint8))
        inputs = self.feature_extractor(images=candidate_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: (1, num_classes, H_model, W_model)
        # 원본 이미지 크기에 맞게 logits을 보간합니다.
        H, W = x.shape[:2]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
        # 예측 결과(각 픽셀의 클래스)를 구합니다.
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        # mIoU를 계산합니다.
        miou = self.compute_miou(pred, self.gt)
        return miou

    def compute_miou(self, pred, gt):
        """
        예측 segmentation과 ground truth 간의 mIoU를 계산합니다.
        
        :param pred: numpy array, shape (H, W), 예측 segmentation (train id)
        :param gt: numpy array, shape (H, W), ground truth segmentation (train id)
        :return: mIoU 값 (실수)
        """
        num_classes = self.num_classes
        ious = []
        for cls in range(num_classes):
            if cls == self.ignore_label:
                continue
            pred_inds = (pred == cls)
            gt_inds = (gt == cls)
            if np.sum(gt_inds) == 0:
                continue
            intersection = np.sum(np.logical_and(pred_inds, gt_inds))
            union = np.logical_or(pred_inds, gt_inds).sum()
            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union
            ious.append(iou)
        if len(ious) == 0:
            return 1.0  # 만약 계산할 수 있는 클래스가 없으면, mIoU는 1.0 (최악의 공격 효과)
        miou = np.mean(ious)
        return miou
