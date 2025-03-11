import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class NES:
    def __init__(self, feature_extractor, model, epsilon=0.05, learning_rate=0.01, samples_per_draw=50, batch_size=8):
        """
        세그멘테이션 모델을 위한 NES Attack 구현 (메모리 효율적)
        Args:
            feature_extractor: 입력 이미지 전처리를 위한 feature extractor
            model: 대상 모델 (예: SegformerForSemanticSegmentation)
            epsilon: 섭동의 최대 크기
            learning_rate: 학습률
            samples_per_draw: 샘플링 수 (대폭 감소시킴)
            batch_size: 배치 처리 크기 (메모리 관리용)
        """
        self.feature_extractor = feature_extractor
        self.model = model
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.samples_per_draw = samples_per_draw
        self.batch_size = batch_size

    def attack(self, x, y, query=50):
        """
        세그멘테이션 모델에 대한 메모리 효율적인 NES 공격 수행
        Args:
            x: 입력 이미지 (PIL Image)
            y: 타겟 마스크 (numpy array)
            query: 반복 횟수 (감소시킴)
        Returns:
            적대적 이미지 (numpy array)
        """
        # PIL Image를 numpy 배열로 변환
        if isinstance(x, Image.Image):
            x_np = np.array(x) / 255.0  # [0,1] 범위로 정규화
            x = torch.from_numpy(x_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
            if x.ndim == 3:  # [H, W, C]
                x = x.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        # 마스크를 텐서로 변환
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        
        # 모델의 디바이스 확인
        device = next(self.model.parameters()).device
        x = x.to(device)
        y = y.to(device)
        
        # 배치 차원이 있는지 확인하고 없으면 추가
        if x.dim() == 3:  # [C, H, W]
            x = x.unsqueeze(0)  # [1, C, H, W]
            
        x_adv = x.clone().detach()  # [1, C, H, W]
        
        for _ in range(query):
            grad_estimate = torch.zeros_like(x_adv)
            
            # 메모리 효율을 위해 배치 처리
            for i in range(0, self.samples_per_draw, self.batch_size):
                batch_size = min(self.batch_size, self.samples_per_draw - i)
                
                # 가우시안 노이즈로 섭동 생성 (작은 배치)
                noise_shape = (batch_size,) + tuple(x_adv.shape[1:])  # [batch_size, C, H, W]
                noise = torch.randn(noise_shape).to(device)
                noise = noise * 0.001
                
                # eval_points 배열 생성
                # 배치 샘플들을 생성 (첫 번째 배치 차원만 반복)
                eval_points = []
                for j in range(batch_size):
                    perturbed = x_adv + self.epsilon * noise[j:j+1]  # [1, C, H, W] + [1, C, H, W]
                    eval_points.append(perturbed)
                eval_points = x_adv.repeat(batch_size, 1, 1, 1) + self.epsilon * noise
                eval_points = torch.clamp(eval_points, 0, 1)

                # feature_extractor 형식에 맞게 변환 [batch_size, C, H, W] -> [batch_size, H, W, C]
                eval_points_np = eval_points.permute(0, 2, 3, 1).cpu().numpy()
                
                # 배치 크기를 고려하여 텐서 준비
                inputs = self.feature_extractor(images=eval_points_np, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # 모델 출력 및 타겟 형태 확인
                logits = outputs.logits  # [batch_size, num_classes, H', W']
                
                # 타겟 마스크 준비 (y 형태에 따라 다르게 처리)
                if y.dim() == 2:  # [H, W]
                    # 배치 차원 추가 및 크기 조정
                    target = y.unsqueeze(0)  # [1, H, W]
                    if target.shape[1:] != logits.shape[2:]:
                        target = F.interpolate(
                            target.float().unsqueeze(1),  # [1, 1, H, W]
                            size=logits.shape[2:],
                            mode='nearest'
                        ).squeeze(1).long()  # [1, H', W']
                elif y.dim() == 3 and y.shape[0] == 1:  # [1, H, W]
                    target = y  # 이미 배치 차원이 있음
                    if target.shape[1:] != logits.shape[2:]:
                        target = F.interpolate(
                            target.float().unsqueeze(1),  # [1, 1, H, W]
                            size=logits.shape[2:],
                            mode='nearest'
                        ).squeeze(1).long()  # [1, H', W']
                else:
                    # 그 외 형태는 에러 출력
                    raise ValueError(f"타겟 마스크의 형태가 잘못되었습니다: {y.shape}")
                
                # 클래스 수 확인 및 유효하지 않은 인덱스 처리
                num_classes = logits.shape[1]  # 모델의 클래스 수
                
                # 타겟에서 유효하지 않은 값을 0으로 대체 (또는 다른 유효한 클래스로)
                target = torch.where(target >= num_classes, torch.zeros_like(target), target)
                
                # 타겟을 배치 크기에 맞게 복제
                target_batch = target.repeat(batch_size, 1, 1)  # [batch_size, H', W']
                
                # 손실 계산 - 메모리 사용 최적화
                losses = F.cross_entropy(
                    logits,  # [batch_size, num_classes, H', W']
                    target_batch,  # [batch_size, H', W']
                    reduction='none',
                    ignore_index=255  # 255는 일반적으로 세그멘테이션에서 무시되는 인덱스
                )
                
                # 손실을 배치 단위로 재구성 (메모리 효율적)
                losses = losses.mean(dim=(1, 2))  # 각 샘플의 평균 손실, [batch_size]
                
                # 배치별 그래디언트 추정 누적
                for j in range(batch_size):
                    grad_estimate += (losses.view(-1, 1, 1, 1) * noise).sum(dim=0) / self.samples_per_draw

                # 명시적 메모리 해제
                del eval_points, outputs, losses, noise
                torch.cuda.empty_cache()
            
            # 적대적 예제 업데이트
            x_adv = x_adv + self.learning_rate * grad_estimate.sign()
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # 명시적 메모리 해제
            torch.cuda.empty_cache()
        
        # 결과를 numpy 배열로 변환
        x_adv_np = x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return x_adv_np

    def __call__(self, x, y):
        return self.attack(x, y)