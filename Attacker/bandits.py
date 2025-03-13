import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

class Bandits:
    """
    Bandits 공격 알고리즘 구현
    
    참고 논문: "Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors"
    """
    def __init__(self, feature_extractor, model, epsilon=0.05, learning_rate=0.01, 
                 fd_eta=0.1, prior_size=50, data_independent=True, batch_size=8):
        self.feature_extractor = feature_extractor
        self.model = model
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.fd_eta = fd_eta  # 유한 차분법을 위한 스텝 크기
        self.prior_size = prior_size  # 이전 그래디언트 활용을 위한 크기
        self.data_independent = data_independent  # 데이터 독립적 사전 정보 사용 여부
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        
    def _compute_loss(self, image, target):
        """
        주어진 이미지와 타겟에 대한 손실 계산
        """
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        
        # 크로스 엔트로피 손실 계산
        outputs = F.interpolate(outputs, size=target.shape, mode="bilinear", align_corners=True)
        outputs = outputs.squeeze(0)
        
        # 타겟 마스크에서 무시해야 할 픽셀(255) 제외
        mask = target != 255
        valid_outputs = outputs[:, mask]
        
        # numpy 배열을 텐서로 변환 (오류 수정)
        if not isinstance(target, torch.Tensor):
            target_tensor = torch.from_numpy(target).to(self.device)
        else:
            target_tensor = target.to(self.device)
        
        valid_target = target_tensor[mask]
        
        # 손실 계산 (음수 로그 가능도)
        log_probs = F.log_softmax(valid_outputs, dim=0)
        loss = F.nll_loss(log_probs.permute(1, 0), valid_target)
        
        return loss.item()
    
    def _estimate_gradient(self, image_np, target, prior, exploration_samples=10):
        """
        Bandits 방식으로 그래디언트 추정
        """
        image_tensor = torch.from_numpy(image_np).float().to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # target이 numpy 배열인 경우 텐서로 변환 (오류 수정)
        if not isinstance(target, torch.Tensor):
            target_tensor = torch.from_numpy(target).long().to(self.device)
        else:
            target_tensor = target.long().to(self.device)
        
        # 기본 손실 계산
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        base_loss = self._compute_loss(image_pil, target_tensor)
        
        # 그래디언트 추정을 위한 탐색
        est_grad = torch.zeros_like(image_tensor)
        
        # 배치 처리를 위한 준비
        for i in range(0, exploration_samples, self.batch_size):
            batch_size = min(self.batch_size, exploration_samples - i)
            
            # 무작위 방향 생성 (사전 정보 활용)
            if prior is not None and not self.data_independent:
                # 데이터 의존적 사전 정보 사용
                noise_shape = (batch_size,) + image_tensor.shape[1:]
                noise = torch.randn(noise_shape, device=self.device)
                noise = noise / torch.norm(noise, p=2, dim=(1, 2, 3), keepdim=True)
                
                # 사전 정보와 무작위 노이즈 결합
                noise = prior.unsqueeze(0) + noise
                noise = noise / torch.norm(noise, p=2, dim=(1, 2, 3), keepdim=True)
            else:
                # 데이터 독립적 또는 사전 정보 없음
                noise_shape = (batch_size,) + image_tensor.shape[1:]
                noise = torch.randn(noise_shape, device=self.device)
                noise = noise / torch.norm(noise, p=2, dim=(1, 2, 3), keepdim=True)
            
            # 유한 차분법으로 그래디언트 추정
            queries = []
            for j in range(batch_size):
                perturbed_image = image_tensor + self.fd_eta * noise[j].unsqueeze(0)
                perturbed_image = torch.clamp(perturbed_image, 0, 1)
                perturbed_np = perturbed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                perturbed_pil = Image.fromarray((perturbed_np * 255).astype(np.uint8))
                queries.append(perturbed_pil)
            
            # 배치 손실 계산
            batch_losses = []
            for query in queries:
                loss = self._compute_loss(query, target_tensor)
                batch_losses.append(loss)
            
            # 그래디언트 추정 업데이트
            for j in range(batch_size):
                est_grad += (batch_losses[j] - base_loss) / self.fd_eta * noise[j].unsqueeze(0)
        
        # 평균 그래디언트 계산
        est_grad /= exploration_samples
        
        # 그래디언트 정규화
        norm = torch.norm(est_grad, p=2)
        if norm > 0:
            est_grad = est_grad / norm
        
        return est_grad.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    def attack(self, image, target, query=200):
        """
        Bandits 공격 수행
        """
        # 이미지와 타겟을 numpy 배열로 변환
        image_np = np.array(image).astype(np.float32) / 255.0
        target_np = np.array(target)
        
        # 초기 사전 정보 설정
        prior = None
        
        # 공격 반복
        steps = query // (self.prior_size + 1)  # 쿼리 예산 내에서 스텝 수 계산
        
        for i in range(steps):
            # 그래디언트 추정
            grad = self._estimate_gradient(image_np, target_np, prior, self.prior_size)
            
            # 사전 정보 업데이트
            if prior is None:
                prior = torch.from_numpy(grad).to(self.device).permute(2, 0, 1)
            else:
                prior = 0.9 * prior + 0.1 * torch.from_numpy(grad).to(self.device).permute(2, 0, 1)
                prior = prior / torch.norm(prior, p=2)
            
            # 적대적 예제 업데이트
            image_np = image_np - self.learning_rate * np.sign(grad)
            
            # 섭동 제한
            orig_img = np.array(image).astype(np.float32) / 255.0
            image_np = np.clip(image_np, orig_img - self.epsilon, orig_img + self.epsilon)
            image_np = np.clip(image_np, 0, 1)
        
        return image_np
