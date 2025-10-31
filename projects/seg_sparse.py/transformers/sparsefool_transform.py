import torch
import numpy as np
import torchattacks
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from mmseg.apis import init_segmentor # 모델 로드를 위해 사용
from mmcv.transforms.utils import cache_randomness # 선택적: 랜덤성 재현 위함

# (선택적) 이미지 정규화/역정규화 함수 정의
# 예시: MMSegmentation 기본 설정과 유사한 경우
# mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
# std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
#
# def denormalize(img_tensor, mean, std):
#     mean = torch.tensor(mean, device=img_tensor.device).view(3, 1, 1)
#     std = torch.tensor(std, device=img_tensor.device).view(3, 1, 1)
#     img_tensor = img_tensor * std + mean
#     return torch.clamp(img_tensor / 255.0, 0, 1) # [0, 1] 범위로 변환
#
# def normalize(img_tensor, mean, std):
#     mean = torch.tensor(mean, device=img_tensor.device).view(3, 1, 1)
#     std = torch.tensor(std, device=img_tensor.device).view(3, 1, 1)
#     img_tensor = (img_tensor * 255.0 - mean) / std
#     return img_tensor


@TRANSFORMS.register_module()
class ApplySparseFool(BaseTransform):
    """Apply SparseFool attack using a pre-loaded fixed model.

    Args:
        model_cfg_path (str): Path to the config file of the model used for attack.
        model_ckpt_path (str): Path to the checkpoint file of the model.
        device (str): Device to load the model and perform attack ('cuda:0', 'cpu', etc.).
        steps (int): Number of steps for SparseFool. Default: 10.
        lam (float): Lambda parameter for SparseFool. Default: 3.
        overshoot (float): Overshoot parameter for SparseFool. Default: 0.02.
        # mean (list[float]): Mean values for denormalization/normalization.
        # std (list[float]): Std values for denormalization/normalization.
        prob (float): The probability to apply this transform. Default: 1.0.
    """
    def __init__(self,
                 model_cfg_path: str,
                 model_ckpt_path: str,
                 device: str = 'cuda:0',
                 steps: int = 10,
                 lam: float = 3.,
                 overshoot: float = 0.02,
                 # mean: list = [123.675, 116.28, 103.53], # 예시 값
                 # std: list = [58.395, 57.12, 57.375],   # 예시 값
                 prob: float = 1.0):
        super().__init__()
        self.prob = prob
        self.steps = steps
        self.lam = lam
        self.overshoot = overshoot
        self.device = device
        # self.mean = np.array(mean, dtype=np.float32)
        # self.std = np.array(std, dtype=np.float32)

        # 공격 생성용 보조 모델 로드
        print(f"Loading auxiliary model for SparseFool from {model_ckpt_path}...")
        self.aux_model = init_segmentor(model_cfg_path, model_ckpt_path, device=self.device)
        self.aux_model.eval()
        print("Auxiliary model loaded.")

        # SparseFool 공격 초기화
        self.attack = torchattacks.SparseFool(self.aux_model, steps=self.steps, lam=self.lam, overshoot=self.overshoot)
        print("SparseFool initialized.")

    # @cache_randomness # 필요시 랜덤성 제어 데코레이터 사용
    def transform(self, results: dict) -> dict:
        """Apply SparseFool attack to the image.

        Args:
            results (dict): Result dict containing the image and other info.

        Returns:
            dict: Result dict with the attacked image.
        """
        if np.random.rand() >= self.prob:
             return results

        img = results['img'] # 보통 이 시점에는 numpy 배열 (H, W, C) 또는 torch 텐서일 수 있음

        # --- 이미지 텐서 변환 및 정규화 처리 ---
        if isinstance(img, np.ndarray):
            # MMSeg 파이프라인에서 보통 ToTensor 후 Normalize 하므로,
            # 이 변환이 Normalize 이전에 온다면 numpy -> tensor 변환 필요
            # 만약 Normalize 이후라면, 역정규화 필요
            # 여기서는 이미지가 torch Tensor이고 정규화된 상태라고 가정 (가장 흔한 경우)
            # img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() # 예시: numpy to tensor
            img_tensor = img.to(self.device)
        else: # torch.Tensor인 경우
            img_tensor = img.to(self.device)

        # 중요: SparseFool은 [0, 1] 범위 입력을 가정하는 경우가 많음
        # 현재 img_tensor가 정규화된 상태라면, 역정규화 후 [0, 1] 범위로 변환 필요
        # img_for_attack = denormalize(img_tensor.clone(), self.mean, self.std)
        # 아래는 임시로 정규화 해제를 가정하지 않고, 값 범위를 강제로 [0, 1]로 맞추는 예시 (부정확할 수 있음)
        img_min, img_max = torch.min(img_tensor), torch.max(img_tensor)
        img_for_attack = (img_tensor - img_min) / (img_max - img_min) if img_max > img_min else img_tensor

        # --- 레이블 준비 (중요: 단순화된 예시) ---
        # gt_semantic_seg는 (H, W) 형태의 텐서. SparseFool은 보통 단일 레이블을 기대.
        # 여기서는 임시로 가장 많이 등장하는 클래스 또는 고정된 클래스를 사용한다고 가정.
        # 실제로는 문제에 맞게 적절한 레이블 전략 필요.
        gt_semantic_seg = results['gt_semantic_seg'] # 보통 LongTensor (H, W)
        # 예시: 더미 레이블 (0번 클래스로 공격 시도) - 실제 문제에 맞게 수정 필요!
        # labels_for_attack = torch.tensor([0], device=self.device).long()
        # 또는 고유 클래스 중 하나 선택 등
        unique_labels = torch.unique(gt_semantic_seg)
        labels_for_attack = unique_labels[0:1].to(device=self.device) # 첫 번째 고유 클래스 사용 (매우 단순화됨)

        # 배치 차원 추가 (공격 라이브러리는 보통 배치 입력을 기대)
        img_for_attack = img_for_attack.unsqueeze(0) # (1, C, H, W)

        # --- 공격 적용 ---
        # 주의: SparseFool의 forward가 labels를 어떻게 사용하는지 확인 필요
        # 만약 labels가 타겟 클래스라면, 현재 클래스와 다른 값을 제공해야 할 수 있음
        try:
             adv_img_tensor = self.attack(img_for_attack, labels_for_attack)
        except Exception as e:
             print(f"Warning: SparseFool attack failed: {e}. Returning original image.")
             return results # 공격 실패 시 원본 반환

        # 배치 차원 제거 및 원래 데이터 타입/범위로 복원
        adv_img_tensor = adv_img_tensor.squeeze(0) # (C, H, W)

        # 중요: 공격 후 이미지를 다시 원래 정규화 상태로 돌려놓아야 함
        # (위에서 역정규화 + [0, 1] 변환을 수행했다면)
        # adv_img_tensor = normalize(adv_img_tensor, self.mean, self.std)
        # 위에서 임시로 [0, 1] 범위를 맞췄다면, 다시 원래 범위로 스케일링 (부정확할 수 있음)
        adv_img_tensor = adv_img_tensor * (img_max - img_min) + img_min


        # MMSeg 파이프라인의 다음 단계를 위해 원래 타입(numpy or tensor)으로 돌려놓기
        if isinstance(img, np.ndarray):
             # 결과를 numpy로 변환해야 할 수 있음 (파이프라인 구성에 따라)
             # results['img'] = adv_img_tensor.cpu().numpy().transpose(1, 2, 0)
             # 여기서는 Tensor로 유지한다고 가정
             results['img'] = adv_img_tensor.cpu() # CPU로 돌려놓기 (다음 transform이 CPU에서 실행될 수 있음)
        else:
             results['img'] = adv_img_tensor.cpu() # CPU로 돌려놓기

        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(steps={self.steps}, lam={self.lam}, overshoot={self.overshoot}, ' \
               f'prob={self.prob}, device=\'{self.device}\')'