# configs/transform.py

custom_imports = dict(
    imports=['transformers.sparsefool_transform'], # '폴더명.파일명'
    allow_failed_imports=False
)

# configs/transform.py 내의 파이프라인 정의 부분

train_pipeline = [
    # ... 다른 변환들 ...
    dict(type='PackSegInputs'), # 예시: PackSegInputs 다음에 적용
    dict(
        type='ApplySparseFool',  # 등록한 변환 클래스 이름
        model_cfg_path='configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py', # 공격용 모델 설정 파일 경로
        model_ckpt_path='checkpoint/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth', # 공격용 모델 체크포인트 경로
        device='cuda:0',         # 사용할 장치
        steps=10,
        lam=3,
        overshoot=0.02,
        prob=0.5                 # 적용 확률 (예: 50%)
        # 필요한 경우 mean, std 등 추가 인자 전달
    ),
    # ... 이후 변환들 ...
]

# 데이터 로더 설정에서 이 파이프라인 사용
train_dataloader = dict(
    # ...
    dataset=dict(
        # ...
        pipeline=train_pipeline
    )
)