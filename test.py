import torch

# CUDA 버전 확인
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA 버전: {cuda_version}")
    
    # 현재 사용 가능한 GPU 정보 출력
    device_count = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
else:
    print("CUDA를 사용할 수 없습니다.")

print(torch.__version__)