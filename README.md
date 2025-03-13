# 도커 이미지 빌드 및 실행 가이드

## 도커 이미지
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

```bash
docker run --gpus all --ipc=host --network=host --name adverserial_segmentation_model -it -v ${pwd}:/workspace pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
```