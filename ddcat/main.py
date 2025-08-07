import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import argparse
import os
import sys
import evaluate

# 현재 디렉토리를 sys.path에 추가 (relative import 해결)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# ddcat 모델들 import
from model import DeepLabV3, DeepLabV3_DDCAT, PSPNet, PSPNet_DDCAT

# 상위 디렉토리에서 dataset 클래스들 import
from dataset import CitySet, ADESet, VOCSet

def load_config(dataset_type, model_type):
    """configs_attack에서 설정 로드"""
    config_map = {
        'deeplabv3': 'config_deeplabv3',
        'deeplabv3_ddcat': 'config_deeplabv3',
        'pspnet': 'config_pspnet', 
        'pspnet_ddcat': 'config_pspnet'
    }
    
    config_name = config_map.get(model_type, 'config_deeplabv3')
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs_attack', dataset_type, f'{config_name}.py'
    )
    
    if not os.path.exists(config_path):
        print(f"경고: 설정 파일을 찾을 수 없습니다: {config_path}")
        return None
    
    # config 파일을 동적으로 import
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    return config_module.config

class DDCATInferenceWithDataset:
    def __init__(self, model_type='deeplabv3_ddcat', model_path=None, device='cuda'):
        """
        DDCAT 모델을 사용한 데이터셋 추론 및 mIoU 계산 클래스
        
        Args:
            model_type (str): 사용할 모델 타입 ('deeplabv3', 'deeplabv3_ddcat', 'pspnet', 'pspnet_ddcat')
            model_path (str): 학습된 모델 가중치 경로
            device (str): 추론에 사용할 디바이스 ('cuda' 또는 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        print(f"디바이스: {self.device}")
        print(f"모델 타입: {model_type}")
        
        # 모델 초기화
        self.model = self._load_model(model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # 모델 가중치 로드
        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
        else:
            print("사전훈련된 가중치를 사용하지 않습니다.")
            
        # 전처리 변환 (BGR -> RGB -> Tensor -> Normalize)
        # dataset.py는 BGR 형태로 반환하므로 RGB로 변환 필요
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # mIoU 계산기 초기화
        self.miou_calculator = evaluate.load("mean_iou")
        
    def _load_model(self, model_type):
        """모델 타입에 따라 모델 초기화"""
        if model_type == 'deeplabv3':
            return DeepLabV3(layers=50, classes=2, pretrained=True)
        elif model_type == 'deeplabv3_ddcat':
            return DeepLabV3_DDCAT(layers=50, classes=2, pretrained=True)
        elif model_type == 'pspnet':
            return PSPNet(layers=50, classes=2, pretrained=True)
        elif model_type == 'pspnet_ddcat':
            return PSPNet_DDCAT(layers=50, classes=2, pretrained=True)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    def _load_weights(self, model_path):
        """학습된 가중치 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"모델 가중치를 성공적으로 로드했습니다: {model_path}")
        except Exception as e:
            print(f"모델 가중치 로드 중 오류 발생: {e}")
    
    def preprocess_image_from_dataset(self, bgr_image, target_size=(512, 512)):
        """dataset.py에서 가져온 BGR 이미지 전처리"""
        # BGR -> RGB 변환
        if len(bgr_image.shape) == 3:
            rgb_image = bgr_image[:, :, ::-1]  # BGR to RGB
        else:
            rgb_image = bgr_image
            
        # PIL Image로 변환
        image = Image.fromarray(rgb_image.astype(np.uint8))
        
        # 원본 크기 저장
        original_size = image.size
        
        # 크기 조정 (8의 배수로 조정 - 모델 요구사항)
        width, height = target_size
        width = ((width - 1) // 8 + 1) * 8 + 1
        height = ((height - 1) // 8 + 1) * 8 + 1
        
        image = image.resize((width, height), Image.BILINEAR)
        
        # 텐서로 변환 및 정규화
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor, original_size, (width, height)
    
    def postprocess_output(self, output, original_size, processed_size):
        """모델 출력 후처리"""
        if isinstance(output, tuple):
            # training 모드에서의 출력인 경우
            output = output[0] if len(output) > 0 else output
            
        # logits를 확률로 변환
        if output.dim() == 4:  # (batch, classes, height, width)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        else:
            predictions = output
            
        # 배치 차원 제거
        if predictions.dim() == 3:
            predictions = predictions.squeeze(0)
        
        # CPU로 이동하고 numpy로 변환
        predictions = predictions.cpu().numpy().astype(np.uint8)
        
        # 원본 크기로 복원
        if original_size != processed_size:
            predictions = cv2.resize(predictions, original_size, interpolation=cv2.INTER_NEAREST)
        
        return predictions
    
    def load_dataset(self, dataset_type, dataset_dir):
        """데이터셋 로드"""
        if dataset_type.lower() == 'cityscapes':
            return CitySet(dataset_dir, use_gt=True)
        elif dataset_type.lower() == 'ade20k':
            return ADESet(dataset_dir, use_gt=True)
        elif dataset_type.lower() == 'voc2012':
            return VOCSet(dataset_dir, use_gt=True)
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset_type}")
    
    def calculate_miou(self, predictions, ground_truths, num_classes, ignore_index=255, reduce_labels=False):
        """mIoU 계산"""
        try:
            miou_result = self.miou_calculator.compute(
                predictions=predictions,
                references=ground_truths,
                num_labels=num_classes,
                ignore_index=ignore_index,
                reduce_labels=reduce_labels
            )
            return miou_result
        except Exception as e:
            print(f"mIoU 계산 중 오류 발생: {e}")
            return None
    
    def evaluate_dataset(self, dataset_type, dataset_dir=None, target_size=(512, 512), num_classes=None, max_samples=None, config=None):
        """데이터셋 전체 평가 및 mIoU 계산"""
        
        # config가 제공되지 않으면 자동 로드
        if config is None:
            config = load_config(dataset_type, self.model_type)
        
        # config에서 설정 가져오기
        if config:
            print(f"configs_attack에서 설정 로드: {config}")
            if dataset_dir is None:
                dataset_dir = config.get('data_dir', f'datasets/{dataset_type}')
            if num_classes is None:
                num_classes = config.get('num_class', 2)
            print(f"설정된 클래스 수: {num_classes}")
            print(f"데이터 디렉토리: {dataset_dir}")
        else:
            # 기본 설정 사용
            if dataset_dir is None:
                dataset_dir = f'datasets/{dataset_type}'
            if num_classes is None:
                if dataset_type.lower() == 'cityscapes':
                    num_classes = 19
                elif dataset_type.lower() == 'ade20k':
                    num_classes = 150
                elif dataset_type.lower() == 'voc2012':
                    num_classes = 21
                else:
                    num_classes = 2  # 기본값
        
        # 데이터셋 디렉토리 존재 확인
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"데이터셋 디렉토리를 찾을 수 없습니다: {dataset_dir}")
        
        # 데이터셋 로드
        print(f"=== {dataset_type.upper()} 데이터셋 평가 시작 ===")
        print(f"데이터 디렉토리: {dataset_dir}")
        dataset = self.load_dataset(dataset_type, dataset_dir)
        print(f"데이터셋 크기: {len(dataset)}개 이미지")
        
        # 샘플 수 제한
        total_samples = len(dataset)
        if max_samples and max_samples < total_samples:
            total_samples = max_samples
            print(f"샘플 수를 {max_samples}개로 제한합니다.")
        
        all_predictions = []
        all_ground_truths = []
        
        for i in range(total_samples):
            print(f"[{i+1}/{total_samples}] 처리 중...")
            
            try:
                # 데이터셋에서 이미지와 GT 가져오기
                bgr_image, filename, gt_image = dataset[i]
                
                print(f"  파일: {filename}")
                print(f"  이미지 크기: {bgr_image.shape}")
                print(f"  GT 크기: {gt_image.shape}")
                
                # 전처리
                image_tensor, original_size, processed_size = self.preprocess_image_from_dataset(
                    bgr_image, target_size
                )
                image_tensor = image_tensor.to(self.device)
                
                # 추론
                with torch.no_grad():
                    output = self.model(image_tensor)
                
                # 후처리
                prediction = self.postprocess_output(output, original_size, processed_size)
                
                # Ground Truth 크기 조정 (예측 결과와 맞춤)
                if gt_image.shape[:2] != prediction.shape:
                    gt_resized = cv2.resize(gt_image, 
                                          (prediction.shape[1], prediction.shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
                else:
                    gt_resized = gt_image
                
                all_predictions.append(prediction)
                all_ground_truths.append(gt_resized)
                
                print(f"  완료 - 예측: {prediction.shape}, GT: {gt_resized.shape}")
                
            except Exception as e:
                print(f"  오류 발생: {e}")
                continue
        
        # mIoU 계산
        if all_predictions:
            print(f"\n=== {len(all_predictions)}개 이미지 mIoU 계산 ===")
            
            # 데이터셋별 reduce_labels 설정
            reduce_labels = (dataset_type.lower() == 'ade20k')
            
            miou_result = self.calculate_miou(
                all_predictions, all_ground_truths, 
                num_classes, reduce_labels=reduce_labels
            )
            
            if miou_result:
                print(f"전체 mIoU: {miou_result['mean_iou']:.4f}")
                print(f"클래스별 IoU:")
                for class_idx, iou in enumerate(miou_result['per_class_iou']):
                    if not np.isnan(iou):
                        print(f"  클래스 {class_idx}: {iou:.4f}")
                
                return miou_result
            else:
                print("mIoU 계산에 실패했습니다.")
                return None
        else:
            print("처리된 이미지가 없습니다.")
            return None
    
def main():
    parser = argparse.ArgumentParser(description='DDCAT 모델을 사용한 데이터셋 평가 및 mIoU 계산 (configs_attack 설정 사용)')
    parser.add_argument('--dataset_type', type=str, required=True,
                      choices=['cityscapes', 'ade20k', 'VOC2012'],
                      help='데이터셋 타입')
    parser.add_argument('--dataset_dir', type=str,
                      help='데이터셋 디렉토리 경로 (기본값: configs_attack에서 자동 설정)')
    parser.add_argument('--model_type', type=str, default='deeplabv3_ddcat',
                      choices=['deeplabv3', 'deeplabv3_ddcat', 'pspnet', 'pspnet_ddcat'],
                      help='사용할 모델 타입')
    parser.add_argument('--model_path', type=str, help='학습된 모델 가중치 경로')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                      help='추론에 사용할 디바이스')
    parser.add_argument('--target_size', type=int, nargs=2, default=[512, 512],
                      help='입력 이미지 크기 (width height)')
    parser.add_argument('--num_classes', type=int, help='클래스 수 (기본값: configs_attack에서 자동 설정)')
    parser.add_argument('--max_samples', type=int, help='평가할 최대 샘플 수 (전체 데이터셋 크기 제한)')
    
    args = parser.parse_args()
    
    print(f"🚀 DDCAT 모델 평가 시작")
    print(f"   데이터셋: {args.dataset_type}")
    print(f"   모델: {args.model_type}")
    
    # configs_attack에서 설정 로드
    config = load_config(args.dataset_type, args.model_type)
    if config:
        print(f"✅ configs_attack에서 설정 로드 완료")
    else:
        print(f"⚠️  configs_attack 설정을 찾을 수 없어 기본 설정을 사용합니다")
    
    try:
        # 추론 객체 생성
        evaluator = DDCATInferenceWithDataset(
            model_type=args.model_type,
            model_path=args.model_path,
            device=args.device
        )
        
        # 데이터셋 평가 실행 (config 전달)
        miou_result = evaluator.evaluate_dataset(
            dataset_type=args.dataset_type,
            dataset_dir=args.dataset_dir,  # None일 수 있음 (config에서 자동 설정)
            target_size=tuple(args.target_size),
            num_classes=args.num_classes,  # None일 수 있음 (config에서 자동 설정)
            max_samples=args.max_samples,
            config=config
        )
        
        if miou_result:
            print(f"\n🎯 최종 결과:")
            print(f"   mIoU: {miou_result['mean_iou']:.4f}")
            print(f"   사용된 모델: {args.model_type}")
            print(f"   데이터셋: {args.dataset_type}")
            if config:
                print(f"   설정 파일: configs_attack/{args.dataset_type}/config_{args.model_type.replace('_ddcat', '')}.py")
        else:
            print("❌ 평가에 실패했습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
