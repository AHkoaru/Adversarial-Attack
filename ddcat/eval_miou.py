import os
import sys
import time
import logging
import argparse
import numpy as np

# 상위 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn

from model import PSPNet, DeepLabV3, PSPNet_DDCAT, DeepLabV3_DDCAT
from dataset import CitySet, ADESet, VOCSet
from sparse_rs import RSAttack
from pixle import Pixle

cv2.ocl.setUseOpenCL(False)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation mIoU Evaluation')
    parser.add_argument('--dataset_type', type=str, choices=['VOC2012', 'ADE20K', 'Cityscapes'], 
                        default='VOC2012', help='Dataset type')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['PSPNet', 'DeepLabV3', 'PSPNet_DDCAT', 'DeepLabV3_DDCAT'], 
                        default='PSPNet', help='Model type')
    parser.add_argument('--layers', type=int, default=50, 
                        help='ResNet layers (50, 101, 152)')
    parser.add_argument('--classes', type=int, default=21, 
                        help='Number of classes')
    parser.add_argument('--zoom_factor', type=int, default=8, 
                        help='Zoom factor')
    parser.add_argument('--input_size', type=int, default=512, 
                        help='Input image size')
    parser.add_argument('--test_gpu', type=int, nargs='+', default=[0], 
                        help='GPU ids for testing')
    parser.add_argument('--num_images', type=int, default=0, 
                        help='Number of images to evaluate (0 for all images)')
    parser.add_argument('--attack', action='store_true', 
                        help='Evaluate with adversarial attack')
    parser.add_argument('--attack_type', type=str, choices=['BIM', 'pixel', 'sparse_rs'], 
                        default='BIM', help='Type of adversarial attack')
    parser.add_argument('--pixle_restarts', type=int, default=250, 
                        help='Number of restarts for Pixle attack')
    parser.add_argument('--pixle_max_iterations', type=int, default=20, 
                        help='Number of max iterations per restart for Pixle attack')
    parser.add_argument('--sparse_rs_queries', type=int, default=5000, 
                        help='Number of queries for Sparse RS attack')
    parser.add_argument('--sparse_rs_eps', type=float, default=0.05, 
                        help='Sparsity parameter for Sparse RS attack')
    parser.add_argument('--sparse_rs_iters', type=int, default=500, 
                        help='Number of iterations for Sparse RS attack')
    parser.add_argument('--save_pred', action='store_true', 
                        help='Save prediction results')
    parser.add_argument('--save_folder', type=str, default='./results', 
                        help='Folder to save results')
    
    args = parser.parse_args()
    return args


def get_logger():
    logger_name = "miou-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def FGSM(input, target, model, clip_min, clip_max, eps=0.2, mean_origin=None, std_origin=None):
    input_variable = input.detach().clone()
    input_variable.requires_grad = True
    model.zero_grad()
    result = model(input_variable)
    
    ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label).cuda()
    loss = criterion(result, target.detach())
    loss.backward()
    res = input_variable.grad

    adversarial_example = input.detach().clone()
    if mean_origin is not None and std_origin is not None:
        # Denormalize
        adversarial_example[:, 0, :, :] = adversarial_example[:, 0, :, :] * std_origin[0] + mean_origin[0]
        adversarial_example[:, 1, :, :] = adversarial_example[:, 1, :, :] * std_origin[1] + mean_origin[1]
        adversarial_example[:, 2, :, :] = adversarial_example[:, 2, :, :] * std_origin[2] + mean_origin[2]
    
    adversarial_example = adversarial_example + eps * torch.sign(res)
    adversarial_example = torch.max(adversarial_example, clip_min)
    adversarial_example = torch.min(adversarial_example, clip_max)
    adversarial_example = torch.clamp(adversarial_example, min=0.0, max=1.0)

    if mean_origin is not None and std_origin is not None:
        # Normalize back
        adversarial_example[:, 0, :, :] = (adversarial_example[:, 0, :, :] - mean_origin[0]) / std_origin[0]
        adversarial_example[:, 1, :, :] = (adversarial_example[:, 1, :, :] - mean_origin[1]) / std_origin[1]
        adversarial_example[:, 2, :, :] = (adversarial_example[:, 2, :, :] - mean_origin[2]) / std_origin[2]
    
    return adversarial_example


def BIM(input, target, model, eps=0.03, k_number=2, alpha=0.01, mean_origin=None, std_origin=None):
    """원본 코드의 BIM 함수 수정"""
    input_unnorm = input.clone().detach()
    if mean_origin is not None and std_origin is not None:
        input_unnorm[:, 0, :, :] = input_unnorm[:, 0, :, :] * std_origin[0] + mean_origin[0]
        input_unnorm[:, 1, :, :] = input_unnorm[:, 1, :, :] * std_origin[1] + mean_origin[1]
        input_unnorm[:, 2, :, :] = input_unnorm[:, 2, :, :] * std_origin[2] + mean_origin[2]
    
    clip_min = input_unnorm - eps
    clip_max = input_unnorm + eps

    adversarial_example = input.detach().clone()
    adversarial_example.requires_grad = True
    for mm in range(k_number):
        adversarial_example = FGSM(adversarial_example, target, model, clip_min, clip_max, 
                                   eps=alpha, mean_origin=mean_origin, std_origin=std_origin)
        adversarial_example = adversarial_example.detach()
        adversarial_example.requires_grad = True
        model.zero_grad()
    return adversarial_example


def net_process(model, image, target, mean, std=None, attack_type='BIM', args=None):
    """원본 코드의 net_process 함수 구현"""
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    target = torch.from_numpy(target).long()

    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    target = target.unsqueeze(0).cuda()

    # 공격 여부에 따라 flip 설정
    if args and args.attack:
        flip = False
    else:
        flip = True

    if flip:
        input = torch.cat([input, input.flip(3)], 0)
        target = torch.cat([target, target.flip(2)], 0)

    # 공격 적용
    if args and args.attack:
        if attack_type == 'BIM':
            mean_origin = [0.406, 0.456, 0.485]  # BGR 순서
            std_origin = [0.225, 0.224, 0.229]   # BGR 순서
            adver_input = BIM(input, target, model, eps=0.03, k_number=2, alpha=0.01, 
                             mean_origin=mean_origin, std_origin=std_origin)
            with torch.no_grad():
                output = model(adver_input)
        else:
            # 다른 공격 방식들은 패치 단위로는 적용하지 않음 (복잡성 때문에)
            with torch.no_grad():
                output = model(input)
    else:
        with torch.no_grad():
            output = model(input)

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, target, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3, args=None):
    """원본 코드의 scale_process 함수 구현"""
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
        target = cv2.copyMakeBorder(target, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=255)

    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            target_crop = target[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            
            # net_process 호출 (attack_type는 BIM만 패치 단위로 지원)
            attack_type = 'BIM' if args and args.attack else None
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, target_crop, mean, std, attack_type, args)
    
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def evaluate_model(model, dataset, args, logger):
    """원본 코드의 test 함수를 기반으로 한 평가 함수"""
    
    # 평가할 이미지 개수 결정
    total_images = len(dataset)
    num_images = total_images if args.num_images == 0 else min(args.num_images, total_images)
    
    if args.attack:
        logger.info(f'Starting evaluation on {num_images}/{total_images} images with {args.attack_type} attack...')
        
        # Dataset configuration for attacks
        if args.dataset_type == 'VOC2012':
            cfg = {"num_class": args.classes, "dataset": "VOC2012"}
        elif args.dataset_type == 'ADE20K':
            cfg = {"num_class": args.classes, "dataset": "ade20k"}
        elif args.dataset_type == 'Cityscapes':
            cfg = {"num_class": args.classes, "dataset": "cityscapes"}
        
        if args.attack_type == 'pixel':
            logger.info(f'Pixle attack - restarts: {args.pixle_restarts}, max_iterations: {args.pixle_max_iterations}')
        elif args.attack_type == 'sparse_rs':
            logger.info(f'Sparse RS attack - queries: {args.sparse_rs_queries}, eps: {args.sparse_rs_eps}, iters: {args.sparse_rs_iters}')
    else:
        logger.info(f'Starting evaluation on {num_images}/{total_images} images...')
    
    # Normalization parameters (BGR 순서)
    value_scale = 255
    mean = [0.406, 0.456, 0.485]  # [B, G, R]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]   # [B, G, R]
    std = [item * value_scale for item in std]
    
    mean_origin = [0.406, 0.456, 0.485]  # BGR 순서
    std_origin = [0.225, 0.224, 0.229]   # BGR 순서
    
    # 원본 코드의 설정값들 (config에서 가져오는 값들을 하드코딩)
    base_size = 512  # 원본 코드의 base_size
    crop_h = 473     # test_h (원본 설정값)
    crop_w = 473     # test_w (원본 설정값)
    scales = [1.0]   # 단일 스케일로 시작 (원본에서는 [0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    
    # Meters for computing mIoU
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    model.eval()
    
    if args.save_pred:
        os.makedirs(args.save_folder, exist_ok=True)
    
    start_time = time.time()
    
    for i in range(num_images):
        image, filename, gt = dataset[i]
        
        # Pixle 또는 Sparse RS 공격을 전체 이미지에 적용
        if args.attack and args.attack_type in ['pixel', 'sparse_rs']:
            if args.attack_type == 'pixel':
                # Pixle 공격 적용 (main.py 방식)
                gt_tensor = torch.from_numpy(gt).unsqueeze(0).cuda()
                
                # 이미지 크기에 따른 패치 크기 계산
                H, W = image.shape[:2]
                attack_pixel_ratio = 0.05  # 기본값
                total_target_pixels_overall = H * W * attack_pixel_ratio
                pixels_per_single_patch_target = total_target_pixels_overall / args.pixle_restarts
                
                # 패치 크기 계산
                target_area_int = int(round(pixels_per_single_patch_target))
                h_found = 1
                for h_candidate in range(int(np.sqrt(target_area_int)), 0, -1):
                    if target_area_int % h_candidate == 0:
                        h_found = h_candidate
                        break
                patch_h_pixels = h_found
                patch_w_pixels = target_area_int // patch_h_pixels
                
                # Pixle 객체 생성
                pixle = Pixle(
                    model,
                    x_dimensions=(patch_w_pixels, patch_w_pixels),
                    y_dimensions=(patch_h_pixels, patch_h_pixels),
                    restarts=args.pixle_restarts,
                    max_iterations=args.pixle_max_iterations,
                    threshold=21000,
                    device='cuda',
                    cfg=cfg
                )
                
                # BGR 이미지를 Pixle 공격에 입력
                img_tensor_bgr = torch.from_numpy(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().cuda()
                
                # Pixle 공격 실행
                results = pixle.forward(img_tensor_bgr, gt_tensor)
                
                # 결과 처리
                if results['adv_images']:
                    # 마지막 (최고) 적대적 이미지 선택
                    adv_img_bgr_tensor = results['adv_images'][-1]
                    image = adv_img_bgr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # 공격 실패 시 원본 이미지 사용
                
            elif args.attack_type == 'sparse_rs':
                # Sparse RS 공격 적용 (rs_eval.py 방식)
                gt_tensor = torch.from_numpy(gt).unsqueeze(0).cuda()
                
                # RSAttack 객체 생성
                attack = RSAttack(
                    model=model,
                    cfg=cfg,
                    norm='L0',
                    n_queries=args.sparse_rs_queries,
                    eps=args.sparse_rs_eps,
                    p_init=0.8,
                    n_restarts=1,
                    seed=0,
                    verbose=False,
                    targeted=False,
                    loss='margin',
                    resc_schedule=True,
                    device='cuda',
                    log_path=None,
                    original_img=image,  # 원본 BGR 이미지 전달
                    d=5,
                    use_decision_loss=False
                )
                
                # BGR 이미지를 Sparse RS 공격에 입력
                img_tensor_bgr = torch.from_numpy(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().cuda()
                
                # 반복 공격 로직 적용
                adv_img_bgr_list = []
                total_queries = args.sparse_rs_iters * args.sparse_rs_queries
                save_steps = [int(total_queries * (i+1) / 5) for i in range(5)]
                
                for iter_idx in range(args.sparse_rs_iters):
                    current_query, adv_img_bgr_tensor = attack.perturb(img_tensor_bgr, gt_tensor)
                    img_tensor_bgr = adv_img_bgr_tensor
                    # 다음 iteration을 위해 업데이트
                    if current_query in save_steps:
                        adv_img_bgr_list.append(adv_img_bgr_tensor)
                
                # 모든 save_steps에 도달하지 못한 경우 마지막 결과로 채우기
                while len(adv_img_bgr_list) < 5:
                    adv_img_bgr_list.append(adv_img_bgr_tensor)
                
                # 마지막 결과 사용 (최종 적대적 이미지)
                final_adv_img_bgr_tensor = adv_img_bgr_list[-1]
                
                # 결과 처리
                image = final_adv_img_bgr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # 원본 이미지 크기
        h, w, _ = image.shape
        
        # 멀티스케일 예측 (원본 코드 방식)
        prediction = np.zeros((h, w, args.classes), dtype=float)
        
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)

            # PSPNet 조건에 맞게 크기 조정
            if (new_h - 1) % 8 != 0:
                new_h = ((new_h - 1) // 8 + 1) * 8 + 1
            if (new_w - 1) % 8 != 0:
                new_w = ((new_w - 1) // 8 + 1) * 8 + 1

            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            target_scale = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # 슬라이딩 윈도우로 예측 (BIM 공격은 패치 단위로 적용됨)
            if args.attack and args.attack_type == 'BIM':
                prediction += scale_process(model, image_scale, target_scale, args.classes, crop_h, crop_w, h, w, mean, std, args=args)
            else:
                # Pixle/Sparse RS는 이미 전체 이미지에 공격이 적용되었으므로 일반 처리
                prediction += scale_process(model, image_scale, target_scale, args.classes, crop_h, crop_w, h, w, mean, std, args=None)
        
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        
        # Compute intersection and union
        intersection, union, target = intersectionAndUnion(prediction, gt, args.classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        
        # Compute current accuracy
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        
        if (i + 1) % 10 == 0 or (i + 1) == num_images:
            elapsed_time = time.time() - start_time
            logger.info(f'Processed [{i+1}/{num_images}] images, current accuracy: {accuracy:.4f}, time: {elapsed_time:.1f}s')
        
        # Save prediction if requested
        if args.save_pred:
            pred_filename = os.path.splitext(filename)[0] + '_pred.png'
            pred_path = os.path.join(args.save_folder, pred_filename)
            cv2.imwrite(pred_path, prediction.astype(np.uint8))
    
    # Compute final metrics
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    return mIoU, mAcc, allAcc, iou_class, accuracy_class


def main():
    args = get_parser()
    logger = get_logger()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    
    logger.info(f"Arguments: {args}")
    
    # Load dataset
    logger.info(f"Loading {args.dataset_type} dataset from {args.dataset_dir}")
    
    if args.dataset_type == 'VOC2012':
        dataset = VOCSet(dataset_dir=args.dataset_dir, use_gt=True)
    elif args.dataset_type == 'ADE20K':
        dataset = ADESet(dataset_dir=args.dataset_dir, use_gt=True)
    elif args.dataset_type == 'Cityscapes':
        dataset = CitySet(dataset_dir=args.dataset_dir, use_gt=True)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
    
    logger.info(f"Loaded {len(dataset)} images")
    
    # Load model
    logger.info("Loading model...")
    if args.model_type == 'PSPNet':
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
    elif args.model_type == 'DeepLabV3':
        model = DeepLabV3(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
    elif args.model_type == 'PSPNet_DDCAT':
        model = PSPNet_DDCAT(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
    elif args.model_type == 'DeepLabV3_DDCAT':
        model = DeepLabV3_DDCAT(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    
    # Load checkpoint
    if os.path.isfile(args.model_path):
        logger.info(f"Loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f"Loaded checkpoint '{args.model_path}'")
    else:
        raise RuntimeError(f"No checkpoint found at '{args.model_path}'")
    
    # Evaluate model
    logger.info("Starting evaluation...")
    start_time = time.time()
    
    mIoU, mAcc, allAcc, iou_class, accuracy_class = evaluate_model(model, dataset, args, logger)
    
    eval_time = time.time() - start_time
    
    # Print results
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f'mIoU: {mIoU:.4f}')
    logger.info(f'mAcc: {mAcc:.4f}')
    logger.info(f'allAcc: {allAcc:.4f}')
    logger.info(f'Evaluation time: {eval_time:.2f} seconds')
    logger.info("")
    
    # Print per-class results
    logger.info("Per-class results:")
    for i in range(args.classes):
        logger.info(f'Class {i:2d}: IoU={iou_class[i]:.4f}, Acc={accuracy_class[i]:.4f}')
    
    logger.info("="*60)


if __name__ == '__main__':
    main() 