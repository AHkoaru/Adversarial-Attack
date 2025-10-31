import evaluate
from mmseg.apis import inference_model
import numpy as np


def eval_miou(model, dataset, adv_examples, gt_images, config):
    """
    Evaluate Mean IoU (Intersection over Union) scores for both benign and adversarial examples.
    
    Args:
        model: Segmentation model to evaluate
        dataset: Dataset containing original images (list of numpy arrays)
        adv_examples: List of adversarial examples (list of numpy arrays)
        gt_images: List of ground truth images
        config: Configuration dictionary containing number of classes
        
    Returns:
        tuple: (benign_miou_score, adv_miou_score) containing Mean IoU scores for both cases
    """
    import sys
    import os
    sys.path.append('/workspace/Robust-Semantic-Segmentation')
    from adv_setting import model_predict
    
    miou = evaluate.load("mean_iou")

    benign_predictions = []
    adv_predictions = []
    
    # Check if model has 'cfg' attribute (mmseg model) or is DataParallel wrapped
    is_mmseg = hasattr(model, 'cfg') or (hasattr(model, 'module') and hasattr(model.module, 'cfg'))
    
    for i in range(len(dataset)):
        if is_mmseg:
            # Use mmseg inference_model for mmseg models
            benign_predictions.append(inference_model(model, dataset[i]).pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8))
            adv_predictions.append(inference_model(model, adv_examples[i]).pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8))
        else:
            # Use model_predict for Robust models
            _, benign_pred = model_predict(model, dataset[i], config)
            benign_predictions.append(benign_pred.cpu().numpy().astype(np.uint8))
            
            _, adv_pred = model_predict(model, adv_examples[i], config)
            adv_predictions.append(adv_pred.cpu().numpy().astype(np.uint8))

    
    if config["dataset"] == "cityscapes":

        benign_to_adv_miou = miou.compute(predictions=adv_predictions,
            references=benign_predictions,
            num_labels=config["num_class"],
            ignore_index=255,
            reduce_labels=False,
            )
        gt_to_adv_miou = miou.compute(predictions=adv_predictions,
            references=gt_images,
            num_labels=config["num_class"],
            ignore_index=255,
            reduce_labels=False,
            )
    elif config["dataset"] == "ade20k":

        benign_to_adv_miou = miou.compute(predictions=adv_predictions,
            references=benign_predictions,
            num_labels=config["num_class"],
            ignore_index=255,
            reduce_labels=False,
            )
        gt_to_adv_miou = miou.compute(predictions=adv_predictions,
            references=gt_images,
            num_labels=config["num_class"],
            ignore_index=255,
            reduce_labels=True,
            )
    elif config["dataset"] == "VOC2012":
        benign_to_adv_miou = miou.compute(predictions=adv_predictions,
            references=benign_predictions,
            num_labels=config["num_class"],
            ignore_index=255,
            reduce_labels=False,
            )
        gt_to_adv_miou = miou.compute(predictions=adv_predictions,
            references=gt_images,
            num_labels=config["num_class"],
            ignore_index=255,
            reduce_labels=False,
            )
    return benign_to_adv_miou, gt_to_adv_miou



def calculate_l0_norm(original_img: np.ndarray, adversarial_img: np.ndarray) -> int:
    """
    Calculate L0 norm between original and adversarial images.
    L0 norm represents the number of pixels that have been modified.

    Args:
        original_img (np.ndarray): Original image array (H, W, 3), uint8
        adversarial_img (np.ndarray): Adversarial image array (H, W, 3), uint8

    Returns:
        int: Number of modified pixels considering all channels
    """
    if original_img.shape != adversarial_img.shape:
        raise ValueError("Images must have the same shape")
    
    if original_img.dtype != np.uint8 or adversarial_img.dtype != np.uint8:
        raise ValueError("Images must be in uint8 format")
    
    return int(np.sum(np.abs(original_img - adversarial_img) > 0))

def calculate_pixel_ratio(original_img: np.ndarray, adversarial_img: np.ndarray) -> float:
    """
    Calculate the ratio of modified pixels to total pixels in the image.
    A pixel is considered modified if any of its channels has changed.

    Args:
        original_img (np.ndarray): Original image array (H, W, 3), uint8
        adversarial_img (np.ndarray): Adversarial image array (H, W, 3), uint8

    Returns:
        float: Ratio of modified pixels (0.0 ~ 1.0)
    """
    if original_img.shape != adversarial_img.shape:
        raise ValueError("Images must have the same shape")
    
    if original_img.dtype != np.uint8 or adversarial_img.dtype != np.uint8:
        raise ValueError("Images must be in uint8 format")
    
    # Count pixels that differ in any channel
    modified_pixels = np.any(original_img != adversarial_img, axis=2).sum()
    total_pixels = original_img.shape[0] * original_img.shape[1]
    
    return float(modified_pixels / total_pixels)

def calculate_impact(original_img: np.ndarray, adversarial_img: np.ndarray, pred_original: np.ndarray, pred_adversarial: np.ndarray) -> float:
    """
    Calculate the impact of the adversarial attack on the segmentation model.
    Impact is measured as the ratio of modified predictions to modified pixels.

    Args:
        original_img (np.ndarray): Original image array (H, W, 3), uint8
        adversarial_img (np.ndarray): Adversarial image array (H, W, 3), uint8
        pred_original (np.ndarray): Original prediction array (H, W), uint8
        pred_adversarial (np.ndarray): Adversarial prediction array (H, W), uint8

    Returns:
        float: Impact score representing how effectively the attack modified predictions
    """
    # Calculate number of modified pixels in the input image
    modified_pixels = np.any(original_img != adversarial_img, axis=2).sum()
    
    # Calculate number of modified predictions
    modified_preds = (pred_original != pred_adversarial).sum() 
    
    # Calculate impact as the ratio of modified predictions to modified pixels
    if modified_pixels == 0:
        return 0.0
    return float(modified_preds / modified_pixels) - 1


if __name__ == '__main__':
    from mmseg.apis import init_model
    from dataset import ADESet
    import evaluate
    import torch
    # 모델 설정 파일과 체크포인트 경로 설정
    config_file = 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py'
    checkpoint_file = 'ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth'
    
    # 모델 초기화
    model = init_model(config_file, None, 'cuda')
    # 2. 체크포인트 로드 (weights_only=False 직접 설정)
    checkpoint = torch.load(checkpoint_file, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    del checkpoint  # 체크포인트 변수 삭제
    torch.cuda.empty_cache()  # GPU 캐시 정리


    dataset_dir = "./datasets/ade20k"
    dataset = ADESet(dataset_dir)

    pred_list1 = []
    pred_list2 = []
    gt_list = []
    for i in range(len(dataset)):
        image, filename, gt = dataset[i]
        pred1 = inference_model(model, image)
        pred2 = inference_model(model, image)
        pred_list1.append(pred1.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8))
        pred_list2.append(pred2.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8))
        gt_list.append(gt)

    miou = evaluate.load("mean_iou")

    miou_score = miou.compute(
        predictions=pred_list2,
        references=gt_list,
        num_labels=150,  # ADE20K는 150개 클래스
        ignore_index=255,
        reduce_labels=True
    )
    
    
    # mIoU 평가

    print(f'ADE20K 검증 세트에 대한 mIoU: {miou_score["mean_iou"]:.4f}')

