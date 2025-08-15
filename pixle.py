from itertools import chain

import numpy as np
import torch
from torch.nn.functional import softmax

from mmseg.apis import inference_model

from tqdm import tqdm

from function import *
from evaluation import *


class Pixle():
    """
    Pixle: a fast and effective black-box attack based on rearranging pixels'
    [https://arxiv.org/abs/2202.02236]

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        x_dimensions (int or float, or a tuple containing a combination of those): size of the sampled patch along ther x side for each iteration. The integers are considered as fixed number of size,
        while the float as parcentage of the size. A tuple is used to specify both under and upper bound of the size. (Default: (2, 10))
        y_dimensions (int or float, or a tuple containing a combination of those): size of the sampled patch along ther y side for each iteration. The integers are considered as fixed number of size,
        while the float as parcentage of the size. A tuple is used to specify both under and upper bound of the size. (Default: (2, 10))
        pixel_mapping (str): the type of mapping used to move the pixels. Can be: 'random', 'similarity', 'similarity_random', 'distance', 'distance_random' (Default: random)
        restarts (int): the number of restarts that the algortihm performs. (Default: 20)
        max_iterations (int): number of iterations to perform for each restart. (Default: 10)
        update_each_iteration (bool): if the attacked images must be modified after each iteration (True) or after each restart (False).  (Default: False)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
        >>> adv_images = attack(images, labels)
    """

    def __init__(
        self,
        model,
        x_dimensions=(6, 6),
        y_dimensions=(5, 5),
        pixel_mapping="random",
        restarts=250,
        max_iterations=20,
        update_each_iteration=False,
        threshold=2250,
        device=None,
        cfg = None,
        is_mmseg_model=False
    ):
        self.model = model
        self.device = device
        self.cfg = cfg
        
        # Check if this is a robust model or mmseg model
        self.is_mmseg_model = is_mmseg_model

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError(
                "restarts must be and integer >= 0 " "({})".format(restarts)
            )

        self.update_each_iteration = update_each_iteration
        self.max_patches = max_iterations

        self.restarts = restarts
        self.pixel_mapping = pixel_mapping.lower()

        self.threshold = threshold
        self.save_interval = restarts // 5
        self.cfg = cfg
        if self.pixel_mapping not in [
            "random",
            "similarity",
            "similarity_random",
            "distance",
            "distance_random",
        ]:
            raise ValueError(
                "pixel_mapping must be one of [random, similarity,"
                "similarity_random, distance, distance_random]"
                " ({})".format(self.pixel_mapping)
            )

        if isinstance(y_dimensions, (int, float)):
            y_dimensions = [y_dimensions, y_dimensions]

        if isinstance(x_dimensions, (int, float)):
            x_dimensions = [x_dimensions, x_dimensions]

        if not all(
            [
                (isinstance(d, (int)) and d > 0)
                or (isinstance(d, float) and 0 <= d <= 1)
                for d in chain(y_dimensions, x_dimensions)
            ]
        ):
            raise ValueError(
                "dimensions of first patch must contains integers"
                " or floats in [0, 1]"
                " ({})".format(y_dimensions)
            )

        self.p1_x_dimensions = x_dimensions
        self.p1_y_dimensions = y_dimensions

        self.supported_mode = ["default", "targeted"]


    def forward(self, images, gt):

        if not self.update_each_iteration:
            adv_images = self.restart_forward(images, gt)
            return adv_images
        else:
            adv_images = self.iterative_forward(images, gt)
            return adv_images



    def restart_forward(self, images, gt):
        if not isinstance(images, torch.Tensor):
            raise ValueError("images must be a torch.Tensor")

        if len(images.shape) == 3:
            images = images.unsqueeze(0) # Shape: (1, C, H, W)
            gt = gt.unsqueeze(0) # Shape: (1, H, W)


        x_bounds = tuple(
            [
                max(1, d if isinstance(d, int) else round(images.size(3) * d))
                for d in self.p1_x_dimensions
            ]
        )

        y_bounds = tuple(
            [
                max(1, d if isinstance(d, int) else round(images.size(2) * d))
                for d in self.p1_y_dimensions
            ]
        )

        results = {
            "query": [], # 각 이미지별 중간 저장 시점 반복 횟수 리스트
            "adv_images": [],   # 각 이미지별 중간 저장 이미지 Tensor 리스트
        }

        images = images.clone().detach().to(self.device)
        gt = gt.clone().detach().to(self.device)

        bs, C, H, W = images.shape

        for idx in range(bs):
            image, gt = images[idx : idx + 1], gt[idx : idx + 1]
            best_image = image.clone()
            pert_image = image.clone()

            initial_loss, loss, callback = self._get_fun(image, gt)
            l0_threshold_captured = self.threshold #W * H * 0.001 = 2097
            best_solution = None

            best_p = initial_loss
            image_probs = [best_p]

            query_count = 0
            update_query = 0
            for r in tqdm(range(self.restarts), desc="Restarts"):
                stop = False
                # if r == 50:
                    # best_img_pred = inference_model(self.model, best_image).pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
                    # visualize_segmentation(best_image, best_img_pred, save_path='./results/pixle_benign1.png', alpha=0.0)
                    # visualize_segmentation(best_image, best_img_pred, save_path='./results/pixle_benign1.png', alpha=1)

                for it in range(self.max_patches):

                    (x, y), (x_offset, y_offset) = self.get_patch_coordinates(
                        image=image, x_bounds=x_bounds, y_bounds=y_bounds
                    )

                    destinations = self.get_pixel_mapping(
                        image, x, x_offset, y, y_offset, destination_image=best_image
                    )

                    solution = [x, y, x_offset, y_offset] + destinations

                    pert_image = self._perturb(
                        source=image, destination=best_image, solution=solution
                    )

                    mean_p = loss(solution=pert_image, solution_as_perturbed=True)

                    query_count += 1
                    if mean_p < best_p:
                        best_p = mean_p
                        best_solution = pert_image
                        update_query = query_count
                    
                    image_probs.append(best_p)

                    # 반복 횟수로 조절해서 필요 없음
                    # if callback(pert_image, l0_threshold_captured):
                    #     best_solution = pert_image
                    #     stop = True
                    #     break
                #L0 check
                # l0_distance = calculate_l0_norm(image.cpu().numpy().astype(np.uint8), pert_image.cpu().numpy().astype(np.uint8))
                # print(f"L0 distance: {l0_distance}")

                if best_solution is None:
                    best_image = pert_image
                else:
                    best_image = best_solution

                if stop:
                    break
            
                #save metrics
                # --- 중간 저장 로직 ---
                if (r+1) % self.save_interval == 0:
                    results["adv_images"].append(best_image)
                    results["query"].append(update_query)
                # --- ----------------------------- ---


        # 최종 adv 이미지 Tensor와 결과 딕셔너리 반환
        return results


    def iterative_forward(self, images, labels):
        assert len(images.shape) == 3 or (
            len(images.shape) == 4 and images.size(0) == 1
        )

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        if self.targeted:
            labels = self.get_target_label(images, labels)

        x_bounds = tuple(
            [
                max(1, d if isinstance(d, int) else round(images.size(3) * d))
                for d in self.p1_x_dimensions
            ]
        )

        y_bounds = tuple(
            [
                max(1, d if isinstance(d, int) else round(images.size(2) * d))
                for d in self.p1_y_dimensions
            ]
        )

        adv_images = []

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        bs, _, _, _ = images.shape

        for idx in range(bs):
            image, label = images[idx : idx + 1], labels[idx : idx + 1]

            best_image = image.clone()

            loss, callback = self._get_fun(image, label, target_attack=self.targeted)

            best_p = loss(solution=image, solution_as_perturbed=True)
            image_probs = [best_p]

            for it in range(self.max_patches):

                (x, y), (x_offset, y_offset) = self.get_patch_coordinates(
                    image=image, x_bounds=x_bounds, y_bounds=y_bounds
                )

                destinations = self.get_pixel_mapping(
                    image, x, x_offset, y, y_offset, destination_image=best_image
                )

                solution = [x, y, x_offset, y_offset] + destinations
               
                pert_image = self._perturb(
                    source=image, destination=best_image, solution=solution
                )

                p = loss(solution=pert_image, solution_as_perturbed=True)

                if p < best_p:
                    best_p = p
                    best_image = pert_image

                image_probs.append(best_p)

                if callback(pert_image, None, True):
                    best_image = pert_image
                    break


            adv_images.append(best_image)

        adv_images = torch.cat(adv_images)

        return adv_images
    #안쓰는 함수
    def _get_prob(self, image, gt):
        result = inference_model(self.model, image)
        # Get prediction logits
        pred_labels = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8) #result shape (1024, 2048)

        # 비교를 위해 gt_labels를 pred_labels와 동일한 장치 및 dtype으로 이동합니다.
        gt_labels = gt.to(device=pred_labels.device, dtype=pred_labels.dtype)

        ignore_index = 255

        # 유효한 ground truth 픽셀 마스크 (ignore_index가 아닌 픽셀)
        valid_gt_mask = (gt_labels != ignore_index)

        # 예측과 ground truth가 *일치하는* 픽셀 마스크 (유효한 gt 내에서)
        # 이곳이 우리가 평균을 계산할 대상 픽셀입니다.
        match_mask = (pred_labels == gt_labels) & valid_gt_mask # Shape: (H, W)

        # 모든 로짓에 대해 softmax 확률을 계산합니다.
        all_probs = softmax(logits, dim=1) # Shape: (1, num_classes, H, W)

        # 각 픽셀 위치에서 클래스들 중 가장 높은 확률 값을 찾습니다.
        max_all_probs, _ = torch.max(all_probs, dim=1) # Shape: (1, H, W)
        max_all_probs = max_all_probs.squeeze(0) # Shape: (H, W)

        # 예측이 맞았고 유효한 GT인 픽셀들만 선택합니다.
        relevant_probs = max_all_probs[match_mask] # 관련 확률값들 (1D 텐서)

        # 선택된 확률 값들의 평균을 계산합니다.
        if relevant_probs.numel() == 0:
            # 만약 모든 유효 픽셀에서 예측이 틀렸거나, 유효 픽셀이 없는 경우
            average_prob = torch.tensor(0.0, device=logits.device)
        else:
            # float 타입으로 변환하여 평균 계산
            average_prob = torch.mean(relevant_probs.float())

        # 계산된 평균 확률 값을 numpy 배열로 변환하여 반환합니다.
        return average_prob.detach().cpu().numpy()
    
    #안쓰는 함수수
    def loss(self, img, label, target_attack=False):

        p = self._get_prob(img)
        p = p[np.arange(len(p)), label]

        if target_attack:
            p = 1 - p

        return p.sum()

    def get_patch_coordinates(self, image, x_bounds, y_bounds):
        c, h, w = image.shape[1:]

        x, y = np.random.uniform(0, 1, 2)

        x_offset = np.random.randint(x_bounds[0], x_bounds[1] + 1)

        y_offset = np.random.randint(y_bounds[0], y_bounds[1] + 1)

        x, y = int(x * (w - 1)), int(y * (h - 1))

        if x + x_offset > w:
            x_offset = w - x

        if y + y_offset > h:
            y_offset = h - y

        return (x, y), (x_offset, y_offset)

    def get_pixel_mapping(
        self, source_image, x, x_offset, y, y_offset, destination_image=None
    ):
        if destination_image is None:
            destination_image = source_image

        destinations = []
        c, h, w = source_image.shape[1:]
        source_image = source_image[0]

        if self.pixel_mapping == "random":
            for i in range(x_offset):
                for j in range(y_offset):
                    dx, dy = np.random.uniform(0, 1, 2)
                    dx, dy = int(dx * (w - 1)), int(dy * (h - 1))
                    destinations.append([dx, dy])
        else:
            for i in np.arange(y, y + y_offset):
                for j in np.arange(x, x + x_offset):
                    pixel = source_image[:, i : i + 1, j : j + 1]
                    diff = destination_image - pixel
                    diff = diff[0].abs().mean(0).view(-1)

                    if "similarity" in self.pixel_mapping:
                        diff = 1 / (1 + diff)
                        diff[diff == 1] = 0

                    probs = torch.softmax(diff, 0).cpu().numpy()

                    indexes = np.arange(len(diff))

                    pair = None

                    linear_iter = iter(
                        sorted(
                            zip(indexes, probs), key=lambda pit: pit[1], reverse=True
                        )
                    )

                    while True:
                        if "random" in self.pixel_mapping:
                            index = np.random.choice(indexes, p=probs)
                        else:
                            index = next(linear_iter)[0]

                        _y, _x = np.unravel_index(index, (h, w))

                        if _y == i and _x == j:
                            continue

                        pair = (_x, _y)
                        break

                    destinations.append(pair)

        return destinations

    def _get_fun(self, img, gt, target_attack=False):
        """
        Calculates initial state and returns loss function and callback.
        Loss is based on the average probability of the TRUE class at initially CORRECTLY predicted pixels.

        Args:
            img (torch.Tensor): Original image tensor (1, C, H, W) on self.device, range [0, 1].
            gt (torch.Tensor): Ground truth tensor (1, H, W) on self.device, dtype long.
            target_attack (bool): Boolean flag.

        Returns:
            initial_loss (float): Average probability of the true class at initially correct pixels.
            func (callable): Function to calculate loss for a perturbed image tensor using the same logic.
            callback (callable): Function to check attack success.
        """
        # --- Calculate initial state ---
        original_img = img.squeeze(0) # Shape: (C, H, W)
        gt = gt.squeeze(0) # Shape: (H, W)
        with torch.no_grad():
            # 1. Get original logits and predictions
            try:
                if self.is_mmseg_model:
                    # Use mmseg API for mmseg models
                    original_result = inference_model(self.model, original_img.permute(1, 2, 0).cpu().numpy()) # Pass Tensor
                    original_logits = original_result.seg_logits.data.to(self.device) # Shape: (C, H, W)
                    original_probs = softmax(original_logits, dim=0) # Shape: (C, H, W)
                    original_pred_labels = original_result.pred_sem_seg.data.squeeze() # Shape: (H, W)
                else:
                    # Use model_predict for Robust models
                    from adv_setting import model_predict
                    img_np = original_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    original_probs, original_pred_labels = model_predict(self.model, img_np, self.cfg)
                    original_logits = torch.log(original_probs)
            except Exception as e:
                print("\n--- Error calling inference (Original Image) ---")
                raise e

            # 2. Create masks
            ignore_index = 255
            num_classes = original_logits.shape[0]
            channel_indices = torch.arange(num_classes, device=self.device) # Shape: (C)

            # 예측이 맞은 픽셀만 선택
            condition_mask = torch.ones_like(original_pred_labels, dtype=torch.bool)

            #gt를 사용해 ignore 픽셀 제거
            if self.cfg['dataset'] == 'cityscapes':
                # Cityscapes: gt에서 255인 픽셀 무시
                valid_gt_mask = gt != 255
            elif self.cfg['dataset'] == 'ade20k':
                # ADE20k: gt에서 0번 클래스인 픽셀 무시 (ADE20K에서 0이 ignore_index)
                valid_gt_mask = gt != 0
            elif self.cfg['dataset'] == 'VOC2012':
                # VOC2012: gt에서 255인 픽셀 무시
                valid_gt_mask = gt != 255
            else:
                raise ValueError(f"Unsupported dataset: {self.cfg['dataset']}")

            correct_masked_pred_labels = torch.where(valid_gt_mask, original_pred_labels, ignore_index)

            #마스크를 (C, H, W) 형태로 변환
            channel_indices_reshaped = channel_indices.view(num_classes, 1, 1)
            channel_indices_reshaped = channel_indices_reshaped.to(correct_masked_pred_labels.device)
            final_mask = channel_indices_reshaped == correct_masked_pred_labels #broadcast

            # 3. Calculate initial loss based on probability of TRUE class at matched locations
            # original_probs (C, H, W) 에서 final_mask (C, H, W)가 True인 위치의 값만 선택
            selected_initial_probs = original_probs[final_mask] # 1D Tensor of selected logits

            # 선택된 확률 값들의 평균을 계산
            initial_loss_val = torch.mean(selected_initial_probs)

        @torch.no_grad()
        def func(solution=None, destination=None, solution_as_perturbed=False, pert_image_tensor=None, **kwargs):
            # 1. Get perturbed image tensor
            if pert_image_tensor is None:
                # ... (Calculate pert_image_tensor using _perturb) ...
                 if solution is None: raise ValueError(...)
                 if not solution_as_perturbed:
                     current_destination = destination if destination is not None else img
                     if not isinstance(current_destination, torch.Tensor): raise TypeError(...)
                     pert_image_tensor = self._perturb(source=img, destination=current_destination, solution=solution)
                 else: pert_image_tensor = solution
                 if not isinstance(pert_image_tensor, torch.Tensor): raise TypeError(...)
                 pert_image_tensor = pert_image_tensor.to(self.device)

            # 2. Get probabilities for the perturbed image
            if self.is_mmseg_model:
                adv_result = inference_model(self.model, pert_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()) # Pass Tensor
                adv_logits = adv_result.seg_logits.data.to(self.device) # Shape: (C, H, W)
                adv_probs = softmax(adv_logits, dim=0) # Shape: (C, H, W)
            else:
                # Use model_predict for Robust models
                from adv_setting import model_predict
                img_np = pert_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                adv_probs, _ = model_predict(self.model, img_np, self.cfg)
            
            adv_correct_probs = adv_probs[final_mask]
            # Average these probabilities
            loss_val = torch.mean(adv_correct_probs.float())

            # Return loss as numpy float. Lower value means the attack is more successful.
            return loss_val.detach().cpu().numpy()
        

        @torch.no_grad()
        def callback(pert_image_tensor=None, l0_threshold_captured=None, **kwargs): # Simplified signature
            """Calculates L0 distance and checks against threshold."""
            if pert_image_tensor is None: raise ValueError("callback needs pert_image_tensor")
            if not isinstance(pert_image_tensor, torch.Tensor): raise TypeError(...)
            pert_image_tensor = pert_image_tensor.to(self.device)

            # Calculate absolute difference (element-wise)
            # Both tensors should be shape (1, C, H, W)
            abs_diff = torch.abs(original_img - pert_image_tensor)

            # Check if difference in *any* channel exceeds epsilon
            # abs_diff > epsilon_captured -> Boolean tensor (1, C, H, W)
            # torch.any(..., dim=1) -> Boolean tensor (1, H, W) where True if any channel changed
            epsilon_captured = 1e-6
            changed_pixels_mask = torch.any(abs_diff > epsilon_captured, dim=1) # Shape (1, H, W)

            # Count number of changed pixels (L0 distance)
            l0_distance = torch.sum(changed_pixels_mask).item()
            
            # Check against threshold
            if l0_distance >= l0_threshold_captured:
                print(f"L0 distance ({l0_distance}) >= threshold ({l0_threshold_captured}). Stopping.") # Optional debug print
                return True # Stop the attack
            else:
                return False # Continue

        # Return the initial loss (numpy float), the loss function, and the callback
        return initial_loss_val.detach().cpu().numpy(), func, callback

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source

        c, h, w = source.shape[1:]

        x, y, xl, yl = solution[:4]
        destinations = solution[4:]

        source_pixels = np.ix_(range(c), np.arange(y, y + yl), np.arange(x, x + xl))

        indexes = torch.tensor(destinations)
        destination = destination.clone().detach().to(self.device)

        s = source[0][source_pixels].view(c, -1)

        destination[0, :, indexes[:, 1], indexes[:, 0]] = s

        return destination
    

if __name__ == "__main__":
    from evaluation import eval_miou
    from dataset import CitySet, ADESet
    from mmseg.apis import init_model
    from tqdm import tqdm
    import setproctitle
    setproctitle.setproctitle("Pixle_Attack_Process")
    
    config = {"num_class": 150,
              "dataset": "ade20k",
              "data_dir": "./datasets/ade20k"}
    
    cf_path = 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py'
    ckpt_path = 'ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples = 1
    dataset = ADESet(dataset_dir=config["data_dir"])
    dataset.images = dataset.images[:min(len(dataset.images), num_samples)]
    dataset.filenames = dataset.filenames[:min(len(dataset.filenames), num_samples)]
    dataset.gt_images = dataset.gt_images[:min(len(dataset.gt_images), num_samples)]

    model = init_model(cf_path, None, device)
    # 2. 체크포인트 로드 (weights_only=False 직접 설정)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    # img, filename, gt = dataset[0]
    img_list = []
    adv_img_list_1 = []
    adv_img_list_2 = []
    adv_img_list_3 = []
    adv_img_list_4 = []
    adv_img_list_result = []
    gt_list = []
    file_path = []
    matrix_results_list = []


    for img, filename, gt in tqdm(dataset, desc="Generating adversarial examples"):

        img_tensor = torch.from_numpy(img.copy()).unsqueeze(0).permute(0, 3, 1, 2) # Shape: (1, C, H, W)
        gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0) # Shape: (1, H, W)

        pixle = Pixle(model,
                      x_dimensions=(0.006, 0.006),
                      y_dimensions=(0.006, 0.006),
                      pixel_mapping="random",
                      restarts=5,
                      max_iterations=1,
                      update_each_iteration=False,
                      threshold=21000,
                      cfg = config)

        results = pixle.forward(img_tensor, gt_tensor)
        results_adv_images = [x.squeeze(0).permute(1, 2, 0).cpu().numpy() for x in results['adv_images']]
        results_iterations_at_save = results['iterations_at_save']

        adv_img_list_1.append(results_adv_images[0])
        adv_img_list_2.append(results_adv_images[1])
        adv_img_list_3.append(results_adv_images[2])
        adv_img_list_4.append(results_adv_images[3])
        adv_img_list_result.append(results_adv_images[4])
        gt_list.append(gt)
        file_path.append(filename)

    benign_pred_1, adv_pred_1, _, adv_miou_score_1 = eval_miou(model, dataset, adv_img_list_1, config)
    benign_pred_2, adv_pred_2, _, adv_miou_score_2 = eval_miou(model, dataset, adv_img_list_2, config)
    benign_pred_3, adv_pred_3, _, adv_miou_score_3 = eval_miou(model, dataset, adv_img_list_3, config)
    benign_pred_4, adv_pred_4, _, adv_miou_score_4 = eval_miou(model, dataset, adv_img_list_4, config)
    benign_pred_rlt, adv_pred_rlt, benign_miou_score_rlt, adv_miou_score_rlt = eval_miou(model, dataset, adv_img_list_result, config)

    print(f"benign_miou_score: {benign_miou_score_rlt['mean_iou']}")
    print(f"adv_miou_score_1: {adv_miou_score_1['mean_iou']}")
    print(f"adv_miou_score_2: {adv_miou_score_2['mean_iou']}")
    print(f"adv_miou_score_3: {adv_miou_score_3['mean_iou']}")
    print(f"adv_miou_score_4: {adv_miou_score_4['mean_iou']}")
    print(f"adv_miou_score: {adv_miou_score_rlt['mean_iou']}")
    print(f"file_path: {file_path}")
    # print(f"matrix_results_list: {matrix_results_list}")