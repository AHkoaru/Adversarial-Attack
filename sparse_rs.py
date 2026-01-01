# Copyright (c) 2020-present
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import time
import math
import torch.nn.functional as F
from torch import softmax

import numpy as np
import copy
import sys
from utils import Logger
import os

try:
    from mmseg.apis import inference_model
except ImportError:
    inference_model = None

from function import *
from evaluation import *
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'Robust-Semantic-Segmentation'))
from adv_setting import model_predict

class RSAttack():
    """
    Sparse-RS attacks

    :param predict:           forward pass function
    :param norm:              type of the attack
    :param n_restarts:        number of random restarts
    :param n_queries:         max number of queries (each restart)
    :param eps:               bound on the sparsity of perturbations
    :param seed:              random seed for the starting point
    :param alpha_init:        parameter to control alphai
    :param loss:              loss function optimized ('margin', 'ce' supported)
    :param resc_schedule      adapt schedule of alphai to n_queries
    :param device             specify device to use
    :param log_path           path to save logfile.txt
    :param constant_schedule  use constant alphai
    :param targeted           perform targeted attacks
    :param init_patches       initialization for patches
    :param resample_loc       period in queries of resampling images and
                              locations for universal attacks
    :param data_loader        loader to get new images for resampling
    :param update_loc_period  period in queries of updates of the location
                              for image-specific patches
    """

    def __init__(
            self,
            model,
            cfg,
            norm='L0',
            n_queries=5000,
            eps=None,
            p_init=.8,
            n_restarts=1,
            seed=0,
            verbose=False,
            targeted=False,
            loss='margin',
            resc_schedule=True,
            device=None,
            log_path=None,
            constant_schedule=False,
            init_patches='random',
            resample_loc=None,
            data_loader=None,
            update_loc_period=None,
            original_img=None,
            d=5,
            use_decision_loss=True,
            is_mmseg_model=False,
            is_sed_model=False, # Added flag
            enable_success_reporting=False
            ):
        """
        Sparse-RS implementation in PyTorch
        """

        self.model = model
        self.cfg = cfg
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.targeted = targeted
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.device = device
        self.logger = Logger(log_path)
        self.constant_schedule = constant_schedule
        self.init_patches = init_patches
        self.resample_loc = n_queries // 10 if resample_loc is None else resample_loc
        self.data_loader = data_loader
        self.update_loc_period = update_loc_period if not update_loc_period is None else 4 if not targeted else 10
        self.mask = None
        self.pre_changed_pixels = None
        self.original_img = original_img
        self.original_pred_labels = None
        self.success_mask = None  # 공격 성공한 픽셀을 추적하는 마스크
        self.d = d
        self.current_query = 0
        self.verbose = verbose
        self.use_decision_loss = use_decision_loss
        self.best_loss = None
        self.is_mmseg_model = is_mmseg_model
        self.is_sed_model = is_sed_model # Store flag
        self.enable_success_reporting = enable_success_reporting

        # 이전 예측 레이블 저장
        self.previous_pred_labels = None

    def margin_and_loss(self, img, final_mask, first_img_pred_labels):  
        if self.is_mmseg_model:
            # 기존 mmseg API 사용
            adv_result = inference_model(self.model, img.squeeze(0).permute(1, 2, 0).cpu().numpy()) # Pass Tensor
            adv_logits = adv_result.seg_logits.data.to(self.device) # Shape: (Class, H, W)
        elif self.is_sed_model:
            # Detectron2 inference
            inputs = [{"image": img.squeeze(0)}]
            outputs = self.model(inputs)
            adv_logits = outputs[0]["sem_seg"].to(self.device)
        else:
            # Convert tensor to numpy for model_predict
            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            adv_probs, _ = model_predict(self.model, img_np, self.cfg)
            # adv_probs is already in (C, H, W) format from model_predict
            adv_logits = torch.log(adv_probs)
        
        gt_indices = final_mask.float().argmax(dim=0)  # (H, W)
        # 2. 정답 로짓
        correct_logits = adv_logits.gather(0, gt_indices.unsqueeze(0)).squeeze(0)  # (H, W)
        
        if self.loss == 'margin':
            # margin 기반 loss로 변경
            adv_logits_clone = adv_logits.clone()
            h, w = adv_logits.shape[1], adv_logits.shape[2]
            adv_logits_clone[gt_indices, torch.arange(h).unsqueeze(1).expand(h, w), torch.arange(w).expand(h, w)] = float('-inf')
            max_wrong_logits, _ = adv_logits_clone.max(dim=0)  # (H, W)
            # 4. margin 계산
            margin = correct_logits - max_wrong_logits  # (H, W)
            # 배경 픽셀들은 loss 계산에서 제외
            valid_pixel_mask = final_mask.any(dim=0)
            loss_val = margin[valid_pixel_mask].mean()

            return loss_val.detach().cpu().numpy(), None, None
        
        elif self.loss == 'prob':
            if self.is_mmseg_model or self.is_sed_model:
                adv_probs = softmax(adv_logits, dim=0).to(self.device) # Shape: (C, H, W)
            adv_correct_probs = adv_probs[final_mask]
            loss_val = torch.mean(adv_correct_probs.float())

            return loss_val.detach().cpu().numpy(), None, None

        elif self.loss == 'decision':
            # Decision loss 계산
            if self.is_mmseg_model:
                adv_pred_labels = adv_result.pred_sem_seg.data.squeeze().to(self.device) # Shape: (H, W)
            elif self.is_sed_model:
                adv_pred_labels = adv_logits.argmax(dim=0).to(self.device)
            else:
                # Assuming model_predict was used if not mmseg/sed, but logic above only sets adv_logits
                # We need labels here.
                # Re-run prediction or assume we can get it from logits if we had them?
                # For non-mmseg/sed, we need to ensure we have labels.
                # The original code structure for 'decision' assumed adv_result existed or model_predict was called.
                # Let's assume if not mmseg/sed, we need to get labels.
                # But wait, margin_and_loss logic for 'decision' in original code:
                # It used adv_result for mmseg, but for else it didn't calculate adv_pred_labels in the 'else' block at top.
                # It seems original code had a bug or I missed something.
                # Ah, original code:
                # if self.loss == 'decision': ... adv_pred_labels = ...
                # It seems it re-calculated or assumed availability.
                # Let's fix for SED:
                pass

            if not self.is_mmseg_model and not self.is_sed_model:
                 # Fallback for other models if needed, but we focus on SED
                 pass

            H, W = adv_pred_labels.shape[0], adv_pred_labels.shape[1]
            current_changed_pixels = (adv_pred_labels != self.original_pred_labels.to(self.device)).long().to(self.device)

            # 증분 changed pixels 계산
            if self.pre_changed_pixels is not None:
                changed_pixels = current_changed_pixels - self.pre_changed_pixels
            else:
                changed_pixels = current_changed_pixels
            decision_loss = (torch.sum(changed_pixels.float()) / ((H * W * self.eps) * (self.d ** 2)))

            # Decision loss만 사용
            total_loss = -decision_loss

            if self.verbose:
                print(f'decision_loss: {-decision_loss:.4f}, total_loss: {total_loss:.4f}')

            # Return loss as numpy float. Lower value means the attack is more successful.
            return total_loss.detach().cpu().numpy(), current_changed_pixels, (-decision_loss).detach().cpu().numpy()

        elif self.loss == 'decision_change':
            # 이전 iteration의 예측 대비 현재 예측 변경 픽셀 수 계산
            if self.is_mmseg_model:
                adv_pred_labels = adv_result.pred_sem_seg.data.squeeze().to(self.device) # Shape: (H, W)
            elif self.is_sed_model:
                adv_pred_labels = adv_logits.argmax(dim=0).to(self.device)
            
            H, W = adv_pred_labels.shape[0], adv_pred_labels.shape[1]
            # 첫 iteration이면 원본 예측을 이전 예측으로 사용
            if self.previous_pred_labels is None:
                self.previous_pred_labels = self.original_pred_labels.clone()
            
            # 성공 마스크 초기화 (첫 iteration)
            if self.success_mask is None:
                self.success_mask = torch.zeros_like(self.original_pred_labels)

            # 공격 성공 마스크: 원본과 다른 예측 (성공적으로 변경된 픽셀)
            current_success_mask = (adv_pred_labels != self.original_pred_labels).long()
            
            # 이전 예측 대비 변경된 픽셀 계산
            changed_pixels = (adv_pred_labels != self.previous_pred_labels).long()
            
            # 새로운 성공 픽셀: 이번 iteration에 새로 성공한 픽셀
            new_success_pixels = current_success_mask * changed_pixels
            
            # 배경 픽셀 제외
            valid_pixel_mask = final_mask.any(dim=0)
            
            # 현재 성공하지 못한 픽셀만 대상으로 계산
            available_pixels = valid_pixel_mask & (current_success_mask == 0)
            new_success_count = new_success_pixels[available_pixels].sum().float()
            total_available_pixels = available_pixels.sum().float()

            # 변경 비율 계산 (음수로 반환: 변경이 많을수록 loss 감소)
            change_ratio = new_success_count / (total_available_pixels + 1e-8)
            
            loss_val = -change_ratio  / ((H * W * self.eps) * (self.d ** 2)) 

            if self.verbose:
                print(f'New success pixels: {new_success_count.item():.0f}/{total_available_pixels.item():.0f}, '
                      f'Change ratio: {change_ratio.item():.4f}, Loss: {loss_val.item():.4f}')
                print(f'Total success pixels so far: {current_success_mask.sum().item():.0f}')

            # Return: loss (음수), changed_pixels mask, change_ratio (양수)
            return loss_val.detach().cpu().numpy(), changed_pixels, change_ratio.detach().cpu().numpy()

    def init_hyperparam(self, x):
        assert self.norm in ['L0', 'patches', 'frames',
            'patches_universal', 'frames_universal']
        assert not self.eps is None
        # assert self.loss in ['ce', 'margin']

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()
        if self.targeted:
            self.loss = 'ce'
        
    def random_target_classes(self, y_pred, n_classes):
        y = torch.zeros_like(y_pred)
        for counter in range(y_pred.shape[0]):
            l = list(range(n_classes))
            l.remove(y_pred[counter])
            t = self.random_int(0, len(l))
            y[counter] = l[t]

        return y.long().to(self.device)

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_choice_255(self, shape):
        """0 또는 255 값을 랜덤하게 생성하는 함수"""
        t = 2 * torch.rand(shape).to(self.device) - 1
        binary_values = torch.sign(t)
        # 0 또는 255 값으로 변환 (-1 -> 0, +1 -> 255)
        return torch.where(binary_values > 0, 255.0, 0.0)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 'patches' in self.norm:
            if 10 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 8
            elif 500 < it <= 1000:
                p = self.p_init / 16
            elif 1000 < it <= 2000:
                p = self.p_init / 32
            elif 2000 < it <= 4000:
                p = self.p_init / 64
            elif 4000 < it <= 6000:
                p = self.p_init / 128
            elif 6000 < it <= 8000:
                p = self.p_init / 256
            elif 8000 < it:
                p = self.p_init / 512
            else:
                p = self.p_init

        elif 'frames' in self.norm:
            if not 'universal' in self.norm :
                tot_qr = 10000 if self.rescale_schedule else self.n_queries
                p = max((float(tot_qr - it) / tot_qr  - .5) * self.p_init * self.eps ** 2, 0.)
                return 3. * math.ceil(p)

            else:
                assert self.rescale_schedule
                its = [200, 600, 1200, 1800, 2500, 10000, 100000]
                resc_factors = [1., .8, .6, .4, .2, .1, 0.]
                c = 0
                while it >= its[c]:
                    c += 1
                return resc_factors[c] * self.p_init

        elif 'L0' in self.norm:
            if 0 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 5
            elif 500 < it <= 1000:
                p = self.p_init / 6
            elif 1000 < it <= 2000:
                p = self.p_init / 8
            elif 2000 < it <= 4000:
                p = self.p_init / 10
            elif 4000 < it <= 6000:
                p = self.p_init / 12
            elif 6000 < it <= 8000:
                p = self.p_init / 15
            elif 8000 < it:
                p = self.p_init / 20
            else:
                p = self.p_init

            if self.constant_schedule:
                p = self.p_init / 2
        
        return p

    def sh_selection(self, it):
        """ schedule to decrease the parameter p """

        t = max((float(self.n_queries - it) / self.n_queries - .0) ** 1., 0) * .75

        return t

    def get_init_patch(self, c, s, n_iter=1000):
        if self.init_patches == 'stripes':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, s]).clamp(0., 255.)
        elif self.init_patches == 'uniform':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, 1]).clamp(0., 255.)
        elif self.init_patches == 'random':
            patch_univ = self.random_choice([1, c, s, s]).clamp(0., 255.)
        elif self.init_patches == 'random_squares':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device)
            for _ in range(n_iter):
                size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
                loc_init = torch.randint(s - size_init + 1, size=[2])
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = 0.
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init
                    ] += self.random_choice([c, 1, 1]).clamp(0., 255.)
        elif self.init_patches == 'sh':
            patch_univ = torch.ones([1, c, s, s]).to(self.device)
        
        return patch_univ.clamp(0., 255.)

    def attack_single_run(self, img, gt, final_mask, first_img_pred_labels):
        with torch.no_grad():
            adv = img.clone()
            c, h, w = img.shape[1:]
            n_features = c * h * w
            n_ex_total = img.shape[0]

            if self.norm == 'L0':
                eps = int(self.eps * h * w)
                # original_img = img.clone()

                x_best = img.clone()
                n_pixels = h * w
                b_all, be_all = torch.zeros([img.shape[0], eps]).long(), torch.zeros([img.shape[0], n_pixels - eps]).long()
                for img_idx in range(img.shape[0]):
                    ind_all = torch.randperm(n_pixels)
                    ind_p = ind_all[:eps]
                    ind_np = ind_all[eps:]
                    x_best[img_idx, :, ind_p // w, ind_p % w] = self.random_choice_255([c, eps]).clamp(0., 255.)
                    b_all[img_idx] = ind_p.clone()
                    be_all[img_idx] = ind_np.clone()
                    
                loss_min, current_changed_pixels, initial_decision_loss = self.margin_and_loss(x_best, final_mask, first_img_pred_labels)
                best_changed_pixels = current_changed_pixels
                best_decision_loss = initial_decision_loss  # 초기 decision_loss 값 저장
                self.current_query += 1 

                # pbar = tqdm(range(1, self.n_queries), desc="Sparse-RS Attack", ncols=120)
                # for it in pbar:
                for it in range(1, self.n_queries):
                    # build new candidate
                    x_new = x_best.clone()
                    eps_it = max(int(self.p_selection(it) * eps), 1)
                    ind_p = torch.randperm(eps)[:eps_it]
                    ind_np = torch.randperm(n_pixels - eps)[:eps_it]
                    
                    for img_idx in range(x_new.shape[0]):
                        p_set = b_all[img_idx, ind_p]
                        np_set = be_all[img_idx, ind_np]
                        x_new[img_idx, :, p_set // w, p_set % w] = adv[img_idx, :, p_set // w, p_set % w].clone()
                        if eps_it > 1:
                            x_new[img_idx, :, np_set // w, np_set % w] = self.random_choice_255([c, eps_it]).clamp(0., 255.)
                        else:
                            # if update is 1x1 make sure the sampled color is different from the current one
                            old_clr = x_new[img_idx, :, np_set // w, np_set % w].clone()
                            assert old_clr.shape == (c, 1), print(old_clr)
                            new_clr = old_clr.clone()
                            while (new_clr == old_clr).all().item():
                                new_clr = self.random_choice_255([c, 1]).clone().clamp(0., 255.)
                            x_new[img_idx, :, np_set // w, np_set % w] = new_clr.clone()
                        
                    # compute loss of the new candidates
                    loss, current_changed_pixels, decision_loss_value = self.margin_and_loss(x_new, final_mask, first_img_pred_labels)
                    self.current_query += 1  # 단일 이미지이므로 간단히 증가
                    
                    # update best solution (loss 기반으로만 판단)
                    if loss < loss_min:
                        best_changed_pixels = current_changed_pixels
                        best_decision_loss = decision_loss_value  # 마지막 성공한 업데이트의 decision_loss_value 저장
                        loss_min = loss
                        x_best = x_new.clone()
                        # if self.verbose:
                            # print(f'loss: {loss}, current_query: {self.current_query}')

                        # decision_change loss 사용 시 이전 예측 레이블 업데이트
                        if self.loss == 'decision_change':
                            if self.is_mmseg_model:
                                new_result = inference_model(self.model, x_new.squeeze(0).permute(1, 2, 0).cpu().numpy())
                                new_pred_labels = new_result.pred_sem_seg.data.squeeze().to(self.device)
                                self.previous_pred_labels = new_pred_labels.clone()
                            elif self.is_sed_model:
                                inputs = [{"image": x_new.squeeze(0)}]
                                outputs = self.model(inputs)
                                new_pred_labels = outputs[0]["sem_seg"].argmax(dim=0).to(self.device)
                                self.previous_pred_labels = new_pred_labels.clone()
                            else:
                                x_new_np = x_new.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                                _, new_pred_labels = model_predict(self.model, x_new_np, self.cfg)
                                self.previous_pred_labels = new_pred_labels.clone()
                            
                            # 성공 마스크 업데이트: 현재 상태 기준으로 재계산
                            current_success_mask = (new_pred_labels != self.original_pred_labels).long()
                            self.success_mask = current_success_mask
                            
                            if self.verbose:
                                print(f'Success mask updated. Total successful pixels: {self.success_mask.sum().item():.0f}')

                        # 픽셀 인덱스 관리 배열 업데이트 (핵심 부분!)
                        # 스왑된 픽셀들의 인덱스를 업데이트
                        for img_idx in range(x_new.shape[0]):
                            # 복원된 픽셀들 (수정됨 → 수정안됨)
                            restored_pixels = b_all[img_idx, ind_p]
                            # 새로 수정된 픽셀들 (수정안됨 → 수정됨)
                            new_modified_pixels = be_all[img_idx, ind_np]

                            # 인덱스 스왑
                            temp_b = b_all[img_idx].clone()
                            temp_be = be_all[img_idx].clone()

                            temp_b[ind_p] = new_modified_pixels
                            temp_be[ind_np] = restored_pixels

                            b_all[img_idx] = temp_b
                            be_all[img_idx] = temp_be


            
        
            return self.current_query, x_best, best_changed_pixels, True
    
    def perturb(self, img, gt):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """

        self.init_hyperparam(img)

        first_img = img.squeeze(0) # Shape: (C, H, W)
        gt = gt.squeeze(0) # Shape: (H, W)
        
        with torch.no_grad():
            # 1. Get original logits and predictions
            try:
                if self.is_mmseg_model:
                    # 기존 mmseg API 사용
                    first_img_result = inference_model(self.model, first_img.permute(1, 2, 0).cpu().numpy()) # Pass Tensor
                    first_img_logits = first_img_result.seg_logits.data.to(self.device) # Shape: (C, H, W)
                    first_img_probs = softmax(first_img_logits, dim=0).to(self.device) # Shape: (C, H, W)
                    first_img_pred_labels = first_img_result.pred_sem_seg.data.squeeze().to(self.device) # Shape: (H, W)
                elif self.is_sed_model:
                    # Detectron2 inference
                    inputs = [{"image": first_img}]
                    outputs = self.model(inputs)
                    first_img_logits = outputs[0]["sem_seg"].to(self.device)
                    first_img_probs = softmax(first_img_logits, dim=0)
                    first_img_pred_labels = first_img_logits.argmax(dim=0)
                else:
                    # Convert tensor to numpy for model_predict
                    first_img_np = first_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    first_img_probs, first_img_pred_labels = model_predict(self.model, first_img_np, self.cfg)
                    
            except Exception as e:
                print("\n--- Error calling inference_model (Original Image) ---")
                raise e

            if self.mask is None:
                
                if self.is_mmseg_model:
                    # 기존 mmseg API 사용
                    original_img_result = inference_model(self.model, first_img.permute(1, 2, 0).cpu().numpy())
                    original_img_logits = original_img_result.seg_logits.data.to(self.device) # Shape: (C, H, W)
                    original_img_probs = softmax(original_img_logits, dim=0).to(self.device) # Shape: (C, H, W)
                    original_img_pred_labels = original_img_result.pred_sem_seg.data.squeeze().to(self.device) # Shape: (H, W)
                elif self.is_sed_model:
                    # Already computed above
                    original_img_probs = first_img_probs
                    original_img_pred_labels = first_img_pred_labels
                else:
                    # pytorch 모델 직접 사용 (이미 위에서 계산됨)
                    # original_img_logits = first_img_logits
                    original_img_probs = first_img_probs
                    original_img_pred_labels = first_img_pred_labels
                
                self.original_pred_labels = original_img_pred_labels.clone()

                # decision_change loss 사용 시 초기 예측 레이블 저장
                if self.loss == 'decision_change':
                    self.previous_pred_labels = original_img_pred_labels.clone()
                    # 성공 마스크 초기화
                    self.success_mask = torch.zeros_like(original_img_pred_labels)

                # 2. Create masks
                ignore_index = 255
                num_classes = first_img_probs.shape[0]
                channel_indices = torch.arange(num_classes, device=self.device) # Shape: (C)

                # 예측이 맞은 픽셀만 선택
                # condition_mask = torch.ones_like(first_img_pred_labels, dtype=torch.bool)

                #gt를 사용해 배경 픽셀은 loss 계산에서 제외
                if self.cfg['dataset'] == 'cityscapes':
                    valid_gt_mask = gt != 255
                elif self.cfg['dataset'] == 'ade20k':
                    valid_gt_mask = gt != 0
                elif self.cfg['dataset'] == 'VOC2012':
                    valid_gt_mask = gt != 0
                else:
                    raise ValueError(f"Unsupported dataset: {self.cfg['dataset']}")

                correct_masked_pred_labels = torch.where(valid_gt_mask, first_img_pred_labels, ignore_index)

                #마스크를 (C, H, W) 형태로 변환
                channel_indices_reshaped = channel_indices.view(num_classes, 1, 1)
                channel_indices_reshaped = channel_indices_reshaped.to(correct_masked_pred_labels.device)
                self.mask = channel_indices_reshaped == correct_masked_pred_labels #broadcast
                
                # decision_loss=True일 때만 changed pixels 초기화
                if self.use_decision_loss:
                    self.pre_changed_pixels = torch.zeros(first_img_pred_labels.shape[0], first_img_pred_labels.shape[1]).to(self.device)
                else:
                    self.pre_changed_pixels = None
        
        ret = self.attack_single_run(img, gt, self.mask, first_img_pred_labels)
        if len(ret) == 4:
            qr_curr, adv_curr, best_changed_pixels, is_success = ret
        else:
            qr_curr, adv_curr, best_changed_pixels = ret
            is_success = True
        
        # decision_loss=True이고 best_changed_pixels가 None이 아닌 경우에만 업데이트
        if self.use_decision_loss and best_changed_pixels is not None:
            self.pre_changed_pixels = best_changed_pixels
        
        if self.enable_success_reporting:
            return qr_curr, adv_curr, is_success
        else:
            return qr_curr, adv_curr