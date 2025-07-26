#!/bin/sh
#configs_attack/ade20k/config_mask2_swin_B.py
#configs_attack/ade20k/config_seg.py
#configs_attack/ade20k/config_pspnet.py
#configs_attack/ade20k/config_deeplabv3.py
#configs_attack/ade20k/config_setr.py

#configs_attack/cityscapes/config_mask2_swin_B.py
#configs_attack/cityscapes/config_seg.py
#configs_attack/cityscapes/config_pspnet.py
#configs_attack/cityscapes/config_deeplabv3.py
#configs_attack/cityscapes/config_setr.py
export CUDA_VISIBLE_DEVICES=0


python rs_eval.py \
    --config configs_attack/ade20k/config_deeplabv3.py\
    --norm patches\
    --eps 0.04

python rs_eval.py \
    --config configs_attack/ade20k/config_seg.py

python rs_eval.py \
    --config configs_attack/ade20k/config_pspnet.py

python rs_eval.py \
    --config configs_attack/ade20k/config_setr.py

python rs_eval.py \
    --config configs_attack/cityscapes/config_deeplabv3.py

python rs_eval.py \
    --config configs_attack/cityscapes/config_seg.py