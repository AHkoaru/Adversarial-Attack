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
# export CUDA_VISIBLE_DEVICES=0

# python px_eval.py \
#     --config configs_attack/ade20k/config_pspnet.py\
#     --attack_pixel 0.05

<<<<<<< HEAD
python rs_eval.py \
    --config configs_attack/cityscapes/config_setr.py \
    --resume \
    --resume_timestamp 20250905_034206 



python rs_eval.py \
    --config configs_attack/voc2012/config_pspnet.py\
    --num_images 1
=======
# python px_eval.py \
#     --config configs_attack/ade20k/config_deeplabv3.py\
#     --attack_pixel 0.05

# python rs_eval.py \
#     --config configs_attack/cityscapes/config_pspnet.py
>>>>>>> 124bc0d0439397f1f25a1bc2bcfcd900fad7ae9a


# python rs_eval.py \
#     --config configs_attack/cityscapes/config_deeplabv3.py


# python px_eval.py \
#     --config configs_attack/VOC2012/config_deeplabv3.py\
#     --attack_pixel 0.05


# python rs_eval.py \
#     --config configs_attack/ade20k/config_pspnet.py\
#     --norm patches\
#     --eps 0.04

# python rs_eval.py \
#     --config configs_attack/ade20k/config_pspnet.py

# python rs_eval.py \
#     --config configs_attack/cityscapes/config_deeplabv3.py

python px_eval.py \
    --config configs_attack/cityscapes/config_deeplabv3.py\
    --attack_pixel 0.05

python px_eval.py \
    --config configs_attack/cityscapes/config_pspnet.py\
    --attack_pixel 0.05

python px_eval.py \
    --config configs_attack/cityscapes/config_seg.py\
    --attack_pixel 0.05

# python px_eval.py \
#     --config configs_attack/cityscapes/config_seg.py\
#     --attack_pixel 0.05



python graph.py \
    --config configs_attack/ade20k/config_pspnet.py\
    --num_images 10

# python rs_eval.py \
#     --config configs_attack/cityscapes/config_deepdlabv3.py

# python rs_eval.py \
#     --config configs_attack/cityscapes/config_seg.py

python px_eval.py \
    --config configs_attack/cityscapes/config_setr.py\
    --attack_pixel 0.05

python zc_eval.py \
    --config configs_attack/VOC2012/config_zegclip.py\
    --num_images 1\
    --iters 10

CUDA_VISIBLE_DEVICES="0" 
python test.py configs/voc12/vpt_seg_fully_vit-b_512x512_20k_12_10.py  ckpt/voc_fully_512_vit_base.pth --eval=mIoU