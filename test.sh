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

export CUDA_VISIBLE_DEVICES=3



# python rs_eval.py \
#     --config configs_attack/cityscapes/config_deeplabv3.py \
#     --n_queries 1000 \
#     --eps 0.01

# python rs_eval.py \
#     --config configs_attack/cityscapes/config_seg.py \
#     --n_queries 1000 \
#     --eps 0.01

python rs_eval.py \
    --config configs_attack/ade20k/config_pspnet.py \
    --num_images 1

python rs_eval.py \
    --config configs_attack/cityscapes/config_setr.py \
    --n_queries 1000 \
    --eps 0.01