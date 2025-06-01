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

python main.py \
    --device cuda:0 \
    --attack_pixel 0.05 \
    --config configs_attack/ade20k/config_setr.py

python rs_eval.py \
    --config configs_attack/ade20k/config_mask2_swin_B.py \
    --device cuda:0 \
    --eps 0.001 \
    --num_images 50 \
    --n_queries 5000