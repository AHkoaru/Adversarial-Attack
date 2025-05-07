#configs_attack/ade20k/config_mask2_swin_B.py
#configs_attack/ade20k/config_seg.py

#configs_attack/cityscapes/config_mask2_swin_B.py
#configs_attack/cityscapes/config_seg.py


python main.py --device cuda:0 --attack_pixel 0.01 --config configs_attack/ade20k/config_seg.py