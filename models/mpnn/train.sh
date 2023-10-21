#!/bin/bash

# train
CUDA_VISIBLE_DEVICES=0 python train.py ./configs/config_Adv.yaml | tee ./logs/train/advection.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./configs/config_Adv.yaml | tee ./logs/test/advection.out

CUDA_VISIBLE_DEVICES=0 python train.py ./configs/config_1DBugers.yaml | tee ./logs/train/1DBugers.out

# test
CUDA_VISIBLE_DEVICES=0 python train.py ./configs/config_Adv.yaml | tee ./logs/test/advection.out
CUDA_VISIBLE_DEVICES=1 python train.py ./configs/config_1DBugers.yaml | tee ./logs/test/1DBugers.out
