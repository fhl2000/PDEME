#!/bin/bash

# train
CUDA_VISIBLE_DEVICES=1,2 python train.py ./configs/config_diff-sorp.yaml | tee -a ./logs/test/diff-sorp.out