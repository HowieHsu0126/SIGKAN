#!/bin/bash

set -x
# CUDA_VISIBLE_DEVICES=0 python train.py --model=gcn --hidden-units=128,128 \
#     --dim=64 --epochs=500 --lr=0.1 --dropout=0.2 --file-dir="/data/hwxu/Dataset/NPU/InflunceLocality/digg/" \
#     --batch=1024 --train-ratio=75 --valid-ratio=12.5 \
#     --class-weight-balanced --instance-normalization --use-vertex-feature



CUDA_VISIBLE_DEVICES=0 python train.py --model sigkan_norm \
                --file-dir "/data/hwxu/Dataset/NPU/InflunceLocality/digg/" \
                --epochs 5 \
                --lr 1e-3 \
                --weight-decay 5e-4 \
                --dropout 0.5 \
                --hidden-units "64,32" \
                --batch 2048 \
                --dim 64 \
                --delta 0.5 \
                --train-ratio 50 \
                --valid-ratio 25 \
                --seed 3047 \
                --use-vertex-feature \
                --shuffle