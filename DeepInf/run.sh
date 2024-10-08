#!/bin/bash

set -x
CUDA_VISIBLE_DEVICES=0 python train.py --model=gcn --hidden-units=128,128 \
    --dim=64 --epochs=500 --lr=0.1 --dropout=0.2 --file-dir="/data/hwxu/Dataset/NPU/InflunceLocality/digg/" \
    --batch=1024 --train-ratio=75 --valid-ratio=12.5 \
    --class-weight-balanced --instance-normalization --use-vertex-feature

