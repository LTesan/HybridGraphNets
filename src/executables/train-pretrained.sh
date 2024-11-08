#!/bin/bash

# Script for training the model
source /mnt/cephfs/home/amb/ltesan/miniconda3/bin/activate pl

# Execute the Python script with train argument and additional training parameters
python main.py  --pretrained  --train --dim_hidden 250 --passes 12 --lambda_d 5. --max_epoch 5000  --batch_size 8  --lr 0.0001

