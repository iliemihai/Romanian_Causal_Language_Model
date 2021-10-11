#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export OUTPUT_DIR=/content/drive/MyDrive/Romanian_GPT_NEO/dataset
export TRAIN_FILE=/content/drive/MyDrive/Romanian_GPT_NEO/dataset/train.txt

python3 create_dataset.py \
    --output_dir="$OUTPUT_DIR" \
    --train_file="$TRAIN_FILE" \
