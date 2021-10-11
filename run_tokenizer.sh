#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export OUTPUT_DIR=/content/drive/MyDrive/Romanian_GPT_NEO/
export TRAIN_FILE=/content/drive/MyDrive/Romanian_GPT_NEO/dataset/train.txt
export VOCAB_SIZE=50257
export MIN_FREQUENCY=2
export SPECIAL_TOKENS='<s>','<pad>','</s>','<unk>','<mask>'

python3 train_tokenizer.py \
    --output_dir="$OUTPUT_DIR"  \
    --train_file="$TRAIN_FILE" \
    --vocab_size=$VOCAB_SIZE \
    --min_frequency=$MIN_FREQUENCY \
    --special_tokens="$SPECIAL_TOKENS"