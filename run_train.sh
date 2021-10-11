#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export OUTPUT_DIR=/home/mihai/Documents/GPT-NEO
export MODEL_TYPE=gpt_neo
export CONFIG_NAME=/home/mihai/Documents/GPT-NEO
export TOKENIZER_NAME=/home/mihai/Documents/GPT-NEO
export TRAIN_FILE=/home/mihai/Documents/GPT-NEO/data/train-fixed.csv
export VALIDATION_FILE=/home/mihai/Documents/GPT-NEO/data/test-fixed.csv
export TEST_FILE=/home/mihai/Documents/GPT-NEO/data/test-fixed.csv
export DATASET_NAME=oscar
export DATASET_CONFIG_NAME=unshuffled_deduplicated_fa
export MAX_SEQUENCE_LENGTH=512
export PER_DEVICE_TRAIN_BATCH_SIZE=16
export PER_DEVICE_EVAL_BATCH_SIZE=16
export NUM_TRAIN_EPOCHS=5.0
export LEARNING_RATE=1e-4
export WARMUP_STEPS=5000
export LOGGING_STEPS=500
export EVAL_STEPS=25000
export SAVE_STEPS=25000


python3 run_clm_flax.py \
    --output_dir="$OUTPUT_DIR"  \
    --model_type="$MODEL_TYPE" \
    --config_name="$CONFIG_NAME" \
    --tokenizer_name="$TOKENIZER_NAME" \
    --dataset_name="$DATASET_NAME" \
    --dataset_config_name="$DATASET_CONFIG_NAME" \
    --block_size=$MAX_SEQUENCE_LENGTH \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --num_train_epochs=$NUM_TRAIN_EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --warmup_steps=$WARMUP_STEPS \
    --logging_step=$LOGGING_STEPS \
    --eval_steps=$EVAL_STEPS \
    --save_steps=$SAVE_STEPS \
    --do_train \
    --do_eval \