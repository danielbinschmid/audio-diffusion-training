#!/bin/bash

# basic flags
EXEC_GENERATE_CFG=true
EXEC_TRAIN=false
EXEC_TEST=false

# config
CFG_PATH="./configs/medley_uncond.yaml"

function train() {
    python ./experiments/unconditional_medley_solos_db.py \
        --mode "train" \
        --cfg_path $CFG_PATH 
}

function test() {
    python ./experiments/unconditional_medley_solos_db.py \
        --mode "test" \
        --cfg_path $CFG_PATH \
        --ddpm_ckpt_path "/path/to/data/models/good_checkpoints/guitar_unconditional_medley_v0/checkpoints/777.pth" \
        --n_inference 20
}

function generate_config() {
    python ./experiments/unconditional_medley_solos_db.py \
        --mode generate_config \
        --cfg_path $CFG_PATH 
}

if [ "$EXEC_GENERATE_CFG" = true ]; then
    generate_config
fi 

if [ "$EXEC_TRAIN" = true ]; then
    train
fi 

if [ "$EXEC_TEST" = true ]; then
    test
fi 
