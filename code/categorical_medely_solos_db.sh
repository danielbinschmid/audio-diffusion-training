#!/bin/bash


# basic flags
EXEC_GENERATE_CFG=false
EXEC_TRAIN=true
EXEC_TEST=false

# config
CFG_PATH="./configs/categorical_v1.yaml"
EXEC_SCRIPT="./experiments/categorical_medley_solos_db.py"

function train() {
    python $EXEC_SCRIPT \
        --mode "train" \
        --cfg_path $CFG_PATH 
}

function test() {
    python $EXEC_SCRIPT \
        --mode "test" \
        --cfg_path $CFG_PATH \
        --ddpm_ckpt_path "/path/to/data/models/good_checkpoints/categorical_medley_v1/checkpoints/491.pth" \
        --n_infer_per_categ 5
}

function generate_config() {
    python $EXEC_SCRIPT \
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
