#!/bin/bash


# basic flags
EXEC_GENERATE_CFG=false
EXEC_TRAIN=true
EXEC_TEST=false

# config
CFG_PATH="./configs/unconditional_guitar_v1.yaml"

function train() {
    python ./experiments/unconditional_moises_db.py \
        --mode "train" \
        --cfg_path $CFG_PATH 
}

function test() {
    python ./experiments/unconditional_moises_db.py \
        --mode "test" \
        --cfg_path $CFG_PATH
}

function generate_config() {
    python ./experiments/unconditional_moises_db.py \
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
