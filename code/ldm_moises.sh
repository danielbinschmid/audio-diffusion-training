#!/bin/bash


# basic flags
EXEC_GENERATE_CFG=false
EXEC_TRAIN=true
EXEC_INFER=false

# config
CFG_PATH="/home/danielbinschmid/audio-diffusion-training/code/configs/ldm_v0.yaml"

function train() {
    python ./experiments/ldm_moisesdb.py \
        --mode "train" \
        --cfg_path $CFG_PATH 
}

function infer() {
    python ./experiments/ldm_moisesdb.py \
        --mode "infer" \
        --cfg_path $CFG_PATH 
}

function generate_config() {
    python ./experiments/ldm_moisesdb.py \
        --mode "generate_config" \
        --cfg_path $CFG_PATH 
}


if [ "$EXEC_GENERATE_CFG" = true ]; then
    generate_config
fi 

if [ "$EXEC_TRAIN" = true ]; then
    train
fi 

if [ "$EXEC_INFER" = true ]; then
    infer
fi 
