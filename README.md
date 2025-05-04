# audio-diffusion-training

This repository has submodules. Pull them explicitely using
```sh
git submodule update --init --recursive
```

## Setup

1. Create venv
2. `pip install --upgrade pip`
3. Install torch `2.7.0`
4. `pip install -r requirements.txt`
5. Install submodules
```sh
$ cd ./code/deps/TorchJaekwon
$ pip install -e .
$ cd ../werkzeug
$ pip install -e .
```

## Inference

1. Use the command line tool for inference and the provided pre-trained models. Install it via 
```sh
$ cd code
$ pip install -e .
```
2. Configure the paths in `code/configs/dm`. Set the paths to the weights and configuration files.
3. Configure the number of samples to generate and sampler configuration in the relevant `infer.yaml` script.
3. Run inference using lookup table below.

|Model|Inference Command|Required Weight Files|
|-|-|-|
|MedleySolosDB Unconditional|`dm medley uncond code/configs/dm/unconditional_medley_v0/infer.yaml`|U-Net: `melddpm_v0`, HifiGAN:`hifigan-ckpt(only generator)`|
|MedleySolosDB Categorical|||
|MoisesDB Mel|||
|MoisesDB LDM|||


## Training

### Mel-Spectrogram Based Diffusion Model

1. Compute normalisation scale of the mel-spectrogram by adapting `code/experiments/compute_melminmax.py`
2. Generate config by running e.g. `code/medley_uncond.sh` after setting
```sh
# basic flags
EXEC_GENERATE_CFG=true
EXEC_TRAIN=false
EXEC_TEST=false
```
3. Set paths in the generated config file.
4. Run training by running e.g. `code/medley_uncond.sh` after setting
```sh
# basic flags
EXEC_GENERATE_CFG=false
EXEC_TRAIN=true
EXEC_TEST=false
```


### Latent Diffusion Model