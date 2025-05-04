# audio-diffusion-training

This repository is a repository with minimal dependencies for training diffusion models for audio, based on 
[TorchJaekwon](https://github.com/jakeoneijk/TorchJaekwon).

This repository has submodules. Pull them explicitely using
```sh
git submodule update --init --recursive
```

## Setup

1. Create venv
2. `pip install --upgrade pip`
3. Install torch `2.7.0` (with torchaudio)
4. `pip install -r requirements.txt`
5. Install submodules
```sh
$ cd ./code/deps/TorchJaekwon
$ pip install -e .
$ cd ../werkzeug
$ pip install -e .
```

### Diffusers Dependency

If you want to run our pre-trained latent diffusion model, you need to install diffusers. For the LDM, diffusers is required because we took the VAE from there. In this case, run in addition:
```sh
$ pip install -r requirements_diffusers_extra.txt
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
|MedleySolosDB Unconditional|`dm medley uncond code/configs/dm/unconditional_medley_v0/infer.yaml`|U-Net: `melddpm_v0.pth`, HifiGAN:`hifigan-ckpt(only generator)`|
|MedleySolosDB Categorical|`dm medley cond code/configs/dm/cond_medley_v0/infer.yaml`|U-Net: `medley_cond.pth`, HifiGAN:`hifigan-ckpt(only generator)`|
|MoisesDB Mel|`dm moises mel code/configs/dm/moises_guitar_mel/infer.yaml`|U-Net: `moises_mel.pth`, HifiGAN:`hifigan-ckpt(only generator)`|
|MoisesDB LDM|`dm moises ldm code/configs/dm/moises_guitar_ldm/infer.yaml`|U-Net: `moises_ldm.pth`, VAE and vocoder: From `diffusers`|


## Training

### Datasets

|Datasets|URL|Note|
|-|-|-|
|MedleySolosDB|Download from [zenodo](https://zenodo.org/records/3464194)||
|MoisesDB|Download from the homepage of [music.ai/research](https://music.ai/research/)|The script will automatically create a training and test split. The split is documented using a csv file. To re-use the split, make sure to backup this csv file safely.|


### Mel-Spectrogram Based Diffusion Model

1. Compute normalisation scale of the mel-spectrogram by adapting `code/experiments/compute_melminmax.py`. Alternatively, re-use the scales pre-computed in the provided pre-trained models in folder `code/configs/dm`
2. Generate config by running e.g. `code/medley_uncond.sh` after setting
```sh
# basic flags
EXEC_GENERATE_CFG=true
EXEC_TRAIN=false
EXEC_TEST=false
```
3. Set paths in the generated config file. Make sure to set the path to the pre-trained neural vocoder.
4. Run training by running e.g. `code/medley_uncond.sh` after setting
```sh
# basic flags
EXEC_GENERATE_CFG=false
EXEC_TRAIN=true
EXEC_TEST=false
```


### Latent Diffusion Model

1. Make sure that a pre-trained VAE and vocoder is prepared. If you want to train our `MelLDM`, make sure to have the dependencies in [requirements_diffusers_extra.txt](requirements_diffusers_extra.txt) installed.
2. Train as described above.