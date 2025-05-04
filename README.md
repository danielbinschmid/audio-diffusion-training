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