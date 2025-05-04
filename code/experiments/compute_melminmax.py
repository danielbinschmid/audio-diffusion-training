import os
import sys

sys.path.append(os.curdir)

from src.datasets import MoisesDBConfig, MoisesDBDataset
from TorchJaekwon.Util.UtilAudioMelSpec import UtilAudioMelSpec
import torch

basefolder = "/home/danielbinschmid/melddpm/data/datasets/moisesdb/moisesdb"
sample_rate: int = 16000
duration_sec: float = 6.0
nfft: int = 1024
hop_size: int = nfft // 4
mel_size: int = 64
frequency_min: float = 0
frequency_max: float = sample_rate // 2
frame_size: int = int((sample_rate * duration_sec) // hop_size)
batch_size: int = 16

util_melgram = UtilAudioMelSpec(
    nfft=nfft,
    hop_size=hop_size,
    sample_rate=sample_rate,
    mel_size=mel_size,
    frequency_min=frequency_min,
    frequency_max=frequency_max,
)
ds_cfg = MoisesDBConfig(
    ds_path="/home/danielbinschmid/melddpm/data/datasets/moisesdb/moisesdb",
    step_length_in_s=2.5,
    recompute_metadata=False,
    train_test_split_track_level=(0.8, 0.2),
    train_test_split_seed=1,
    empty_waveform_thresshold=0.01,
    stem_mode="Bass",
)
train_ds = MoisesDBDataset(cfg=ds_cfg.set_mode("train"))
test_ds = MoisesDBDataset(cfg=ds_cfg.set_mode("test", recompute_metadata=False))

# find min and max of frequency bins over whole dataset
from tqdm import tqdm

melspecs = []

for idx in tqdm(range(len(train_ds))):
    melspecs.append(
        util_melgram.get_hifigan_mel_spec(audio=train_ds[idx][0], return_type="Tensor")
    )
all_melgrams_train = torch.cat(melspecs)

mel_max = torch.max(all_melgrams_train, dim=2).values
mel_max = torch.max(mel_max, dim=0).values

mel_min = torch.min(all_melgrams_train, dim=2).values
mel_min = torch.min(mel_min, dim=0).values

import json


def serialize(tensor, fpath):
    tensor_list = tensor.tolist()
    with open(fpath, "w") as json_file:
        json.dump(tensor_list, json_file)


serialize(
    tensor=mel_max,
    fpath="/home/danielbinschmid/melddpm/code/configs/moisesdb/mel_max_bass.json",
)
serialize(
    tensor=mel_min,
    fpath="/home/danielbinschmid/melddpm/code/configs/moisesdb/mel_min_bass.json",
)
