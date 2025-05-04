from tqdm import tqdm
from src.datasets.medley_solo_db import MedleySolosDBConstants
from src.sampler.sampler import DiffusersSchedulerConfig
from schmid_werkzeug.pydantic import DynBaseModel, load_cfg, load_cfg_dynamic
from TorchJaekwon.Util.UtilAudio import UtilAudio
import torch
import torch.nn as nn
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import (
    GuidedDiffusionUnet,
    GuidedDiffusionUnetConfig,
)
import random
from src.models.MelDDPM import MelDDPM, MelDDPMConfig, MelDDPMOutput
from src.models.hifigan import HifiGAN, HifiGANConfig
from typing import Literal, Optional
from pydantic import BaseModel
from schmid_werkzeug import get_sample_batches
from src.sampler import SamplerConfig, Sampler
from uuid import uuid4
from .shared_utils import get_melddpm
import os
from typing import Dict, List
import numpy as np


def infer_uniform_distribution(
    mel_ddpm: MelDDPM,
    n_samples_per_category: int,
    out_folder: str,
    max_bs: int,
    num_steps: int,
    device,
    sampler_type="dpm",
):
    n_categories = mel_ddpm.cfg.n_categories + 1  # + 1 for unconditional

    global_idx_per_categ = {}
    batches = get_sample_batches(n_samples_per_category, max_bs // n_categories)
    sampler_cfg = SamplerConfig(
        type=sampler_type,
        diffusers_scheduler_cfg=DiffusersSchedulerConfig(num_steps=num_steps),
    )
    sampler = Sampler(cfg=sampler_cfg)
    out_path = os.path.join(out_folder, str(uuid4()))
    categ_to_name = MedleySolosDBConstants.IDX_TO_DESCR
    os.makedirs(out_path, exist_ok=False)

    for categ_idx in range(n_categories):
        os.makedirs(os.path.join(out_path, categ_to_name[categ_idx]))
        global_idx_per_categ[categ_idx] = 0

    for batch_idx, batch in enumerate(batches):
        cond_vector = [
            categ_idx for categ_idx in range(n_categories) for _ in range(batch)
        ]
        cond = {"class_label": torch.tensor(cond_vector).to(device)}

        outp: MelDDPMOutput = sampler.sample(
            ddpm=mel_ddpm, cond=cond,
        )
        audio_hat = outp.x

        for categ_idx in range(n_categories):
            categ_name = categ_to_name[categ_idx]
            categ_out_folder = os.path.join(out_path, categ_name)

            out_audio = audio_hat[int(categ_idx * batch) : int((categ_idx + 1) * batch)]

            for j in range(batch):
                UtilAudio.write(
                    audio_path=os.path.join(
                        categ_out_folder, f"out_{global_idx_per_categ[categ_idx]}.wav"
                    ),
                    audio=out_audio[j],
                    sample_rate=mel_ddpm.cfg.sample_rate,
                )
                global_idx_per_categ[categ_idx] += 1


def create_index_list(distribution: Dict[int, int]) -> List[int]:
    """
    Convert a class distribution dictionary into a list of indices.
    
    Args:
        distribution (Dict[int, int]): A dictionary where keys are class indices
                                       and values are the number of samples.
    
    Returns:
        List[int]: A list of indices with the same distribution.
    """
    index_list = []
    for class_idx, count in distribution.items():
        index_list.extend([class_idx] * count)
    random.shuffle(index_list)
    index_list = np.array(index_list)
    return index_list


def infer_mirrored_distribution(
    mel_ddpm: MelDDPM,
    max_bs: int,
    num_steps: int,
    out_folder: str,
    device,
    sampler_type="dpm",
):
    global_idx: int = 0
    index_list: np.ndarray = create_index_list(
        MedleySolosDBConstants.TEST_SET_DISTRIBUTION
    )
    batches = get_sample_batches(len(index_list), max_bs)
    sampler_cfg = SamplerConfig(
        type=sampler_type,
        diffusers_scheduler_cfg=DiffusersSchedulerConfig(num_steps=num_steps),
    )
    out_folder = os.path.join(out_folder, str(uuid4()))

    sampler = Sampler(cfg=sampler_cfg)

    for batch_idx, batch in enumerate(batches):
        batch_start = batch_idx * max_bs
        batch_end = batch_start + batch

        class_labels = index_list[batch_start:batch_end].tolist()

        cond = {"class_label": torch.tensor(class_labels).to(device)}

        outp: MelDDPMOutput = sampler.sample(
            ddpm=mel_ddpm, cond=cond,
        )
        for idx in range(batch):
            y = class_labels[idx]
            audio = outp.x[idx]

            UtilAudio.write(
                audio_path=os.path.join(out_folder, f"out_{global_idx}_label-{y}.wav"),
                audio=audio,
                sample_rate=mel_ddpm.cfg.sample_rate,
            )
            global_idx += 1


class CondMedleyV0Cfg(DynBaseModel):
    vocoder_cfg: str = "./vocoder.yaml"
    vocoder_weights: str = "/home/danielbinschmid/melddpm/data/models/pretrained_opensource/musicldm_hifigan/hifigan-ckpt(only generator).ckpt"
    unet_cfg: str = "./unet.yaml"
    weights: str = "/home/danielbinschmid/melddpm/data/models/good_checkpoints/categorical_medley_v1/checkpoints/507.pth"
    melddpm_cfg: str = "./melddpm.yaml"
    original_base_path: str = "./"

    def get_melddpm(self, device="cuda") -> MelDDPM:
        m = lambda fp: self.map(fp)

        vocoder_cfg = load_cfg(m(self.vocoder_cfg), HifiGANConfig)
        unet_cfg = load_cfg(m(self.unet_cfg), GuidedDiffusionUnetConfig)
        melddpm_cfg = load_cfg(m(self.melddpm_cfg), MelDDPMConfig)

        melddpm = get_melddpm(
            ddpm_cfg=melddpm_cfg,
            vocoder_cfg=vocoder_cfg,
            vocoder_weights=self.vocoder_weights,
            unet_backbone_cfg=unet_cfg,
            device=device,
            mel_ddpm_weights=self.weights,
        )
        return melddpm


class InferenceCfg(BaseModel):
    pipeline_cfg: str
    out: str
    device: str
    max_bs: int = 32
    num_steps: int = 200
    sampler_type: Literal["standard", "dpm"]
    mirror_test_distribution: bool = True
    n_samples_per_category_for_uniform_inference: int = 10


def inference(inference_cfg: InferenceCfg):
    cfg: CondMedleyV0Cfg = load_cfg_dynamic(inference_cfg.pipeline_cfg, CondMedleyV0Cfg)
    assert cfg is not None
    ddpm = cfg.get_melddpm(inference_cfg.device)

    if inference_cfg.mirror_test_distribution:
        infer_mirrored_distribution(
            ddpm,
            out_folder=inference_cfg.out,
            max_bs=inference_cfg.max_bs,
            num_steps=inference_cfg.num_steps,
            device=inference_cfg.device,
            sampler_type=inference_cfg.sampler_type,
        )
    else:
        infer_uniform_distribution(
            ddpm,
            n_samples_per_category=inference_cfg.n_samples_per_category_for_uniform_inference,
            out_folder=inference_cfg.out,
            max_bs=inference_cfg.max_bs,
            num_steps=inference_cfg.num_steps,
            device=inference_cfg.device,
            sampler_type=inference_cfg.sampler_type,
        )
