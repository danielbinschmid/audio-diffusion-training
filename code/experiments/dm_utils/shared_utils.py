import os
from uuid import uuid4
from pydantic import BaseModel
import torch
import torch.nn as nn
from TorchJaekwon.Util.UtilAudio import UtilAudio
from schmid_werkzeug.ml_workflow import get_sample_batches
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import (
    GuidedDiffusionUnet,
    GuidedDiffusionUnetConfig,
)
from src.models.MelDDPM import MelDDPM, MelDDPMConfig, MelDDPMOutput
from src.models.hifigan import HifiGAN, HifiGANConfig
from typing import Literal, Optional

from src.sampler.sampler import DiffusersSchedulerConfig, Sampler, SamplerConfig


def get_melddpm(
    ddpm_cfg: MelDDPMConfig,
    vocoder_cfg: HifiGANConfig,
    vocoder_weights: str,
    unet_backbone_cfg: GuidedDiffusionUnetConfig,
    device,
    mel_ddpm_weights: Optional[str] = None,
) -> MelDDPM:
    diffusion_denoiser: nn.Module = GuidedDiffusionUnet(cfg=unet_backbone_cfg)

    vocoder = HifiGAN(h=vocoder_cfg)
    vocoder_state_dict = torch.load(vocoder_weights, weights_only=True)
    vocoder.load_state_dict(state_dict=vocoder_state_dict["generator"])

    mel_ddpm: nn.Module = MelDDPM(
        cfg=ddpm_cfg,
        vocoder=vocoder,
        model=diffusion_denoiser,
        model_output_type="v_prediction",
        unconditional_prob=0,
    )

    mel_ddpm = mel_ddpm.to(device)

    if mel_ddpm_weights is not None:
        state_dict = torch.load(
            mel_ddpm_weights, weights_only=True, map_location=device
        )
        mel_ddpm.load_state_dict(state_dict)
    return mel_ddpm


class InferenceCfgUncond(BaseModel):
    pipeline_cfg: str
    out: str
    n_samples: int = 10
    device: str
    max_bs: int = 32
    num_steps: int = 200
    sampler_type: Literal["standard", "dpm"]


def infer_uncond_melddpm(
    mel_ddpm: MelDDPM,
    n_samples: int,
    out_folder: str,
    max_bs: int,
    num_steps: int,
    sampler="dpm",
):
    global_idx = 0
    batches = get_sample_batches(n_samples, max_bs)
    sampler_cfg = SamplerConfig(
        type=sampler,
        diffusers_scheduler_cfg=DiffusersSchedulerConfig(num_steps=num_steps),
    )
    sampler = Sampler(cfg=sampler_cfg)
    out_path = os.path.join(out_folder, str(uuid4()))
    os.makedirs(out_path, exist_ok=False)

    for i, batch in enumerate(batches):

        outp: MelDDPMOutput = sampler.sample(
            ddpm=mel_ddpm,
            cond={"class_label": torch.tensor([[1] for _ in range(batch)])},
        )

        audio_hat = outp.x
        for i in range(batch):
            UtilAudio.write(
                audio_path=os.path.join(out_path, f"out_{global_idx}.wav"),
                audio=audio_hat[i],
                sample_rate=mel_ddpm.cfg.sample_rate,
            )
            global_idx += 1
