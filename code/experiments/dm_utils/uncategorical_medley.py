from src.sampler.sampler import DiffusersSchedulerConfig
from schmid_werkzeug.pydantic import DynBaseModel, load_cfg, load_cfg_dynamic
from TorchJaekwon.Util.UtilAudio import UtilAudio
import torch
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import (
    GuidedDiffusionUnetConfig,
)
from src.models.MelDDPM import MelDDPM, MelDDPMConfig, MelDDPMOutput
from src.models.hifigan import HifiGANConfig
from typing import Literal
from pydantic import BaseModel
from schmid_werkzeug import get_sample_batches
from src.sampler import SamplerConfig, Sampler
from uuid import uuid4
from .shared_utils import get_melddpm
import os


def infer(
    mel_ddpm: MelDDPM, n_samples: int, out_folder: str, max_bs: int, num_steps: int
):
    global_idx = 0
    batches = get_sample_batches(n_samples, max_bs)
    sampler_cfg = SamplerConfig(
        type="dpm",
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


class UncategoricalMedleyV0Cfg(DynBaseModel):
    vocoder_cfg: str = "./vocoder.yaml"
    vocoder_weights: str = "/home/danielbinschmid/melddpm/data/models/pretrained_opensource/musicldm_hifigan/hifigan-ckpt(only generator).ckpt"
    unet_cfg: str = "./unet.yaml"
    weights: str = "/home/danielbinschmid/melddpm/data/models/good_checkpoints/melddpm_v1/2024-12-03.pth"
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
    n_samples: int = 10
    device: str
    max_bs: int = 32
    num_steps: int = 200
    sampler_type: Literal["standard", "dpm"]


def inference(inference_cfg: InferenceCfg):
    cfg: UncategoricalMedleyV0Cfg = load_cfg_dynamic(
        inference_cfg.pipeline_cfg, UncategoricalMedleyV0Cfg
    )
    assert cfg is not None
    ddpm = cfg.get_melddpm(inference_cfg.device)
    infer(
        ddpm,
        inference_cfg.n_samples,
        out_folder=inference_cfg.out,
        max_bs=inference_cfg.max_bs,
        num_steps=inference_cfg.num_steps,
    )
