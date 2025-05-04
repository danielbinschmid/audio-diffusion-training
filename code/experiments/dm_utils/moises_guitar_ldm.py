"""
Runs with MusicLDM from diffusers
"""

import os
from typing import Optional
from uuid import uuid4
from TorchJaekwon.Util.UtilAudio import UtilAudio
from schmid_werkzeug.ml_workflow import get_sample_batches
from schmid_werkzeug.pydantic import DynBaseModel, load_cfg, load_cfg_dynamic
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import (
    GuidedDiffusionUnet,
    GuidedDiffusionUnetConfig,
)
from src.models.mel_ldm import MelLDM, MelLDMConfig
from src.sampler.sampler import DiffusersSchedulerConfig, Sampler, SamplerConfig
from src.utils.audio_diffusers_output import AudioDiffusersOutput
from src.utils.noise_sampling import sample_noise
from .shared_utils import InferenceCfgUncond
from schmid_werkzeug import print_info
import torch


def infer(
    ldm: MelLDM,
    n_samples: int,
    max_bs: int,
    num_steps: int,
    out_folder: str,
    device,
    sampler_t: str = "dpm",
):
    out_path = os.path.join(out_folder, str(uuid4()))
    os.makedirs(out_path, exist_ok=False)
    global_idx = 0
    batch_sizes = get_sample_batches(n_samples=n_samples, max_bs=max_bs)

    for batch_idx, batch_size in enumerate(batch_sizes):

        x_start = sample_noise(
            mode="random",
            ddpm=ldm,
            cond=ldm.get_unconditional_condition(
                cond_shape=(batch_size,), condition_device=device,
            ),
            seed=batch_idx,
        )
        sampler_cfg = SamplerConfig(
            type=sampler_t,
            diffusers_scheduler_cfg=DiffusersSchedulerConfig(num_steps=num_steps),
        )
        sampler = Sampler(cfg=sampler_cfg)
        outp: AudioDiffusersOutput = sampler.sample(
            ddpm=ldm,
            cond=torch.tensor([[1] for i in range(batch_size)]),
            x_start=x_start.to(device),
        )

        audio_hat = outp.x
        for i in range(batch_size):
            UtilAudio.write(
                audio_path=os.path.join(out_path, f"out_{global_idx}.wav"),
                audio=audio_hat[i],
                sample_rate=16000,
            )
            global_idx += 1


def get_ldm(
    unet_cfg: GuidedDiffusionUnetConfig,
    ldm_cfg: MelLDMConfig,
    device,
    ldm_ckpt: Optional[str] = None,
):
    unet = GuidedDiffusionUnet(unet_cfg).to(device)

    ldm = MelLDM(
        cfg=ldm_cfg,
        device=device,
        sample_rate=16000,
        model=unet,
        model_output_type="v_prediction",
        unconditional_prob=0,
    )
    ldm = ldm.to(device)

    if ldm_ckpt is not None:
        print_info(f"Loading checkpoint {ldm_ckpt}")
        state_dict = torch.load(ldm_ckpt, weights_only=True, map_location=device)
        ldm.load_state_dict(state_dict)

    return ldm


class GuitarLDMCfg(DynBaseModel):
    unet_cfg: str = "./unet.yaml"
    ldm_cfg: str = "./ldm.yaml"
    original_base_path: str = "./"
    ldm_ckpt: str = "/path/to/ckpt.pth"

    def get_ldm(self, device) -> MelLDM:
        m = lambda fp: self.map(fp)
        ldm_cfg: MelLDMConfig = load_cfg(m(self.ldm_cfg), CfgClass=MelLDMConfig)
        unet_cfg: GuidedDiffusionUnetConfig = load_cfg(
            m(self.unet_cfg), CfgClass=GuidedDiffusionUnetConfig
        )
        ldm = get_ldm(
            unet_cfg=unet_cfg, ldm_cfg=ldm_cfg, device=device, ldm_ckpt=self.ldm_ckpt
        )
        return ldm


def inference(inference_cfg: InferenceCfgUncond):
    cfg: GuitarLDMCfg = load_cfg_dynamic(inference_cfg.pipeline_cfg, GuitarLDMCfg)
    assert cfg is not None
    ldm = cfg.get_ldm(inference_cfg.device)
    infer(
        ldm=ldm,
        n_samples=inference_cfg.n_samples,
        max_bs=inference_cfg.max_bs,
        num_steps=inference_cfg.num_steps,
        out_folder=inference_cfg.out,
        device=inference_cfg.device,
        sampler_t=inference_cfg.sampler_type,
    )
