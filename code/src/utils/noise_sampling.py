from typing import Literal, Optional, Any
import torch
from src.sampler import Sampler, SamplerConfig, DiffusersSchedulerConfig
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM


def sample_noise(
    mode: Literal["random", "ddim_inversion"],
    ddpm: DDPM,
    cond: Optional[Any] = None,
    audio_latent: Optional[torch.Tensor] = None,
    num_steps_ddim_inversion: int = 100,
    seed: int | None = None,
) -> torch.Tensor:
    if mode == "random":
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        return ddpm.sample_rndn_x0(cond=cond, generator=generator)
    elif mode == "ddim_inversion":
        assert audio_latent is not None

        # 2. inverse
        sampler_cfg = SamplerConfig(
            type="ddim_inverse",
            diffusers_scheduler_cfg=DiffusersSchedulerConfig(
                num_steps=num_steps_ddim_inversion, cfg_scale=1.0
            ),
        )
        sampler = Sampler(cfg=sampler_cfg)
        x_inversed = sampler.sample(ddpm=ddpm, cond=cond, x_start=audio_latent)
        return x_inversed.raw_diffusion_output
    else:
        raise NotImplementedError()
