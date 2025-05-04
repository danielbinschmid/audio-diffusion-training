from pydantic import BaseModel
from typing import Literal
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM, DDPMOutput
import torch
from TorchJaekwon.Model.Diffusion.External.diffusers.DiffusersWrapper import (
    DiffusersWrapper,
)
from TorchJaekwon.Model.Diffusion.External.diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from TorchJaekwon.Model.Diffusion.External.diffusers.schedulers.scheduling_ddim import (
    DDIMScheduler,
)
from TorchJaekwon.Model.Diffusion.External.diffusers.schedulers.scheduling_ddim_inverse import (
    DDIMInverseScheduler,
)
from typing import Optional
from TorchJaekwon.Model.Diffusion.Editing import DeltaHBase


class DiffusersSchedulerConfig(BaseModel):
    num_steps: int = 100
    cfg_scale: float = 1.0  # 3.5
    use_asyrp: bool = False


class SamplerConfig(BaseModel):
    type: Literal["standard", "dpm", "ddim", "ddim_inverse"] = "standard"
    diffusers_scheduler_cfg: DiffusersSchedulerConfig = DiffusersSchedulerConfig()


class Sampler:
    def __init__(self, cfg: SamplerConfig):
        self.cfg = cfg

    def _sample(
        self,
        ddpm: DDPM,
        cond: torch.Tensor | dict,
        x_start: Optional[torch.Tensor] = None,
        delta_h: Optional[DeltaHBase] = None,
        inference_adapter: Optional[torch.nn.Module] = None,
    ):

        if type(cond) == torch.Tensor:
            cond = {"class_label": cond}

        if self.cfg.type == "standard":
            if delta_h is not None:
                raise NotImplementedError()

            if inference_adapter is not None:
                raise NotImplementedError()

            if x_start is not None:
                raise NotImplementedError()
            ddpm_output = ddpm.forward(cond=cond, stage="infer", is_cond_unpack=True,)
            return ddpm_output

        elif (
            self.cfg.type == "dpm"
            or self.cfg.type == "ddim"
            or self.cfg.type == "ddim_inverse"
        ):

            # get corresponding diffusers scheduling class
            if self.cfg.type == "dpm":
                diffusers_scheduler_class = DPMSolverMultistepScheduler
            elif self.cfg.type == "ddim":
                diffusers_scheduler_class = DDIMScheduler
            elif self.cfg.type == "ddim_inverse":
                diffusers_scheduler_class = DDIMInverseScheduler
            else:
                raise ValueError("")

            ddpm_output: DDPMOutput = DiffusersWrapper.infer(
                ddpm_module=ddpm,
                diffusers_scheduler_class=diffusers_scheduler_class,
                x_shape=None,
                cond=cond,
                is_cond_unpack=True,
                num_steps=self.cfg.diffusers_scheduler_cfg.num_steps,
                cfg_scale=self.cfg.diffusers_scheduler_cfg.cfg_scale,
                x_start=x_start,
                delta_h=delta_h,
                use_asyrp=self.cfg.diffusers_scheduler_cfg.use_asyrp,
                inference_adapter=inference_adapter,
            )
            return ddpm_output
        else:
            raise NotImplementedError(f"Sampler {self.cfg.type} not implemented.")

    def sample(
        self,
        ddpm: DDPM,
        cond: torch.Tensor | dict,
        x_start: Optional[torch.Tensor] = None,
        use_no_grad: bool = True,
        delta_h: Optional[DeltaHBase] = None,
        inference_adapter: Optional[torch.nn.Module] = None,
    ) -> DDPMOutput:
        if use_no_grad:
            with torch.no_grad():
                return self._sample(
                    ddpm,
                    cond,
                    x_start=x_start,
                    delta_h=delta_h,
                    inference_adapter=inference_adapter,
                )
        else:
            return self._sample(
                ddpm,
                cond,
                x_start=x_start,
                delta_h=delta_h,
                inference_adapter=inference_adapter,
            )
