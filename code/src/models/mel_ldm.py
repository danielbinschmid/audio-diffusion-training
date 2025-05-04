import torch
from torch import Tensor
from src.models.musicldm import MusicLDMWrapper
from typing import Optional, Tuple, Union
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM
from pydantic import BaseModel
from src.utils.audio_diffusers_output import AudioDiffusersOutput
from typing import List


class MelLDMConfig(BaseModel):
    latent_shape: List[int] = [8, 16, 125]
    audio_length_in_s: float = 5.0


class MelLDM(DDPM):
    def __init__(self, cfg: MelLDMConfig, device, sample_rate: int, **kwargs) -> None:
        """
        vocoder: for postprocessing
        delta_h: for conditioning
        """
        super().__init__(**kwargs)
        self.cfg = cfg
        self.sample_rate = sample_rate
        mldm_wrapper = MusicLDMWrapper(
            device=device,
            collect_h_space=False,
            audio_length_in_s=self.cfg.audio_length_in_s,
            dtype=torch.float32,
        )
        self.mldm = mldm_wrapper

    def preprocess(
        self, x_start: Tensor, cond: Optional[Union[dict, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, None]:
        if x_start is not None:
            with torch.no_grad():
                latent = self.mldm.audios_to_latents(x_start.squeeze())
                latent: torch.Tensor = self.mldm.get_vae_scaling_factor()  * latent
                latent = latent.permute(0, 1, 3, 2)
        else:
            latent = None
        return latent, cond, None

    def postprocess(self, x: Tensor, additional_data_dict) -> AudioDiffusersOutput:
        x = x.permute(0, 1, 3, 2)
        latent_scaled = 1 / self.mldm.get_vae_scaling_factor() * x

        audio, mel_spectrogram, latents = self.mldm.decode_latent(
            latents=latent_scaled, audio_length_in_s=5.0
        )
        return AudioDiffusersOutput(
            x=audio,
            melgram=mel_spectrogram,
            raw_latent=latents,
            sample_rate=self.sample_rate,
        )

    def get_x_shape(self, cond) -> tuple:
        x_shape = (cond["class_label"].shape[0], *self.cfg.latent_shape)
        return x_shape

    def get_unconditional_condition(
        self,
        cond: Optional[Union[dict, Tensor]] = None,
        cond_shape: Optional[tuple] = None,
        condition_device: Optional[torch.device] = None,
    ) -> dict:
        return {"class_label": torch.zeros(cond_shape).to(condition_device)}
