from typing import Optional, Tuple, Union, Literal
from torch import Tensor

import torch

from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM
from TorchJaekwon.Util.UtilAudioMelSpec import UtilAudioMelSpec
from TorchJaekwon.Util.UtilTorch import UtilTorch
from pydantic import BaseModel
import torch.nn as nn
from schmid_werkzeug import print_info
from schmid_werkzeug.torch_utils import load_json_list_as_tensor

from src.utils.audio_diffusers_output import AudioDiffusersOutput
from typing import Optional
import torch
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM

class MelDDPMOutput(AudioDiffusersOutput):
    def __init__(
        self,
        audio: Tensor,
        melgram: torch.Tensor,
        raw_diffusion_output: torch.Tensor,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(
            x=audio,
            melgram=melgram,
            raw_latent=raw_diffusion_output,
            sample_rate=sample_rate,
        )

    def to_cpu(self) -> None:
        self.melgram = self.melgram.to("cpu")
        self.raw_diffusion_output = self.raw_diffusion_output.to("cpu")
        super().to_cpu()
        self.raw_latent.to("cpu")


class MelDDPMConfig(BaseModel):
    nfft: int
    hop_size: int
    sample_rate: int
    mel_size: int
    frequency_min: float
    frequency_max: float
    frame_size: int
    mel_min_cfg_path: str
    mel_max_cfg_path: str
    n_categories: int = 0
    cls_embedding_dim: int = 4


class MelDDPM(DDPM):
    def __init__(self, cfg: MelDDPMConfig, vocoder, **kwargs) -> None:
        """
        vocoder: for postprocessing
        delta_h: for conditioning
        """
        super().__init__(**kwargs)
        self.cfg = cfg
        self.mel_size: int = cfg.mel_size
        self.frame_size: int = cfg.frame_size
        self.mel_spec_util = UtilAudioMelSpec(
            cfg.nfft,
            cfg.hop_size,
            cfg.sample_rate,
            cfg.mel_size,
            cfg.frequency_min,
            cfg.frequency_max,
        )
        mel_max: Tensor = load_json_list_as_tensor(cfg.mel_max_cfg_path).view(
            1, 1, -1, 1
        )
        self.mel_max: Tensor = UtilTorch.register_buffer(
            model=self, variable_name="mel_max", value=mel_max
        )

        mel_min: Tensor = load_json_list_as_tensor(cfg.mel_min_cfg_path).view(
            1, 1, -1, 1
        )
        self.mel_min: Tensor = UtilTorch.register_buffer(
            model=self, variable_name="mel_min", value=mel_min
        )

        self.vocoder = (
            vocoder
            if vocoder is not None
            else lambda x: torch.rand(x.shape[0], 1, x.shape[-1] * cfg.hop_size)
        )

        self.is_categorical = cfg.n_categories > 0
        if self.is_categorical:
            print_info(
                f"Instantiated categorical MelDDPM with {cfg.n_categories} categories"
            )
            self.category_embeddings = nn.Embedding(
                num_embeddings=cfg.n_categories
                + 1,  # one more category for unconditional category
                embedding_dim=cfg.cls_embedding_dim,
            )
        else:
            self.category_embeddings = None

    def preprocess(
        self, x_start: Tensor, cond: Optional[Union[dict, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, None]:
        if x_start is not None:
            mel_spec: Tensor = self.mel_spec_util.get_hifigan_mel_spec(x_start)
            mel_spec = self.normalize_mel(mel_spec)
        else:
            mel_spec = None

        if self.is_categorical:
            # assert existence
            assert cond is not None
            assert cond["class_label"] is not None

            # get learnable embedding
            class_label = cond["class_label"].clone().long().squeeze()
            cond["class_label"] = self.category_embeddings.forward(
                input=class_label
            ).squeeze()  # of shape (batch_size, cls_embedding_dim)
        additional_data_dict = None
        return mel_spec, cond, additional_data_dict

    def normalize_mel(self, mel_spec: Tensor) -> Tensor:
        if mel_spec.device != self.mel_min.device:
            mel_spec = mel_spec.to(self.mel_min.device)
        return (mel_spec - self.mel_min) / (self.mel_max - self.mel_min) * 2 - 1

    def denormalize_mel(self, mel_spec: Tensor) -> Tensor:
        return (mel_spec + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min

    def postprocess(
        self, x: Tensor, additional_data_dict
    ) -> Tensor:  # [batch, 1, mel, time]
        mel_spec: Tensor = self.denormalize_mel(x)
        pred_audio: Tensor = self.vocoder(mel_spec.squeeze(1))
        return MelDDPMOutput(audio=pred_audio, melgram=mel_spec, raw_diffusion_output=x)

    def get_x_shape(self, cond) -> tuple:
        x_shape = (cond["class_label"].shape[0], 1, self.mel_size, self.frame_size)
        return x_shape

    def get_unconditional_condition(
        self,
        cond: Optional[Union[dict, Tensor]] = None,
        cond_shape: Optional[tuple] = None,
        condition_device: Optional[torch.device] = None,
    ) -> dict:
        if self.is_categorical:
            batch_size = cond["class_label"].shape[0]
            unc_idx = torch.tensor(
                [self.cfg.n_categories for _ in range(batch_size)], dtype=torch.long
            ).to(condition_device)
            emb = self.category_embeddings.forward(unc_idx)

            return {"class_label": emb}
        else:
            return {"class_label": torch.zeros(cond_shape).to(condition_device)}
