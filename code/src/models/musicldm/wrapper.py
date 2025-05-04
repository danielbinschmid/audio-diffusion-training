
from diffusers.pipelines.musicldm.pipeline_musicldm import MusicLDMPipeline
import torch
from typing import List, Optional, Tuple, Union
from .stft import TacotronSTFT
from abc import ABC
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class LDMWrapper(ABC):
    def __init__(self, manual_seed: Optional[int] = None) -> None:
        self.generator = (
            torch.Generator().manual_seed(manual_seed)
            if manual_seed is not None
            else None
        )


class MusicLDMWrapper(LDMWrapper):
    pipe: MusicLDMPipeline

    def __init__(
        self,
        device: str,
        manual_seed: Optional[int] = None,
        audio_length_in_s: float = 10,
        dtype=torch.float32,
    ) -> None:
        super().__init__(manual_seed=manual_seed)

        self.repo_id = "ucsd-reach/musicldm"
        self.dtype = dtype
        self.pipe = MusicLDMPipeline.from_pretrained(
            self.repo_id, torch_dtype=self.dtype
        )
        self.device = device
        self.pipe = self.pipe.to(device)
        self.audio_length = audio_length_in_s

        # waveform to melgram
        self.stft = TacotronSTFT().to(self.device)

    def mel_spectrogram_to_latent(
        self, 
        mel_spectrogram: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    ) -> torch.Tensor:
        mel_spectrogram = mel_spectrogram.to(self.pipe._execution_device)
        output_distr: DiagonalGaussianDistribution = self.pipe.vae.encode(x=mel_spectrogram).latent_dist
        sample = output_distr.sample(generator)
        return sample

    def audios_to_melgrams(self, audios: torch.Tensor) -> torch.Tensor:
        """
        audios: shape (bs, wavsamples, 2) for stereo or (bs, wavsamples,) for mono

        returns (bs, n_mel_scale, time)
        """
        audios_tmp = audios.clone()

        if len(audios_tmp.shape) == 3 and audios_tmp.shape[2] == 2:
            # stereo to mono
            audios_tmp = audios_tmp.mean(axis=2)
        if len(audios_tmp.shape) == 4:
            ValueError()
        assert len(audios_tmp.shape) == 2  # (bs, wavsamples)

        mel_output, _, _ = self.stft.mel_spectrogram(y=audios_tmp)

        return mel_output

    def audios_to_latents(self, audios: torch.Tensor):
        """
        audios: shape (bs, wavsamples, 2) for stereo or (bs, wavsamples,) for mono
        """

        # to melgram
        mel_grams = self.audios_to_melgrams(
            audios=audios
        )  # shape (bs, n_mel_scale, time)

        # "formatting"
        mel_grams = mel_grams.permute(0, 2, 1)  # shape (bs, time, n_mel_scale)
        mel_grams = mel_grams.unsqueeze(1)  # (bs, 1, time, n_mel_scale)
        mel_grams = mel_grams.to(self.dtype)

        # to latent
        latent = self.mel_spectrogram_to_latent(
            mel_spectrogram=mel_grams.to(self.dtype), generator=self.generator
        )
        return latent
    
    def get_vae_scaling_factor(self):
        return self.pipe.vae.config.scaling_factor

    def mel_spectrogram_to_waveform(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        mel_spectrogram = mel_spectrogram.to(self.pipe._execution_device)

        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.pipe.vocoder.forward(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.float()
        return waveform
    
    def decode_latent(self, latents: torch.Tensor, audio_length_in_s: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)

        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        mel_spectrogram = self.pipe.vae.decode(latents).sample
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :original_waveform_length]
        return audio, mel_spectrogram, latents


