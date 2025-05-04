from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPMOutput
from TorchJaekwon.Util.UtilAudio import UtilAudio
import torch
from typing import Optional, Union
import os


class DiffusersBaseOutput(DDPMOutput):
    x: torch.Tensor
    raw_latent: Optional[torch.Tensor]

    def __init__(self, x, raw_latent):
        super().__init__(x=x)
        self.raw_latent = raw_latent

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)

        torch.save(self.x, os.path.join(folder, "x.pth"))
        if self.raw_latent is not None:
            torch.save(self.raw_latent, os.path.join(folder, "raw_latent.pth"))

    @classmethod
    def load(cls, folder: str) -> "DiffusersBaseOutput":
        x = torch.load(
            os.path.join(folder, "x.pth"), map_location="cpu", weights_only=True
        )

        if os.path.exists(os.path.join(folder, "raw_latent.pth")):
            raw_latent = torch.load(
                os.path.join(folder, "raw_latent.pth"),
                map_location="cpu",
                weights_only=True,
            )
        else:
            raw_latent = None

        return DiffusersBaseOutput(x=x, raw_latent=raw_latent)


class AudioDiffusersOutput(DiffusersBaseOutput):
    x: torch.Tensor
    """Audio"""
    melgram: torch.Tensor
    raw_latent: Optional[torch.Tensor]
    sample_rate: int

    def __init__(self, x, melgram, raw_latent, sample_rate):
        super().__init__(x, raw_latent)
        self.melgram = melgram
        self.sample_rate = sample_rate

    def save(self, folder: str) -> None:
        super().save(folder)
        torch.save(self.melgram, os.path.join(folder, "melgram.pth"))

        UtilAudio.write(
            audio_path=os.path.join(folder, "audio.wav"),
            audio=self.x.detach().cpu()[0],
            sample_rate=self.sample_rate,
        )

    @classmethod
    def load(cls, folder: str) -> "AudioDiffusersOutput":
        base = DiffusersBaseOutput.load(folder)
        melgram = torch.load(
            os.path.join(folder, "melgram.pth"), map_location="cpu", weights_only=True
        )
        audio, sr = UtilAudio.read(
            audio_path=os.path.join(folder, "audio.wav"), module_name="torchaudio"
        )
        return AudioDiffusersOutput(
            x=base.x, melgram=melgram, raw_latent=base.raw_latent, sample_rate=sr
        )


DiffusersOutput = Union[DiffusersBaseOutput, AudioDiffusersOutput]
