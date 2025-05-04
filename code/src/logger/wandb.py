import wandb
from typing import Dict, Optional, Literal
from numpy import ndarray
from TorchJaekwon.Util.UtilAudioSTFT import UtilAudioSTFT
import os
from pydantic import BaseModel
from src.vis.plots import array_to_figure
from src.models.MelDDPM import MelDDPMOutput, MelDDPM
from typing import List


class WandbLoggerConfig(BaseModel):
    project_name: str
    experiment_name: str
    log_dir: str
    mode: Literal["online", "offline", "disabled"] = "online"

    def set_mode(
        self, mode: Literal["online", "offline", "disabled"]
    ) -> "WandbLoggerConfig":
        copy_ = self.model_copy()
        copy_.mode = mode
        return copy_


class WandbLogger:
    def __init__(self, cfg: WandbLoggerConfig) -> None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        wandb.init(
            project=cfg.project_name,
            name=cfg.experiment_name,
            dir=cfg.log_dir,
            mode=cfg.mode,
        )
        self.log_dir = cfg.log_dir

    def log_loss(self, loss: float) -> None:
        wandb.log({"loss": loss})

    def log_custom_float(self, custom_key: str, val: float) -> None:
        wandb.log({custom_key: val})

    def log_epoch_avg_loss(self, epoch_loss: float, lr: Optional[float] = None) -> None:
        log_dict = {"epoch_loss": epoch_loss}
        if lr is not None:
            log_dict["lr"] = lr
        wandb.log(log_dict)

    def plot_wav(
        self,
        name: str,  # test case name, you could make structure by using /. ex) 'audio/test_set_1'
        audio_dict: Dict[str, ndarray],  # {'audio name': 1d audio array},
        sample_rate: int,
        global_step: int,
    ) -> None:

        wandb_audio_list = list()
        for audio_name in audio_dict:
            wandb_audio_list.append(
                wandb.Audio(
                    audio_dict[audio_name], caption=audio_name, sample_rate=sample_rate
                )
            )
        wandb.log({name: wandb_audio_list})

    def plot_lines(self, name: str, img_dict: Dict[str, ndarray]):
        wandb_img_list = []
        for img_name, img_array in img_dict.items():
            save_path = os.path.join(self.log_dir, f"temp_img_{img_name}.png")
            array_to_figure(
                img_array, save_path=save_path,
            )
            wandb_img_list.append(wandb.Image(save_path, caption=img_name))

        wandb.log({name: wandb_img_list})

    def plot_spec(
        self,
        name: str,  # test case name, you could make structure by using /. ex) 'mel/test_set_1'
        spec_dict: Dict[str, ndarray],  # {'name': 2d array},
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        transposed=False,
        global_step=0,
    ):
        wandb_mel_list = list()
        for audio_name in spec_dict:
            UtilAudioSTFT.spec_to_figure(
                spec_dict[audio_name],
                vmin=vmin,
                vmax=vmax,
                transposed=transposed,
                save_path=f"""{self.log_dir}/temp_img_{audio_name}.png""",
            )
            wandb_mel_list.append(
                wandb.Image(
                    f"""{self.log_dir}/temp_img_{audio_name}.png""", caption=audio_name
                )
            )
        wandb.log({name: wandb_mel_list})

    def finish(self) -> None:
        wandb.finish()

    def plot_melddpm_outputs(
        self, ddpm_outputs: List[MelDDPMOutput], ddpm: MelDDPM, id: str = ""
    ) -> None:
        audio_dict = {}
        spec_dict = {}

        for batch_idx, ddpm_output in enumerate(ddpm_outputs):
            n_samples = ddpm_output.x.shape[0]

            for sample_idx in range(n_samples):
                audio_dict[f"{batch_idx}_{sample_idx}"] = (
                    ddpm_output.x[sample_idx, 0].detach().cpu().numpy()
                )
                spec_dict[f"{batch_idx}_{sample_idx}"] = (
                    ddpm_output.melgram[sample_idx, 0].detach().cpu().numpy()
                )

        self.plot_wav(
            name=f"{id}_generated_waveform",
            audio_dict=audio_dict,
            sample_rate=ddpm.cfg.sample_rate,
            global_step=0,
        )
        self.plot_spec(
            name=f"{id}_generated_melgram", spec_dict=spec_dict, global_step=0,
        )
