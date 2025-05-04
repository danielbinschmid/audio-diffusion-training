import torch
from src.utils.audio_diffusers_output import AudioDiffusersOutput
from src.logger.wandb import WandbLogger
from src.models.mel_ldm import MelLDM


class LoggingHooks:
    def __init__(self, monitor_every_nth_epoch, n_samples_inference, sample_rate):
        self.monitor_every_nth_epoch = monitor_every_nth_epoch
        self.n_samples_inference = n_samples_inference
        self.sample_rate = sample_rate

    def log_melgram_audio_hook(
        self, epoch_idx, ddpm: MelLDM, logger: WandbLogger
    ) -> None:
        if epoch_idx % self.monitor_every_nth_epoch == 0:
            outp: AudioDiffusersOutput = ddpm.forward(
                cond={
                    "class_label": torch.tensor(
                        [[1] for i in range(self.n_samples_inference)]
                    )
                },
                stage="infer",
                is_cond_unpack=True,
            )
            audio_hat = outp.x
            mel_gram = outp.melgram
            audio_dict = {}
            spec_dict = {}
            for i in range(self.n_samples_inference):
                audio_dict[f"{i}"] = audio_hat[i].detach().cpu().numpy()
                spec_dict[f"{i}"] = mel_gram[i].detach().cpu().numpy().T

            logger.plot_wav(
                name=f"generated_waveform",
                audio_dict=audio_dict,
                sample_rate=self.sample_rate,
                global_step=epoch_idx,
            )
            logger.plot_spec(
                name="generated_melgram", spec_dict=spec_dict, global_step=epoch_idx,
            )
