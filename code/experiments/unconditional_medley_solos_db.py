import os
import sys

sys.path.append(os.curdir)

import torch
import torch.nn as nn
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import (
    GuidedDiffusionUnet,
    GuidedDiffusionUnetConfig,
)
from src.models.MelDDPM import MelDDPM, MelDDPMOutput, MelDDPMConfig
from src.datasets import MedleySolosDBDataset, MedleySolosDBDatasetConfig
from src.models.hifigan import HifiGAN, HifiGANConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from src.trainer import DiffusionTrainer, DiffusionTrainerConfig
from src.logger.wandb import WandbLogger, WandbLoggerConfig
from pydantic import BaseModel
from schmid_werkzeug.pydantic import save_cfg, load_cfg
import argparse
from typing import Literal
from TorchJaekwon.Util.UtilAudio import UtilAudio


class ExperimentConfig(BaseModel):
    trainer_cfg: DiffusionTrainerConfig
    ds_cfg: MedleySolosDBDatasetConfig
    vocoder_cfg: HifiGANConfig
    vocoder_pretrained_weights: str
    unet_backbone_cfg: GuidedDiffusionUnetConfig
    melddpm_cfg: MelDDPMConfig
    wandb_cfg: WandbLoggerConfig
    lr: float
    lr_decay_step_size: int
    monitor_every_nth_epoch: int
    use_full_ds_for_training: bool = False  # whether to use val and test set for training as well


def get_cfg(
    checkpoints_base_path: str = "/path/to/data/checkpoints/melddpm",
    logging_base_path: str = "/path/to/data/logging/medleysolosdb_melddpm",
    mel_min_json: str = "/path/to/code/configs/medley_solos_db/mel_min.json",
    mel_max_json: str = "/path/to/code/configs/medley_solos_db/mel_max.json",
    vocoder_pretrained_weights: str = "/data/models/pretrained_opensource/musicldm_hifigan/hifigan-ckpt(only generator).ckpt",
    medley_ds_path: str = "/path/to/data/datasets/medleysolosdb",
) -> ExperimentConfig:
    # CONFIG -------------------------------------------------------------------
    PROJECT_NAME = "mel_ddpm"
    EXPERIMENT_NAME = "mel_ddpm_v1"

    CHECKPOINTS_PATH = os.path.join(checkpoints_base_path, f"{EXPERIMENT_NAME}")
    LOGGING_PATH = os.path.join(logging_base_path, f"medleysolosdb_{EXPERIMENT_NAME}")

    sample_rate: int = 16000
    duration_sec: float = 47552 / sample_rate
    nfft: int = 1024
    hop_size: int = 160  # nfft // 4
    mel_size: int = 64
    frequency_min: float = 0
    frequency_max: float = sample_rate // 2
    frame_size: int = int((sample_rate * duration_sec) // hop_size)
    batch_size: int = 16
    num_epochs = 2000
    num_workers = 16
    monitor_every_nth_epoch = 10
    device: str = "cuda"
    lr = 0.0008
    lr_decay_step_size = 40

    mel_min_cfg_path = mel_min_json
    mel_max_cfg_path = mel_max_json
    # --------------------------------------------------------------------------

    # CONFIG PREP --------------------------------------------------------------
    trainer_cfg = DiffusionTrainerConfig(
        device=device,
        num_epochs=num_epochs,
        num_workers_dataloader=num_workers,
        batch_size=batch_size,
        checkpoint_path=CHECKPOINTS_PATH,
    )
    ds_config = MedleySolosDBDatasetConfig(
        base_folder=medley_ds_path, sample_rate=sample_rate, mono=True
    )
    hifigan_config = HifiGANConfig(num_mels=mel_size, upsample_initial_channel=nfft)

    backbone_unet_cfg = GuidedDiffusionUnetConfig(
        image_size=None,
        in_channel=1,
        inner_channel=64,
        out_channel=1,
        res_blocks=2,
        attn_res=[8],
    )

    melddpm_cfg = MelDDPMConfig(
        nfft=nfft,
        hop_size=hop_size,
        sample_rate=sample_rate,
        mel_size=mel_size,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frame_size=frame_size,
        mel_min_cfg_path=mel_min_cfg_path,
        mel_max_cfg_path=mel_max_cfg_path,
    )

    wandb_cfg = WandbLoggerConfig(
        project_name=PROJECT_NAME, experiment_name=EXPERIMENT_NAME, log_dir=LOGGING_PATH
    )

    cfg = ExperimentConfig(
        trainer_cfg=trainer_cfg,
        ds_cfg=ds_config,
        vocoder_cfg=hifigan_config,
        unet_backbone_cfg=backbone_unet_cfg,
        melddpm_cfg=melddpm_cfg,
        vocoder_pretrained_weights=vocoder_pretrained_weights,
        wandb_cfg=wandb_cfg,
        lr=lr,
        lr_decay_step_size=lr_decay_step_size,
        monitor_every_nth_epoch=monitor_every_nth_epoch,
    )
    return cfg


def setup_trainer(cfg: ExperimentConfig) -> DiffusionTrainer:
    logger = WandbLogger(cfg=cfg.wandb_cfg)

    # diffusion backbone
    diffusion_denoiser: nn.Module = GuidedDiffusionUnet(cfg=cfg.unet_backbone_cfg)
    # vocoder
    vocoder = HifiGAN(h=cfg.vocoder_cfg)
    vocoder_state_dict = torch.load(cfg.vocoder_pretrained_weights, weights_only=True)
    vocoder.load_state_dict(state_dict=vocoder_state_dict["generator"])
    # full ddpm
    mel_ddpm: nn.Module = MelDDPM(
        cfg=cfg.melddpm_cfg,
        vocoder=vocoder,
        model=diffusion_denoiser,
        model_output_type="v_prediction",
        unconditional_prob=0,
    )

    # datasets
    ds_train = MedleySolosDBDataset(config=cfg.ds_cfg.set_mode("training"))
    ds_test = MedleySolosDBDataset(config=cfg.ds_cfg.set_mode("test"))
    ds_validation = MedleySolosDBDataset(config=cfg.ds_cfg.set_mode("validation"))

    if cfg.use_full_ds_for_training:
        ds_train = ds_train + ds_test + ds_validation

    # optim tools
    optimizer = AdamW(
        mel_ddpm.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.0
    )
    scheduler = StepLR(optimizer, step_size=cfg.lr_decay_step_size, gamma=0.5)

    # trainer
    trainer = DiffusionTrainer(
        trainer_cfg=cfg.trainer_cfg,
        logger=logger,
        train_ds=ds_train,
        scheduler=scheduler,
        ddpm=mel_ddpm,
    )
    return trainer


def train(cfg: ExperimentConfig):

    # load cfg and trainer
    trainer = setup_trainer(cfg)

    # set up hook
    def log_melgram_audio_hook(epoch_idx, ddpm: MelDDPM, logger: WandbLogger) -> None:
        if epoch_idx % cfg.monitor_every_nth_epoch == 0:
            outp: MelDDPMOutput = ddpm.forward(
                cond={"class_label": torch.tensor([[1]])},
                stage="infer",
                is_cond_unpack=True,
            )
            audio_hat = outp.x
            mel_gram = outp.melgram
            logger.plot_wav(
                name=f"generated_waveform",
                audio_dict={"": audio_hat[0, 0].detach().cpu().numpy()},
                sample_rate=cfg.melddpm_cfg.sample_rate,
                global_step=epoch_idx,
            )
            logger.plot_spec(
                name="generated_melgram",
                spec_dict={"": mel_gram[0, 0].detach().cpu().numpy()},
                global_step=epoch_idx,
            )

    trainer.register_hook(hook_type="on_epoch_end", hook=log_melgram_audio_hook)

    # run
    trainer.fit_ddpm()


def infer(cfg: ExperimentConfig, ddpm_ckpt_path: str, n_samples: int):
    trainer = setup_trainer(cfg)
    trainer.load_ddpm_ckpt(ddpm_ckpt_path)
    trainer.ddpm = trainer.ddpm.to(cfg.trainer_cfg.device)
    outp: MelDDPMOutput = trainer.ddpm.forward(
        cond={"class_label": torch.tensor([[1] for i in range(n_samples)])},
        stage="infer",
        is_cond_unpack=True,
    )
    audio_hat = outp.x
    for i in range(n_samples):
        UtilAudio.write(
            audio_path=f"out/out_{i}.wav",
            audio=audio_hat[i],
            sample_rate=cfg.melddpm_cfg.sample_rate,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument parser for experiment configurations."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "generate_config"],  # Use `choices` for validation
        help="Either 'train', 'test', or 'generate_config'.",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="cfg.yaml",
        help="yaml config path required for running train or test or destination where to save config to.",
    )
    parser.add_argument(
        "--ddpm_ckpt_path", type=str, default="model.pth", help="Needed for inference.",
    )
    parser.add_argument(
        "--n_inference",
        type=int,
        default=1,
        help="Number of samples to generate during inference.",
    )
    args = parser.parse_args()

    exec_mode: Literal["train", "test", "generate_config"] = args.mode
    cfg_path: str = args.cfg_path
    ddpm_ckpt_path: str = args.ddpm_ckpt_path
    n_inference: int = args.n_inference

    if exec_mode == "train":
        cfg = load_cfg(cfg_path, ExperimentConfig)
        train(cfg)
    elif exec_mode == "test":
        cfg = load_cfg(cfg_path, ExperimentConfig)
        infer(cfg, ddpm_ckpt_path=ddpm_ckpt_path, n_samples=n_inference)
    elif exec_mode == "generate_config":
        cfg = get_cfg()
        save_cfg(cfg=cfg, yaml_path=cfg_path)
    else:
        raise ValueError("Wrong exec mode.")
