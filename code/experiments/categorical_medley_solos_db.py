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
from src.datasets import (
    MedleySolosDBDataset,
    MedleySolosDBDatasetConfig,
    MedleySolosDBConstants,
)
from src.models.hifigan import HifiGAN, HifiGANConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from src.trainer import DiffusionTrainer, DiffusionTrainerConfig
from src.logger.wandb import WandbLogger, WandbLoggerConfig
from pydantic import BaseModel
from TorchJaekwon.Util.UtilAudio import UtilAudio
from schmid_werkzeug.pydantic import save_cfg, load_cfg
import argparse
from typing import Literal


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
    unconditional_prob: float


def get_cfg(
    checkpoints_base_path: str = "/path/to/data/checkpoints/melddpm_categorical_v1",
    logging_base_path: str = "/path/to/data/logging/medleysolosdb_melddpm_categorical_v1",
    mel_min_json: str = "/path/to/code/configs/mel_ddpm/mel_min.json",
    mel_max_json: str = "/path/to/code/configs/mel_ddpm/mel_max.json",
    vocoder_pretrained_weights: str = "/data/models/pretrained_opensource/musicldm_hifigan/hifigan-ckpt(only generator).ckpt",
    medley_ds_path: str = "/path/to/data/datasets/medleysolosdb",
) -> ExperimentConfig:
    # CFG -------------------------------------------------------------------------
    PROJECT_NAME = "melddpm_categorical"
    EXPERIMENT_NAME = "melddpm_categorical_v1"

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
    batch_size: int = 32
    num_epochs = 2000
    num_workers = 16
    monitor_every_nth_epoch = 10
    device: str = "cuda"
    lr = 0.0008
    lr_decay_step_size = 40
    unconditional_prob = 0.1
    n_categories = 8
    category_embedding_dim = 4
    n_res_blocks = 3
    conditioning_mechanism = "inp_channel"

    mel_min_cfg_path = mel_min_json
    mel_max_cfg_path = mel_max_json
    # --------------------------------------------------------------------------

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
        in_channel=1
        if conditioning_mechanism != "inp_channel"
        else 1 + category_embedding_dim,
        inner_channel=64,
        out_channel=1,
        res_blocks=n_res_blocks,
        attn_res=[8],
        conditioning_mechanism=conditioning_mechanism,
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
        n_categories=n_categories,
        embedding_dim=category_embedding_dim,
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
        unconditional_prob=unconditional_prob,
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
        unconditional_prob=cfg.unconditional_prob,
    )

    # datasets
    ds_train = MedleySolosDBDataset(config=cfg.ds_cfg.set_mode("training"))
    ds_test = MedleySolosDBDataset(config=cfg.ds_cfg.set_mode("test"))
    ds_validation = MedleySolosDBDataset(config=cfg.ds_cfg.set_mode("validation"))

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
                cond={
                    "class_label": torch.tensor(
                        [i for i in range(cfg.melddpm_cfg.n_categories + 1)]
                    ).to(device=cfg.trainer_cfg.device)
                },
                stage="infer",
                is_cond_unpack=True,
            )

            generated_waveforms = {}
            generated_melgrams = {}
            for categ_idx in range(cfg.melddpm_cfg.n_categories + 1):
                audio_hat = outp.x
                mel_gram = outp.melgram
                idx_to_label = MedleySolosDBConstants.IDX_TO_DESCR
                instrument_name = idx_to_label[categ_idx]
                generated_waveforms[instrument_name] = (
                    audio_hat[categ_idx, 0].detach().cpu().numpy()
                )
                generated_melgrams[instrument_name] = (
                    mel_gram[categ_idx, 0].detach().cpu().numpy()
                )
            logger.plot_wav(
                name="generated_waveform",
                audio_dict=generated_waveforms,
                sample_rate=cfg.melddpm_cfg.sample_rate,
                global_step=epoch_idx,
            )
            logger.plot_spec(
                name=f"generated_melgram",
                spec_dict=generated_melgrams,
                global_step=epoch_idx,
            )

    trainer.register_hook(hook_type="on_epoch_end", hook=log_melgram_audio_hook)

    trainer.fit_ddpm()


def infer(cfg: ExperimentConfig, ddpm_ckpt_path: str, n_samples_per_category: int):
    trainer = setup_trainer(cfg)
    trainer.load_ddpm_ckpt(ddpm_ckpt_path)
    trainer.ddpm = trainer.ddpm.to(cfg.trainer_cfg.device)

    cond_vector = [
        categ_idx
        for categ_idx in range(cfg.melddpm_cfg.n_categories + 1)
        for n in range(n_samples_per_category)
    ]
    outp: MelDDPMOutput = trainer.ddpm.forward(
        cond={"class_label": torch.tensor(cond_vector).to(cfg.trainer_cfg.device)},
        stage="infer",
        is_cond_unpack=True,
    )
    audios = outp.x

    categ_to_name = MedleySolosDBConstants.IDX_TO_DESCR
    for idx, cond_idx in enumerate(cond_vector):
        instr_name = categ_to_name[cond_idx]
        UtilAudio.write(
            audio_path=f"out/out_{idx}_{instr_name}.wav",
            audio=audios[idx],
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
        "--n_infer_per_categ",
        type=int,
        default=1,
        help="Number of samples to generate during inference/ testing for every category.",
    )
    args = parser.parse_args()

    exec_mode: Literal["train", "test", "generate_config"] = args.mode
    cfg_path: str = args.cfg_path
    ddpm_ckpt_path: str = args.ddpm_ckpt_path
    n_samples_per_category: int = args.n_infer_per_categ

    if exec_mode == "train":
        cfg = load_cfg(cfg_path, ExperimentConfig)
        train(cfg)
    elif exec_mode == "test":
        cfg = load_cfg(cfg_path, ExperimentConfig)
        infer(
            cfg,
            ddpm_ckpt_path=ddpm_ckpt_path,
            n_samples_per_category=n_samples_per_category,
        )
    elif exec_mode == "generate_config":
        cfg = get_cfg()
        save_cfg(cfg=cfg, yaml_path=cfg_path)
    else:
        raise ValueError("Wrong exec mode.")
