from pydantic import BaseModel
from src.datasets.moises_db_monophonic import (
    MoisesDBMonophonicConfig,
    MoisesDBConstants,
)
from src.trainer.trainer import DiffusionTrainerConfig
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import (
    GuidedDiffusionUnetConfig,
)
from src.models.mel_ldm import MelLDMConfig
from src.logger.wandb import WandbLoggerConfig
from src.models.embedding_sampler.mert import MertConfig


class LinearConfig(BaseModel):
    in_feats: int = 1024
    out_feats: int = 1024


class OptimiserConfig(BaseModel):
    lr: float = 1e-4
    step_size_lr_scheduler: int = 10


class InferenceConfig(BaseModel):
    n_samples: int = 160
    max_bs: int = 32
    n_denoising_steps: int = 1000


class ExperimentConfig(BaseModel):
    ds_cfg: MoisesDBMonophonicConfig
    mert_cfg: MertConfig
    trainer_cfg: DiffusionTrainerConfig
    unet_cfg: GuidedDiffusionUnetConfig
    mel_ldm_cfg: MelLDMConfig
    logger_cfg: WandbLoggerConfig
    do_repa: bool
    mlp_cfg: LinearConfig
    device: str
    n_samples_inference: int
    monitor_every_nth_epoch: int
    optimiser_cfg: OptimiserConfig
    ldm_ckpt: str | None = None
    from_epoch: int = 0
    inference_cfg: InferenceConfig = InferenceConfig()
    load_test_ds: bool = False


def get_cfg() -> ExperimentConfig:
    excluded_labels = [
        MoisesDBConstants.GUITAR_STEM_SOURCE_LABEL_TO_IDX["distorted_electric_guitar"]
    ]

    device: str = "cuda:1"

    ds_cfg = MoisesDBMonophonicConfig(
        ds_path="/home/danielbinschmid/melddpm/data/datasets/moisesdb/moisesdb",
        sample_rate=16000,
        mono=True,
        window_length_in_s=5,
        step_length_in_s=1,
        stem_mode="Guitar",
        recompute_metadata=False,
        train_val_test_split_seed=0,
        train_val_test_split_track_level=(0.8, 0.0, 0.2),
        precomputed_test_metadata_path="/home/danielbinschmid/melddpm/data/datasets/moisesdb/monophonic_guitar/v1/preprocessed_extracted_Guitar_monophonic_test_metadata.csv",
        precomputed_train_metadata_path="/home/danielbinschmid/melddpm/data/datasets/moisesdb/monophonic_guitar/v1/preprocessed_extracted_Guitar_monophonic_train_metadata.csv",
        precomputed_val_metadata_path="/home/danielbinschmid/melddpm/data/datasets/moisesdb/monophonic_guitar/v1/preprocessed_extracted_Guitar_monophonic_val_metadata.csv",
        empty_waveform_thresshold=1e-2,
        exclude_label=excluded_labels,
    )

    mert_cfg = MertConfig(embedding_idx=12)

    trainer_cfg = DiffusionTrainerConfig(
        device=device,
        num_epochs=2000,
        num_workers_dataloader=16,
        batch_size=16,
        checkpoint_path="/home/danielbinschmid/melddpm/data3/ckpt",
        repa_gamma=0.1,
    )

    unet_cfg = GuidedDiffusionUnetConfig(
        image_size=None,
        in_channel=8,
        inner_channel=64,
        out_channel=8,
        res_blocks=2,
        attn_res=[8],
        dropout=0,
        channel_mults=[1, 2, 4, 8, 16],
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_new_attention_order=False,
    )

    ldm_cfg = MelLDMConfig()

    logger_cfg = WandbLoggerConfig(
        project_name="repa",
        experiment_name="ldm_filtered_data",
        log_dir="/home/danielbinschmid/melddpm/data3/logging/wandb",
        mode="online",
    )

    do_repa = True

    mlp_cfg = LinearConfig(in_feats=1024, out_feats=1024)

    optimiser_cfg = OptimiserConfig(lr=1e-4, step_size_lr_scheduler=10)

    n_samples_inference = 10

    monitor_every_nth_epoch = 1

    exp_cfg = ExperimentConfig(
        ds_cfg=ds_cfg,
        mert_cfg=mert_cfg,
        trainer_cfg=trainer_cfg,
        unet_cfg=unet_cfg,
        mel_ldm_cfg=ldm_cfg,
        logger_cfg=logger_cfg,
        do_repa=do_repa,
        mlp_cfg=mlp_cfg,
        device=device,
        n_samples_inference=n_samples_inference,
        monitor_every_nth_epoch=monitor_every_nth_epoch,
        optimiser_cfg=optimiser_cfg,
    )

    return exp_cfg
