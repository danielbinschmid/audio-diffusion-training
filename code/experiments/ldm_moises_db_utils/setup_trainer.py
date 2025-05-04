from .config import ExperimentConfig
from src.trainer.trainer import DiffusionTrainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from src.logger import WandbLogger
from src.transforms import RandomPitchShift
from src.datasets.moises_db_monophonic import MoisesDBMonophonicDataset
from src.models.mel_ldm import MelLDM
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import GuidedDiffusionUnet
import torch
from src.models.embedding_sampler.mert import Mert
from schmid_werkzeug import print_info


def setup_trainer(cfg: ExperimentConfig) -> DiffusionTrainer:

    logger = WandbLogger(cfg=cfg.logger_cfg)

    augmentations = [
        RandomPitchShift(
            sample_rate=cfg.ds_cfg.sample_rate,
            min_n_steps=-3,
            max_n_steps=3,
            bins_per_octave=12,
            p=0.15,
        )
    ]
    ds = MoisesDBMonophonicDataset(cfg=cfg.ds_cfg, transforms=augmentations)
    ds.set_mode("train")
    unet = GuidedDiffusionUnet(cfg.unet_cfg).to(cfg.device)

    num_trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters: {num_trainable_params / 1000000} million"
    )
    ldm = MelLDM(
        cfg=cfg.mel_ldm_cfg,
        device=cfg.device,
        sample_rate=16000,
        model=unet,
        model_output_type="v_prediction",
        unconditional_prob=0,
    )
    ldm = ldm.to(cfg.device)

    linear = torch.nn.Linear(
        in_features=cfg.mlp_cfg.in_feats,
        out_features=cfg.mlp_cfg.out_feats,
        device=cfg.device,
    )

    mert = Mert(device=cfg.device, cfg=cfg.mert_cfg, sample_rate=cfg.ds_cfg.sample_rate)
    emb_sample = mert.sample_time_expanded

    if cfg.do_repa:
        params = list(unet.parameters()) + list(linear.parameters())
    else:
        params = unet.parameters()

    optimizer = AdamW(
        params, lr=cfg.optimiser_cfg.lr, betas=(0.9, 0.98), weight_decay=0.0
    )
    scheduler = StepLR(
        optimizer, step_size=cfg.optimiser_cfg.step_size_lr_scheduler, gamma=0.5
    )

    if cfg.load_test_ds:
        test_ds = MoisesDBMonophonicDataset(cfg=cfg.ds_cfg)
        test_ds.set_mode("test")
    else:
        test_ds = None

    trainer = DiffusionTrainer(
        trainer_cfg=cfg.trainer_cfg,
        logger=logger,
        train_ds=ds,
        test_ds=test_ds,
        scheduler=scheduler,
        ddpm=ldm,
        repa_embedding_sampler=emb_sample if cfg.do_repa else None,
        repa_align=linear,
    )

    if cfg.ldm_ckpt is not None:
        print_info(f"Loading checkpoint {cfg.ldm_ckpt}")
        trainer.load_ddpm_ckpt(cfg.ldm_ckpt, device=cfg.device)

    return trainer
