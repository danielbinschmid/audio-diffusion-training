from experiments.ldm_moises_db_utils import (
    ExperimentConfig,
    setup_trainer,
)
import torch
from TorchJaekwon.Util.UtilAudio import UtilAudio
import os
from src.utils.audio_diffusers_output import AudioDiffusersOutput
from src.utils.noise_sampling import sample_noise
from src.sampler import Sampler, SamplerConfig, DiffusersSchedulerConfig
from schmid_werkzeug.ml_workflow import get_sample_batches


def infer(cfg: ExperimentConfig):
    trainer = setup_trainer(cfg)

    batch_sizes = get_sample_batches(
        n_samples=cfg.inference_cfg.n_samples, max_bs=cfg.inference_cfg.max_bs
    )

    for batch_idx, batch_size in enumerate(batch_sizes):

        x_start = sample_noise(
            mode="random",
            ddpm=trainer.ddpm,
            cond=trainer.ddpm.get_unconditional_condition(
                cond_shape=(batch_size,), condition_device=trainer.config.device,
            ),
            seed=batch_idx,
        )
        sampler_cfg = SamplerConfig(
            type="dpm",
            diffusers_scheduler_cfg=DiffusersSchedulerConfig(
                num_steps=cfg.inference_cfg.n_denoising_steps
            ),
        )
        sampler = Sampler(cfg=sampler_cfg)
        outp: AudioDiffusersOutput = sampler.sample(
            ddpm=trainer.ddpm,
            cond=torch.tensor([[1] for i in range(batch_size)]),
            x_start=x_start.to(cfg.trainer_cfg.device),
        )

        audio_hat = outp.x
        for i in range(batch_size):
            fpath = os.path.join(
                "out",
                f"out_batch-idx-{batch_idx}_sample-{i}_seed-{batch_idx}_bs-{batch_size}.wav",
            )
            UtilAudio.write(
                audio_path=fpath, audio=audio_hat[i], sample_rate=outp.sample_rate,
            )
