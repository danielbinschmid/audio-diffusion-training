from experiments.ldm_moises_db_utils import (
    ExperimentConfig,
    setup_trainer,
    LoggingHooks,
)
from schmid_werkzeug import print_info


def train(cfg: ExperimentConfig):
    print_info(f"Trainig with config {cfg}")

    trainer = setup_trainer(cfg)
    logging_hooks = LoggingHooks(
        monitor_every_nth_epoch=cfg.monitor_every_nth_epoch,
        n_samples_inference=cfg.n_samples_inference,
        sample_rate=cfg.ds_cfg.sample_rate,
    )

    trainer.register_hook(
        hook_type="on_epoch_end", hook=logging_hooks.log_melgram_audio_hook
    )

    trainer.fit_ddpm(from_epoch=cfg.from_epoch)

    return trainer
