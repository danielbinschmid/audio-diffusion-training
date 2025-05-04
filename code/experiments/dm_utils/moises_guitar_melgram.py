from schmid_werkzeug.pydantic import DynBaseModel, load_cfg, load_cfg_dynamic
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import (
    GuidedDiffusionUnetConfig,
)
from src.models.MelDDPM import MelDDPM, MelDDPMConfig
from src.models.hifigan import HifiGANConfig
from .shared_utils import get_melddpm
from .shared_utils import InferenceCfgUncond, infer_uncond_melddpm


class MoisesGuitarMelCfg(DynBaseModel):
    vocoder_cfg: str = "./vocoder.yaml"
    vocoder_weights: str = "/home/danielbinschmid/melddpm/data/models/pretrained_opensource/musicldm_hifigan/hifigan-ckpt(only generator).ckpt"
    unet_cfg: str = "./unet.yaml"
    weights: str = "/home/danielbinschmid/melddpm/data/models/good_checkpoints/melddpm_v1/2024-12-03.pth"
    melddpm_cfg: str = "./melddpm.yaml"
    original_base_path: str = "./"

    def get_melddpm(self, device="cuda") -> MelDDPM:
        m = lambda fp: self.map(fp)

        vocoder_cfg = load_cfg(m(self.vocoder_cfg), HifiGANConfig)
        unet_cfg = load_cfg(m(self.unet_cfg), GuidedDiffusionUnetConfig)
        melddpm_cfg = load_cfg(m(self.melddpm_cfg), MelDDPMConfig)

        melddpm = get_melddpm(
            ddpm_cfg=melddpm_cfg,
            vocoder_cfg=vocoder_cfg,
            vocoder_weights=self.vocoder_weights,
            unet_backbone_cfg=unet_cfg,
            device=device,
            mel_ddpm_weights=self.weights,
        )
        return melddpm


def inference(inference_cfg: InferenceCfgUncond):
    cfg: MoisesGuitarMelCfg = load_cfg_dynamic(
        inference_cfg.pipeline_cfg, MoisesGuitarMelCfg
    )
    assert cfg is not None
    ddpm = cfg.get_melddpm(inference_cfg.device)
    infer_uncond_melddpm(
        ddpm,
        inference_cfg.n_samples,
        out_folder=inference_cfg.out,
        max_bs=inference_cfg.max_bs,
        num_steps=inference_cfg.num_steps,
    )
