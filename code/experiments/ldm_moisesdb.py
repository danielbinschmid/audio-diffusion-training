import sys
import os

sys.path.append(os.curdir)

from experiments.ldm_moises_db_utils import get_cfg, ExperimentConfig, train, infer
import argparse
from typing import Literal
from schmid_werkzeug.pydantic import load_cfg, save_cfg


def main():

    parser = argparse.ArgumentParser(
        description="Argument parser for experiment configurations."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate_config", "infer"],  # Use `choices` for validation
        help="Either 'train' or 'generate_config'.",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="cfg.yaml",
        help="yaml config path required for running train or test or destination where to save config to.",
    )

    args = parser.parse_args()

    exec_mode: Literal["train", "generate_config", "infer"] = args.mode
    cfg_path: str = args.cfg_path

    if exec_mode == "train":
        cfg: ExperimentConfig = load_cfg(cfg_path, ExperimentConfig)
        train(cfg)
    elif exec_mode == "generate_config":
        cfg = get_cfg()
        save_cfg(cfg=cfg, yaml_path=cfg_path)
    elif exec_mode == "infer":
        cfg: ExperimentConfig = load_cfg(cfg_path, ExperimentConfig)
        infer(cfg)
    else:
        raise ValueError("Wrong exec mode.")


if __name__ == "__main__":
    main()
