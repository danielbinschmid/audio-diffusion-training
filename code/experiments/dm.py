import experiments.dm_utils.uncategorical_medley as unc_medl
import experiments.dm_utils.cond_medley as cond_medl
#import experiments.dm_utils.moises_guitar_melgram as moises_mel
#import experiments.dm_utils.moises_guitar_ldm as moises_ldm
from schmid_werkzeug.pydantic import load_cfg
import argparse


def medley_command(args):
    mode = args.mode
    inference_cfg_path = args.inference_cfg

    if mode == "cond":
        inference_cfg: cond_medl.InferenceCfg = load_cfg(
            inference_cfg_path, cond_medl.InferenceCfg
        )
        cond_medl.inference(inference_cfg)
    elif mode == "uncond":
        inference_cfg: unc_medl.InferenceCfg = load_cfg(
            inference_cfg_path, unc_medl.InferenceCfg
        )
        unc_medl.inference(inference_cfg)
    else:
        raise NotImplementedError()


def moises_command(args):
    mode = args.mode
    inference_cfg_path = args.inference_cfg
    raise ValueError(f"Invalid mode {mode}")
    if mode == "mel":
        inference_cfg: moises_mel.InferenceCfgUncond = load_cfg(
            inference_cfg_path, moises_mel.InferenceCfgUncond
        )
        moises_mel.inference(inference_cfg)
    elif mode == "ldm":
        inference_cfg: moises_ldm.InferenceCfgUncond = load_cfg(
            inference_cfg_path, moises_ldm.InferenceCfgUncond
        )
        moises_ldm.inference(inference_cfg)
    else:
        raise ValueError(f"Invalid mode {mode}")


def main():
    parser = argparse.ArgumentParser(description="My command line tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: medley
    medley_parser = subparsers.add_parser("medley", help="")
    medley_parser.add_argument(
        "mode",
        choices=["cond", "uncond"],
        help="run conditional or unconditional model",
    )
    medley_parser.add_argument("inference_cfg", help="inference_cfg")
    medley_parser.set_defaults(func=medley_command)

    # Subcommand: moises
    moises_parser = subparsers.add_parser("moises", help="")
    moises_parser.add_argument(
        "mode", choices=["mel", "ldm"], help="run mel or ldm model",
    )
    moises_parser.add_argument("inference_cfg", help="inference_cfg")
    moises_parser.set_defaults(func=moises_command)

    # exec.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
