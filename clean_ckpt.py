"""
Clean a Lightning checkpoint for inference-only use.

Removes optimizer states, lr_scheduler states, and other training-only data,
keeping only the model state_dict and embedded config (cfg).

If the checkpoint does not already contain a 'cfg' key, you can optionally
supply --cfg_path to embed one.

Usage:
    python clean_ckpt.py --ckpt_path /path/to/last.ckpt
    python clean_ckpt.py --ckpt_path /path/to/last.ckpt --output_path /path/to/clean.ckpt
    python clean_ckpt.py --ckpt_path /path/to/last.ckpt --cfg_path /path/to/config.yaml
"""
import argparse
import os

import torch
from omegaconf import OmegaConf


def clean_ckpt(ckpt_path: str, output_path: str = None, cfg_path: str = None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" not in ckpt:
        raise KeyError(f"'state_dict' not found in checkpoint: {ckpt_path}")

    clean = {"state_dict": ckpt["state_dict"]}

    # Embed config
    if "cfg" in ckpt:
        clean["cfg"] = ckpt["cfg"]
        print(f"[clean_ckpt] cfg found in checkpoint, keeping it.")
    elif cfg_path is not None:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg = OmegaConf.load(cfg_path)
        clean["cfg"] = OmegaConf.to_container(cfg, resolve=True)
        print(f"[clean_ckpt] cfg embedded from: {cfg_path}")
    else:
        print("[clean_ckpt] WARNING: no cfg found in checkpoint and no --cfg_path provided. "
              "The cleaned checkpoint will not contain config.")

    if output_path is None:
        base, ext = os.path.splitext(ckpt_path)
        output_path = f"{base}_clean{ext}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(clean, output_path)

    orig_size = os.path.getsize(ckpt_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[clean_ckpt] original:  {ckpt_path} ({orig_size:.1f} MB)")
    print(f"[clean_ckpt] cleaned:   {output_path} ({new_size:.1f} MB)")
    print(f"[clean_ckpt] size reduction: {orig_size - new_size:.1f} MB ({(1 - new_size / orig_size) * 100:.1f}%)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean checkpoint for inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the Lightning .ckpt file")
    parser.add_argument("--output_path", type=str, default=None, help="Output path (default: <name>_clean.ckpt)")
    parser.add_argument("--cfg_path", type=str, default=None,
                        help="Path to config .yaml to embed (only needed if ckpt has no cfg)")
    args = parser.parse_args()

    clean_ckpt(args.ckpt_path, args.output_path, args.cfg_path)
