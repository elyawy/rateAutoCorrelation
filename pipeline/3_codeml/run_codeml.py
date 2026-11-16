#!/usr/bin/env python3
"""Thin codeml runner that exposes the paper's modes: sim, real, core."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.codeml import create_control_file, run_codeml
import config


def _parse_args():
    parser = argparse.ArgumentParser(description="Run codeml in sim / real / core mode")
    parser.add_argument("--mode", choices=["sim", "real", "core"], default="sim")
    parser.add_argument("--seq", type=str, default=str(Path(config.DATA_DIR) / "alignment.phy"))
    parser.add_argument("--tree", type=str, default=str(Path(config.DATA_DIR) / "tree.nwk"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--run-dir", type=str, default=".")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    control_file = create_control_file(
        config.CODEML_TEMPLATE,
        run_dir,
        args.seq,
        args.tree,
        args.alpha,
    )
    print(f"mode={args.mode} seq={args.seq} tree={args.tree} control={control_file}")
    success, elapsed = run_codeml(control_file, run_dir, timeout=config.CODEML_TIMEOUT)
    print(f"success={success} elapsed={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
