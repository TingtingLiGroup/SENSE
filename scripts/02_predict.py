#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microdroplet_ml.config import DEFAULT_CONFIG_PATH, load_config
from microdroplet_ml.pipeline import run_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the released model bundles."
    )
    parser.add_argument("--input", required=True, help="Input .xlsx/.csv with sequences")
    parser.add_argument("--models", required=True, help="Directory containing model .pkl files")
    parser.add_argument("--out", required=True, help="Output .xlsx/.csv path")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_path = run_predict(
        input_path=args.input,
        models_dir=args.models,
        out_path=args.out,
        config=config,
    )
    print(f"Prediction completed: {out_path}")


if __name__ == "__main__":
    main()
