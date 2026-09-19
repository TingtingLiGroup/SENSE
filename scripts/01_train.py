#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microdroplet_ml.config import DEFAULT_CONFIG_PATH, load_config
from microdroplet_ml.pipeline import run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the associative-framework and solubility models."
    )
    parser.add_argument("--input", required=True, help="Input .xlsx/.csv file")
    parser.add_argument("--out-dir", required=True, help="Directory for model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional head(N) subset for smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    written = run_train(
        input_path=args.input,
        out_dir=args.out_dir,
        config=config,
        seed=args.seed,
        max_rows=args.max_rows,
    )

    print("Training completed. Outputs:")
    for name, path in sorted(written.items()):
        rel = Path(path)
        print(f"- {name}: {rel}")


if __name__ == "__main__":
    main()
