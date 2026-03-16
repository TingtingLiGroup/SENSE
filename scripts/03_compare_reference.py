#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microdroplet_ml.config import DEFAULT_CONFIG_PATH, load_config


def compare_tables(generated: Path, reference: Path, metric_tolerance: float) -> tuple[bool, str]:
    g = pd.read_csv(generated)
    r = pd.read_csv(reference)

    if list(g.columns) != list(r.columns):
        return False, "Column mismatch"
    if len(g) != len(r):
        return False, "Row-count mismatch"

    if "value" in g.columns and "metric" in g.columns:
        max_delta = 0.0
        groups = sorted(set(zip(g["model"], g["metric"])))
        for model_name, metric_name in groups:
            g_values = sorted(
                g[(g["model"] == model_name) & (g["metric"] == metric_name)]["value"].tolist()
            )
            r_values = sorted(
                r[(r["model"] == model_name) & (r["metric"] == metric_name)]["value"].tolist()
            )
            if len(g_values) != len(r_values):
                return False, f"Metric count mismatch for {model_name}/{metric_name}"
            deltas = [abs(a - b) for a, b in zip(g_values, r_values, strict=True)]
            if deltas:
                max_delta = max(max_delta, max(deltas))
        if max_delta > metric_tolerance:
            return False, f"Metric delta too large: {max_delta:.6f} > {metric_tolerance:.6f}"
        return True, f"OK (max metric delta={max_delta:.6f})"

    return True, "OK"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated outputs with reference outputs")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--metric-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config used for the generated outputs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = config["outputs"]
    checks = []

    if config.get("performance_group") is not None:
        checks.append(outputs["model_performance"])

    for group in config["prediction_groups"]:
        checks.append(outputs[group["train_prediction_output"]])

    ok = True
    for file_name in checks:
        generated = Path(args.generated_dir) / file_name
        reference = Path(args.reference_dir) / file_name
        result, msg = compare_tables(generated, reference, args.metric_tolerance)
        print(f"{file_name}: {msg}")
        ok = ok and result

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
