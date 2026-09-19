#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microdroplet_ml.config import DEFAULT_CONFIG_PATH, load_config


def compare_tables(
    generated: Path,
    reference: Path,
    value_tolerance: float,
) -> tuple[bool, str]:
    g = pd.read_csv(generated)
    r = pd.read_csv(reference)

    if "sequence_id" in g.columns:
        g = g.sort_values("sequence_id").reset_index(drop=True)
        r = r.sort_values("sequence_id").reset_index(drop=True)
    elif {"model", "metric", "fold"}.issubset(g.columns):
        order = ["model", "metric", "fold"]
        g = g.sort_values(order).reset_index(drop=True)
        r = r.sort_values(order).reset_index(drop=True)

    if list(g.columns) != list(r.columns):
        return False, "Column mismatch"
    if len(g) != len(r):
        return False, "Row-count mismatch"

    numeric_columns = [column for column in g.columns if pd.api.types.is_numeric_dtype(g[column])]
    text_columns = [column for column in g.columns if column not in numeric_columns]

    for column in text_columns:
        if not g[column].fillna("").equals(r[column].fillna("")):
            return False, f"Value mismatch in text column: {column}"

    max_delta = 0.0
    for column in numeric_columns:
        generated_values = g[column].to_numpy(dtype=float)
        reference_values = r[column].to_numpy(dtype=float)
        if not np.allclose(
            generated_values,
            reference_values,
            atol=value_tolerance,
            rtol=0,
            equal_nan=True,
        ):
            delta = float(np.nanmax(np.abs(generated_values - reference_values)))
            return False, f"Numeric delta in {column}: {delta:.6f} > {value_tolerance:.6f}"
        if len(generated_values):
            max_delta = max(
                max_delta,
                float(np.nanmax(np.abs(generated_values - reference_values))),
            )

    return True, f"OK (max numeric delta={max_delta:.6f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated outputs with reference outputs")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--value-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--prediction-tolerance",
        type=float,
        default=0.2,
        help="Tolerance for refitted probabilities across numerical backends",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config used for the generated outputs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = config["outputs"]
    checks = [outputs[key] for key in config["reference_checks"]]

    ok = True
    for file_name in checks:
        generated = Path(args.generated_dir) / file_name
        reference = Path(args.reference_dir) / file_name
        if not generated.exists() or not reference.exists():
            print(f"{file_name}: missing generated or reference file")
            ok = False
            continue
        tolerance = (
            args.prediction_tolerance
            if "predictions" in file_name
            else args.value_tolerance
        )
        result, msg = compare_tables(generated, reference, tolerance)
        print(f"{file_name}: {msg}")
        ok = ok and result

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
