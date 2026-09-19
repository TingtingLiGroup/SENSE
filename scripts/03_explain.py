#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tabpfn import TabPFNClassifier
from tabpfn_extensions import interpretability

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microdroplet_ml.config import DEFAULT_CONFIG_PATH, load_config
from microdroplet_ml.features import feature_generation
from microdroplet_ml.io import read_table, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the cross-validated SHAP feature importance analysis."
    )
    parser.add_argument("--input", required=True, help="Curated training .csv/.xlsx")
    parser.add_argument("--out", required=True, help="Output .csv path")
    parser.add_argument(
        "--seed", type=int, default=42, help="Permutation-explainer random seed"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    config = load_config(args.config)
    seq_col = config.get("sequence_column", "sequence")
    name_col = config.get("name_column", "sequence_id")
    min_length = int(config.get("min_length", 14))
    n_folds = int(config.get("cv_folds", 3))
    cv_random_state = int(config.get("cv_random_state", 13))
    label_col = config["labels"]["associative_framework"]
    feature_cols = config["feature_sets"]["associative_framework"]

    data = feature_generation(
        read_table(args.input), seq_col=seq_col, name_col=name_col
    )
    data = data[
        data[label_col].isin(["Positive", "Negative"])
        & (data["length"] >= min_length)
    ].reset_index(drop=True)
    data_x = data[feature_cols].to_numpy()
    data_y = data[label_col].map({"Positive": 1, "Negative": 0}).to_numpy()

    fold_columns: dict[str, np.ndarray] = {}
    splitter = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=cv_random_state,
    )
    for fold, (train_idx, validation_idx) in enumerate(splitter.split(data_x), 1):
        model = TabPFNClassifier()
        model.fit(data_x[train_idx], data_y[train_idx])
        explanation = interpretability.shap.get_shap_values(
            estimator=model,
            test_x=data_x[validation_idx],
            attribute_names=pd.Index(feature_cols),
            algorithm="permutation",
        )
        mean_absolute = np.abs(explanation.values).mean(axis=0)
        if mean_absolute.ndim == 2:
            mean_absolute = mean_absolute[:, 0]
        fold_columns[f"fold_{fold}"] = mean_absolute

    result = pd.DataFrame({"feature": feature_cols, **fold_columns})
    fold_names = list(fold_columns)
    result["mean_abs_shap"] = result[fold_names].mean(axis=1)
    write_table(result, args.out, index=False)
    print(f"SHAP analysis completed: {args.out}")


if __name__ == "__main__":
    main()
