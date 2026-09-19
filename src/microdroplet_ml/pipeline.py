from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import feature_generation
from .io import read_table, write_table
from .modeling import add_model_predictions, evaluate_models, fit_models


def _get_feature_sets(config: dict[str, Any]) -> dict[str, list[str]]:
    feature_sets = config.get("feature_sets")
    if not isinstance(feature_sets, dict):
        raise ValueError("Missing 'feature_sets' in config")
    return feature_sets


def _get_outputs(config: dict[str, Any]) -> dict[str, str]:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Missing 'outputs' in config")
    return outputs


def _get_prediction_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    groups = config.get("prediction_groups")
    if not isinstance(groups, list):
        raise ValueError("Missing 'prediction_groups' in config")
    return groups


def _to_binary_associative_framework(series: pd.Series) -> np.ndarray:
    invalid = sorted(set(series.dropna()) - {"Positive", "Negative"})
    if invalid:
        raise ValueError(f"Invalid associative-framework labels: {invalid}")
    return series.map({"Positive": 1, "Negative": 0}).to_numpy(dtype=int)


def _to_binary_solubility(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="raise")
    invalid = sorted(set(numeric.dropna()) - {0, 1})
    if invalid:
        raise ValueError(f"Invalid solubility labels: {invalid}")
    return numeric.to_numpy(dtype=int)


def _dump_pickle(obj: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return path


def _coerce_model_device_cpu(models: Any) -> None:
    # Legacy TabPFN pickles may store a CUDA runtime device, which is not loadable
    # on CPU-only machines unless remapped.
    if not isinstance(models, dict):
        return
    for model in models.values():
        if hasattr(model, "device_"):
            try:
                import torch

                model.device_ = torch.device("cpu")
            except Exception:
                pass
        if hasattr(model, "device"):
            try:
                model.device = "cpu"
            except Exception:
                pass


def _load_pickle_cpu_compatible(path: Path) -> Any:
    with path.open("rb") as fh:
        try:
            models = pickle.load(fh)
            _coerce_model_device_cpu(models)
            return models
        except RuntimeError as exc:
            message = str(exc)
            if "deserialize object on a CUDA device" not in message:
                raise

    # Fallback: remap any torch storages to CPU during unpickle.
    import torch

    original_loader = torch.storage._load_from_bytes

    def _cpu_load_from_bytes(raw_bytes: bytes) -> Any:
        return torch.load(
            io.BytesIO(raw_bytes),
            map_location=torch.device("cpu"),
            weights_only=False,
        )

    torch.storage._load_from_bytes = _cpu_load_from_bytes
    try:
        with path.open("rb") as fh:
            models = pickle.load(fh)
    finally:
        torch.storage._load_from_bytes = original_loader

    _coerce_model_device_cpu(models)
    return models


def run_train(
    input_path: str | Path,
    out_dir: str | Path,
    config: dict[str, Any],
    seed: int = 42,
    max_rows: int | None = None,
) -> dict[str, Path]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_col = config.get("sequence_column", "Seq")
    name_col = config.get("name_column", "name")
    min_length = int(config.get("min_length", 14))
    cv_folds = int(config.get("cv_folds", 3))
    cv_random_state = int(config.get("cv_random_state", 13))

    feature_sets = _get_feature_sets(config)
    outputs = _get_outputs(config)
    model_groups = _get_prediction_groups(config)
    labels = config.get("labels", {})
    associative_label_col = labels.get(
        "associative_framework", "associative_framework_label"
    )
    solubility_label_col = labels.get("solubility", "solubility_label")

    input_df = read_table(input_path)
    if max_rows is not None:
        input_df = input_df.head(max_rows).copy()

    data_features = feature_generation(input_df, seq_col=seq_col, name_col=name_col)

    written: dict[str, Path] = {}
    feature_path = write_table(
        data_features,
        output_dir / outputs["features_table"],
        index=False,
    )
    written["features_table"] = feature_path

    data_associative = data_features[
        data_features[associative_label_col].notna()
        & (data_features["length"] >= min_length)
    ].reset_index(drop=True)

    data_associative_y = _to_binary_associative_framework(
        data_associative[associative_label_col]
    )
    data_solubility = data_features[
        data_features[solubility_label_col].notna()
        & (data_features["length"] >= min_length)
    ].reset_index(drop=True)
    data_solubility_y = _to_binary_solubility(
        data_solubility[solubility_label_col]
    )

    task_inputs: dict[str, tuple[pd.DataFrame, np.ndarray]] = {
        "associative_framework": (data_associative, data_associative_y),
        "solubility": (data_solubility, data_solubility_y),
    }

    for group in model_groups:
        task = group["task"]
        feature_set_name = group["feature_set"]
        model_output_key = group["model_output"]
        prediction_output_key = group["train_prediction_output"]

        if task not in task_inputs:
            raise ValueError(f"Unsupported task in config: {task}")

        group_df, group_y = task_inputs[task]
        feature_cols = feature_sets[feature_set_name]
        group_x = group_df[feature_cols].to_numpy()

        cv_output_key = group.get("cv_output")
        if cv_output_key:
            cv_table = evaluate_models(
                group_x,
                group_y,
                seed=seed,
                n_fold=cv_folds,
                cv_random_state=cv_random_state,
            )
            written[cv_output_key] = write_table(
                cv_table,
                output_dir / outputs[cv_output_key],
                index=False,
            )

        models = fit_models(group_x, group_y, seed=seed)
        group_with_pred, _ = add_model_predictions(group_df, group_x, group_y, models)

        label_col = (
            associative_label_col
            if task == "associative_framework"
            else solubility_label_col
        )
        prediction_columns = [
            name_col,
            "sequence",
            label_col,
            *feature_cols,
            *models.keys(),
        ]
        group_with_pred = group_with_pred[prediction_columns]

        written[model_output_key] = _dump_pickle(
            models, output_dir / outputs[model_output_key]
        )
        written[prediction_output_key] = write_table(
            group_with_pred,
            output_dir / outputs[prediction_output_key],
            index=False,
        )

    return written


def run_predict(
    input_path: str | Path,
    models_dir: str | Path,
    out_path: str | Path,
    config: dict[str, Any],
) -> Path:
    seq_col = config.get("sequence_column", "Seq")
    name_col = config.get("name_column", "name")
    feature_sets = _get_feature_sets(config)
    model_groups = _get_prediction_groups(config)

    data = read_table(input_path)
    data_features = feature_generation(data, seq_col=seq_col, name_col=name_col)

    model_root = Path(models_dir)
    out_df = data_features.copy()

    for group in model_groups:
        group_name = group["name"]
        feature_set_name = group["feature_set"]
        model_file = group["model_file"]
        prefix = group.get("prefix", group_name)

        model_path = model_root / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        models: dict[str, object] = _load_pickle_cpu_compatible(model_path)

        feature_cols = feature_sets[feature_set_name]
        data_x = out_df[feature_cols].to_numpy()

        for model_name, model in models.items():
            out_df[f"{prefix}_{model_name}"] = model.predict_proba(data_x)[:, 1]

    return write_table(out_df, out_path, index=False)
