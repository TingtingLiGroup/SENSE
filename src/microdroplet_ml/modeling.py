from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier


@dataclass(frozen=True)
class DatasetSpec:
    feature_set_name: str
    feature_columns: list[str]
    target_name: str


def build_models(seed: int) -> dict[str, object]:
    return {
        "TabPFN": TabPFNClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=seed),
        "SVC": SVC(probability=True, random_state=seed),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=seed,
        ),
    }


def k_fold_cv(
    data_x: np.ndarray,
    data_y: np.ndarray,
    model: object,
    n_fold: int = 3,
    random_state: int = 13,
) -> tuple[list[float], list[float]]:
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    fold_aurocs: list[float] = []
    fold_auprcs: list[float] = []

    for train_idx, val_idx in kf.split(data_x):
        x_train, x_val = data_x[train_idx], data_x[val_idx]
        y_train, y_val = data_y[train_idx], data_y[val_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        model.fit(x_train, y_train)
        proba = model.predict_proba(x_val)
        if proba.shape[1] == 1:
            class_value = int(getattr(model, "classes_", np.array([0]))[0])
            y_pred_proba = np.full(len(x_val), float(class_value))
        else:
            y_pred_proba = proba[:, 1]

        _ = roc_curve(y_val, y_pred_proba)
        _ = precision_recall_curve(y_val, y_pred_proba)
        auroc = float(roc_auc_score(y_val, y_pred_proba))
        auprc = float(average_precision_score(y_val, y_pred_proba))

        fold_aurocs.append(auroc)
        fold_auprcs.append(auprc)

    if not fold_aurocs or not fold_auprcs:
        raise ValueError("No valid CV folds were produced. Increase sample size or adjust folds.")

    return fold_aurocs, fold_auprcs


def evaluate_models(
    data_x: np.ndarray,
    data_y: np.ndarray,
    seed: int,
    n_fold: int = 3,
    cv_random_state: int = 13,
) -> pd.DataFrame:
    rows: list[tuple[str, str, float]] = []
    models = build_models(seed)

    for model_name, model in models.items():
        fold_aurocs, fold_auprcs = k_fold_cv(
            data_x,
            data_y,
            model,
            n_fold=n_fold,
            random_state=cv_random_state,
        )
        rows.extend((model_name, "auroc", v) for v in fold_aurocs)
        rows.extend((model_name, "auprc", v) for v in fold_auprcs)

    return pd.DataFrame(rows, columns=["model", "metric", "value"])


def fit_models(data_x: np.ndarray, data_y: np.ndarray, seed: int) -> dict[str, object]:
    models = build_models(seed)
    for model in models.values():
        model.fit(data_x, data_y)
    return models


def add_model_predictions(
    df: pd.DataFrame,
    data_x: np.ndarray,
    data_y: np.ndarray,
    models: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    metrics_rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        proba = model.predict_proba(data_x)
        if proba.shape[1] == 1:
            class_value = int(getattr(model, "classes_", np.array([0]))[0])
            y_pred_proba = np.full(len(data_x), float(class_value))
        else:
            y_pred_proba = proba[:, 1]
        out[model_name] = y_pred_proba
        metrics_rows.append(
            {
                "model": model_name,
                "auroc": float(roc_auc_score(data_y, y_pred_proba)),
                "auprc": float(average_precision_score(data_y, y_pred_proba)),
            }
        )

    return out, pd.DataFrame(metrics_rows)
