from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_released_training_counts() -> None:
    data = pd.read_csv(ROOT / "data/training/peptide_model_training_data.csv")
    assert len(data) == 50
    assert data["sequence"].str.len().min() >= 14
    assert data["sequence"].is_unique
    assert data["sequence_id"].is_unique
    assert data["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+").all()
    assert data["associative_framework_label"].value_counts().to_dict() == {
        "Negative": 24,
        "Positive": 20,
    }
    assert data["solubility_label"].value_counts().to_dict() == {1: 35, 0: 15}


def test_final_feature_sets_and_artifacts() -> None:
    with (ROOT / "configs/default.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["feature_sets"]["associative_framework"] == [
        "norm_rg",
        "norm_re",
        "epsilon_mf",
        "epsilon_cf",
        "asphericity",
        "fraction_aliphatic",
        "hydrophobicity",
        "FCR",
    ]
    assert config["feature_sets"]["solubility"] == [
        "fraction_aliphatic",
        "hydrophobicity",
        "pI",
        "fraction_aromatic",
        "rg",
    ]

    for model_file in (
        "associative_framework_model_bundle.pkl",
        "solubility_model_bundle.pkl",
    ):
        path = ROOT / "models" / model_file
        assert path.is_file()
        assert path.stat().st_size < 100 * 1024 * 1024


def test_reference_tables_match_release_contract() -> None:
    cv = pd.read_csv(ROOT / "data/reference/associative_framework_cv_metrics.csv")
    assert len(cv) == 30
    assert set(cv["fold"]) == {1, 2, 3}
    assert set(cv["metric"]) == {"auroc", "auprc"}

    shap = pd.read_csv(
        ROOT / "data/reference/associative_framework_shap_importance.csv"
    )
    assert len(shap) == 8
    assert shap.loc[shap["mean_abs_shap"].idxmax(), "feature"] == "norm_rg"
