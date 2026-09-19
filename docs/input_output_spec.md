# Input and output specification

## Training

```bash
uv run python scripts/01_train.py \
  --input data/training/peptide_model_training_data.csv \
  --out-dir outputs/retrained \
  --seed 42
```

Required training columns are defined in `docs/data_manifest.md`. The input may
be CSV or XLSX. Sequences must use uppercase canonical one-letter amino-acid
codes. Records shorter than 14 residues are excluded from fitting.

Outputs:

| File | Contents |
|---|---|
| `peptide_features.csv` | Generated sequence and ensemble descriptors |
| `associative_framework_cv_metrics.csv` | Fold, model, metric, value |
| `solubility_cv_metrics.csv` | Fold, model, metric, value |
| `associative_framework_model_bundle.pkl` | Five final fitted classifiers |
| `solubility_model_bundle.pkl` | Five final fitted classifiers |
| `associative_framework_training_predictions.csv` | Selected features and full-data refit probabilities |
| `solubility_training_predictions.csv` | Selected features and full-data refit probabilities |

## Prediction

```bash
uv run python scripts/02_predict.py \
  --input data/demo/peptides.csv \
  --models models \
  --out outputs/demo_predictions.csv
```

Prediction input requires `sequence`; `sequence_id` is optional. The output
retains the input, appends generated features, and adds:

- `associative_framework_<ModelName>` for each of the five classifiers;
- `solubility_<ModelName>` for each of the five classifiers.

Values are positive-class probabilities between 0 and 1.

## SHAP interpretation

`scripts/03_explain.py` produces one row per associative-framework feature and
the columns `fold_1`, `fold_2`, `fold_3`, and `mean_abs_shap`.

## Reference comparison

`scripts/04_compare_reference.py` checks file schemas, identifiers, labels, and
numeric values against `data/reference/`. It exits with a non-zero status when
any required output is missing or exceeds the requested tolerance.
