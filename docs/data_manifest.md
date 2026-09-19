# Data manifest

## Training data

`data/training/peptide_model_training_data.csv` is the final model input. The
peptides were experimentally characterized in this study after the upstream IDR
screening workflow. Only records used by at least one released model are kept.

The source experimental table contained 55 records. Five sequences shorter than
14 residues were excluded before model fitting, leaving 50 records:

- associative-framework task: 44 records with binary experimental labels
  (20 `Positive`, 24 `Negative`);
- solubility task: all 50 records (35 class `1`, 15 class `0`).

Six records without a binary associative-framework label are retained because
they are used by the solubility classifier. Free-text notes and experimental
columns not used by either model are not included in the released training
table.

### Columns

| Column | Meaning | Used by |
|---|---|---|
| `sequence_id` | Stable peptide identifier | Both tasks |
| `sequence` | Canonical amino-acid sequence | Both tasks |
| `associative_framework_label` | `Positive`, `Negative`, or blank | Associative-framework task |
| `solubility_annotation` | Original experimental concentration/state annotation | Label provenance |
| `solubility_label` | `0` for annotations containing `insolub`; `1` otherwise | Solubility task |

## Reference data

`data/reference/` contains compact, machine-readable outputs from the final
workflow:

- `peptide_features.csv`: model feature values for the 50 retained sequences;
- `associative_framework_cv_metrics.csv`: the three folds underlying the
  reported AUROC/AUPRC mean and standard deviation;
- `associative_framework_training_predictions.csv`: full-data refit outputs;
- `solubility_cv_metrics.csv`: three-fold solubility metrics;
- `solubility_training_predictions.csv`: full-data refit outputs;
- `associative_framework_shap_importance.csv`: fold-level and mean absolute
  SHAP values for the eight final features.

SHA256 checksums for all released data and model files are recorded in
`SHA256SUMS` at the repository root.
