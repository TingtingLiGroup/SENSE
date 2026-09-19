# Reproducibility guide

## Evaluation design

The experimentally labelled dataset is small, so performance is estimated with
three-fold shuffled cross-validation. The split random state is 13. Each of the
five classifier families is evaluated on the same folds using AUROC and AUPRC.
The figure reports the mean and standard deviation across the three folds.

After evaluation, each classifier is refitted on all available task-specific
records and stored in the corresponding model bundle. Probabilities in the
`*_training_predictions.csv` files come from these full-data refits and are not
used as cross-validation performance estimates.

## Fixed inputs and software

- Curated training input: `data/training/peptide_model_training_data.csv`
- Scikit-learn model seed: 42; TabPFN default random state: 0
- Cross-validation split seed: 13
- Feature and model definitions: `configs/default.yaml`
- Exact environment: `uv.lock`
- Integrity hashes: `SHA256SUMS`

Use `uv sync --frozen`; running `uv sync` without `--frozen` can update the
locked resolution in future environments.

## End-to-end commands

```bash
uv sync --frozen

uv run python scripts/01_train.py \
  --input data/training/peptide_model_training_data.csv \
  --out-dir outputs/retrained \
  --seed 42

uv run python scripts/04_compare_reference.py \
  --generated-dir outputs/retrained \
  --reference-dir data/reference \
  --value-tolerance 0.01
```

SHAP is reproduced separately because it requires many repeated TabPFN
evaluations:

```bash
uv run python scripts/03_explain.py \
  --input data/training/peptide_model_training_data.csv \
  --out outputs/associative_framework_shap_importance.csv \
  --seed 42
```

## Acceptance criteria

- Training retains 44 associative-framework records (20 positive, 24 negative)
  and 50 solubility records (35 positive, 15 negative).
- Each task produces 30 cross-validation rows: 5 classifiers × 2 metrics × 3
  folds.
- Output schemas and row identifiers match the released references.
- Maximum absolute cross-validation metric deviation is no greater than 0.01.
- Maximum probability deviation is no greater than 0.2; this wider bound
  accommodates TabPFN drift and the non-converged MLP optimization observed in
  the original workflow while still detecting material changes.

Some Sparrow/FINCHES features and TabPFN probabilities can show small
cross-platform floating-point differences. Larger deviations should be treated
as a failed reproduction and investigated before use.
