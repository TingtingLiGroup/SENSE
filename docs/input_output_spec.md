# Input Output Specification

This document defines expected inputs, command usage, and generated outputs.

## CLI entrypoints

- Train: `scripts/01_train.py`
- Predict: `scripts/02_predict.py`
- Compare with reference: `scripts/03_compare_reference.py`

## 1) Training

Command:

```bash
uv run python scripts/01_train.py \
  --input <xlsx_or_csv> \
  --out-dir <output_dir> \
  --seed 42 \
  --config configs/default.yaml
```

Optional:

- `--max-rows N` for smoke tests
- `--config configs/full.yaml` to train all four retained model groups

### Required training input columns

- `Seq`: peptide sequence
- `name`: sample id (auto-generated if missing)
- `label`: microcompartment label (`Positive` or `Negative`)
- `Pep. Conc.`: solubility text label (`insolub` is mapped to class 0, others to class 1)

### Training outputs

Written to `--out-dir`:

- `features_generated.csv`
- `model_performance_on_microcompartment_3fold.csv`
- `models_alldata_20250425.pkl`
- `data_micro_prediction_physiochemical_20250425.csv`
- `models_sol_alldata_20250425.pkl`
- `data_sol_prediction_physiochemical_20250425.csv`

Additional outputs when using `configs/full.yaml`:

- `models_physical_alldata_20250425.pkl`
- `data_micro_prediction_physical_20250425.csv`
- `models_sol_chemical_alldata_20250425.pkl`
- `data_sol_prediction_chemical_20250425.csv`

## 2) Prediction

Command:

```bash
uv run python scripts/02_predict.py \
  --input <xlsx_or_csv> \
  --models <model_dir> \
  --out <xlsx_or_csv> \
  --config configs/default.yaml
```

### Required prediction input columns

- `Seq`: peptide sequence
- `name`: optional; auto-generated if missing

### Prediction output columns

Output includes all input columns + generated features + prediction columns:

- `micro_physchem_<ModelName>`
- `micro_physical_<ModelName>`
- `sol_physchem_<ModelName>`
- `sol_chemical_<ModelName>`

Where `<ModelName>` is one of:

- `TabPFN`
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `MLPClassifier`

## 3) Reference comparison

Command:

```bash
uv run python scripts/03_compare_reference.py \
  --generated-dir <train_output_dir> \
  --reference-dir data/reference \
  --metric-tolerance 0.01 \
  --config configs/default.yaml
```

Current checks:

- column and row alignment for CSV outputs enabled by the selected config
- grouped metric deviation bound on `model_performance_on_microcompartment_3fold.csv`

## 4) CPU compatibility note

`scripts/02_predict.py` supports loading legacy TabPFN pickles that contain CUDA device tags by remapping to CPU during load. This allows inference on CPU-only machines with legacy reference model bundles.
