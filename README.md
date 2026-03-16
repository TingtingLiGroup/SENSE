# microdroplet-ml-open

Open, script-first ML pipeline for peptide microcompartment and solubility prediction.

This repository provides:

- deterministic feature generation from peptide sequences
- model training for the default two-model release, with optional legacy groups
- inference with saved model bundles
- reproducibility checks against reference outputs

## 1) Quick start

Sync environment:

```bash
uv sync
```

Run training:

```bash
uv run python scripts/01_train.py \
  --input data/raw/experiments_sequences_20250425-V18-modified.xlsx \
  --out-dir artifacts/train_run \
  --seed 42
```

Run prediction with existing model bundles:

```bash
uv run python scripts/02_predict.py \
  --input data/raw/experiments_sequences_20250425-V18-modified.xlsx \
  --models data/reference \
  --out artifacts/predict_from_reference.csv
```

Compare a training run with reference outputs:

```bash
uv run python scripts/03_compare_reference.py \
  --generated-dir artifacts/train_run \
  --reference-dir data/reference \
  --metric-tolerance 0.01
```

Run the full four-model configuration when needed:

```bash
uv run python scripts/01_train.py \
  --input data/raw/experiments_sequences_20250425-V18-modified.xlsx \
  --out-dir artifacts/train_run_full \
  --seed 42 \
  --config configs/full.yaml
```

## 2) Default and optional models

Default commands use these two saved model bundles:

- `models_alldata_20250425.pkl` (micro, physiochemical feature set)
- `models_sol_alldata_20250425.pkl` (solubility, physiochemical feature set)

Optional legacy bundles are still kept in the repository and are enabled by
`configs/full.yaml`:

- `models_physical_alldata_20250425.pkl` (micro, physical feature set)
- `models_sol_chemical_alldata_20250425.pkl` (solubility, chemical feature set)

Each bundle contains five classifiers:

- `TabPFN`
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `MLPClassifier`

See full feature definitions and model mapping in:

- `docs/models_and_features.md`

## 3) Inputs and outputs

Training input (`scripts/01_train.py`) expects a table containing at least:

- sequence column: `Seq`
- sample name column: `name` (if missing, auto-generated)
- labels used by training:
  - `label` for microcompartment task (`Positive`/`Negative`)
  - `Pep. Conc.` for solubility task (`insolub` interpreted as class 0)

Main outputs include:

- `features_generated.csv`
- `model_performance_on_microcompartment_3fold.csv`
- two model bundle `.pkl` files by default
- two training-set prediction `.csv` files by default

Using `--config configs/full.yaml` restores the four-bundle / four-prediction
artifact set.

Detailed I/O schema and output column conventions:

- `docs/input_output_spec.md`

## 4) Performance and importance artifacts

The notebook barplot and importance table correspond to:

- `data/reference/model_performance_on_microcompartment_3fold.csv`
- `data/reference/feature_importance_all_features_20250425.csv`

## 5) Reproducibility notes

- Same environment is deterministic in repeated runs.
- Cross-environment numerical drift can appear in some feature columns from third-party physics predictors.
- Legacy TabPFN pickles with CUDA device tags are handled in `02_predict.py` via CPU-compatible loading.

See:

- `docs/reproducibility.md`
- `docs/data_manifest.md`

## 6) Repository structure

- `src/microdroplet_ml/` core pipeline code
- `scripts/01_train.py` train entrypoint
- `scripts/02_predict.py` predict entrypoint
- `scripts/03_compare_reference.py` regression check
- `configs/default.yaml` feature sets, output names, prediction groups
- `data/raw/` required input files
- `data/reference/` reference outputs and model bundles
- `docs/` user-facing project docs
