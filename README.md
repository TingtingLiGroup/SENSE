# SENSE

This repository contains the machine-learning workflow used to predict
associative-framework formation and peptide solubility from amino-acid
sequence.

**Repository:** https://github.com/TingtingLiGroup/SENSE

## Scope

This release starts from the experimentally curated peptide table and covers
feature generation, three-fold cross-validation, final model fitting, SHAP
interpretation, and prediction on new sequences. It corresponds to the model
workflow described in the manuscript's **Machine learning models** Methods
section.

Candidate discovery and experimental peptide selection precede the released
training table and are outside the scope of this model repository.

The release contains two final tasks:

- **Associative-framework classification:** 44 experimentally labelled
  sequences (20 positive, 24 negative), 8 features, and 5 classifier families.
- **Solubility classification:** 50 sequences (35 soluble/non-insoluble,
  15 insoluble), 5 features, and the same 5 classifier families.

The classifier families are TabPFN, logistic regression, random forest, support
vector classifier, and multilayer perceptron.

## System requirements

Tested configuration:

- Ubuntu 22.04 x86-64
- Python 3.12.11 (Python 3.11 is also supported by the lock file)
- CPU-only prediction is supported
- at least 8 GB RAM and 8 GB free disk space are recommended
- an NVIDIA GPU is recommended, but not required, for retraining and SHAP

The first environment installation and a fresh TabPFN training run require
internet access to obtain packages and, if not already cached, pretrained
TabPFN weights. Exact Python dependencies are recorded in `uv.lock`.

## Installation

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the
locked environment:

```bash
git clone https://github.com/TingtingLiGroup/SENSE.git
cd SENSE
uv sync --frozen
```

Installation typically takes 5–15 minutes on a normal broadband connection.
PyTorch is the largest dependency, so download time is connection-dependent.

## Quick demo

Run the two released model bundles on the included two-sequence demo:

```bash
uv run python scripts/02_predict.py \
  --input data/demo/peptides.csv \
  --models models \
  --out outputs/demo_predictions.csv
```

Expected result: `outputs/demo_predictions.csv` containing the input sequences,
generated features, and ten probability columns (five per prediction task).
The first run is slower because predictor resources are initialized. The demo
completed in 43 seconds in the tested CPU-only environment.

## Reproduce model training and cross-validation

```bash
uv run python scripts/01_train.py \
  --input data/training/peptide_model_training_data.csv \
  --out-dir outputs/retrained \
  --seed 42
```

This command generates:

- the sequence feature table;
- three-fold AUROC and AUPRC values for both tasks;
- two model bundles, each containing the five classifier families;
- training-table probabilities from the models refitted on all task-specific
  samples.

With dependencies and pretrained resources already installed, the complete
training command took 48 seconds in the tested 8-vCPU environment. Hardware and
cache state can materially affect this time.

Compare a completed run with the released reference tables:

```bash
uv run python scripts/04_compare_reference.py \
  --generated-dir outputs/retrained \
  --reference-dir data/reference \
  --value-tolerance 0.01
```

Small numerical differences can occur across operating systems or numerical
backends. See `docs/reproducibility.md` for interpretation and acceptance
criteria.

## Reproduce SHAP importance

```bash
uv run python scripts/03_explain.py \
  --input data/training/peptide_model_training_data.csv \
  --out outputs/associative_framework_shap_importance.csv \
  --seed 42
```

This reproduces the three-fold mean absolute SHAP values for the eight
associative-framework features. SHAP repeatedly evaluates TabPFN and may take
tens of minutes on CPU; a GPU is recommended. The values used for the
manuscript plot are included in
`data/reference/associative_framework_shap_importance.csv`.

## Predict your own sequences

Supply a CSV or XLSX file with:

- `sequence`: required, using the 20 canonical one-letter amino-acid codes;
- `sequence_id`: optional; generated automatically when absent.

Then run:

```bash
uv run python scripts/02_predict.py \
  --input your_sequences.csv \
  --models models \
  --out outputs/your_predictions.csv
```

The probabilities are intended for research ranking and hypothesis generation,
not as calibrated clinical or safety estimates.

## Released files

- `data/training/peptide_model_training_data.csv`: final curated model input
- `data/demo/peptides.csv`: minimal inference example
- `models/associative_framework_model_bundle.pkl`: five fitted 8-feature models
- `models/solubility_model_bundle.pkl`: five fitted 5-feature models
- `data/reference/`: cross-validation, feature, prediction, and SHAP references
- `configs/default.yaml`: task, feature, output, and random-state definitions
- `src/microdroplet_ml/`: reusable feature and model pipeline
- `scripts/`: train, predict, explain, and validate entry points

Detailed schemas and feature definitions are in `docs/`.

## License

The repository is released under the MIT License. Third-party packages and
pretrained models remain subject to their respective licenses and terms.
