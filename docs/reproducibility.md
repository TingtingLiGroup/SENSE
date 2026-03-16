# Reproducibility Guide

## Goal

Reproduce the main training outputs from the cleaned pipeline and compare them with reference files.

## Commands

1. Sync dependencies:

```bash
uv sync
```

2. Run training:

```bash
uv run python scripts/01_train.py \
  --input data/raw/experiments_sequences_20250425-V18-modified.xlsx \
  --out-dir artifacts/train_run \
  --seed 42
```

3. Compare with references:

```bash
uv run python scripts/03_compare_reference.py \
  --generated-dir artifacts/train_run \
  --reference-dir data/reference \
  --metric-tolerance 0.01
```

To reproduce the retained four-model variant instead of the default two-model
release, add `--config configs/full.yaml` to both commands.

## Acceptance criteria

- Output files exist with expected names.
- Row count and column order match reference files.
- For cross-validation metric file, max absolute delta for `value` is `<= 0.01`.
- No hardcoded credentials or host-specific absolute paths in source code.

## Notes

- Random Forest / SVC / MLP use fixed random seed through CLI `--seed`.
- CV split random state follows notebook logic (`13`) via config.
- Repeated runs in the same environment are deterministic.
- Cross-environment drift may appear in a subset of physics-derived feature columns (`rg`, `re`, `asphericity`, and normalized variants) due to third-party numerical stacks.
- TabPFN prediction drift is usually larger than classical models when feature values drift.
- Legacy reference pickles with CUDA tags are loadable on CPU via the compatibility path in `run_predict`.
