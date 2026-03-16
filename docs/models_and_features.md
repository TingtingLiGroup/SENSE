# Models And Features

This document describes exactly which feature sets and model bundles are used.

## Model bundles

Default config: `configs/default.yaml`

- `models_alldata_20250425.pkl`
- `models_sol_alldata_20250425.pkl`

Full config: `configs/full.yaml`

- `models_alldata_20250425.pkl`
- `models_physical_alldata_20250425.pkl`
- `models_sol_alldata_20250425.pkl`
- `models_sol_chemical_alldata_20250425.pkl`

### `models_alldata_20250425.pkl`

- Task: microcompartment classification
- Feature set: `micro_physiochemical`
- Features:
  - `norm_rg`
  - `norm_re`
  - `epsilon_mf`
  - `epsilon_cf`
  - `asphericity`
  - `fraction_aliphatic`
  - `hydrophobicity`
  - `FCR`

### `models_physical_alldata_20250425.pkl`

- Task: microcompartment classification
- Feature set: `micro_physical`
- Features:
  - `norm_rg`
  - `norm_re`
  - `epsilon_mf`
  - `epsilon_cf`
  - `asphericity`

### `models_sol_alldata_20250425.pkl`

- Task: solubility classification
- Feature set: `sol_physiochemical`
- Features:
  - `fraction_aliphatic`
  - `hydrophobicity`
  - `pI`
  - `fraction_aromatic`
  - `rg`

### `models_sol_chemical_alldata_20250425.pkl`

- Task: solubility classification
- Feature set: `sol_chemical`
- Features:
  - `fraction_aliphatic`
  - `hydrophobicity`
  - `pI`
  - `fraction_aromatic`

## Base models inside each bundle

Each bundle stores the same five classifiers:

- `TabPFN`
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `MLPClassifier`

## Feature generation sources

Feature computation is implemented in `src/microdroplet_ml/features.py` and includes:

- sequence-derived counts/fractions (`A`, `C`, ..., `Y`, `frac_A`, ..., `frac_Y`)
- sticker spacing (`sticker_distance`)
- physics predictors from `sparrow` and `afrc` (`rg`, `re`, `scaled_rg`, `scaled_re`, `asphericity`, `scaling_exponent`, `prefactor`, `mean_rg`, `mean_re`, `norm_rg`, `norm_re`)
- interaction terms from `finches` (`epsilon_mf`, `epsilon_cf`, `fourier_peakratio`)
- composition/biochemical terms (`FCR`, `NCPR`, `fraction_*`, `hydrophobicity`, `pI`)

## Performance and importance reference files

- CV metrics used by notebook barplot:
  - `data/reference/model_performance_on_microcompartment_3fold.csv`
- Feature importance table used in analysis:
  - `data/reference/feature_importance_all_features_20250425.csv`
