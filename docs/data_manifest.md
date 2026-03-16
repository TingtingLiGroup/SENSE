# Data Manifest

This manifest lists files intentionally included for the v1 open pipeline.

| File | Source | Purpose | Regenerable | SHA256 |
|---|---|---|---|---|
| `data/raw/experiments_sequences_20250425-V18-modified.xlsx` | `microdroplet/data/experiments_sequences_20250425-V18-modified.xlsx` | Main training input for model pipeline | No (primary curated input) | `8a3c83e0da3c7da87c87af335a9529ceafd66673074dcea4843a576ab2f3d833` |
| `data/reference/model_performance_on_microcompartment_3fold.csv` | `microdroplet/data/model_performance_on_microcompartment_3fold.csv` | Reference CV metrics | Yes | `0a9feba0eace2564fe8d9e51812668ea50824a6977793297315a4440fdc68ff2` |
| `data/reference/models_alldata_20250425.pkl` | `microdroplet/data/models_alldata_20250425.pkl` | Reference microcompartment model bundle | Yes | `28c071b4cce5c472864fd01c49cd62b73f5b152f08ed97c055725f184ec502d0` |
| `data/reference/models_physical_alldata_20250425.pkl` | `microdroplet/data/models_physical_alldata_20250425.pkl` | Reference physical-feature model bundle | Yes | `961f483f6c064f45740f4d8e2b7abfe21516c10feb611b37a9fa5caebf354cc0` |
| `data/reference/models_sol_alldata_20250425.pkl` | `microdroplet/data/models_sol_alldata_20250425.pkl` | Reference solubility physiochemical model bundle | Yes | `c4f949a79d5b5e9d2eb443bcd2c920d3d9bbf27490d51dc8d6e8c02c9dadf58b` |
| `data/reference/models_sol_chemical_alldata_20250425.pkl` | `microdroplet/data/models_sol_chemical_alldata_20250425.pkl` | Reference solubility chemical model bundle | Yes | `e6c96152c2e000acf8e4a0cc4fdc1bb6754e0509636e70d02c5e66e428a42a9e` |
| `data/reference/data_micro_prediction_physiochemical_20250425.csv` | `microdroplet/data/data_micro_prediction_physiochemical_20250425.csv` | Reference training predictions (micro physiochemical) | Yes | `cf1a619b41df7bd4642d3fb2c068166a5bc9f995cb369e3ccd382d3d1b76b55f` |
| `data/reference/data_micro_prediction_physical_20250425.csv` | `microdroplet/data/data_micro_prediction_physical_20250425.csv` | Reference training predictions (micro physical) | Yes | `eb43a474f4715f84aafc863aebbe573fd4e5026f9924cbfcf75033fcb5a42501` |
| `data/reference/data_sol_prediction_physiochemical_20250425.csv` | `microdroplet/data/data_sol_prediction_physiochemical_20250425.csv` | Reference training predictions (sol physiochemical) | Yes | `54c8e2ef852baf86068b8d633380ade0904ec9fdc57b815c5cc1c100d9c07a0c` |
| `data/reference/data_sol_prediction_chemical_20250425.csv` | `microdroplet/data/data_sol_prediction_chemical_20250425.csv` | Reference training predictions (sol chemical) | Yes | `d94b9936ddaa4224b4b7770470e3bf9ce78a417b5dd8706864450b9c1908608c` |
| `data/reference/feature_importance_all_features_20250425.csv` | `microdroplet/data/feature_importance_all_features_20250425.csv` | Reference interpretation output | Yes | `a8e170aed718bdb1586b2175c790116cc8d7fd74b039b16e3eb08bb98e8f9a59` |

## Explicitly excluded

The following directories/files from the legacy project are intentionally not included because they are outside the main model pipeline or too large for repository distribution:

- `data/msa_fasta/`
- `data/msa_a3m/`
- `data/zhangxin_result.rar`
