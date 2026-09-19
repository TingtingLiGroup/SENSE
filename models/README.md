# Released model bundles

The two pickle files are the final full-data refits used by the released
workflow:

- `associative_framework_model_bundle.pkl`: five classifiers fitted on 44
  records and the 8 associative-framework features;
- `solubility_model_bundle.pkl`: five classifiers fitted on 50 records and the
  5 solubility features.

Each bundle is a dictionary keyed by `TabPFN`, `LogisticRegression`,
`RandomForestClassifier`, `SVC`, and `MLPClassifier`. Use
`scripts/02_predict.py` to load the bundles and generate features in the correct
order. The release files were rewritten with CPU-mapped PyTorch storage so they
can be loaded on machines without CUDA.

Python pickle files can execute code during loading. Only load these artifacts
from this repository or another trusted, checksum-verified source.
