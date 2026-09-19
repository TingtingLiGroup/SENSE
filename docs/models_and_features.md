# Models and features

## Associative-framework model

Artifact: `models/associative_framework_model_bundle.pkl`

Training set: 44 sequences, comprising 20 positive and 24 negative experimental
labels.

| Internal name | Interpretation |
|---|---|
| `norm_rg` | Predicted radius of gyration normalized by the AFRC analytical expectation |
| `norm_re` | Predicted end-to-end distance normalized by the AFRC analytical expectation |
| `epsilon_mf` | FINCHES Mpipi mean-field homotypic interaction strength |
| `epsilon_cf` | FINCHES CALVADOS contact-fluctuation interaction strength |
| `asphericity` | Predicted chain-shape asphericity |
| `fraction_aliphatic` | Fraction of aliphatic residues |
| `hydrophobicity` | Sequence hydrophobicity |
| `FCR` | Fraction of charged residues |

## Solubility model

Artifact: `models/solubility_model_bundle.pkl`

Training set: 50 sequences, comprising 35 positive and 15 insoluble-class
records.

| Internal name | Interpretation |
|---|---|
| `fraction_aliphatic` | Fraction of aliphatic residues |
| `hydrophobicity` | Sequence hydrophobicity |
| `pI` | Predicted isoelectric point |
| `fraction_aromatic` | Fraction of aromatic residues |
| `rg` | Predicted absolute radius of gyration |

## Classifier families

Each model bundle contains fitted instances named:

- `TabPFN`
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `MLPClassifier`

The scikit-learn settings are defined in
`src/microdroplet_ml/modeling.py`. TabPFN uses its released default architecture
and default random state of 0. Random forest uses 100 trees; the multilayer
perceptron uses hidden layers of 100 and 50 units. All task and feature mappings
are centralized in `configs/default.yaml`.

## Feature software

- Sparrow provides sequence-to-ensemble predictions (`rg`, `re`, and
  `asphericity`).
- AFRC supplies analytical polymer expectations used to normalize chain
  dimensions.
- FINCHES supplies Mpipi and CALVADOS homotypic interaction descriptors.
- Sparrow and Biopython supply sequence composition, hydrophobicity, charge,
  and isoelectric-point properties.

Exact versions and Git commits are locked in `uv.lock`.
