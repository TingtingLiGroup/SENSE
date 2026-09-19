# SENSE model card

## Intended use

The released models rank peptide sequences by predicted
associative-framework-forming propensity and predicted solubility. They are
intended for research screening and experimental prioritization.

## Training data

The models use a small experimentally curated peptide dataset from this study.
Associative-framework labels reflect the experimental phenotype defined in the
manuscript. Solubility class 0 denotes an annotation containing `insolub`; all
other experimentally retained annotations are class 1.

## Evaluation

Five classifier families were compared using shuffled three-fold
cross-validation. AUROC and AUPRC are reported per fold and as mean ± standard
deviation. Final released classifiers were refitted on all task-specific data.

## Limitations

- The training sets contain 44 and 50 sequences, respectively.
- Performance estimates therefore have substantial sampling uncertainty.
- The training sequences arise from the experimental design space in this
  study; predictions far outside that sequence space require experimental
  validation.
- Output probabilities are not calibrated clinical, toxicological, or safety
  estimates.
- Feature generation depends on third-party ensemble and interaction models and
  inherits their assumptions and applicable domains.

## Recommended use

Use predictions to rank candidates, inspect agreement across classifier
families, and experimentally validate selected sequences. Do not interpret a
single probability or threshold as definitive evidence of material state.
