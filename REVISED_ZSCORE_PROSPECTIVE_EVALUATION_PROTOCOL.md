# Revised Z-Scoring Prospective Evaluation Protocol (Frozen Before New Data)

**Status:** Frozen **before** any new dataset is downloaded or analyzed. Authorized
only because `POSTHOC_ZSCORE_TRANSPORT_AUDIT.md` returned
`ZSCORE_OBSERVATION_VALID_FOR_NEW_HYPOTHESIS_GENERATION`. This protocol locks the
analysis so the post-hoc z-scoring hypothesis can be tested prospectively on an
untouched dataset without further degrees of freedom.

## Revised primary hypothesis

Under cross-device / cross-task domain shift, **subject resting-baseline z-scoring
improves transport of EEG workload decoding more than absolute features or mean
subtraction.**

## Evidence roles (locked)

- **Development evidence only:** MAT and STEW. **Neither may count as prospective
  validation** — both were used to generate this hypothesis.
- **Prospective validation:** exactly one **untouched** dataset, selected by the
  metadata-only screen (`UNTOUCHED_DATASET_ELIGIBILITY_SCREEN.md`) and only after a
  fresh provenance audit passes. Direction is **MAT→untouched** transport (MAT
  selected the original method family; same predeclared training source).

## Locked methods

- **Primary transport method:** subject resting-baseline **z-scoring** (center by
  baseline mean, scale by baseline std, computed per target subject from
  **unlabeled baseline/rest** only).
- **Comparator 1:** absolute features (no per-subject calibration).
- **Comparator 2:** mean subtraction (baseline centering only).
- **Model:** L2 logistic regression with the already-fixed parameters
  (`make_model("logistic_l2", C=1.0)`: `solver=liblinear`, `class_weight=balanced`,
  `max_iter=5000`). **No tuning.**
- **Transport feature set:** the already-frozen **96 unit-compatible features**
  from `results/stew_sensitivity/transport_compatible_feature_spec.yaml`
  (skew, kurtosis, hjorth_mobility, hjorth_complexity, spectral_entropy,
  band_rel_{δ,θ,α,β}, ratio_{θ/α,β/α,θ/β} × 8 channels), invariant under
  `x→a·x+b`. **No feature added or removed after viewing new-dataset outcomes.**
- **Fitting discipline:** imputation, scaling, and the model are fit on **MAT
  training data only**; the target dataset's workload labels are **never** used for
  training, preprocessing fitting, feature selection, calibration-method
  selection, or hyperparameter tuning.

## Channel rule

- Use **only defensibly harmonized channels available across MAT, STEW, and the
  selected untouched dataset.** Start from the locked 8 (F3,F4,F7,F8,O1,O2,T3↔T7,
  T4↔T8); if the untouched dataset lacks one, restrict to the harmonized
  intersection and document it **before** seeing outcomes. **Do not expand** the
  feature/channel space after viewing new-dataset outcomes.

## Endpoints (locked)

- **Primary endpoint:** macro subject-level ROC-AUC on the untouched target dataset.
- **Primary comparison:** **z-scoring − mean subtraction**, paired subject-bootstrap
  95% CI (2000 resamples).
- **Secondary comparison:** **z-scoring − absolute**, paired subject-bootstrap 95% CI.
- Also report: per-method macro subject mean/median AUC, subject SD, subject-bootstrap
  95% CI, and a full-pipeline macro-subject permutation test (≥200) for z-scoring.

## Decision rules (locked, before seeing outcomes)

- **Success:** z-scoring achieves **above-chance** transport **and** a **positive
  paired CI excluding zero against mean subtraction**.
- **Partial:** above-chance z-scoring **but** the paired CI (vs mean subtraction)
  **includes zero**.
- **Failure:** z-scoring is **at/below chance**, or does **not** outperform the
  failed centering baseline (mean subtraction).

## Anti-degrees-of-freedom rules

- **No model tuning, no feature reselection, no endpoint switching** after seeing
  any new target outcome.
- A single untouched dataset is the prospective test; serial dataset shopping until
  one "works" is prohibited.
- The result is reported regardless of direction; failure is informative.

## Honesty constraints

- This remains an exploratory hypothesis until the prospective test is run; until
  then z-scoring transport is **not** confirmed, replicated, or successful.
- No rejected claim is revived (no universal axis, no PAC/gamma/source/clinical,
  no confirmed transfer).
