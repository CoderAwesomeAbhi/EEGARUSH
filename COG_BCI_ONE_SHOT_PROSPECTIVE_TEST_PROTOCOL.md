# COG-BCI One-Shot Prospective Test Protocol (Frozen Before Download)

**Status:** Frozen **before** any COG-BCI signal file is downloaded or inspected.
Authorized only because `COG_BCI_METADATA_ELIGIBILITY_AUDIT.md` returned
`COG_BCI_METADATA_ELIGIBLE_FOR_PRETEST_LOCK`. This locks every analysis degree of
freedom so the post-hoc z-scoring transport hypothesis gets exactly one honest
prospective test. Builds on `REVISED_ZSCORE_PROSPECTIVE_EVALUATION_PROTOCOL.md`.

## Dataset role

- COG-BCI (Hinss et al. 2023; Zenodo `10.5281/zenodo.6874128`) is the **single
  untouched prospective evaluation dataset** for the z-scoring transport hypothesis.
- **MAT and STEW remain development evidence only.**
- **No method, feature, channel, task, baseline, or endpoint may change after
  COG-BCI outcomes are accessed.**

## Primary paradigm selection (by construct, not performance)

- **Primary paradigm: MATB.** Chosen because it is operational/multitasking
  workload, construct-aligned with the STEW stress-test domain — **not** chosen by
  expected or observed performance. Metadata confirms MATB is present with discrete
  levels (easy/medium/difficult), so it can provide a valid fixed workload contrast.
- **N-back must NOT affect the primary verdict.** It may be analyzed only later as
  explicitly **secondary/exploratory**, after the primary result is locked and
  reported.
- (Contingency already discharged: MATB is present and valid, so
  `COG_BCI_PRIMARY_PARADIGM_INVALID_DO_NOT_DOWNLOAD` does not apply.)

## Baseline selection

- **Calibration baseline = eyes-open resting only** (COG-BCI has 1-min eyes-open
  resting at session start `RS_Beg` and end `RS_End`). **Eyes-closed rest is NOT
  used** for the primary model (eye-state alpha/spectral change is an avoidable
  construct mismatch).
- The primary outcome is rest-versus-task classification, so a **separate
  scored-rest segment** is required, **non-overlapping** with the calibration
  baseline.
- **Predeclared segment mapping** (to be confirmed in the future provenance audit):
  - **Calibration baseline:** eyes-open resting from `RS_Beg`.
  - **Scored-rest (label 0):** eyes-open resting from `RS_End` (disjoint from calibration).
  - **Scored-task (label 1):** MATB **difficult** (see below).
  - If the audit shows only one eyes-open resting recording exists, calibration and
    scored-rest will be taken from fixed **non-overlapping halves** of that single
    eyes-open recording (calibration = first half, scored-rest = second half). This
    fallback is fixed now, before any download.
- Windowing follows the frozen pipeline: **4 s windows, 50 % overlap**, equal
  scored-rest and scored-task window counts per subject where segment length allows.

## Primary outcome (matches the trained transport problem)

- **Train:** the already-frozen **MAT transport-compatible source pipeline** (MAT
  rest-vs-arithmetic eval rows; imputer + scaler + L2 logistic fit on **MAT only**).
- **Sampling representation (CORRECTED — see
  `COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md`):** both source MAT **and**
  target COG-BCI are represented at **128 Hz** before feature extraction, using the
  **identical** deterministic anti-aliased resampling of the frozen transport
  pipeline (`scipy.signal.resample_poly`, `up=32`, `down=125`). The earlier
  "500 Hz native — no resample needed" statement was incorrect (it conflated MAT's
  native 500 Hz acquisition with the transport pipeline's frozen 128 Hz feature
  representation) and is **superseded**. The 96 features are scale/offset-invariant
  but **not** sampling-rate-invariant, so the 128-Hz-trained model requires 128-Hz
  target features. **Native-500-Hz COG-BCI analysis is forbidden** from influencing
  the primary verdict. This correction was made **before** any COG-BCI predictive
  metric was computed and reopens no other degree of freedom.
- **Test:** COG-BCI **eyes-open rest vs the single predeclared MATB workload
  condition = MATB "difficult"** (the **highest documented** workload level, fixed
  before download).
- **Primary endpoint:** **macro subject-level ROC-AUC for z-scoring** on
  rest-versus-MATB-difficult classification in COG-BCI.

## Locked comparisons, model, features, channels

- **Primary method:** baseline **z-scoring** (per-subject center+scale from the
  unlabeled eyes-open calibration baseline only).
- **Primary comparator:** **mean subtraction**.
- **Secondary comparator:** **absolute** features.
- **Model:** fixed **L2 logistic regression** (`C=1.0`, `liblinear`,
  `class_weight=balanced`, `max_iter=5000`). No tuning.
- **Features:** the frozen **96 transport-compatible** features only
  (`results/stew_sensitivity/transport_compatible_feature_spec.yaml`); no addition,
  removal, or reselection.
- **Channels:** the **eight** harmonized channels (F3,F4,F7,F8,O1,O2,T3↔T7,T4↔T8)
  — **all eight are documented in COG-BCI metadata**, so the 8-channel set is locked.
  If the provenance audit unexpectedly finds any of the eight missing/unusable,
  **stop before modeling and report incompatibility** (do not expand or substitute).
- **Label discipline:** COG-BCI workload labels are **never** used for training,
  preprocessing fitting, feature selection, calibration-method selection, or tuning;
  z-scoring/mean-subtraction stats come only from the unlabeled eyes-open baseline.

## Primary comparison and success rule (locked)

- **Primary comparison:** paired subject-bootstrap ΔAUC = **z-scoring − mean
  subtraction** (2000 resamples); secondary = z-scoring − absolute.
- **Full success:** z-scoring AUC **above chance** AND the 95 % paired CI for
  (z-scoring − mean subtraction) **excludes zero in the positive direction**.
- **Partial:** z-scoring above chance but the superiority CI **includes zero**.
- **Failure:** z-scoring **at/below chance**, or **does not outperform** mean
  subtraction.
- **Exactly one** primary analysis is run. **No** post-outcome retuning, task
  switching, channel expansion, feature reselection, or endpoint change.

## Locked analysis order (for the future download stage)

1. **Source provenance audit only** (files, subject IDs, sampling rate, channel
   names/order, true rest vs MATB-difficult, license, integrity).
2. **Construct / channel / baseline verification** (8 channels present;
   eyes-open rest identifiable; MATB-difficult identifiable; non-overlapping
   calibration vs scored-rest design realizable).
3. **Freeze executable config** (exact segment→window mapping, frozen features,
   frozen model) — no outcomes viewed yet.
4. **Run the one-shot primary prospective test exactly once.**
5. **Report success / partial / failure honestly**, regardless of direction.

## Honesty constraints

- Until step 4 runs under a passed provenance audit, z-scoring transport is **not**
  confirmed/replicated/successful.
- No rejected claim is revived (no universal axis, no PAC/gamma/source/clinical, no
  confirmed transfer). MAT/STEW results are unchanged.
