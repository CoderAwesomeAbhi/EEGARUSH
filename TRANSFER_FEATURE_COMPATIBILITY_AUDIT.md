# Transfer Feature Compatibility Audit (Revised, Predeclared, Before Any Results)

**Status:** Revised after identifying a measurement-unit transport problem.
Frozen **before** any STEW AUC, MAT→STEW AUC, transfer metric, or permutation
result was computed or inspected. No transfer metric using amplitude-dependent
raw-scale features was ever computed (so none requires invalidation).

## 1. The measurement-unit problem

- **MAT** raw EDF amplitudes are in **microvolts (µV)** (EDF physical units;
  resampled traces range ≈ ±80 µV — physiological).
- **STEW** official Emotiv EPOC files are **raw 16-bit A/D counts** with a large
  DC offset (per-channel baseline ≈ 4200, values ≈ 330–8000). These are **not**
  physically comparable µV.
- Resampling MAT 500 → 128 Hz fixes **sampling-rate** compatibility **only**. It
  does **not** make amplitude-dependent features comparable across devices.

## 2. Was an official unit conversion found? — NO

Searched, per the corrected protocol:
- **Official zip archive:** contains only `ratings.txt` and `subNN_{lo,hi}.txt`.
  **No readme, no documentation, no calibration/conversion file.**
- **IEEE DataPort dataset page:** explicitly provides no voltage calibration,
  µV conversion, gain, or LSB scaling; states only Emotiv EPOC, 128 Hz, 14
  channels, 16-bit A/D.
- **Publication metadata (Lim, Mountstephens, Teo 2018):** documents the device
  and channel order but no per-recording µV conversion.

The Emotiv EPOC's *nominal* LSB is a device datasheet figure, not a verified
per-recording calibration, and it does not address the large non-physiological DC
offset. **Therefore no defensible verified unit conversion exists.** Consequently
the MAT→STEW transport pipeline must use a **scale/offset-invariant feature
subset frozen before any transfer result is viewed.**

## 3. Two separate pipelines

### A. Within-STEW exploratory sensitivity (same device/unit system)
- Training and testing are both within STEW → amplitude units are internally
  consistent. The full locked **`no_gamma_184`** 8-channel family is permitted.
- Candidate method: **mean subtraction**; comparator **absolute**; secondary
  diagnostic **z-scoring**. Model: fixed **L2 logistic regression** (C=1.0).
- Labeled **cross-task/device sensitivity only** — never replication/confirmation.

### B. MAT→STEW transport (cross device/unit system)
- May use **only** features invariant under the arbitrary device map
  `x → a·x + b` (gain `a>0`, offset `b`). No verified unit conversion exists, so
  amplitude-dependent features are forbidden.
- Same candidate method (mean subtraction), comparator (absolute), and diagnostic
  (z-scoring); MAT resampled to 128 Hz; imputer/scaler/model fit on **MAT only**;
  STEW per-subject calibration computed from **unlabeled STEW baseline** only;
  **STEW workload labels never used** for training/selection/tuning.

## 4. Per-feature invariance audit (every template individually)

Invariance was checked **mathematically and empirically**. Empirical test:
real STEW windows transformed by 42 random `(a∈[0.3,4.0], b∈[-2000,2000])` draws;
max relative change recorded per template (tolerance 1e-6).

| Template (per channel) | Invariant under x→a·x+b? | Max rel. change | Reason |
|---|---|---|---|
| `stat_mean` | **NO** | 3.3e+00 | → a·mean+b |
| `stat_std` | **NO** | 3.0e+00 | → a·std |
| `stat_var` | **NO** | 1.5e+01 | → a²·var |
| `stat_rms` | **NO** | 3.3e+00 | depends on a,b |
| `stat_ptp` | **NO** | 3.0e+00 | → a·ptp |
| `stat_shannon_entropy` | **NO** | 1.5e-03 | value-histogram binning is numerically **not** shift/scale stable — **excluded** (do not assume entropy safe) |
| `hjorth_activity` | **NO** | 1.5e+01 | = a²·var |
| `band_abs_{delta,theta,alpha,beta}` | **NO** | 1.5e+01 | PSD scales by a² |
| `stat_skew` | **YES** | 1.1e-11 | standardized 3rd moment (a>0) |
| `stat_kurtosis` | **YES** | 3.9e-11 | standardized 4th moment |
| `hjorth_mobility` | **YES** | 1.1e-14 | √(var(Δx)/var(x)); a² cancels, Δ removes b |
| `hjorth_complexity` | **YES** | 2.3e-13 | ratio of difference-variances |
| `spectral_entropy` | **YES** | 1.5e-14 | entropy of **normalized** PSD; `detrend='constant'` removes offset b |
| `band_rel_{delta,theta,alpha,beta}` | **YES** | <1e-13 | band/total power → a² cancels; detrend removes b |
| `ratio_{theta_alpha,beta_alpha,theta_beta}` | **YES** | <1e-12 | ratio of bandpowers → a² cancels |

(Gamma already excluded by the no-gamma set.)

## 5. Frozen feature subsets

- **Within-STEW (pipeline A):** all **184** `no_gamma_184` features.
- **MAT→STEW transport (pipeline B):** the **96** invariant features
  (12 templates × 8 channels):
  `skew, kurtosis, hjorth_mobility, hjorth_complexity, spectral_entropy,
  band_rel_{delta,theta,alpha,beta}, ratio_{theta_alpha,beta_alpha,theta_beta}`.
- **Excluded from transport (88):** `stat_mean/std/var/rms/ptp`,
  `stat_shannon_entropy`, `hjorth_activity`, `band_abs_{delta,theta,alpha,beta}`.

The exact lists are frozen in
`results/stew_sensitivity/transport_compatible_feature_spec.yaml`.

## 6. Honesty constraints

- The 128 Hz MAT re-extraction + transport subset is a **new exploratory
  transport analysis**, not a replacement for the original 500 Hz MAT validation
  (`mean-subtraction macro subject AUC = 0.880102`, permutation p = 0.004975).
- Transport uses scale-invariant features **because** STEW units are
  undocumented; this is a limitation, not a strength.
- STEW remains non-comparable and exploratory regardless of outcome. This audit
  was frozen before any transfer result was viewed.
