# COG-BCI Executable Freeze Decision

**No-results freeze.** No AUC, model prediction, separability statistic,
permutation statistic, feature importance, bootstrap interval, or any
outcome-dependent result was computed in this task. No COG-BCI raw/source files
are committed (all live under git-ignored `data/raw/`).

This decision finalizes whether the COG-BCI one-shot prospective z-scoring
transport test is cleared for its single execution. It is based on:

- `COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md`
- `REVISED_ZSCORE_PROSPECTIVE_EVALUATION_PROTOCOL.md`
- `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`
- `results/stew_sensitivity/transport_compatible_feature_spec.yaml`
- `COG_BCI_SOURCE_PROVENANCE_AUDIT.md` / `COG_BCI_SOURCE_PROVENANCE_DECISION.md`
- the four `results/cog_bci_provenance/` CSVs.

---

## 1. What is unambiguous and ready

The provenance audit (`COG_BCI_SOURCE_VALID_FREEZE_EXECUTABLE_CONFIG_BEFORE_ONE_SHOT_TEST`)
established, and this task confirmed, that the following are fully determined and
satisfiable without changing any locked rule:

| Element | Frozen value | Status |
|---|---|---|
| Dataset role | COG-BCI = single untouched prospective test | OK |
| Subjects | all 29 (`sub-01`…`sub-29`); no pre-outcome exclusions | OK |
| Session | `ses-S1` only (first labeled session, non-performance rule) | OK |
| Calibration recording | `RS_Beg_EO` (eyes-open) | OK |
| Scored-rest (label 0) | `RS_End_EO` (eyes-open) | OK |
| Scored-task (label 1) | `MATBdiff` (MATB difficult) | OK |
| Channels | 8 locked F3,F4,F7,F8,O1,O2,T7,T8 → MAT …,T3,T4 | present in all checked recordings |
| Segment rule | first 30 s each; 4 s windows; 50 % overlap | OK |
| Window counts | **14 calibration / 14 scored-rest / 14 scored-task** for **all 29 subjects** | verified (sampling-rate-independent) |
| Features | frozen 96 transport-invariant | OK |
| Model | L2 logistic, `C=1.0`, `liblinear`, `class_weight=balanced`, `max_iter=5000`, fit on MAT only | OK |
| Endpoint | macro subject-level ROC-AUC | OK |
| Primary comparison | paired subject-bootstrap ΔAUC (z-scoring − mean subtraction) | OK |
| Success rule | above chance AND positive paired CI excluding zero | OK |
| Units | physical µV; no STEW-style unit ambiguity | OK |
| N-back | excluded from primary | OK |

Window counts were derived from the audited durations in
`results/cog_bci_provenance/cog_bci_locked_channel_verification.csv`
(every `RS_Beg_EO`/`RS_End_EO` ≈ 60 s, every `MATBdiff` ≈ 299 s, all ≥ 30 s) →
`floor((30−4)/2)+1 = 14` windows per 30 s segment, equal for scored-rest and
scored-task. See `results/cog_bci_provenance/cog_bci_primary_session_window_plan.csv`.

Window counts are **time-defined** (seconds), so they are identical (14/14/14)
regardless of which sampling rate the features are extracted at. The window plan
is therefore **not** the source of the block.

---

## 2. The blocking issue — sampling-representation protocol ambiguity

The executable-freeze task's **Critical Pipeline-Lock Issue** required that, before
any result, the most conservative *no-drift* sampling implementation be frozen, and
that **any conflict** between the conservative default and a committed protocol be
reported as a protocol ambiguity with the one-shot test **not run**.

Inspecting the committed documents reveals exactly such a conflict, on the single
most outcome-affecting implementation detail.

### 2.1 What the hypothesis-generating pipeline actually used: 128 Hz

The post-hoc z-scoring transport hypothesis, and the trained MAT model that the
one-shot test transports to COG-BCI, were produced by the frozen **MAT→STEW
transport pipeline**. Its frozen spec is explicit
(`results/stew_sensitivity/transport_compatible_feature_spec.yaml`, line 26):

```yaml
mat_resample: {method: scipy.signal.resample_poly, up: 32, down: 125,
               native_hz: 500, target_hz: 128.0}
```

`TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md` confirms this repeatedly:
"Resampling MAT 500 → 128 Hz fixes sampling-rate compatibility"; pipeline B
"MAT resampled to 128 Hz"; "The 128 Hz MAT re-extraction + transport subset."

So the trained imputer + scaler + L2 logistic model, and the 96 features it
expects, are defined at **128 Hz**.

### 2.2 Why sampling rate matters for these specific 96 features

The 96 transport features are scale/offset-invariant (`x → a·x + b`) — that is what
the compatibility audit verified. They are **not** sampling-rate-invariant. Of the
12 templates per channel, the following change systematically with sampling rate:

- `band_rel_{delta,theta,alpha,beta}` — PSD bin layout and band edges relative to
  Nyquist depend on `fs`;
- `spectral_entropy` — entropy of the normalized PSD, `fs`-dependent;
- `hjorth_mobility`, `hjorth_complexity` — built from first/second differences,
  whose variance ratios depend on the sampling interval;
- `ratio_{theta_alpha,beta_alpha,theta_beta}` — ratios of the above bandpowers.

A model fit on 128-Hz MAT features applied to features extracted at a different
rate sees a systematically shifted feature distribution. This is silent drift, and
it would directly bias the very ΔAUC the test exists to measure.

### 2.3 What the committed COG-BCI protocol says: 500 Hz native, no resample

`COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md` (Primary outcome section) states the
training pipeline is:

> "the already-frozen MAT transport-compatible source pipeline (MAT rest-vs-arithmetic
> eval rows; imputer + scaler + L2 logistic fit on **MAT only**; **500 Hz native — no
> resample needed since COG-BCI is also 500 Hz**)."

The same "500 Hz, no MAT resample needed" claim is repeated in
`COG_BCI_SOURCE_PROVENANCE_AUDIT.md` ("Sampling rate 500 Hz matches MAT (no MAT
resample needed for this target)") and carried into
`COG_BCI_SOURCE_PROVENANCE_DECISION.md`.

### 2.4 The conflict

These two committed, pre-data freezes cannot both be correct:

- The **frozen transport spec** (and the compatibility audit) define the
  hypothesis-generating pipeline at **128 Hz**.
- The **COG-BCI protocol/audit/decision** assert that the *same* "already-frozen
  transport pipeline" is **500 Hz native** and needs **no resample**.

The COG-BCI protocol appears to have conflated MAT's *native acquisition* rate
(500 Hz) with the rate at which the *transport pipeline* actually froze its features
and model (128 Hz). Because COG-BCI is also natively 500 Hz, "no resample needed"
*looks* harmless — but it silently redefines the prospective pipeline to a sampling
representation that the trained model was never fit on (or forces a new native-500-Hz
MAT model, which is itself a different, un-frozen pipeline).

Either branch is drift:

1. **Use the frozen 128-Hz MAT model** → COG-BCI **must** be resampled 500→128 Hz
   (the conservative no-drift default). This contradicts the committed protocol's
   "no resample / 500 Hz native."
2. **Honor the protocol's native-500-Hz wording** → requires a *new* native-500-Hz
   MAT model + 500-Hz feature extraction, which is not the pipeline that generated
   the z-scoring hypothesis and is not "already-frozen."

This is a genuine, material protocol ambiguity, not a cosmetic config detail.

---

## 3. Why I do not resolve it unilaterally

The task instructed: adopt the conservative no-drift default (resample COG-BCI to
128 Hz) **only** absent conflict; if the committed protocol explicitly mandates a
native-500-Hz pipeline, do not silently switch; **and if there is any conflict,
report it as a protocol ambiguity and do not run.**

There **is** a conflict. Adopting the 128-Hz default here would override the
committed COG-BCI protocol's explicit "500 Hz native, no resample" statement —
i.e., I would be changing a frozen, committed protocol *after* metadata were seen.
That is precisely the post-hoc degree-of-freedom the one-shot discipline forbids.
The honest action is to surface the conflict and stop, not to pick the branch I
think is scientifically correct and quietly proceed.

(Scientifically, branch 1 — resample COG-BCI to 128 Hz to match the frozen model —
is the correct no-drift resolution. But endorsing it must be done by an explicit,
documented protocol amendment, ideally with sign-off, not inside this freeze.)

---

## 4. Other freeze tasks completed (consistent with a blocked verdict)

- **Exact-input materialization & hashing (partial, honest):** the 10 locally
  present per-subject archives (`sub-01`…`sub-10`) were MD5-verified byte-for-byte
  against the Zenodo manifest, and their exact locked `ses-S1` inputs
  (`RS_Beg_EO`, `RS_End_EO`, `MATBdiff` × `.set`/`.fdt` = 60 files) were
  materialized locally and **SHA-256** hashed (exact-input hashing, not
  full-archive MD5). See
  `results/cog_bci_provenance/cog_bci_execution_input_hash_manifest.csv`.
- The remaining 19 subjects (`sub-11`…`sub-29`) were **not** downloaded in full:
  pulling ~19 GB for a test that cannot validly run as specified is not warranted.
  Their archive MD5s are on record; full local materialization + verification is a
  gated prerequisite for the eventual run, recorded as deferred.
- **Window plan** frozen for all 29 subjects (14/14/14).
- **Executable config** written as `COG_BCI_EXECUTABLE_CONFIG.yaml` with
  `status: BLOCKED_PROTOCOL_AMBIGUITY_DO_NOT_RUN`.
- **Run-once script** `scripts/run_cog_bci_one_shot_prospective.py` created but
  **not executed**; it refuses to run without `--execute-locked-one-shot`, aborts
  if any result output exists, and additionally aborts while the config status is
  `BLOCKED_*`.

---

## 5. What must happen before a run is ever cleared

1. **Resolve the sampling ambiguity by explicit amendment.** Reconcile
   `COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md` with the frozen 128-Hz transport
   spec — almost certainly by adopting: *resample COG-BCI 500→128 Hz with
   `scipy.signal.resample_poly(up=32, down=125)` before extracting the 96 features,
   matching the frozen MAT model.* Flip `sampling_pipeline.resolved` to `true` and
   the config `status` to ready only after that amendment is committed.
2. **Complete full local materialization** of all 29 subjects (download + MD5
   verify); the run must not depend on live remote range reads.
3. Re-run this freeze's dry validation; then run the one-shot test **exactly once**.

---

## VERDICT

```
COG_BCI_EXECUTABLE_CONFIG_BLOCKED_PROTOCOL_AMBIGUITY_DO_NOT_RUN
```

The locked scientific rules (baseline, task, channels, features, model, endpoint,
success rule, subjects, session, windows) are all satisfiable and frozen. The
**single unresolved item is the sampling representation**, where two committed
freezes conflict. Per the executable-freeze stopping rule, the one-shot prospective
test is **NOT** cleared to run until that ambiguity is resolved by an explicit
documented protocol amendment. No COG-BCI predictive metric was computed.
