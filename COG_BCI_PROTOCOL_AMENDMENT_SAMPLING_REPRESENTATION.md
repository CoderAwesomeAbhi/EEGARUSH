# COG-BCI Protocol Amendment — Sampling Representation

**Type:** Pre-outcome protocol correction (internal pipeline-consistency only).
**Status:** Adopted **before** any COG-BCI predictive metric was computed or viewed.
No AUC, prediction, confidence interval, permutation, correlation, feature
importance, or any outcome-dependent statistic exists for COG-BCI.

This amendment resolves the single blocking ambiguity identified in
`COG_BCI_EXECUTABLE_FREEZE_DECISION.md`
(`COG_BCI_EXECUTABLE_CONFIG_BLOCKED_PROTOCOL_AMBIGUITY_DO_NOT_RUN`). It changes
**only** the sampling representation. It reopens **no** other degree of freedom.

---

## 1. The prior protocol contained an incorrect native-500-Hz statement

`COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md` (Primary outcome) previously stated
that the training pipeline was the "already-frozen MAT transport-compatible source
pipeline (... **500 Hz native — no resample needed since COG-BCI is also 500 Hz**)."
The same "500 Hz / no MAT resample needed" claim appeared in
`COG_BCI_SOURCE_PROVENANCE_AUDIT.md` and `COG_BCI_SOURCE_PROVENANCE_DECISION.md`.

That statement was **incorrect**. It conflated MAT's native acquisition rate
(500 Hz) with the rate at which the transport pipeline actually froze its features
and trained its model.

## 2. The conflict was discovered before any COG-BCI predictive metric

The discrepancy was found during the executable-freeze stage by comparing the
committed protocol against the authoritative frozen transport specification. No
COG-BCI signal was passed through any model; no predictive metric was produced.
Because nothing outcome-dependent has been computed, the representation can be
corrected prospectively without biasing the test.

## 3. The authoritative frozen representation is 128 Hz

The post-hoc z-scoring transport hypothesis, and the trained MAT source model that
the one-shot test transports to COG-BCI, were produced by the frozen **MAT→STEW
transport pipeline**, whose specification is explicit:

- `results/stew_sensitivity/transport_compatible_feature_spec.yaml`:
  `mat_resample: {method: scipy.signal.resample_poly, up: 32, down: 125,
  native_hz: 500, target_hz: 128.0}`
- `scripts/run_stew_sensitivity_and_transfer.py`: `TARGET_SFREQ = 128.0`;
  MAT loaded at 500 Hz then `resample_poly(data, up=32, down=125, axis=1)` **before**
  extracting the frozen 96 transport-compatible features.
- `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`: "Resampling MAT 500 → 128 Hz"; the
  "128 Hz MAT re-extraction + transport subset."

The 96 transport features are scale/offset-invariant (`x → a·x + b`) but they are
**not** sampling-rate-invariant: `band_rel_{delta,theta,alpha,beta}`,
`spectral_entropy`, `hjorth_mobility`, `hjorth_complexity`, and the band ratios all
change with sampling rate. A model trained on 128-Hz MAT features must therefore be
applied to target features extracted at the **same** 128 Hz.

### Correction (the only change this amendment makes)

The one-shot COG-BCI prospective test uses a single, identical sampling
representation for source and target:

- **Source MAT:** 500 Hz → 128 Hz via `scipy.signal.resample_poly`, `up=32`,
  `down=125` (deterministic, anti-aliased polyphase).
- **Target COG-BCI:** 500 Hz → 128 Hz via the **identical** deterministic
  anti-aliased resampling method (`scipy.signal.resample_poly`, `up=32`, `down=125`)
  **before** feature extraction.
- **Features:** the frozen **96** transport-compatible features only.

## 4. This amendment is not based on COG-BCI performance

The correction is justified solely by consistency with the already-frozen
hypothesis-generating transport pipeline. It does **not** depend on, and was made
without access to, any COG-BCI outcome. It does **not** reopen any other degree of
freedom: subjects (all 29), session (`ses-S1`), baseline (`RS_Beg_EO`), scored-rest
(`RS_End_EO`), scored-task (`MATBdiff`), channels (8), feature set (96), model
(L2 logistic, fixed parameters), endpoint (macro subject-level ROC-AUC), comparator
order (primary z-scoring vs mean subtraction; secondary vs absolute), window rule
(4 s, 50 % overlap, 14 rest + 14 task scored windows per subject), and the success
rule are all **unchanged**.

## 5. Native-500-Hz analysis is prohibited from the primary verdict

A native-500-Hz COG-BCI analysis is **forbidden** for the primary prospective test,
because it would change the data representation after the hypothesis was generated
and would apply the 128-Hz-trained model to a representation it was never fit on.
Native-500-Hz results may not influence the primary verdict in any way.

---

## Effect

With this amendment adopted, the only freeze blocker is resolved. The executable
config sampling rule is set to the corrected 128 Hz representation
(`sampling_pipeline.resolved: true`), and the freeze may proceed to a ready state
once all locked source inputs are local and hash-recorded (see
`results/cog_bci_provenance/cog_bci_execution_input_hash_manifest.csv`).
