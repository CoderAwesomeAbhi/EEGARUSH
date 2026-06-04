# Final Negative-Paper Claim Ledger

Gate: `FINAL_RESULT_INTEGRITY_AUDIT.md` →
`FINAL_RESULTS_VALID_UNLOCK_NEGATIVE_MANUSCRIPT_REWRITE`. This ledger binds every
manuscript claim to executed analyses with exact result-file provenance. Anything
not listed under "Allowed" or "Allowed With Caution" is forbidden.

---

## 1. Claims Allowed

Each claim below is directly supported by an executed, committed analysis.

1. **Within-MAT, mean-subtraction calibration decodes rest vs mental arithmetic
   above chance at the subject level.** Macro subject-level ROC-AUC = **0.880102**;
   full-pipeline within-subject label-permutation null mean 0.500600, 95%
   [0.441057, 0.553729], **empirical p = 0.004975** (200 perms).
   *Provenance:* `MAT_MEAN_SUBTRACTION_SUBJECT_LEVEL_RESULTS.md`,
   `results/raw_rebuilt/mat_subject_level_metrics.csv`,
   `MAT_MACRO_SUBJECT_NULL_RESULTS.md`,
   `results/raw_rebuilt/mat_macro_subject_full_pipeline_permutation.csv`.

2. **Within-STEW, mean-subtraction calibration decodes low vs high workload above
   chance at the subject level.** Macro subject-level ROC-AUC = **0.839498**;
   permutation null mean 0.502071, 95% [0.458830, 0.539908], **p = 0.004975**.
   *Provenance:* `STEW_EXPLORATORY_SENSITIVITY_RESULTS.md`,
   `results/stew_sensitivity/stew_within_metrics.csv`,
   `stew_permutation_summary.csv`. (Label: exploratory cross-task/device sensitivity,
   same-device train/test.)

3. **The predeclared MAT→STEW mean-subtraction transport failed (below chance).**
   Macro subject AUC = **0.447598** (absolute 0.472045); paired
   mean-subtraction−absolute 95% CI [−0.070057, +0.022646] includes zero.
   *Provenance:* `MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`,
   `results/stew_sensitivity/mat_to_stew_transfer_metrics.csv`,
   `mat_to_stew_transfer_bootstrap.csv`.

4. **A z-scoring transport advantage was observed only as a post-hoc secondary
   diagnostic on MAT→STEW (hypothesis-generating, not confirmatory).** MAT→STEW
   z-scoring macro subject AUC = **0.682823**; computation verified leakage-free and
   predeclared, but its *elevation as a method of interest* was post-hoc.
   *Provenance:* `MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`,
   `POSTHOC_ZSCORE_TRANSPORT_AUDIT.md`
   (`ZSCORE_OBSERVATION_VALID_FOR_NEW_HYPOTHESIS_GENERATION`).

5. **The z-scoring hypothesis was frozen and tested exactly once on the untouched
   COG-BCI dataset.** Single run marker; frozen config/script checksums match run
   materials; all degrees of freedom locked before outcomes were seen.
   *Provenance:* `REVISED_ZSCORE_PROSPECTIVE_EVALUATION_PROTOCOL.md`,
   `COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md`,
   `COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md`,
   `COG_BCI_EXECUTABLE_CONFIG.yaml`, `results/cog_bci_one_shot/RUN_MARKER.json`,
   `config_and_script_checksums.json`, `FINAL_RESULT_INTEGRITY_AUDIT.md`.

6. **Prospective MAT→COG-BCI z-scoring transport failed below chance.** z-scoring
   macro subject AUC = **0.435961** (below 0.5); mean subtraction 0.392153; absolute
   0.367875. Primary paired ΔAUC (z − mean-subtraction) = +0.043807, 95% CI
   **[−0.076359, +0.175233]** (includes zero). All three modes below chance.
   *Provenance:* `COG_BCI_ONE_SHOT_PROSPECTIVE_RESULTS.md`,
   `results/cog_bci_one_shot/cog_bci_primary_results.json`,
   `cog_bci_one_shot_summary.csv`, `cog_bci_paired_comparisons_summary.csv`.

7. **Therefore: baseline-relative calibration supports within-domain workload
   discrimination but, in these datasets, neither mean subtraction nor the
   prospectively tested z-scoring provides supported cross-domain transport.**
   *Provenance:* claims 1–6 jointly; `COG_BCI_FINAL_PROJECT_DECISION.md`
   (`PROSPECTIVE_ZSCORE_TRANSPORT_FAILED_WRITE_NEGATIVE_METHODS_PAPER`).

8. **Methodological-rigor claims (process, not outcome):** preregistration-style
   freezing, leakage controls (MAT-only fitting; unlabeled-baseline calibration),
   scale/offset-invariant transport feature set, harmonized 128 Hz representation,
   single-shot prospective discipline, and checksum-locked execution.
   *Provenance:* `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`,
   `results/stew_sensitivity/transport_compatible_feature_spec.yaml`,
   the COG-BCI freeze/amendment/checklist docs, `FINAL_RESULT_INTEGRITY_AUDIT.md`.

## 2. Claims Allowed Only With Caution

Permitted **only** with explicit hedging wording ("in these datasets", "under the
tested domain shifts", "exploratory", "prospective negative result",
"hypothesis-generating").

- **C1.** "z-scoring removes per-subject multiplicative scale and was the only mode
  above chance in the MAT→STEW *exploratory* diagnostic" — must be framed as an
  exploratory, mechanistic-sounding *observation that did not survive prospective
  test*, never as evidence z-scoring works. (`MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`.)
- **C2.** "Within-domain decoding was strong (AUC ≈ 0.84–0.88)" — must be qualified
  as **within-dataset only** (MAT and STEW separately), not cross-dataset.
- **C3.** "Mean subtraction was not proven superior to absolute features" — true
  within MAT and within STEW (paired CIs cross zero); state as a null, not a win.
  (`MAT_MEAN_SUBTRACTION_SUBJECT_LEVEL_RESULTS.md`, `STEW_EXPLORATORY_SENSITIVITY_RESULTS.md`.)
- **C4.** "STEW is non-comparable / undocumented units" — caution about device-unit
  ambiguity; frame as a limitation motivating invariant features, not a result.
  (`TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`.)
- **C5.** Any generality statement must say "under the specific MAT→STEW and
  MAT→COG-BCI device/task shifts tested here", never "EEG workload transfer in
  general".
- **C6.** N-back, eyes-closed rest, MATB easy/medium, and COG-BCI sessions S2/S3 may
  be named **only** as *prespecified-excluded / not-analyzed* scope boundaries —
  **no** results from them may be reported or implied.

## 3. Claims Forbidden

The following are explicitly prohibited anywhere in the manuscript:

- ❌ an invariant / universal / shared workload axis or dimension;
- ❌ cross-dataset / cross-device / cross-task transport **success** (it failed);
- ❌ z-scoring **validation**, confirmation, replication, or "promising/works" framing
  of the COG-BCI outcome;
- ❌ any claim that the **COG-BCI result was positive**, partial-positive, or
  trending-positive (z-scoring was below chance);
- ❌ clinical deployment, clinical readiness, or real-world BCI utility;
- ❌ mechanistic neuroscience conclusions inferred from feature behavior
  (e.g., physiological interpretation of band ratios / Hjorth / spectral entropy);
- ❌ source-localization or anatomical-generator claims;
- ❌ PAC (phase-amplitude coupling) or gamma-mechanism claims;
- ❌ superiority of mean subtraction over absolute features (CIs include zero);
- ❌ generalizable transfer "with more data/tuning" speculation presented as result;
- ❌ DS007262-based confirmation or any revived old invariant-axis / z-scoring-centered
  narrative.

## 4. Required Manuscript Story (locked)

> **Strong within-domain decoding did not translate into cross-domain transport; a
> promising post-hoc z-scoring rescue failed in a predeclared, untouched, one-shot
> prospective test.**

Concretely: mean-subtraction calibration decodes workload above chance **within**
MAT (0.880) and **within** STEW (0.840), but the predeclared MAT→STEW
mean-subtraction **transport failed** (0.448, below chance). A post-hoc z-scoring
advantage on STEW (0.683) generated a single new hypothesis, which was frozen and
tested **once** on untouched COG-BCI and **failed below chance** (0.436; ΔAUC vs
mean-subtraction CI includes zero). Conclusion: baseline-relative calibration aids
**within-domain** discrimination but does **not** deliver supported **cross-domain**
transport under the tested EEG domain shifts. This is reported as a rigorous negative
transport/reproducibility result.
