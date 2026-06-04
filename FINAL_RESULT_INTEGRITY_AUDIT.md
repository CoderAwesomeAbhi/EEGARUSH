# Final Result Integrity Audit

**Read-only audit.** No model was run, no result was recomputed, no COG-BCI rerun,
no `main.tex` edit, no DS007262 analysis, no alternative COG-BCI task/session
inspected. This audit only reads committed result files (JSON/CSV/Markdown) and
verifies internal consistency before any manuscript rewrite.

Scope: the consumed one-shot COG-BCI prospective test and the development-evidence
MAT/STEW results that frame it.

---

## Check 1 — Exactly one prospective execution

`results/cog_bci_one_shot/RUN_MARKER.json`:
`{"executed_utc": "2026-06-04T19:26:22.982458+00:00", "single_one_shot": true}`.
One run marker, one execution. The frozen one-shot script aborts if any result
output already exists, preventing a second result-producing run. **PASS.**

## Check 2 — Result files report consistent values

| Quantity | `cog_bci_primary_results.json` | `cog_bci_one_shot_summary.csv` | `cog_bci_paired_comparisons_summary.csv` | `..._RESULTS.md` / `..._DECISION.md` |
|---|---|---|---|---|
| z-scoring macro subject AUC | 0.435961 | 0.435961 | — | 0.435961 |
| mean-subtraction macro subject AUC | 0.392153 | 0.392153 | — | 0.392153 |
| absolute macro subject AUC | 0.367875 | 0.367875 | — | 0.367875 |
| primary ΔAUC (z − mean-sub) mean | +0.043807 | — | +0.043807 | +0.043807 |
| primary ΔAUC 95% CI | [−0.076359, +0.175233] | — | [−0.076359, +0.175233] | [−0.076359, +0.175233] |

All JSON values reproduce exactly in both summary CSVs and in both markdown reports
(machine-checked to < 1e-6). n = 29 target subjects for every method. **PASS.**

## Check 3 — Frozen config/script checksums align with run materials

| Source | config SHA-256 | script SHA-256 |
|---|---|---|
| Frozen pre-run (`cog_bci_frozen_run_materials_checksums.json`) | `0273e308…c263c0` | `679d183f…a3ca2` |
| Run-time record (`results/cog_bci_one_shot/config_and_script_checksums.json`) | `0273e308…c263c0` | `679d183f…a3ca2` |
| Live committed files (recomputed now) | `0273e308…c263c0` | `679d183f…a3ca2` |

All three agree exactly. The config and script that ran are byte-identical to the
frozen, committed materials; neither was altered before, during, or after the run.
**PASS.**

## Check 4 — Classification follows the predeclared rule

Predeclared rule (`COG_BCI_EXECUTABLE_CONFIG.yaml`, `COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md`):
- full success = z-scoring above chance AND paired 95% CI (z − mean-sub) excludes zero positively;
- partial = z-scoring above chance but CI includes zero;
- failure = z-scoring at/below chance, or does not outperform mean subtraction.

Observed: z-scoring AUC 0.435961 < 0.5 (**below chance**) → above-chance precondition
not met; primary CI [−0.076359, +0.175233] includes zero. Derived classification =
**FAILURE**, which is exactly what `COG_BCI_FINAL_PROJECT_DECISION.md` records. **PASS.**

## Check 5 — MAT and STEW results consistent across reports and CSVs

| Result | Report value | CSV value | Source CSV |
|---|---|---|---|
| MAT mean-subtraction macro subject AUC | 0.880102 | 0.8801020408 | `results/raw_rebuilt/mat_subject_level_metrics.csv` |
| MAT null permutation observed / p | 0.880102 / 0.004975 | 0.880102 / 0.004975 | `mat_macro_subject_full_pipeline_permutation.csv`, `MAT_MACRO_SUBJECT_NULL_RESULTS.md` |
| STEW within mean-subtraction macro AUC | 0.839498 | 0.8394982993 | `results/stew_sensitivity/stew_within_metrics.csv` |
| STEW within permutation observed / p | 0.839498 / 0.004975 | 0.839498 / 0.004975 | `stew_permutation_summary.csv` |
| MAT→STEW transfer: absolute / mean-sub / zscore | 0.472045 / 0.447598 / 0.682823 | 0.472045 / 0.447598 / 0.682823 | `mat_to_stew_transfer_metrics.csv` |

Reported MAT/STEW values match their committed CSVs exactly. **PASS.**

## Check 6 — No legitimate "success" reading of the COG-BCI z-scoring result

z-scoring transport is **below chance** (0.435961) with a subject-bootstrap CI
[0.359602, 0.509338] that sits at/below 0.5; it does not significantly exceed mean
subtraction (primary CI includes zero) or absolute (secondary CI includes zero). All
three calibration modes transport below chance. There is **no** metric, comparison,
or subgroup in the committed outputs that supports interpreting the COG-BCI z-scoring
result as success or partial success. **PASS** (no success reading is available).

## Check 7 — No crash / metric mismatch / leakage / rerun / invalid execution

- **Crash:** the one-shot run completed and wrote all expected outputs
  (`cog_bci_primary_results.json`, `RUN_MARKER.json`,
  `config_and_script_checksums.json`); no partial/error artifact present.
- **Metric mismatch:** none — identical macro-subject-AUC + 2000× subject-bootstrap
  code path across all three methods (same as MAT/STEW), values consistent across files.
- **Leakage:** the frozen pipeline fits imputer/scaler/L2 on **MAT only**; COG-BCI
  workload labels are used only to score; per-subject z-scoring/mean-subtraction
  stats come only from the unlabeled `RS_Beg_EO` baseline (disjoint from scored
  segments). This matches the leakage-free design verified in
  `POSTHOC_ZSCORE_TRANSPORT_AUDIT.md`.
- **Rerun:** single run marker; output-exists guard prevents a second run.
- **Invalid execution:** all 29 subjects scored; pre-run gates (input hashes,
  config/script checksums, channel/condition/window availability) all passed before
  execution.

**PASS.**

*(Historical note, not a defect: `POSTHOC_ZSCORE_TRANSPORT_AUDIT.md` records a
`NameError` during STEW summary-JSON assembly that occurred **after** all STEW
transfer CSVs were written and did not affect any metric; it was fixed for
reproducibility. It does not concern the COG-BCI one-shot run, which completed
cleanly.)*

---

## Summary

All seven integrity checks pass. The one-shot COG-BCI prospective test executed
exactly once on byte-identical frozen materials; its results are internally
consistent across JSON/CSV/Markdown; the FAILURE classification follows the
predeclared rule; the MAT/STEW development results are consistent with their CSVs;
and there is no available reading of the COG-BCI z-scoring result as success and no
sign of crash, leakage, mismatch, rerun, or invalid execution.

## VERDICT

```
FINAL_RESULTS_VALID_UNLOCK_NEGATIVE_MANUSCRIPT_REWRITE
```
