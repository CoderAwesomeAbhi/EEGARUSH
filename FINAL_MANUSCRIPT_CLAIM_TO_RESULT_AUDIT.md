# Final Manuscript Claim-to-Result Audit

Maps every major abstract / results / conclusion statement in
`paper/tex/main.tex` (negative-methods rewrite) to committed result evidence,
and explicitly confirms that all forbidden claims are absent. Gate:
`FINAL_RESULT_INTEGRITY_AUDIT.md` → `FINAL_RESULTS_VALID_UNLOCK_NEGATIVE_MANUSCRIPT_REWRITE`.
Bound by `FINAL_NEGATIVE_PAPER_CLAIM_LEDGER.md`. No model was rerun; all values
are read from committed result files.

---

## 1. Claim → evidence map

| # | Manuscript statement (abstract / results / conclusion) | Value(s) | Committed source |
|---|---|---|---|
| 1 | Within-MAT mean-subtraction decodes rest vs. arithmetic above chance | macro subject AUC **0.880102**; perm null mean 0.500600, 95% [0.441057, 0.553729], **p = 0.004975** (200 perms) | `MAT_MEAN_SUBTRACTION_SUBJECT_LEVEL_RESULTS.md`; `results/raw_rebuilt/mat_subject_level_metrics.csv`; `MAT_MACRO_SUBJECT_NULL_RESULTS.md`; `results/raw_rebuilt/mat_macro_subject_full_pipeline_permutation.csv` |
| 2 | MAT mean subtraction NOT superior to absolute | absolute 0.841553; paired mean-sub − absolute 95% CI **[−0.018566, +0.093112]** (incl. 0) | `MAT_MEAN_SUBTRACTION_SUBJECT_LEVEL_RESULTS.md` (paired bootstrap table) |
| 3 | Within-STEW mean-subtraction decodes low vs. high workload above chance | macro subject AUC **0.839498**; perm null mean 0.502071, 95% [0.458830, 0.539908], **p = 0.004975** | `STEW_EXPLORATORY_SENSITIVITY_RESULTS.md`; `results/stew_sensitivity/stew_within_metrics.csv`, `stew_permutation_summary.csv` |
| 4 | STEW mean subtraction NOT superior to absolute; z highest but diagnostic | absolute 0.815795; z 0.898703; paired mean-sub − absolute 95% CI **[−0.013180, +0.060717]** (incl. 0) | `STEW_EXPLORATORY_SENSITIVITY_RESULTS.md`; `results/stew_sensitivity/stew_within_metrics.csv`, `stew_paired_bootstrap.csv` |
| 5 | STEW is exploratory / non-comparable (task, device, montage, units, sfreq) | — (qualitative) | `STEW_SENSITIVITY_AND_TRANSFER_DECISION.md` ("Required statements"); `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md` |
| 6 | MAT→STEW mean-subtraction transport FAILS (below chance), not better than absolute | mean-sub **0.447598**; absolute 0.472045; paired CI **[−0.070057, +0.022646]** (incl. 0) | `MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`; `results/stew_sensitivity/mat_to_stew_transfer_metrics.csv`, `mat_to_stew_transfer_bootstrap.csv` |
| 7 | z-scoring only above-chance transport mode = post-hoc, hypothesis-generating | MAT→STEW z **0.682823**; clean/leakage-free but post-hoc elevation | `MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`; `POSTHOC_ZSCORE_TRANSPORT_AUDIT.md` (`ZSCORE_OBSERVATION_VALID_FOR_NEW_HYPOTHESIS_GENERATION`) |
| 8 | Hypothesis frozen + tested exactly once on untouched COG-BCI (checksum-locked) | single run marker; config/script SHA-256 match across frozen/run/live | `results/cog_bci_one_shot/RUN_MARKER.json`, `config_and_script_checksums.json`; `FINAL_RESULT_INTEGRITY_AUDIT.md` (Checks 1, 3) |
| 9 | COG-BCI protocol: ses-S1, RS_Beg_EO calib / RS_End_EO rest / MATBdiff task; 128 Hz both | — (protocol) | `COG_BCI_ONE_SHOT_PROSPECTIVE_RESULTS.md`; `COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md` |
| 10 | COG-BCI prospective FAILS below chance, all three modes | z **0.435961** (CI [0.359602, 0.509338]), mean-sub 0.392153, absolute 0.367875; n = 29 | `COG_BCI_ONE_SHOT_PROSPECTIVE_RESULTS.md`; `results/cog_bci_one_shot/cog_bci_one_shot_summary.csv`, `cog_bci_primary_results.json` |
| 11 | Primary ΔAUC (z − mean-sub) CI includes zero | mean **+0.043807**, 95% CI **[−0.076359, +0.175233]** | `results/cog_bci_one_shot/cog_bci_paired_comparisons_summary.csv`, `cog_bci_primary_results.json` |
| 12 | Secondary ΔAUC (z − absolute) CI includes zero | mean +0.068086, 95% CI [−0.060173, +0.190715] | `results/cog_bci_one_shot/cog_bci_paired_comparisons_summary.csv` |
| 13 | Final verdict / negative conclusion | `PROSPECTIVE_ZSCORE_TRANSPORT_FAILED_WRITE_NEGATIVE_METHODS_PAPER` | `COG_BCI_FINAL_PROJECT_DECISION.md` |
| 14 | Process/rigor claims (freezing, leakage control, invariant features, 128 Hz, run-once, checksums) | — (process) | `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`; `results/stew_sensitivity/transport_compatible_feature_spec.yaml`; `FINAL_RESULT_INTEGRITY_AUDIT.md` |
| 15 | Datasets/provenance (MAT 36/72 EDF/500 Hz/µV; STEW 48/128 Hz/counts; COG-BCI 29/ses-S1/500→128 Hz) | — (provenance) | `docs/EEGARUSH_CHECKPOINT.md`; `results/cog_bci_provenance/cog_bci_subject_session_condition_summary.csv` |

Every numeric value above was machine-verified against its source CSV/JSON in
`FINAL_RESULT_INTEGRITY_AUDIT.md` (Checks 2, 5). The manuscript reproduces these
values exactly; no value in the paper lacks a committed source.

## 2. Tables / figures → source

| Artifact | Content | Source |
|---|---|---|
| Table 1 | datasets / role / task / baseline / device-sampling / n | checkpoint + COG-BCI provenance |
| Table 2 | within-domain AUCs (MAT, STEW) + perm p | rows 1–4 above |
| Table 3 | MAT→STEW transport AUCs | row 6–7 above |
| Table 4 | COG-BCI one-shot AUCs + primary/secondary ΔAUC CIs | rows 10–12 above |
| Fig 1 | sequential design / hypothesis-control flow | composed from rows 1, 3, 6, 7, 10–11 |
| Fig 2 | within-domain AUC bars | `results/raw_rebuilt/mat_subject_level_metrics.csv`, `results/stew_sensitivity/stew_within_metrics.csv` |
| Fig 3 | MAT→STEW transport AUC bars | `results/stew_sensitivity/mat_to_stew_transfer_metrics.csv` |
| Fig 4 | COG-BCI AUC bars + primary ΔAUC CI | `results/cog_bci_one_shot/cog_bci_one_shot_summary.csv`, `cog_bci_paired_comparisons_summary.csv` |

Figures are generated by `scripts/make_negative_manuscript_figures.py`, which
reads only the committed summary files above (no model, no raw data).

## 3. Forbidden-claim absence check

Source-text sweep of `paper/tex/main.tex` and `paper/tex/references.tex`
confirms each forbidden claim is **ABSENT**:

| Forbidden claim | Status | Note |
|---|---|---|
| invariant / universal / shared workload axis | ABSENT | "invariant" appears only as "scale/offset-invariant features" (a transport feature property), never as a workload axis |
| cross-dataset/device/task transport **success** | ABSENT | transport explicitly reported as failed (0.448, 0.436) |
| z-scoring validation / confirmation / replication / "works" | ABSENT | z-scoring framed as post-hoc hypothesis that failed prospectively; "replicat*" used only as "not a strict replication" / "did not replicate" |
| COG-BCI positive / partial / trending-positive | ABSENT | reported below chance, classification failure |
| clinical / deployment / diagnostic / real-time readiness | ABSENT | "clinical" appears only in explicit disclaimers (Limitations, Ethics) |
| mechanistic neuroscience from feature behavior | ABSENT | "we make no neural-mechanism claim" |
| source localization / anatomical generators | ABSENT | none present |
| PAC / phase-amplitude coupling | ABSENT | none present |
| gamma-mechanism | ABSENT | "gamma" appears only as "no-gamma … gamma bands were removed" |
| superiority of mean subtraction over absolute | ABSENT | reported as null (paired CIs include zero) |
| "with more data/tuning it would transfer" speculation | ABSENT | none present |
| DS007262 confirmation / old invariant-axis / z-centric narrative | ABSENT | DS007262 fully removed from manuscript |
| publication-acceptance-likelihood language | ABSENT | none present |

## 4. Verdict

Every major manuscript claim maps to a committed, executed result with exact
provenance; all values match the integrity-audited sources; all forbidden
claims are absent. The manuscript is consistent with the locked negative story.
