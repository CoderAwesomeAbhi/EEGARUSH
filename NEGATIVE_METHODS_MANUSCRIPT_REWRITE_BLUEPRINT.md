# Negative-Methods Manuscript Rewrite Blueprint

Gate: integrity audit passed (`FINAL_RESULTS_VALID_UNLOCK_NEGATIVE_MANUSCRIPT_REWRITE`)
and claims bound by `FINAL_NEGATIVE_PAPER_CLAIM_LEDGER.md`. **This is a planning
document only — `main.tex` is NOT edited in this task.** The current
`paper/tex/main.tex` is built on a stale z-scoring-centric + DS007262-confirmatory
narrative and must be reframed as a rigorous negative transport/reproducibility study.

---

## 1. Final title options

1. *"Strong Within-Domain EEG Workload Decoding Does Not Transfer Across Devices and
   Tasks: A Preregistered Negative Test of Baseline-Relative Calibration."*
2. *"When a Promising Rescue Fails Its One Shot: A Frozen Prospective Test of
   Baseline-Relative EEG Workload Transfer (MAT → STEW → COG-BCI)."*
3. *"Baseline Calibration Improves Within-Dataset EEG Workload Decoding but Fails
   Cross-Domain Transport: A Single-Shot Prospective Reproducibility Study."*

Recommended: option 3 (states the within/across contrast and the prospective
discipline without overclaiming).

## 2. Revised central question

> Does subtracting or standardizing each subject's resting EEG baseline produce a
> workload-decoding signal that **transfers** across provenance-distinct EEG datasets
> (different devices and tasks) — or only within each dataset?

Answer delivered: within-dataset yes; cross-domain transport no, including a
predeclared one-shot prospective test of the post-hoc z-scoring rescue.

## 3. Abstract claim boundaries

- State within-MAT (0.880) and within-STEW (0.840) above-chance subject-level
  decoding with permutation p ≈ 0.005 — labeled within-dataset.
- State the predeclared MAT→STEW mean-subtraction **transport failure** (0.448).
- State that z-scoring was a **post-hoc** STEW diagnostic (0.683) → a hypothesis.
- State the **single** frozen prospective COG-BCI test and its **below-chance
  failure** (z 0.436; ΔAUC vs mean-subtraction CI includes zero).
- Closing sentence = the locked story (§Required Manuscript Story in the ledger).
- **Forbidden in abstract:** any "replicates", "transfers", "z-scoring works",
  "universal axis", clinical, or mechanism language.

## 4. Introduction story

1. Workload decoding is easy within a dataset but rarely tested for genuine transfer.
2. Baseline-relative calibration (mean subtraction; z-scoring) is a plausible route
   to subject/device-invariant features — motivation, not a finding.
3. We test it with preregistration-style freezing across three provenance-distinct
   datasets (MAT µV EDF; STEW undocumented Emotiv counts; COG-BCI BrainProducts).
4. Frame the paper as a **negative transport/reproducibility** contribution: honest
   single-shot prospective testing of a post-hoc rescue, with full leakage controls.
5. Remove all prior framing that positioned z-scoring as the headline method or
   DS007262 as the confirmatory test.

## 5. Methods section structure

1. **Datasets & provenance** — MAT (36 subj, 500 Hz, µV), STEW (48 subj, 128 Hz,
   undocumented counts), COG-BCI (29 subj, ses-S1, 500 Hz, µV). Roles: MAT+STEW =
   development; COG-BCI = untouched prospective test.
2. **Frozen feature space** — no-gamma 184 within-dataset; the **96**
   scale/offset-invariant transport features for cross-device work; explain the
   unit-invariance audit. **Gamma removed.**
3. **Sampling representation** — harmonized **128 Hz** via
   `resample_poly(up=32, down=125)` for MAT and COG-BCI; cite the amendment.
4. **Calibration modes** — absolute, mean subtraction (candidate), z-scoring
   (secondary diagnostic); per-subject stats from **unlabeled** baseline only.
5. **Model & evaluation** — fixed L2 logistic (C=1.0, liblinear, balanced,
   max_iter=5000); macro subject-level ROC-AUC; LOSO within-dataset; MAT-only fit for
   transport; subject-bootstrap CIs; within-subject label-permutation nulls.
6. **Preregistration / freezing & single-shot discipline** — protocol freezes,
   checksum-locked one-shot execution, prohibition on post-outcome changes.
7. **Leakage controls** — explicit statement (MAT-only fitting; unlabeled-baseline
   calibration; disjoint calibration/scored segments).

## 6. Results section order

1. **Within-MAT** subject-level decoding (mean subtraction 0.880; permutation
   p = 0.004975); note mean-sub not superior to absolute (CI crosses zero).
2. **Within-STEW** exploratory sensitivity (mean subtraction 0.840; p = 0.004975);
   same non-superiority caveat; z-scoring highest but diagnostic only.
3. **MAT→STEW transport FAILURE** (mean subtraction 0.448 below chance; not better
   than absolute) — the central negative pivot.
4. **Post-hoc z-scoring diagnostic** on STEW (0.683) framed as hypothesis-generating
   (cite the validity audit).
5. **Frozen one-shot COG-BCI prospective test** — primary endpoint table (z 0.436,
   mean-sub 0.392, absolute 0.368; all below chance) and primary ΔAUC
   [−0.076, +0.175]; classification = failure.
6. Integrity/reproducibility summary (single run, checksum match).

## 7. Required figures / tables

- **T1** Dataset/provenance/role table (device, units, sfreq, n, condition mapping).
- **T2** Within-dataset subject-level AUCs (MAT, STEW) × calibration mode + permutation.
- **T3** MAT→STEW transport AUCs × mode + paired deltas.
- **T4** COG-BCI one-shot: per-method macro subject AUC + subject CI; primary &
  secondary paired ΔAUC. (From `results/cog_bci_one_shot/` summaries.)
- **F1** Pipeline/decision flow: development (MAT, STEW) → post-hoc hypothesis →
  frozen prospective COG-BCI → failure.
- **F2** Bar/forest plot of macro subject AUC with chance line at 0.5 across
  within-MAT, within-STEW, MAT→STEW, MAT→COG-BCI (shows above-chance within,
  below-chance across).
- Optional **F3** within-subject permutation null vs observed (MAT, STEW).
- **Remove** old z-score-centric and DS007262 figures/tables.

## 8. Discussion interpretation

- Within-domain success ≠ transfer; high LOSO AUC is not evidence of device/task
  generalization.
- Mean subtraction and z-scoring both fail to confer cross-domain transport under the
  tested shifts; the post-hoc z-scoring lead did not survive a single honest
  prospective test.
- Value is methodological: preregistration, leakage control, unit-invariant features,
  harmonized sampling, and single-shot prospective testing of a post-hoc rescue.
- Explicitly state what is **not** claimed (mirror the forbidden list).

## 9. Limitations

- Two development datasets; one prospective dataset (one shot by design).
- STEW units undocumented → invariant-feature restriction (limitation, not strength).
- Cross-domain shift is large (device + task simultaneously); cannot separate the two.
- Below-chance transport may reflect anti-aligned decision geometry; not interpreted
  mechanistically.
- Primary paradigm restricted to MATB-difficult vs eyes-open rest; N-back, other MATB
  levels, eyes-closed rest, and COG-BCI S2/S3 were prespecified out of scope.
- No permutation was computed for the COG-BCI run (frozen script scope); not needed
  given below-chance primary endpoint.

## 10. What must be REMOVED from `main.tex`

Current `paper/tex/main.tex` (487 lines) requires:

- **Title (l.27)** replaced (drop "Replicates in Exploratory Datasets").
- **Abstract (l.43–67)** rewritten: remove z-scored-model headline AUCs as the
  positive result; remove DS007262 graded-transfer framing; insert MAT/STEW
  mean-subtraction within-domain + transport-failure + COG-BCI one-shot failure.
- **Methods:** "Frozen Feature Space" (l.126) — remove **gamma** bands and the
  "z-scored 200-feature" primary framing; "Calibration-Duration Curve" (l.156),
  "Effect-Direction Stability" (l.165) — keep only if reframed as within-dataset
  caveats; **delete "Frozen DS007262 Confirmatory Test" (l.185)** entirely.
- **Results:** "Baseline-Relative Calibration Improved Exploratory Decoding"
  (l.209) and the z-score-primary tables (l.230–235, 243–245, 264–316) — replace
  with mean-subtraction-primary within-domain results; **delete "Frozen DS007262
  Graded Transfer Failed" (l.320)** and substitute the MAT→STEW transport failure +
  COG-BCI one-shot prospective failure.
- **Discussion:** "Interpretation of the Baseline-Relative Workload Transfer
  Hypothesis" (l.398) and "Resting Phenotyping" (l.416) — remove
  transfer-optimistic and phenotyping speculation; align to negative conclusion.
- **Conclusion (l.441)** rewritten to the locked story.
- **Data/Code Availability (l.453)** updated to cite COG-BCI freeze/run artifacts.
- Global: purge every residual claim on the forbidden list (universal/invariant axis,
  z-scoring validation, clinical, PAC/gamma mechanism, source localization,
  cross-dataset transport success, DS007262).

## 11. Frontiers-style positioning (no acceptance-likelihood claim)

- Target a Frontiers methods/registered-report-adjacent venue (e.g., Frontiers in
  Neuroscience / Human Neuroscience, Methods or Brain-Computer Interfaces section).
- Position as a **rigorous negative transport/reproducibility** study: preregistered
  freezes, leakage-controlled pipeline, unit-invariant transport features,
  single-shot prospective test of a post-hoc rescue, full checksum-verified
  provenance.
- Emphasize that informative negative results and honest prospective failure are
  contributions to reproducible BCI methodology.
- **Do not** assert or estimate acceptance likelihood anywhere.

---

**Next task (separate, explicitly unlocked):** apply this blueprint to
`paper/tex/main.tex` under the claim ledger. Not performed here.
