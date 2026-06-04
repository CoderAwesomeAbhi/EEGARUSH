# Manuscript Changelog — Negative Reframe

Rewrite of `paper/tex/main.tex` from the stale z-scoring-centric +
DS007262-confirmatory narrative to a rigorous negative cross-domain transport /
reproducibility methods study. Authorized by
`FINAL_RESULTS_VALID_UNLOCK_NEGATIVE_MANUSCRIPT_REWRITE`; bound by
`FINAL_NEGATIVE_PAPER_CLAIM_LEDGER.md`; structured per
`NEGATIVE_METHODS_MANUSCRIPT_REWRITE_BLUEPRINT.md`. No new model, COG-BCI rerun,
new task/session/dataset, or rescue analysis was performed.

## Title

- **Old:** "Baseline-Relative EEG Workload Decoding Replicates in Exploratory
  Datasets but Fails a Frozen Graded External Transfer Test."
- **New:** "Strong Within-Domain EEG Workload Decoding Fails to Transport Across
  Heterogeneous Task and Device Domains: A Prospectively Evaluated Negative
  Methods Study."

## Authors

- Order set to **Arush Ravipati**, **Abhijay Gangarapu** (per task instruction);
  existing equal-contribution (`$^{\ast}$`) notation retained. No new
  affiliations, emails, funding, or IRB invented.

## Abstract

- Removed z-scored-model headline AUCs (0.865995 / 0.791896) as a positive
  result and the DS007262 graded-transfer framing.
- Inserted: within-MAT (0.880102) and within-STEW (0.839498) mean-subtraction
  decoding with p = 0.004975; MAT→STEW mean-subtraction transport failure
  (0.447598); post-hoc z-scoring (0.682823) as hypothesis only; COG-BCI one-shot
  prospective failure (z 0.435961, all modes below chance; primary ΔAUC CI
  [−0.076359, +0.175233] includes zero); negative transport conclusion.

## Introduction

- Reframed around reproducibility/transportability; within-dataset AUC ≠
  generalization. Baseline centering/standardization introduced only as candidate
  calibration methods, not mechanisms. Removed prior framing positioning z-scoring
  as the headline method and DS007262 as the confirmatory test.

## Methods

- Replaced single-stage "exploratory vs frozen DS007262" structure with the
  sequential hypothesis-control design (development MAT+STEW → post-hoc hypothesis
  → untouched prospective COG-BCI).
- Added raw-data provenance subsection (MAT EDF integrity; STEW unit ambiguity;
  COG-BCI source audit).
- **Frozen Feature Space:** removed gamma bands; replaced the "200-feature
  z-scored primary" framing with the no-gamma 184 within-dataset family and the
  **96** scale/offset-invariant transport features + unit-incompatibility audit.
- Corrected channels to the 8 harmonized (F3, F4, F7, F8, O1, O2, T7→T3, T8→T4);
  corrected MAT sampling to 500 Hz (was mis-stated 256 Hz).
- Added transparent **sampling-representation amendment** (128 Hz via
  `resample_poly(up=32, down=125)` for MAT and COG-BCI; native-500-Hz forbidden).
- Added calibration-mode definitions, MAT-only-fit transport, leakage controls,
  COG-BCI one-shot protocol, and run-once/checksum controls.
- **Deleted** "Frozen DS007262 Confirmatory Test", "Calibration-Duration Curve",
  and "Effect-Direction Stability (universal axis)" sections.

## Results

- **Deleted** "Baseline-Relative Calibration Improved Exploratory Decoding"
  (z-score-primary tables), the calibration-duration result, the
  coefficient-direction/"universal axis" result, and "Frozen DS007262 Graded
  Transfer Failed".
- **Added** in order: provenance; within-MAT (0.880102, p = 0.004975, not
  superior to absolute); within-STEW (0.839498, p = 0.004975, z diagnostic only);
  MAT→STEW transport failure (0.447598); post-hoc z-scoring (0.682823); COG-BCI
  one-shot failure (z 0.435961; primary ΔAUC CI includes zero); final decision.
- Replaced all old tables/figures (ROC curves, duration curve, direction
  stability, DS007262 level curve) with Tables 1–4 and Figures 1–4 generated only
  from committed summaries.

## Discussion / Limitations / Conclusion

- Discussion: within-domain success ≠ transport; explicit no-mechanism statement;
  task/device/reference/cohort treated as plausible (not proven) contributors;
  methodological value of prospective falsification.
- Removed transfer-optimistic "Interpretation of the Baseline-Relative Transfer
  Hypothesis" and "Resting Phenotyping" speculation.
- Limitations: confounded device+task shift; single prospective target; fixed
  linear model/feature family; STEW non-comparable; predeclared out-of-scope
  conditions; no COG-BCI permutation needed; no clinical/deployment/mechanistic/
  anatomical/causal claim; no acceptance-likelihood claim.
- Conclusion rewritten to the locked story.

## Figures / code / references

- New figure generator: `scripts/make_negative_manuscript_figures.py` →
  `paper/figures/figure_neg_design_flow.png`, `figure_neg_within_domain_auc.png`,
  `figure_neg_mat_to_stew.png`, `figure_neg_cogbci_oneshot.png` (committed
  summaries only; colour-blind-safe palette).
- Old stale figures (DS007262, PAC comodulogram, two-stage, etc.) are no longer
  referenced by `main.tex`.
- `references.tex`: added COG-BCI (`hinss2023`); DS007262 entry no longer cited.

## Compilation

- Built with `tectonic main.tex` (repo has no `pdflatex`); PDF produced at
  `paper/tex/main.pdf`. No undefined references or citations; only cosmetic
  overfull-hbox warnings. Scientific results unchanged during LaTeX fixes.
