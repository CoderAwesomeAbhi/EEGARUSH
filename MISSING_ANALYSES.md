# Missing Analyses

Strict checklist of experiments required to prove the Baseline-Relative Workload Transfer Hypothesis. These items are not fully present in the current Python scripts and artifacts.

## Definition And Preregistration

- [ ] Define the Baseline-Relative Workload Transfer Hypothesis mathematically before testing: target-domain task state should be predictable from baseline-relative feature displacement using only target baseline data and source-domain labels.
- [ ] Pre-register primary endpoints, allowed features, subject-level splits, dataset-level splits, exclusion criteria, and statistical tests.
- [ ] Separate confirmatory analyses from exploratory biological interpretation.

## Reproducible External SNWA Pipeline

- [ ] Add a single executable script that regenerates `external_validation_ds007262/ds007262_low_high_predictions.csv` from raw or harmonized feature inputs.
- [ ] Ensure SNWA feature selection, normalization parameters, and model weights are learned only from source training folds.
- [ ] Ensure target-domain workload labels are never used for feature selection, calibration, normalization, or threshold choice.
- [ ] Save fold-wise SNWA weights, selected features, baseline medians/MADs, and prediction checksums for auditability.

## True Baseline-Relative Transfer Tests

- [ ] Train on MAT and test on DS007262 using only DS007262 resting/baseline data for subject normalization.
- [ ] Train on STEW and test on DS007262 using only DS007262 resting/baseline data for subject normalization.
- [ ] Train on MAT plus STEW and test on DS007262 with no target workload calibration.
- [ ] Reverse the transfer directions to test whether the axis is dataset-general rather than MAT-specific.
- [ ] Compare baseline-relative SNWA against absolute features, pooled z-scoring, per-subject task-only normalization, and dataset-level normalization.

## Leakage And Ablation Controls

- [ ] Shuffle baseline records across subjects and verify transfer performance collapses.
- [ ] Remove baseline data entirely and verify the baseline-relative method loses its advantage.
- [ ] Vary available baseline length to test minimum rest duration needed for stable normalization.
- [ ] Repeat all feature selection inside nested training folds only.
- [ ] Add subject, session, and dataset identity prediction controls to quantify confounding.

## Mathematical Invariance Testing

- [ ] Test whether workload displacement vectors align across subjects after baseline normalization using cosine-angle, bootstrap confidence intervals, and sign-consistency tests.
- [ ] Test cross-dataset representational alignment using Procrustes, CCA, RSA, or another explicit geometry test.
- [ ] Compare SNWA against Riemannian tangent-space alignment with matched train/test splits.
- [ ] Test whether the learned SNWA direction is stable under channel subset, reference montage, feature family, and dataset perturbations.

## Biological Mechanism Validation

- [ ] Perform source localization with a specified montage, head model, inverse method, noise covariance, regularization, and quality-control workflow.
- [ ] Test whether source-level frontal/midline theta effects map to plausible fronto-hippocampal or fronto-medial generators.
- [ ] Test alpha changes with source or sensor models capable of separating occipital alpha, thalamocortical interpretations, and generic visual/task effects.
- [ ] Add an explicit limitation that 19-channel scalp EEG is not sufficient to prove hippocampal or thalamic generators without stronger source evidence.
- [ ] Validate any hyperarousal/default-mode-network interpretation with independent measures such as resting-state connectivity, validated arousal scales, pupilometry, HRV, or fMRI/MEG where available.

## Baseline Phenotyping Validation

- [ ] Replicate resting Hjorth complexity and temporal-occipital connectivity predictors in an independent cohort.
- [ ] Predefine the baseline phenotype score before testing predictive utility.
- [ ] Use nested cross-validation or external validation for subject-level screening.
- [ ] Report whether any resting predictors survive FDR correction, not only uncorrected correlations.
- [ ] Test whether baseline phenotype improves decoding or triage prospectively, not retrospectively.

## Statistical Evidence Standards

- [ ] Use subject-level bootstrap confidence intervals for every headline AUC and transfer result.
- [ ] Use permutation tests at the subject or dataset level for SNWA transfer significance.
- [ ] Compare SNWA against full-model and ablated baselines with paired tests across identical held-out subjects.
- [ ] Correct across K values, feature families, datasets, and endpoint families.
- [ ] Add hierarchical mixed-effects models to separate subject, dataset, task, and feature-family variance.

## Reproducibility And Portability

- [ ] Remove hard-coded local paths from exploratory scripts.
- [ ] Add a manifest linking every manuscript table/figure to the exact script and output file that generated it.
- [ ] Add checksums for static CSVs used as evidence.
- [ ] Add one command that regenerates all manuscript-supporting outputs without manual notebook or path edits.
- [ ] Add CI smoke tests for feature extraction, LOSO splitting, SNWA train/test separation, and external-transfer prediction generation.

