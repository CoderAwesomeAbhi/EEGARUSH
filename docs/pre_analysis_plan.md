# Retrospective Pre-Analysis Plan

**Project:** Frontal Theta Oscillations Predict Arithmetic Performance
**Author:** Abhijay Gangarapu
**Date:** May 28, 2026

---

This document retrospectively distinguishes pre-planned analyses from
exploratory decisions made after data access. The analysis was **not formally
pre-registered**; this document serves to document the analysis trajectory
for transparency.

## Pre-Planned Analyses (Designed Before Data Access)

1. **Primary hypothesis:** Frontal theta power (F3/F4 mean) will be
   significantly higher during mental arithmetic than rest, replicating across
   the PhysioNet MAT dataset.

2. **Secondary hypothesis:** Alpha power will decrease during arithmetic.

3. **Classification approach:** Leave-one-subject-out cross-validation with
   SVM RBF, Logistic Regression, and Random Forest.

4. **Feature set:** Absolute and relative bandpower, Hjorth parameters,
   entropy, time-domain statistics, and band ratios — all standard in EEG
   workload literature.

5. **Evaluation metrics:** Accuracy, sensitivity, specificity, F1, ROC-AUC,
   PR-AUC, with 95% subject-level bootstrap confidence intervals.

6. **Statistical tests:** Paired t-tests and Wilcoxon signed-rank tests with
   Benjamini-Hochberg FDR correction for feature-level analysis.

## Exploratory/Post-Hoc Additions (After Data Access)

1. **Subject-Normalized Workload Axis (SNWA):** Developed after initial LOSO
   results to provide an interpretable one-dimensional workload score. The
   rest-normalization and effect-size weighting scheme was iterated based on
   feature-ranking stability across folds.

2. **Multi-dataset expansion:** The addition of STEW and DS007262 was
   motivated by the desire to test generalizability after observing promising
   MAT results. These datasets were added sequentially, not as part of the
   original design.

3. **K selection for SNWA:** The choice of K=8 was made by comparing
   performance across K ∈ {3, 5, 8, 12, 20} on the combined dataset. This
   parameter was not pre-specified.

4. **Cross-dataset transfer analysis:** Designed after observing that
   within-dataset LOSO performed well but could not address dataset
   specificity.

5. **Negative controls (permutation, circular shift, Gaussian features):**
   Added to address reviewer concerns about dataset artifact detection.

6. **Feature family stability analysis:** Post-hoc categorization of features
   into families to understand which feature types contribute most
   consistently.

7. **Gradient Boosting and XGBoost models:** Added after initial LR/SVM/RF
   results to test whether ensemble methods improved performance.

8. **Calibration analysis (Brier score, ECE):** Added during revision to
   assess probability calibration quality, which is not part of standard EEG
   classification reporting.

9. **DS007262 difficulty thresholds (0.6-1.5 vs 6.0-6.9):** Selected
   post-hoc after inspecting the dataset distribution. The original dataset
   has 8 difficulty levels; the extreme-group contrast was designed to
   maximize the workload signal.

## Pre-Registration Status

This analysis was **not pre-registered**. All results should be interpreted
as exploratory. The four hypotheses stated in the Introduction were defined
before the multi-dataset analysis but after initial MAT-only results were
available. Confirmatory replication in a prospective, pre-registered study
is needed before strong claims can be made.
