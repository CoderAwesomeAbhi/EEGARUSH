# Methods Template

## Dataset

This study used the PhysioNet EEG During Mental Arithmetic Tasks dataset. Each subject had two EEG recordings: a background EEG recording before the arithmetic task and an EEG recording during a mental arithmetic task. The `_1` files were labeled as rest/baseline, and the `_2` files were labeled as cognitive workload.

## Preprocessing

EEG recordings were read from EDF format. Each recording was segmented into overlapping time windows of 4 seconds with 50% overlap. Features were extracted independently from each window. Model evaluation used subject-wise splitting so that EEG windows from the same participant were not placed into both training and testing sets.

## Feature Extraction

Three categories of features were extracted:

1. Spectral features:
   - delta bandpower
   - theta bandpower
   - alpha bandpower
   - beta bandpower
   - gamma bandpower
   - relative bandpower
   - theta/alpha, beta/alpha, and theta/beta ratios
   - spectral entropy

2. Statistical features:
   - mean
   - standard deviation
   - variance
   - RMS
   - peak-to-peak amplitude
   - skewness
   - kurtosis
   - Shannon entropy
   - Hjorth activity, mobility, and complexity

3. Connectivity features:
   - pairwise Pearson correlations between EEG channels
   - mean channel correlation
   - left-right hemisphere correlation summaries

## Machine Learning

The following supervised classifiers were trained and compared:

- Logistic Regression
- Random Forest
- Support Vector Machine with RBF kernel
- Gradient Boosting
- XGBoost, if available

Missing values were imputed using the training-set median, and features were standardized inside each training fold.

## Evaluation

The main evaluation used subject-wise holdout testing and leave-one-subject-out cross-validation. Performance metrics included accuracy, sensitivity, specificity, PPV, NPV, F1-score, ROC-AUC, PR-AUC, and confusion matrices. Bootstrap confidence intervals were computed by resampling subjects.

## Statistical Analysis

For feature-level analysis, windows were first averaged within each subject and condition. Paired tests then compared each subject's rest EEG features against that subject's workload EEG features. Paired t-tests, Wilcoxon signed-rank tests, Cohen's dz effect sizes, and Benjamini-Hochberg FDR correction were used.
