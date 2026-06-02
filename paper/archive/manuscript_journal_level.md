# A Leakage-Aware Subject-Normalized EEG Workload Axis for Mental Arithmetic: Subject-Wise Validation and External Stress Testing

## Abstract

Electroencephalography (EEG) workload classifiers are vulnerable to inflated performance when overlapping windows from the same participant are split across training and test sets. This secondary analysis tested whether interpretable EEG features can distinguish rest from mental arithmetic workload under subject-wise validation, and whether a compact workload axis transfers to an independent arithmetic workload dataset. The primary dataset was PhysioNet EEG During Mental Arithmetic Tasks, containing 36 participants with paired baseline/rest and mental-arithmetic EEG recordings. Signals were segmented into 4-second windows with 50% overlap, producing 4,266 windows and 805 numeric features spanning spectral power, relative bandpower, band ratios, time-domain morphology, Hjorth parameters, entropy, regional summaries, hemispheric summaries, and connectivity/correlation. The strongest standard leave-one-subject-out (LOSO) classifier was an RBF-kernel support vector machine (ROC-AUC 0.796, accuracy 0.734, F1 0.519). I developed the Subject-Normalized Workload Axis (SNWA), a one-dimensional score that rest-normalizes features within subject and selects features only inside training folds. SNWA achieved ROC-AUC 0.761, accuracy 0.744, and F1 0.567 at K=8 features. An independent full-feature LOSO recheck from the regenerated feature table confirmed the core signal (SGD logistic ROC-AUC 0.779). External validation on OpenNeuro DS007262, an independent CC0 arithmetic workload EEG dataset, tested low-vs-high arithmetic difficulty in 18 subjects; SNWA K=12 achieved ROC-AUC 0.615 and F1 0.546, while full-feature logistic regression was near chance (ROC-AUC 0.537). Nested feature stability, ablations, negative controls, calibration, and subject-level reliability analyses support a moderate, interpretable EEG workload axis with substantial subject variability. The findings do not support clinical, diagnostic, attention-monitoring, or individual cognitive-assessment claims.

## Introduction

Mental arithmetic is a compact experimental manipulation for increasing cognitive workload. EEG offers a noninvasive measure of brain electrical activity during such tasks, but machine-learning analyses of EEG are especially vulnerable to validation errors. The major danger in windowed EEG classification is subject leakage: a model can learn stable subject-specific structure if windows from the same participant appear in both training and test sets. This can make random window splits look impressively accurate while failing to measure generalization to unseen people.

The scientific goal of this project is therefore not to maximize a leaderboard number. The goal is to ask whether a workload-related EEG signal survives a stricter validation standard: held-out subjects, nested feature selection, negative controls, calibration, and external stress testing. The project uses public EEG data and should be understood as secondary computational neuroscience, not as a device-development or clinical study.

This work makes four contributions. First, it formalizes why random window splitting causes leakage and demonstrates the effect empirically. Second, it evaluates standard interpretable-feature classifiers under subject-wise validation. Third, it introduces SNWA, a one-dimensional subject-normalized workload score designed to be explainable and leakage-aware. Fourth, it tests whether the axis generalizes beyond the original dataset using OpenNeuro DS007262, an independent arithmetic workload EEG dataset.

## Dataset

### Primary Dataset: PhysioNet EEGMAT

The primary dataset was PhysioNet EEG During Mental Arithmetic Tasks v1.0.0. Each participant has a baseline/rest recording and a mental-arithmetic recording. The analysis used 36 subjects. The `_1` EDF files were treated as baseline/rest and assigned label 0; the `_2` EDF files were treated as arithmetic workload and assigned label 1.

The EEGMAT feature table contained:

- Subjects: 36
- Total windows: 4,266
- Rest windows: 3,186
- Workload windows: 1,080
- Numeric feature columns: 805
- Total feature-table columns including metadata: 812

### External Dataset: OpenNeuro DS007262

External validation used OpenNeuro DS007262, *Cognitive Workload 8-level arithmetic*, a CC0 dataset with 18 released subjects. DS007262 is not an exact rest-versus-task replication of EEGMAT. Instead, it contains graded arithmetic trials. I therefore used a workload-gradient contrast:

- Low arithmetic difficulty: `0.6-1.5`
- High arithmetic difficulty: `6.0-6.9`

This produced 360 trial windows across 18 subjects. The contrast is scientifically useful because it tests whether the EEG workload axis transfers to independent arithmetic difficulty data, but it should not be described as exact rest-vs-arithmetic replication.

## Preprocessing and Feature Extraction

EEGMAT EDF files were read with MNE-Python, converted to microvolts, and segmented into 4-second windows with 50% overlap. DS007262 BrainVision files were downloaded from public OpenNeuro S3, read with MNE-Python, and converted to microvolts. For DS007262, one 4-second window was extracted from each selected low/high arithmetic event onset.

For each window, the pipeline extracted interpretable features:

- Absolute bandpower in delta, theta, alpha, beta, and gamma bands
- Relative bandpower in the same bands
- Theta/alpha, beta/alpha, and theta/beta ratios
- Time-domain morphology: mean, standard deviation, variance, RMS, peak-to-peak amplitude, skewness, kurtosis, and Shannon entropy
- Hjorth activity, mobility, and complexity
- Spectral entropy
- Regional and hemispheric summaries
- Pairwise channel-correlation/connectivity summaries

The feature set was intentionally interpretable so that feature-family behavior could be audited rather than treated as a black-box embedding.

## Leakage Theorem

Suppose subject \(s\) contributes \(m_s\) windows. Each window is independently assigned to the training set with probability \(p\) and to the test set with probability \(1-p\). Subject \(s\) avoids leakage only in two disjoint cases:

1. All \(m_s\) windows go to training, with probability \(p^{m_s}\).
2. All \(m_s\) windows go to testing, with probability \((1-p)^{m_s}\).

Therefore, the probability that subject \(s\) appears in both train and test is:

\[
P(\text{leakage}_s)=1-p^{m_s}-(1-p)^{m_s}.
\]

For EEGMAT, each subject contributes many windows, so with \(p=0.75\), this probability is essentially 1 for every subject. This explains why random window splits are not valid evidence of generalization to unseen participants.

## Validation Design

The main validation standard was leave-one-subject-out cross-validation. In each fold, one subject was held out completely. All preprocessing steps that could learn from data, including imputation, scaling, model fitting, SNWA feature ranking, SNWA weights, and SNWA calibration, were fit only on training subjects.

Grouped subject holdout was also reported as a secondary baseline. Random window splits were included only as a leakage demonstration and not as a main result.

## Standard Models

The standard classifiers were:

- Logistic regression
- Random forest
- RBF-kernel support vector machine
- Gradient boosting
- XGBoost

Models were wrapped in scikit-learn pipelines with median imputation and standard scaling where appropriate. The full-feature baseline used the 805 numeric EEGMAT features.

## Subject-Normalized Workload Axis

SNWA was designed to make the workload signal explainable as a one-dimensional score. For each subject \(s\), window \(w\), and feature \(f\), features were normalized to the subject's rest baseline:

\[
z_{s,w,f} = \frac{x_{s,w,f} - \text{median}_{rest,s}(f)}
{\text{MAD}_{rest,s}(f) + 10^{-8}}.
\]

Inside each outer LOSO fold:

1. The held-out subject was excluded.
2. Features were ranked using only training subjects.
3. For each feature, subject-level rest and workload means were computed.
4. Paired workload-minus-rest differences were computed across training subjects.
5. Features were ranked by signed paired effect size and significance.
6. Top K values were tested for K in {3, 5, 8, 12, 20}.
7. Features were weighted by signed effect size.
8. A scalar workload score was computed.
9. A one-dimensional logistic calibration model was fit on training subjects only.
10. The calibrated score was evaluated on the held-out subject.

For DS007262 external validation, the same idea was applied using low arithmetic difficulty as the reference condition and high arithmetic difficulty as the workload condition. This is a difficulty-gradient version of SNWA, not a rest-baseline version.

## Negative Controls

Three negative controls were used:

- Label permutation within training subjects
- Circular label shifts within subjects
- Gaussian random features matched to the data shape

A subject-ID prediction analysis was also included to estimate how much subject-specific structure exists in the features. The real model beat the negative controls in the permutation summary, supporting that the observed signal is not explained solely by chance label structure.

## Feature Stability and Ablations

Nested LOSO feature stability ranked features using only training subjects for each fold. The most frequent top-20 feature families were:

| Feature family | Top-20 count | Unique features |
|---|---:|---:|
| Relative bandpower | 169 | 12 |
| Time-domain morphology | 162 | 20 |
| Spectral absolute bandpower | 151 | 15 |
| Band ratios | 107 | 6 |
| Regional summaries | 79 | 4 |
| Hjorth | 41 | 6 |
| Hemispheric summaries | 10 | 3 |
| Connectivity/correlation | 1 | 1 |

The most stable individual SNWA features included central beta power, occipital/temporal/parietal relative gamma, parietal peak-to-peak amplitude, occipital/parietal regional gamma summaries, occipital peak-to-peak amplitude, and occipital skewness. Connectivity features were almost absent from the stable top features, despite being a large and superficially attractive feature family.

Feature-family ablations showed that no single family produced perfect transfer. Regional/hemispheric summaries, spectral power, band ratios, morphology, and Hjorth features carried measurable subject-wise signal. Connectivity/correlation did not emerge as a stable driver of the workload axis.

## Results

### EEGMAT LOSO Baseline

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| SVM RBF | 0.734 | 0.519 | 0.796 | 0.545 |
| Gradient boosting | 0.727 | 0.456 | 0.764 | 0.426 |
| Logistic regression | 0.719 | 0.533 | 0.763 | 0.466 |
| XGBoost | 0.723 | 0.469 | 0.761 | 0.430 |
| Random forest | 0.741 | 0.253 | 0.760 | 0.451 |

The best ROC-AUC came from the RBF-kernel SVM. Logistic regression had the strongest F1 among the original standard models. Random forest had high specificity but weak sensitivity, illustrating why accuracy alone is misleading in an imbalanced window table.

### SNWA on EEGMAT

| K | Accuracy | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|
| 8 | 0.744 | 0.567 | 0.761 | 0.597 |
| 12 | 0.742 | 0.549 | 0.738 | 0.574 |
| 20 | 0.742 | 0.539 | 0.726 | 0.563 |
| 5 | 0.725 | 0.523 | 0.723 | 0.545 |
| 3 | 0.708 | 0.495 | 0.688 | 0.510 |

SNWA did not dominate the best full SVM by ROC-AUC, but it improved interpretability and achieved the strongest F1 among the compact axis models. The best SNWA setting was K=8.

### Independent LOSO Recheck

The exact original SVM-probability LOSO rerun was too slow in the sandbox. To avoid relying only on a copied baseline table, independent full-feature LOSO models were refit from the regenerated EEGMAT feature table.

| Recheck model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| SGD logistic | 0.762 | 0.545 | 0.779 | 0.517 |
| Logistic regression | 0.719 | 0.533 | 0.763 | 0.466 |
| ExtraTrees | 0.748 | 0.192 | 0.759 | 0.447 |

The independent logistic result exactly matched the committed logistic LOSO metrics, and the SGD logistic recheck confirmed moderate subject-wise signal from the regenerated feature table.

### External Validation on OpenNeuro DS007262

| External model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| SNWA K=12 | 0.617 | 0.546 | 0.615 | 0.643 |
| SNWA K=5 | 0.586 | 0.536 | 0.614 | 0.643 |
| SNWA K=8 | 0.586 | 0.524 | 0.614 | 0.644 |
| SNWA K=3 | 0.578 | 0.510 | 0.593 | 0.637 |
| SNWA K=20 | 0.581 | 0.492 | 0.586 | 0.618 |
| Full-feature logistic | 0.508 | 0.484 | 0.537 | 0.522 |

The external validation result is modest but important. SNWA transferred better than full-feature logistic regression to a different arithmetic workload dataset, but performance was far from high enough to claim a universal EEG workload detector.

### Calibration and Subject-Level Reliability

Calibration and reliability analyses showed that pooled window metrics hide meaningful subject-level heterogeneity. Some subjects were classified well, while others were difficult. This variability is not a nuisance to hide; it is one of the central findings and a key reason not to make individual assessment claims.

## Discussion

The strongest supported claim is that a leakage-aware, subject-normalized EEG workload axis shows moderate subject-wise mental-arithmetic classification in EEGMAT and modest external transfer to independent arithmetic difficulty data. The feature-stability results suggest that the signal is not an arbitrary high-dimensional artifact: relative bandpower, spectral power, morphology, and posterior/parietal/occipital features recur across nested folds. At the same time, the external ROC-AUC of approximately 0.615 shows that transfer is limited. This is a serious scientific result precisely because it is not inflated into a claim of perfect decoding.

The random-window comparison demonstrates how easy it would be to overstate the project. Random splits produced dramatically higher apparent performance, but the leakage theorem shows why those numbers are not valid evidence of cross-subject generalization. Subject-wise validation is the correct standard for the main claim.

SNWA's value is not that it beats every full model on every metric. Its value is that it is low-dimensional, nested, subject-normalized, interpretable, and externally stress-tested. In a science-fair or research-defense setting, this is easier to explain and harder to dismiss than a black-box model trained on hundreds of features with unclear stability.

## Limitations

This is a secondary analysis of public EEG datasets. No new human-subject data were collected. EEGMAT provides rest-vs-arithmetic recordings, while DS007262 provides graded arithmetic trials; the external validation is therefore a workload-gradient test rather than exact rest-vs-task replication. Windows within a subject are correlated, so subject-level reliability is emphasized. Some subjects remain difficult, and this limits any claim of individual-level cognitive-state inference. SNWA assumes an available baseline or low-workload reference condition. The analysis should not be interpreted as diagnosis, attention monitoring, medical-device validation, or individual cognitive assessment.

## Ethics Statement

This study used publicly available EEG datasets. No new human-subject data were collected. The project avoids clinical, diagnostic, surveillance, or individual assessment claims. The intended contribution is methodological: showing how leakage-aware validation and interpretable modeling can produce a more honest EEG workload analysis.

## Data and Code Availability

The primary EEGMAT dataset is available from PhysioNet. The external DS007262 dataset is available from OpenNeuro under CC0. Code, generated outputs, SNWA analyses, independent LOSO recheck outputs, and external validation outputs are organized in this repository workspace.


