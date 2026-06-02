# Prospective Validation Plan & OSF Pre-registration
## Subject Variability in EEG Workload Classification

### Pre-registration ID (OSF)
Embargoed until data collection is complete. Register at [osf.io](https://osf.io) under the project name "Resting EEG Predictors of Workload Classifiability."

### Hypotheses (3 pre-registered)

**H1: Resting EEG predicts LOSO classifiability in new participants.**
- Spearman ρ > 0.3 between predicted and actual LOSO AUC
- Predictors: Hjorth complexity (negative), temporal-occipital connectivity (positive), occipital theta power (negative)
- Source: cross-validated from MAT (N=36) and STEW (N=48) datasets

**H2: SNWA discriminates rest from arithmetic only in participants predicted classifiable.**
- Two-stage screening: participants with predicted AUC > 0.6 receive SNWA
- SNWA AUC significantly above chance (one-sided t-test, α = 0.05)
- Effect: Cohen's d ≥ 0.7 (based on MAT replication)

**H3: Riemannian geometry alignment improves cross-participant transfer.**
- Covariance-based tangent space features + Riemannian alignment
- Compared to raw feature baseline in LOSO
- Minimum detectable improvement: ΔAUC = 0.05

### Experimental Design

**Participants:** N ≥ 10 (target N = 15), recruited through school/community.
Inclusion: age 14-18, normal or corrected vision, no neurological conditions.
Exclusion: self-reported ADHD, epilepsy, or medication affecting CNS.

**Protocol (25 minutes per participant):**
1. Consent and screening (5 min)
2. Headset fitting: 8-channel dry EEG (OpenBCI Cyton/Daisy or similar) (5 min)
3. Resting baseline: eyes open, fixate cross, 2 minutes
4. Arithmetic task: serial subtraction by 7 from 4-digit number, 5 minutes
5. Resting recovery: eyes open, 2 minutes
6. Arithmetic task 2: serial subtraction by 13 from 4-digit number, 5 minutes
7. Debrief (3 min)

**Equipment:** Consumer EEG (8+ channels, dry electrodes, ≤ $500).

### Power Analysis

From MAT data: SNWA K8 AUC = 0.761, per-subject SD = 0.16.
For one-sided t-test vs chance (AUC = 0.5):
- α = 0.05, power = 0.80
- Minimum effect d = 0.70
- Required N = 15 (G*Power, one-sample t-test)

For replication (same effect as MAT d = 0.84): required N = 10.
Target: N = 15 to account for attrition.

### Analysis Plan

**Primary analysis:** Per-subject LOSO AUC using same pipeline as source study.
**Secondary analysis:** 
1. Spearman correlation between resting EEG features and per-subject AUC
2. SNWA transfer performance
3. Riemannian alignment effect

### Exclusion Criteria
- Participants with < 50% valid trials (excessive motion artifact)
- Less than 10 windows per condition
- Technical failure during recording

### SRC/IRB Approval Status
- [ ] SRC application submitted (date: _______)
- [ ] SRC approval received (date: _______)  
- [ ] Parental consent form prepared
- [ ] Participant assent form prepared

### Data Availability
All de-identified data will be deposited to OSF upon project completion.
Analysis code is already public at the project repository.
