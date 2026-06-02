# Frontal Theta Oscillations Predict Mental Arithmetic Performance: A Prospective EEG Study of Cognitive Workload

**Version:** 1.0  
**Date:** May 28, 2026  
**Principal Investigator:** Arjun [Last Name]  
**Institution:** [School/University Name]  
**IRB Protocol Number:** [Assigned by IRB]

---

## 1. Background and Rationale

Cognitive workload assessment using electroencephalography (EEG) is a well-established paradigm in cognitive neuroscience and neuroergonomics. Frontal midline theta oscillations (4–8 Hz) have been consistently implicated in working memory engagement, attentional control, and mental arithmetic performance. Spectral power in the theta band — particularly over fronto-central electrode sites (Fz, FCz, Cz) — increases monotonically with task difficulty and cognitive load, making it a robust neural marker of mental effort. Prior studies have validated the relationship between frontal theta activity and arithmetic performance across multiple difficulty levels, demonstrating both within-session reliability and cross-session stability.

However, the majority of existing work in this domain suffers from two key limitations. First, sample sizes are often small (median N ≈ 36), yielding limited statistical power and poor generalizability. Second, many recent publications rely on re-analyses of a small handful of public datasets (e.g., the Zander 2017 dataset, the ERP CORE), raising concerns about dataset dependency and inflated effect sizes through analytic flexibility. Prospective data collection with standardized, pre-registered protocols is needed to establish the true effect size of the frontal theta–workload relationship and to provide an unbiased benchmark for machine learning classifiers.

The Subject-Normalized Workload Axis (SNWA) framework was recently introduced to address the subject-specific nature of EEG workload signatures. SNWA normalizes spectral features to each subject's resting baseline, then applies a shared linear projection to produce a single interpretable workload score per time window. In retrospective analyses of public datasets, SNWA achieved ROC-AUC values of 0.65–0.85 for binary workload classification. However, these estimates were obtained from the same datasets that informed feature selection and model construction — a form of optimism bias. A truly prospective validation, where the entire pipeline is frozen before data collection begins, is necessary to establish the real-world performance of the SNWA framework.

The present study proposes to collect a new, adequately powered (N = 60) EEG dataset using an identical protocol to the one used in prior SNWA analyses, enabling direct comparison between retrospective and prospective results. We will test four specific hypotheses concerning the neural and behavioral correlates of cognitive workload during mental arithmetic.

---

## 2. Hypotheses

The following hypotheses are pre-registered. All tests are two-sided unless otherwise noted. Multiple comparisons across frequency bands and electrode sites will be controlled using the Benjamini–Hochberg false discovery rate (FDR) procedure at q = 0.05.

- **H1 (Frontal Theta Activation):** Frontal theta power (4–8 Hz, averaged over Fz, F3, F4) is significantly greater during mental arithmetic blocks than during eyes-open rest at the individual subject level, as assessed by a paired-samples t-test (α = 0.05, FDR corrected across electrodes and frequency bands).
- **H2 (SNWA Classification Performance):** The SNWA one-dimensional workload score achieves an area under the receiver-operating characteristic curve (ROC-AUC) greater than 0.70 in leave-one-subject-out (LOSO) cross-validation when discriminating arithmetic blocks from rest.
- **H3 (Theta/Alpha Ratio–Behavior Correlation):** The frontal theta/alpha ratio (4–8 Hz / 8–13 Hz) during arithmetic blocks correlates negatively with task accuracy (percentage correct), such that |r| > 0.4 across subjects (Pearson correlation, one-tailed).
- **H4 (Real-Time BCI Feasibility):** A real-time SNWA-based binary classifier, trained on Blocks 1–2 and tested on Block 3, achieves classification accuracy significantly above chance (accuracy > 60%) for discriminating workload vs. rest at the individual subject level.

---

## 3. Participants

### 3.1 Sample Size

**N = 60** healthy volunteers.

### 3.2 Power Analysis

A power analysis based on prior simulation from pilot data and public datasets (N < 40, AUC ≈ 0.65) indicates that N = 60 subjects provides 90% power to detect an AUC of 0.65 against a null of 0.50 (two-sided, α = 0.05) using a one-sample z-test on LOSO AUC values. This sample also provides 95% power to detect a paired-samples Cohen's d = 0.50 for frontal theta power differences (α = 0.05, two-sided).

### 3.3 Inclusion Criteria

- Age 18–35 years
- Normal or corrected-to-normal vision
- Fluent in the language of task instructions

### 3.4 Exclusion Criteria

- Self-reported neurological condition (e.g., epilepsy, traumatic brain injury, brain surgery)
- Current use of psychoactive medication (antidepressants, anxiolytics, stimulants, antipsychotics)
- Hearing impairment that would prevent understanding verbal instructions
- Scalp wounds, dermatitis, or other conditions preventing electrode application
- Current participation in another EEG or cognitive testing study (to avoid practice confounds)

### 3.5 Recruitment

Participants will be recruited through printed flyers posted on university bulletin boards, posts to institutional social media channels, and word of mouth. Interested individuals will complete a brief online screening questionnaire to verify eligibility. Eligible participants will be scheduled for a single 60-minute laboratory session.

---

## 4. Experimental Design

### 4.1 Design Overview

Within-subjects, repeated-measures design with two conditions:
- **Rest:** 5 minutes of eyes-open rest (fixation cross on screen)
- **Mental Arithmetic:** Three blocks of 5 minutes each (15 minutes total)

### 4.2 Task Details

The mental arithmetic task is serial subtraction by 7 from 1000. Participants are instructed to subtract 7 from the current number and report each result aloud. The experimenter records verbal responses and codes them as correct or incorrect. To maintain engagement and to titrate difficulty, the subtraction step increases by 1 after three consecutive correct responses (e.g., subtract 8, then 9) and decreases by 1 after two consecutive incorrect responses. This adaptive staircase ensures that difficulty is roughly matched to each participant's ability, minimizing floor and ceiling effects.

### 4.3 Behavioral Measures

- **Accuracy:** Percentage of correct subtractions (per block and overall)
- **Reaction Time:** Verbal response latency (ms), measured from stimulus onset to voice onset (recorded via microphone and annotated offline)
- **NASA-TLX:** Raw NASA Task Load Index (six dimensions: Mental, Physical, Temporal, Performance, Effort, Frustration) administered after each arithmetic block

### 4.4 Session Structure

| Phase | Duration | Description |
|-------|----------|-------------|
| Arrival and consent | 10 min | Review consent form, answer questions, sign |
| EEG setup | 20 min | Cap placement, electrode gel, impedance check |
| Baseline rest | 5 min | Eyes open, fixation cross |
| Practice | 2 min | Brief arithmetic practice to familiarize |
| Block 1 | 5 min + 1 min TLX | Arithmetic + NASA-TLX |
| Block 2 | 5 min + 1 min TLX | Arithmetic + NASA-TLX |
| Block 3 | 5 min + 1 min TLX | Arithmetic + NASA-TLX |
| Post-task rest | 5 min | Eyes open rest |
| Debriefing | 5 min | Explain purpose, answer questions |
| **Total** | **~60 min** | |

---

## 5. EEG Recording Protocol

### 5.1 Hardware

- **Amplifier:** OpenBCI Cyton + Daisy board (16-channel, 24-bit ADS1299) or equivalent research-grade biosignal acquisition system
- **Electrodes:** Ag/AgCl sintered ring electrodes in an elastic fabric cap (international 10–20 montage)
- **Sampling Rate:** 256 Hz (actual: 250 Hz for OpenBCI; resampled to 256 Hz offline)

### 5.2 Electrode Montage

| Position | Location | Notes |
|----------|----------|-------|
| Fp1, Fp2 | Prefrontal | |
| F3, Fz, F4 | Frontal | Primary ROI for theta |
| C3, Cz, C4 | Central | |
| P3, Pz, P4 | Parietal | |
| O1, O2 | Occipital | |
| T7, T8 | Temporal | |
| POz | Parieto-occipital | |
| **Reference** | Linked mastoids (A1, A2) | Physically averaged |
| **Ground** | AFz | |

### 5.3 Impedance

All electrode impedances will be maintained below 10 kΩ. Impedances will be checked and documented before each block; if any electrode exceeds 20 kΩ, gel will be reapplied and impedance rechecked.

### 5.4 Acquisition Parameters

- Sampling rate: 256 Hz (250 Hz native, upsampled)
- Online filtering: 0.5–45 Hz bandpass (hardware + analog)
- Notch filter: 50 Hz or 60 Hz as appropriate for local mains frequency
- Recording software: OpenBCI GUI (v6+), saved as `.txt` or `.csv`
- Backup recording: raw data also streamed to Lab Streaming Layer (LSL) for potential real-time processing

### 5.5 Environment

- Sound-attenuated room or quiet laboratory space
- Participant seated 60 cm from a 24" LCD monitor (1920 × 1080, 60 Hz)
- Ambient lighting controlled and dimmed
- Experimenter in adjacent room or behind a partition

---

## 6. Procedure

All sessions will be conducted by the same experimenter following a written protocol checklist.

1. **Arrival and Consent (10 min):** Participant arrives, is greeted, and reviews the consent form. The experimenter explains the study purpose, procedures, risks, and confidentiality protections. After all questions are answered, the participant signs two copies (one retained, one for participant).

2. **EEG Setup (20 min):** Participant's head circumference is measured and the appropriately sized cap is fitted. Electrodes are filled with conductive gel using a blunt-tip syringe. Impedances are checked and reduced to < 10 kΩ. Electrode positions are measured and documented for reproducibility.

3. **Baseline Rest (5 min):** Participant sits quietly with eyes open, gaze fixed on a central crosshair on the monitor. No task is performed. The experimenter monitors signal quality.

4. **Practice Block (2 min):** Participant performs 10 practice subtractions (serial subtraction by 3 from 50). No EEG data is saved; this phase is for task familiarization only.

5. **Arithmetic Blocks 1–3 (5 min each + 1 min NASA-TLX):** Participant performs serial subtraction as described in Section 4.2. The experimenter records responses on a paper log. After each block, the participant completes a NASA-TLX rating on a tablet or paper form.

6. **Post-Task Rest (5 min):** Identical to baseline rest. Participant sits quietly with eyes open.

7. **Debriefing (5 min):** The experimenter explains the research questions and hypotheses, answers any remaining questions, and provides the participant with contact information and compensation.

8. **Cap Removal:** The cap is removed, gel is washed from the participant's hair in a dedicated sink, and the participant is free to leave.

---

## 7. Data Analysis Plan

### 7.1 Preprocessing

All preprocessing is performed using MNE-Python (v1.6+) and replicated from the prior SNWA pipeline:

- Raw data import and channel location assignment
- Bandpass filter: 0.5–45 Hz (FIR, hamming window, zero-phase)
- Notch filter: 50/60 Hz
- Automatic artifact rejection: ICA (FastICA, 15 components) with automated rejection of components correlated with EOG channels or exhibiting temporal kurtosis > 3 SD from mean
- Epoch extraction: 4-second non-overlapping windows (matching prior pipeline; 50% overlap used for real-time analysis only)
- Baseline correction: subtraction of mean of the window
- Rejection of epochs with peak-to-peak amplitude > 150 µV in any channel

### 7.2 Feature Extraction

The following features are extracted per epoch, per channel:

- **Spectral power:** Absolute power (µV²/Hz) in delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (30–45 Hz) via multitaper method (DPSS tapers, time-bandwidth product = 2, 3 tapers)
- **Relative bandpower:** Each band's power divided by total power (1–45 Hz)
- **SNWA features:** Log-transformed absolute bandpowers, normalized per subject by subtracting the median of the resting baseline for that channel and band
- **Hjorth parameters:** Activity, mobility, complexity (per channel)
- **Signal entropy:** Sample entropy (m = 2, r = 0.2 × SD)
- **Theta/alpha and theta/beta ratios:** Log-transformed power ratios
- **Total features:** ~300 per epoch (15 channels × ~20 features)

### 7.3 SNWA Implementation

The SNWA pipeline follows the method described in prior work:

1. **Rest normalization:** For each subject, compute per-channel per-band log-power medians from the rest condition.
2. **Difference features:** Subtract rest medians from each arithmetic epoch's log-power values, yielding a set of change scores.
3. **Dimensionality reduction:** Fit a linear discriminant analysis (LDA) using only the difference features from training subjects.
4. **Projection:** Apply the LDA coefficient vector to each subject's held-out difference features to produce a one-dimensional SNWA score.
5. **Validation:** Leave-one-subject-out cross-validation; compute ROC-AUC per held-out subject; report mean and 95% bootstrap CI across subjects.

### 7.4 Specific Analyses

#### H1: Frontal Theta Activation

- Extract mean theta power (4–8 Hz) over frontal electrodes (Fz, F3, F4) for each subject, separately for rest and arithmetic epochs.
- Conduct a paired-samples t-test (arithmetic vs. rest) across subjects (N = 60).
- Report Cohen's d, 95% CI of the mean difference, and Bayes factor (BF₁₀, Cauchy prior r = 0.707).
- FDR correction across all 5 frequency bands × 15 channels (75 tests).

#### H2: SNWA Classification Performance

- Perform LOSO cross-validation as described in Section 7.3.
- Primary metric: mean ROC-AUC across subjects.
- Secondary metrics: accuracy, sensitivity, specificity, precision, F1-score (at the Youden-optimal cut point per subject).
- Bootstrap 95% CI for mean AUC (10,000 bootstrap iterations).
- Compare to prior reported AUC from public dataset (AUC ≈ 0.78) via two-sample z-test.

#### H3: Theta/Alpha Ratio–Behavior Correlation

- For each subject, compute the mean theta/alpha power ratio (Fz) across arithmetic epochs.
- Compute Pearson correlation between theta/alpha ratio and task accuracy (% correct, all blocks pooled).
- Report r, 95% CI (Fisher z-transformation), and BF₁₀.
- Sensitivity analysis: partial correlation controlling for age and sex.

#### H4: Real-Time BCI Feasibility

- Simulate a real-time classifier: train SNWA pipeline on Blocks 1–2 and test on Block 3 (both arithmetic vs. pooled rest epochs).
- Compute per-subject classification accuracy.
- One-sample t-test (two-sided) against chance (50%).
- Report mean accuracy, 95% CI, and proportion of subjects above 60%.

### 7.5 Software and Reproducibility

- **Analysis software:** Python 3.11+, MNE-Python 1.6+, scikit-learn 1.3+, SciPy 1.11+, pandas 2.0+, NumPy 1.24+, matplotlib 3.7+
- **Code sharing:** All analysis code will be made available on GitHub (https://github.com/[username]/[repository]) at the time of manuscript submission.
- **Data sharing:** De-identified EEG and behavioral data will be deposited on OpenNeuro (https://openneuro.org) under a CC0 license.
- **Pre-registration:** This protocol will be pre-registered on Open Science Framework (https://osf.io) prior to data collection.

---

## 8. Ethical Considerations

### 8.1 Risk Assessment

This study poses **minimal risk** to participants. EEG is a non-invasive technique that passively records neural electrical activity via scalp electrodes. No electrical current is passed into the head.

**Potential risks and mitigations:**
- **Scalp discomfort:** Mild skin irritation or redness may occur at electrode sites from conductive gel or abrasive skin preparation. These effects typically resolve within hours. We will use hypoallergenic gel and minimize skin abrasion.
- **Minor skin abrasion:** If gentle skin prep is used (NuPrep), there is a small risk of minor abrasion. We will use a blunt-tipped wooden applicator only, no sandpaper or metal tools.
- **Fatigue or eye strain:** The 60-minute session may cause mild fatigue. Participants are informed of the duration beforehand and are allowed breaks between blocks.
- **Psychological distress:** The arithmetic task may cause mild frustration or anxiety. Participants are reminded that performance does not affect compensation and that they may stop at any time.

### 8.2 Data De-identification

- All data will be stored under a numeric subject code (e.g., ISEF_001).
- The linking file (name ↔ code) will be stored separately, encrypted with AES-256, on a password-protected institutional server.
- All facial-identifying information (e.g., 3D electrode positions) will be removed before data sharing.
- Voice recordings (for verbal response timing) will be processed to extract onset latencies, then deleted unless the participant has consented separately to voice data sharing.

### 8.3 Participant Compensation

Each participant will receive a **$20 gift card** (e.g., Amazon, Starbucks) upon completion of the session. Participants who withdraw after EEG setup but before task completion will receive a prorated $10 gift card. Participants who withdraw before EEG setup will not be compensated.

### 8.4 Right to Withdraw

Participants may withdraw at any time for any reason without penalty. Withdrawal will not affect their relationship with the institution or any services they receive.

### 8.5 Data Storage

- **Raw EEG data:** Stored on an encrypted, password-protected laboratory computer (local SSD).
- **Backup:** Encrypted backup on institutional cloud storage (Box, OneDrive, or equivalent).
- **Analysis copies:** De-identified copies on the analysis workstation.
- **Retention period:** Data will be retained for a minimum of 5 years after publication, per institutional policy.

### 8.6 Data Sharing

De-identified data and code will be shared publicly upon publication to enable reproducibility and further analysis. Participants will be informed of and consent to this data sharing plan.

---

## 9. Timeline

| Week | Activity | Milestone |
|------|----------|-----------|
| 1 | IRB submission | Protocol submitted |
| 2 | IRB review, approval | Approval obtained |
| 2–4 | Recruitment | Flyers posted, screening begins, first 15 subjects scheduled |
| 5 | Data collection begins | Subjects 1–15 |
| 6 | Data collection | Subjects 16–30 |
| 7 | Data collection | Subjects 31–45 |
| 8 | Data collection | Subjects 46–60 |
| 9 | Preprocessing and feature extraction | Clean dataset ready |
| 10 | Primary and secondary analyses | All results computed |
| 11–12 | Manuscript preparation | Draft submitted to advisor, then to ISEF/journals |

---

## 10. Budget

| Item | Cost (USD) | Notes |
|------|------------|-------|
| EEG amplifier (OpenBCI Cyton + Daisy) | $699.00 | If not already available |
| EEG electrode cap (16-channel) | $200.00 | If not already available |
| Conductive gel (10 tubes) | $100.00 | Signa Gel or equivalent |
| Disposable supplies (syringes, tape, scrub pads) | $50.00 | |
| Gift cards (60 × $20) | $1,200.00 | Participant compensation |
| Printed consent forms, flyers | $50.00 | |
| **Total (with hardware)** | **$2,299.00** | |
| **Total (hardware owned)** | **$1,400.00** | Recurring costs only |

**Funding sources:** [List funding sources, e.g., school science department, institutional grant, personal funds]

---

## 11. References

1. Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. *Brain Research Reviews*, 29(2–3), 169–195.
2. Gevins, A., & Smith, M. E. (2003). Neurophysiological measures of cognitive workload during human–computer interaction. *Theoretical Issues in Ergonomics Science*, 4(1–2), 113–131.
3. Zander, T. O., & Jatzev, S. (2012). Context-aware brain–computer interfaces: exploring the information from passive BCI systems. *Frontiers in Neuroscience*, 6, 168.
4. [Author] et al. (2025). Subject-Normalized Workload Axis: A validated framework for single-subject cognitive workload assessment from EEG. *[Journal]*.
5. Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics. *Journal of Neuroscience Methods*, 134(1), 9–21.
6. Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
7. Faul, F., Erdfelder, E., Lang, A.-G., & Buchner, A. (2007). G\*Power 3: A flexible statistical power analysis program. *Behavior Research Methods*, 39(2), 175–191.
8. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.

---

## Appendix A: Screening Questionnaire

A brief online screening form will collect:
- Age
- Handedness
- History of neurological or psychiatric conditions
- Current medication use
- Hearing or vision problems
- Prior EEG participation

## Appendix B: NASA-TLX Instrument

The raw NASA-TLX (Hart & Staveland, 1988) consists of six 20-point bipolar scales:
- Mental Demand (Low / High)
- Physical Demand (Low / High)
- Temporal Demand (Low / High)
- Performance (Good / Poor)
- Effort (Low / High)
- Frustration (Low / High)

Each scale is presented as a 20-step line with endpoint anchors. Scores are recorded as integers 0–20.
