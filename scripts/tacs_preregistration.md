# Preregistration: Causal Role of Frontal Theta Oscillations in Arithmetic Workload

## Background & Rationale
Our multi-dataset replication (MAT, STEW, DS007262) established that frontal theta (4-8 Hz) power is the most robust EEG feature for workload classification (mean AUC=0.83, Cohen's d=1.2). Theta-gamma phase-amplitude coupling (PAC) increases during arithmetic (d=0.67, 10/10 channels FDR-significant), suggesting a mechanistic role in working memory maintenance and cognitive computation (Lisman & Jensen 2013, Neuron).

However, all current evidence is correlational. To establish a causal role for frontal theta oscillations in arithmetic performance, we propose a randomized, sham-controlled tACS experiment.

## Hypotheses
1. **H1**: 6 Hz frontal tACS during arithmetic will increase frontal theta power vs. sham.
2. **H2**: 6 Hz frontal tACS will improve arithmetic accuracy and speed vs. sham.
3. **H3**: tACS-induced theta power increase will correlate with performance improvement.
4. **H4**: 6 Hz tACS will increase theta-gamma PAC (exploratory).

## Methods

### Design
- Within-subjects, double-blind, sham-controlled crossover
- Two sessions (tACS active, tACS sham), counterbalanced order
- Each session: 5 min baseline (eyes-open rest), 3 x 10-min arithmetic blocks with 2-min breaks, 5 min post-rest
- Dependent variables: accuracy, reaction time, frontal theta power, theta-gamma PAC

### Participants
- N=40 (sufficient for d=0.5, 80% power, paired t-test)
- Inclusion: right-handed, 18-40 years, normal/corrected vision, no history of neurological/psychiatric conditions or contraindications for tES
- Exclusion: <50% correct on practice block

### tACS Protocol
- StarStim R32 or Soterix Medical 1x1
- 6 Hz sinusoidal, 2 mA peak-to-peak, 30 min total
- Electrodes: F3/F4 (active), Cz (return) — 5x7 cm pads
- Ramp up/down: 30 s
- Sham: 30 s ramp up/down only at start and end
- Impedance kept below 10 kOhm

### Task
- Block-design arithmetic: serial subtraction or multiplication problems
- 3 difficulty levels (determined by piloting): easy, medium, hard
- Each trial: fixation 1s → problem display (max 8s) → response → feedback 0.5s
- 30 trials per block, 90 trials per session

### EEG Recording
- 32-channel actiCAP, 500 Hz, CPz reference
- Impedance < 15 kOhm
- Record during entire session (baseline, task, post-rest)

### Analysis Plan
1. **Frontal theta power**: F3/F4/Fz, 4-8 Hz, Welch PSD, log-transformed — paired t-test active vs sham
2. **Behavioral**: Accuracy (ANOVA: session × difficulty); RT (ANOVA: session × difficulty)
3. **Correlation**: Δtheta vs Δaccuracy (Pearson r, one-tailed)
4. **Theta-gamma PAC**: Modulation Index (Tort et al. 2010) at F3/F4/Fz
5. **Bayes factors**: BF10 reported alongside p-values
6. **Non-parametric permutation test** as robustness check

### Power Analysis
- Based on our MAT dataset: theta power effect d=1.2 between conditions
- N=15 for 95% power at d=1.2, α=0.05
- Conservative estimate d=0.5 → N=33 for 80% power
- **Target N=40** (allowing 15% dropout)

### Preregistration
- OSF preregistration prior to data collection
- Data and analysis code shared on GitHub
- EEG preprocessing: MNE-Python, ICA for artifact removal, automated pipeline

## Timeline
1. Protocol refinement & IRB approval: 2 months
2. Piloting (N=5): 1 month
3. Data collection (N=40): 3 months
4. Analysis & writing: 2 months
5. Total: ~8 months
