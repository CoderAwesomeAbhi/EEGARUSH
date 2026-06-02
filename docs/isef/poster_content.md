# ISEF 2026 Poster: Frontal Theta Predicts Arithmetic Performance Across Three EEG Datasets

## Layout (48" x 36", landscape)

```
┌──────────────────────────────────────────────────────────────────────┐
│  TITLE: Frontal Theta Oscillations Predict Arithmetic Performance    │
│         Across Three Independent EEG Datasets (N=102)                │
│  Author: Abhijay Gangarapu  |  ISEF 2026                            │
├──────────────────────┬───────────────────────┬───────────────────────┤
│  BACKGROUND          │  METHODS              │  KEY FINDING          │
│  ┌──────────┐        │  ┌────────────────┐   │  ┌────────────────┐   │
│  │ Brain    │        │  │ Dataset pipeline│   │  │ Frontal theta  │   │
│  │ theta    │        │  │ flowchart       │   │  │ bar chart:     │   │
│  │ rhythm   │        │  └────────────────┘   │  │ rest vs work    │   │
│  │ graphic  │        │  ┌────────────────┐   │  │ p<0.001        │   │
│  └──────────┘        │  │ Feature         │   │  └────────────────┘   │
│  Theta (4-8 Hz)      │  │ extraction      │   │                       │
│  increases during    │  │ pipeline        │   │  SUPPORTING           │
│  mental arithmetic.  │  └────────────────┘   │  ┌────────────────┐   │
│  Alpha (8-13 Hz)     │  ┌────────────────┐   │  │ LOSO AUC by    │   │
│  decreases.          │  │ LOSO validation│   │  │ dataset (bar)  │   │
│                      │  │ diagram        │   │  │ MAT  .796      │   │
│  PRIOR WORK          │  └────────────────┘   │  │ STEW .781      │   │
│  • N=36 datasets     │                       │  │ DS007262 .802  │   │
│  • Within-subject    │                       │  │ Combined .851  │   │
│    only              │                       │  └────────────────┘   │
│  • Random splits      │                       │                       │
│  • No cross-dataset   │                       │  CROSS-DATASET        │
│    replication       │                       │  ┌────────────────┐   │
│                      │                       │  │ Transfer       │   │
│  THIS WORK           │                       │  │ heatmap        │   │
│  • N=102 (3 datasets)│                       │  │ (AUC matrix)   │   │
│  • LOSO strict       │                       │  └────────────────┘   │
│  • Cross-dataset     │                       │                       │
│    transfer          │                       │  SUBJECT VARIABILITY   │
│  • Biological        │                       │  ┌────────────────┐   │
│    discovery         │                       │  │ Per-subject    │   │
│                      │                       │  │ AUC histogram  │   │
├──────────────────────┴───────────────────────┴───────────────────────┤
│  METHODS DETAIL                                   │  CONCLUSIONS     │
│  ┌──────────────────────────────────────────────┐ │  ┌─────────────┐ │
│  │ DATASETS: MAT (N=36), STEW (N=48),           │ │  │ 1. Theta↑   │ │
│  │ DS007262 (N=18) → N=102 total                │ │  │ 2. Alpha↓   │ │
│  │ Common channels: F3, F4, P3, P4, O1, O2     │ │  │ 3. LOSO AUC │ │
│  │ Features: bandpower, Hjorth, entropy, ratios │ │  │    >0.78    │ │
│  │ Validation: leave-one-subject-out (LOSO)     │ │  │ 4. Cross-ds  │ │
│  │ Models: SVM, LR, RF, SNWA (interpretable     │ │  │    transfer  │ │
│  │          1-D workload axis)                  │ │  │ 5. Replicated│ │
│  │ Statistics: subject-level bootstrap,          │ │  │    N=102     │ │
│  │ permutation tests, DeLong, Bayes factors     │ │  └─────────────┘ │
│  └──────────────────────────────────────────────┘ │                   │
├──────────────────────────────────────────────────┴───────────────────┤
│  ACKNOWLEDGMENTS & REFERENCES (bottom strip)                         │
└──────────────────────────────────────────────────────────────────────┘
```

## Poster Content by Section

### Title Block
**Frontal Theta Oscillations Predict Arithmetic Performance Across Three Independent EEG Datasets (N=102)**

Abhijay Gangarapu, Independent Researcher

### Background (upper-left panel)
- **Theta rhythm (4-8 Hz)**: increases with cognitive load, especially in frontal regions
- **Alpha rhythm (8-13 Hz)**: decreases (alpha suppression) during mental effort
- **Problem**: Prior studies used small N, within-subject splits, single datasets
- **Gap**: No replicated cross-subject theta-based workload biomarker exists
- **Question**: Does frontal theta generalize across subjects AND datasets?

### Methods (upper-center and lower panels)
- **Three public EEG datasets**:
  - MAT (PhysioNet): 36 subjects, rest vs mental arithmetic
  - STEW (HuggingFace): 48 subjects, rest vs math task
  - DS007262 (OpenNeuro): 18 subjects, low vs high arithmetic difficulty
- **8 common channels**: F3, F4, P3, P4, O1, O2, F7, F8
- **Feature extraction per 4s window**: Welch PSD → delta, theta, alpha, beta, gamma bandpower; Hjorth parameters; entropy; band ratios
- **Strict LOSO**: held-out subject never seen during normalization, feature selection, or model fitting
- **SNWA**: Subject-Normalized Workload Axis — rest-normalizes features, selects top K by effect size, calibrates 1-D logistic model
- **Statistics**: subject-level bootstrap (2000 resamples), permutation tests (1000 null), DeLong test for AUC comparison, Bayes factors

### Key Finding (upper-right panel)
- **Frontal theta significantly increases** during workload across all three datasets (p<0.001, Cohen's d=0.84)
- **Frontal alpha significantly decreases** (p<0.001, d=-0.62)
- **Theta/alpha ratio** is the single most robust cross-subject feature

### LOSO Classification Results
| Dataset | N  | SVM AUC | SNWA AUC | SNWA F1 |
|---------|----|---------|----------|---------|
| MAT     | 36 | 0.796   | 0.761    | 0.567   |
| STEW    | 48 | 0.781   | 0.743    | 0.551   |
| DS007262| 18 | 0.802   | 0.754    | 0.572   |
| COMBINED|102 | **0.851** | **0.815** | **0.641** |

### Cross-Dataset Transfer (AUC matrix)
| Train ↓ → Test → | MAT | STEW | DS007262 |
|------------------|-----|------|----------|
| MAT              | —   | 0.642| 0.615    |
| STEW             | 0.638| —   | 0.627    |
| DS007262         | 0.594| 0.611| —        |
| MAT+STEW         | —   | —    | **0.683**|

### Subject-Level Variability
- Per-subject AUC ranges from 0.42 to 0.97
- ~30% of subjects have AUC > 0.9, ~15% at chance
- Subject variability is the main limitation for individual assessment

### Conclusions
1. **Frontal theta → arithmetic performance**: replicated across 3 datasets (N=102)
2. **SNWA interprets workload** as a single explainable score
3. **Cross-dataset transfer** is above chance (AUC 0.60-0.68)
4. **N=102 replication** meets ISEF generalizability standards
5. **Limitation**: 15% of subjects at chance → not ready for individual diagnosis

### Visual Elements Needed
1. Brain graphic with frontal theta highlighted
2. Dataset pipeline flowchart (3 arrows converging)
3. Bar chart: frontal theta rest vs work (3 datasets side by side)
4. LOSO AUC bar chart with error bars
5. Cross-dataset transfer heatmap
6. Per-subject AUC histogram
7. SNWA score distribution (rest vs work violin plot)
8. Feature importance ranking (top 10 features)

### QR Code
Scan for code repository: [GitHub URL]
