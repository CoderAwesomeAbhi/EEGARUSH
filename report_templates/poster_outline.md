# Science Fair Poster Outline

## Title
Machine Learning Classification of Cognitive Brain States Using EEG Spectral and Statistical Features

## Question
Can EEG spectral, statistical, and connectivity features classify baseline/rest versus cognitive workload during mental arithmetic?

## Hypothesis
EEG features related to neural oscillations, entropy, and channel relationships will differ between rest and workload, allowing machine learning models to classify cognitive state above chance.

## Background
- EEG measures electrical brain activity.
- Neural oscillations are commonly analyzed by frequency bands.
- Mental arithmetic induces cognitive workload.
- Machine learning can learn feature patterns that distinguish conditions.

## Dataset
- PhysioNet EEG During Mental Arithmetic Tasks
- Rest/background EEG: `_1` files
- Mental arithmetic EEG: `_2` files
- 23 EEG channels
- EDF format

## Procedure
1. Download EEG data.
2. Segment signals into 4-second windows.
3. Extract spectral, statistical, and connectivity features.
4. Run paired feature statistics.
5. Train ML classifiers.
6. Evaluate using subject-wise testing.
7. Compare models and feature types.

## Results
Insert:
- confusion matrix
- model comparison table
- ROC curve
- bandpower graph
- top feature importance graph
- top significant features table

## Conclusion
Write this only after running the real dataset. Do not invent performance numbers.

## Limitations
- Dataset is controlled, not real-world EEG.
- Binary rest/workload labels are not the same as every possible mental state.
- EEG can contain subject-specific patterns.
- Subject-wise evaluation is essential.

## Future Work
- Test multi-class states.
- Add deep learning using raw EEG.
- Use more datasets.
- Improve artifact rejection.
- Test real-time classification.
