# 60-Second Judge Explanation

My project asks whether EEG brain signals can be used to classify cognitive state, specifically baseline rest versus cognitive workload during mental arithmetic. I used an open EEG dataset where each subject has a rest recording and a mental arithmetic recording.

The system has four main parts. First, it reads the raw EEG recordings. Second, it splits them into small time windows. Third, it extracts features from each window, including brainwave bandpower in theta, alpha, beta, and gamma bands, statistical features like variance and entropy, and connectivity features based on correlations between EEG channels. Fourth, it trains machine learning models to classify each window as rest or workload.

A major part of my project is that I did not only report accuracy. I also calculated sensitivity, specificity, positive predictive value, negative predictive value, ROC-AUC, precision-recall AUC, and confidence intervals. I also ran statistical tests to identify which EEG features significantly changed between rest and workload.

The most important scientific control is subject-wise evaluation. That means data from the same person is not placed in both training and testing sets, which prevents the model from simply memorizing individual EEG patterns. This makes the evaluation more realistic.
