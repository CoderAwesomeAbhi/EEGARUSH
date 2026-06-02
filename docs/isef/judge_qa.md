# ISEF 2026 — Judge Q&A Preparation

## Tier 1: Must-Know (Confident, Detailed Answers)

### Q1: What is the main biological finding?
**A:** Frontal theta power (4-8 Hz) significantly increases during mental arithmetic compared to rest, replicated across three independent datasets (N=102, p<0.001, Cohen's d=0.84). Alpha power (8-13 Hz) simultaneously decreases. The theta/alpha ratio is the single most robust cross-subject feature. This confirms Klimesch's (1999) theta/alpha framework at an unprecedented scale and with strict subject-wise validation.

### Q2: How is your work different from prior studies?
**A:** Three key differences:
1. **Strict LOSO validation**: prior work often uses random window splits or within-subject partitions, leaking subject identity. I hold out every subject completely.
2. **Multi-dataset replication (N=102)**: most studies use one small dataset (N=36 or less). I integrate three.
3. **Cross-dataset transfer**: I train on one dataset, test on another — a much harder, more realistic test of generalizability.

### Q3: What is SNWA and why is it better than a black-box classifier?
**A:** SNWA (Subject-Normalized Workload Axis) rest-normalizes each feature within subject, selects the top K features by effect size inside each training fold, weights them, and calibrates a one-dimensional logistic model. It's better because:
- **Interpretable**: you can see exactly which brain features drive the decision
- **Compact**: 8 features vs 805 in the full model
- **Comparable performance**: AUC 0.815 vs 0.851 for full SVM
- **Better calibration**: Brier score 0.184 vs 0.205 for SVM

### Q4: How do you know it's not just artifacts or noise?
**A:** Four negative controls:
1. **Label permutation**: shuffle labels within each LOSO fold → AUC drops to ~0.5
2. **Circular shift**: shift labels within subject → chance performance
3. **Gaussian features**: replace real features with random noise → AUC ~0.5
4. **Permutation test**: 1000 null permutations, real AUC exceeds 99% of null (p<0.001)
All controls confirm the signal is real and task-related.

### Q5: Why only 8 common channels across datasets?
**A:** MAT uses 19 channels (10-20 system), STEW uses 14 (Emotiv EPOC+), DS007262 uses 19 (10-20). The 8-channel intersection (F3, F4, P3, P4, O1, O2, F7, F8) is conservative but sufficient — frontal theta is captured by F3 and F4. Having fewer channels that exist in all datasets is better than having more channels that require imputation or mismatched montages.

### Q6: What are the limitations?
**A:** Five main limitations:
1. **15% of subjects at chance**: per-subject AUC ranges from 0.42 to 0.97
2. **Not prospective**: all data are from existing public datasets
3. **Moderate cross-dataset transfer**: AUC 0.60-0.68, indicating dataset-specific confounds
4. **No real-time validation**: the demo simulates real-time but uses pre-recorded data
5. **Correlated windows**: overlapping windows reduce effective sample size

---

## Tier 2: Should Know (Brief, Clear Answers)

### Q7: What statistical methods did you use?
**A:** Subject-level bootstrap (2000 resamples), permutation tests (1000 null), DeLong test for AUC comparison, Wilcoxon signed-rank for paired comparisons, Benjamini-Hochberg FDR correction, and Bayes factors for null/alternative evidence.

### Q8: Couldn't you use deep learning for better accuracy?
**A:** I could, but that would sacrifice interpretability. The goal is not to maximize accuracy on one dataset, but to understand whether a reliable physiological signal exists. SNWA's 8 features matching a full SVM shows the signal is simple and biological, not a black-box fluke.

### Q9: How did you handle class imbalance?
**A:** MAT has 3,186 rest vs 1,080 workload windows (3:1 ratio). I used class-balanced models (class_weight="balanced"), and reported F1, sensitivity, specificity alongside accuracy, since accuracy alone is misleading on imbalanced data.

### Q10: What is subject-level bootstrap and why use it?
**A:** Instead of resampling windows (which are correlated within subject), I resample entire subjects with replacement 2000 times. This preserves the subject as the independent unit and gives correct confidence intervals for cross-subject generalization.

### Q11: Can this work with consumer EEG headsets?
**A:** Possibly — the 8 common channels include F3/F4, which consumer headsets like Muse or Emotiv EPOC+ have. The SNWA model is lightweight and could run on a smartphone. But I haven't tested this yet; the prospective study protocol proposes exactly this experiment.

### Q12: What was the hardest part?
**A:** Feature harmonization across datasets. Each dataset uses different channel naming (FP1 vs Fp1), different montages, different sampling rates, and different task designs. Getting them to align required careful mapping and conservative intersection.

---

## Tier 3: Nice to Know (Impressive Depth)

### Q13: What is the leakage theorem?
**A:** For subject s with m_s windows, if each window is independently assigned to train with probability p, the probability the subject appears in both train AND test is: P(leakage_s) = 1 - p^{m_s} - (1-p)^{m_s}. For typical window counts (m_s > 50) and p=0.75, this is essentially 1.0, meaning random splits guarantee subject leakage.

### Q14: Why is feature selection inside LOSO critical?
**A:** If you select features on the full dataset before LOSO, the held-out subject's data influences which features are used. This violates subject-independence and inflates performance. In my pipeline, feature ranking, selection, and weighting all happen strictly inside each training fold.

### Q15: What's next for this research?
**A:** 1) Prospective data collection with consumer EEG (protocol in the repository). 2) Real-time closed-loop BCI where the workload score modulates task difficulty. 3) Clinical validation with attention-deficit populations. 4) Cross-task transfer (does theta predict workload for other cognitive tasks?).

### Q16: How did you verify reproducibility?
**A:** Full reproducibility package: requirements.txt with pinned versions, Makefile, environment.yml, smoke test with synthetic data, and a Colab notebook that runs the entire pipeline from scratch. The random seed (42) is fixed everywhere.

### Q17: How do you handle multiple comparisons?
**A:** Benjamini-Hochberg FDR correction at q=0.05 for feature-level tests. For the primary LOSO results, I report unadjusted values but only claim significance at p<0.05 after correction.

### Q18: What is the Bayes factor analysis showing?
**A:** For the primary LOSO AUC (SVM: 0.796), the Bayes factor BF_10 > 100, meaning extreme evidence for the alternative (real signal). For cross-dataset transfer (AUC ~0.62), BF_10 ≈ 3-10, meaning moderate evidence — real but weaker.

---

## Quick Reference: Key Numbers

| Category | Value | Interpretation |
|----------|-------|----------------|
| Total N | 102 | 3 datasets combined |
| MAT AUC | 0.796 | SVM LOSO on primary dataset |
| STEW AUC | 0.781 | Replicated on second dataset |
| DS007262 AUC | 0.802 | Replicated on third dataset |
| Combined AUC | 0.851 | N=102 pooled analysis |
| SNWA AUC | 0.815 | 8-feature interpretable model |
| Cross-transfer | 0.60-0.68 | Above chance but modest |
| Frontal theta d | 0.84 | Large effect size |
| Subjects at chance | ~15% | Main limitation |
| Permutation p | <0.001 | Real signal confirmed |

## Potential Judge Profiles

**Neuroscience judge**: Focus on theta/alpha framework, Klimesch citation, biological plausibility, frontal lobe role in arithmetic. Emphasize the replicated effect across 3 datasets.

**Engineering/CS judge**: Focus on SNWA's efficiency, LOSO methodology, leakage theorem, cross-dataset transfer. Emphasize that fewer features achieve comparable performance.

**Statistics judge**: Focus on bootstrap, permutation tests, DeLong, Bayes factors, FDR correction. Emphasize that all inference is subject-level, not window-level.

**Clinical judge**: Focus on limitations (15% at chance), no clinical claims, path toward real-world validation. Emphasize ethical caution and the prospective study protocol.
