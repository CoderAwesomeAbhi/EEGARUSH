# ISEF 2026 — 60-Second Pitch Script

## The "Elevator Pitch"

---

**Intro (10s):**
"Did you know your brain produces a specific electrical signal when you do math? It's called frontal theta — a 4-8 Hz oscillation that gets stronger the harder you think. But here's the problem: most studies only test this on one small group, with one dataset. My question was: does this signal actually generalize across different people and different experiments?"

---

**Methods (15s):**
"I analyzed three independent public EEG datasets — 102 participants total, each doing mental arithmetic. I extracted brainwave features, then used a strict validation where I hold out every person completely — their data never touches the model during training. I also built an interpretable one-dimensional workload axis called SNWA that shows you exactly which brain features drive the classification."

---

**Results (20s):**
"The result is clear: frontal theta power increases during arithmetic in ALL three datasets with p<0.001. My classifiers achieved 78-85% AUC across datasets. When I trained on one dataset and tested on another — a much harder test — performance stayed above chance at 60-68% AUC. That tells us there's a real, generalizable brain signal here, not just dataset-specific noise."

---

**Significance (15s):**
"This matters because it transforms workload detection from a lab curiosity into a validated scientific finding. It means EEG-based cognitive monitoring could eventually help with adaptive learning tools, attention-aware interfaces, and even early detection of cognitive fatigue — all grounded in a brain signal that we now know generalizes across people and experiments."

---

## Backup: Extended Version (90 seconds)

**Intro (12s):**
"Your brain changes its electrical patterns when you concentrate. Specifically, frontal theta power goes up, alpha power goes down. Scientists have known this for decades — but nobody has proven it generalizes across different datasets with strict subject-independent validation. That's what I did."

**Methods (20s):**
"I pulled together three public datasets: PhysioNet EEGMAT (36 subjects), STEW (48 subjects), and OpenNeuro DS007262 (18 subjects) — 102 total. I extracted 805 features per 4-second window: bandpower, Hjorth parameters, entropy, channel correlations. Then I used leave-one-subject-out validation — each person is completely held out during training — and repeated this independently for each dataset plus a combined analysis."

**Results (25s):**
"Frontal theta increased during workload across all three datasets — highly significant, large effect size. My classifiers achieved 0.78-0.85 ROC-AUC. The SNWA model — a single interpretable workload score using just 8 features — matched the performance of complex black-box models. Cross-dataset transfer gave 0.60-0.68 AUC, proving the brain signal generalizes beyond any single recording setup."

**Significance (18s):**
"This is the largest multi-dataset replication of the theta-arithmetic effect I know of. It provides a validated, interpretable biomarker for cognitive workload. Applications include adaptive learning systems, BCI for accessibility, and cognitive health monitoring. My code is fully open-source and reproducible."

**Close (5s):**
"I'm happy to walk through the live demo or the real-time BCI prototype — thank you!"

---

## Key Stats to Memorize

| Metric | Value |
|--------|-------|
| Total subjects | N=102 |
| Datasets | 3 independent |
| LOSO AUC (combined) | 0.851 |
| SNWA AUC (combined) | 0.815 |
| Cross-dataset transfer | 0.60-0.68 AUC |
| Frontal theta effect | p<0.001, d=0.84 |
| Features in SNWA | 8 (interpretable) |
| Subject-level AUC range | 0.42-0.97 |

## Common Questions to Anticipate

**Q: Why not deep learning?**
A: Interpretability. I show that 8 simple features match complex models. Science needs understanding, not just accuracy.

**Q: What about artifacts?**
A: No artifact rejection — kurtosis and peak-to-peak amplitude are used as *features*, so artifacts become informative rather than noise.

**Q: Can this be used for ADHD diagnosis?**
A: Not yet — 15% of subjects are at chance. Individual-level reliability needs more work before clinical use.

**Q: How did you get 3 datasets to work together?**
A: Eight common channels (F3, F4, P3, P4, O1, O2, F7, F8) and identical feature extraction code applied to all three.
