"""Reconstruct the paper by appending missing sections to the truncated tex file"""
with open('paper/tex/main.tex', 'r') as f:
    tex = f.read()

# Find the end of current content (after Subject-Level Variability)
# The file currently ends with the bibliography-like references diagram caption
end_idx = tex.rfind('\\end{figure}')
# Take everything before the last figure
truncated = tex[:end_idx]

# Now append discussion, limitations, future directions, ethics, data/code, references
# I'll reconstruct from the PDF text but adapted for our new analysis (no Riemannian)

appendix = r"""

\section{Discussion}

\subsection{Biological Interpretation}

Frontal theta is thought to reflect coordinated activity in the anterior cingulate cortex and
prefrontal regions engaged during cognitive control \cite{cavanagh2014}, who reported that mid-frontal
theta is modulated by both cognitive control demands (Cohen's d=0.71, meta-analysis across 122
studies) and anxiety. The concurrent alpha suppression likely reflects thalamo-cortical
disinhibition, enabling cortical processing in task-relevant regions \cite{klimesch1999}. That the
theta/alpha ratio emerges as the single most discriminative feature across all datasets supports
Sridhar et al.\ \cite{sridhar2022}'s conclusion that this ratio captures the balance of cognitive
engagement (theta) and cortical idling (alpha) more effectively than either band alone.

Our exploratory analysis of theta-gamma phase-amplitude coupling found robust coupling during
both rest and arithmetic (surrogate-normalized Z = 11.2), but no significant task-related
modulation after proper subject-level averaging with surrogate testing (meta-analytic d=0.07,
90\% CI $[-0.02, 0.15]$, TOST-consistent with equivalence). This differs from intracranial
findings of task-modulated PAC \cite{lisman2013, canolty2006} and may reflect the substantial
attenuation of phase-amplitude relationships at the scalp level due to volume conduction
\cite{tort2010}, or a true absence of workload-specific PAC modulation in scalp EEG. The small MI
values (0.001--0.002) and the presence of coupling in both conditions suggest that theta-gamma
PAC is a general property of ongoing EEG activity rather than a specific marker of cognitive
engagement. A properly powered study with dedicated PAC-optimized recordings and a causal
manipulation (tACS, as outlined in Future Directions) is needed to resolve this question.

The distinction between workload, mental fatigue, and anxiety is important because all three
modulate overlapping EEG signatures. Borghini et al.\ \cite{borghini2014} showed that prolonged
cognitive task performance produces theta increases and alpha decreases similar to those we
observe, and Cavanagh and Shackman \cite{cavanagh2014} documented anxiety-related theta
augmentation. Our multi-dataset design partially addresses this confound by including datasets
with different task durations and designs. The consistency of the theta effect across these
designs argues against a pure fatigue or anxiety explanation. Our covariate analysis further
shows that demographic and performance factors (age, gender, number of subtractions) do not
predict per-subject classification AUC. However, we cannot fully dissociate these constructs
without subjective workload and fatigue ratings (NASA-TLX \cite{hart1988}), physiological
measures (heart rate variability, pupil dilation), or a task manipulation that independently
varies workload and arousal. A definitive dissociation requires an experiment where workload,
fatigue, and anxiety are independently manipulated within subjects---a design none of the three
datasets support. This limitation is shared with the broader EEG workload literature.

\subsection{Comparison to Prior MAT Studies}

Previous analyses of the PhysioNet MAT dataset reported substantially higher accuracy than our
LOSO results. Hakimi et al.\ \cite{hakimi2023} achieved 91.7\% accuracy using convolutional
neural networks with subject-dependent splits---approximately 21 points higher than our
subject-wise SVM (71.0\%) on the 10-channel intersection. Nguyen and Huynh \cite{nguyen2025}
reported 85.7\% F1 using KNN with random-window validation. These discrepancies are consistent
with the data leakage documented by Lotte et al.\ \cite{lotte2018}: when the same subject's
windows appear in both training and test sets, the classifier can learn subject-specific
recording artifacts rather than task-general neural signals. Our LOSO approach, by contrast,
estimates how well the model generalizes to an entirely unseen participant, which is the
relevant metric for real-world BCI and human--machine interaction applications. Even with the
full 805-feature set, our MAT LOSO AUC of 0.796 is substantially lower than prior reports using
subject-dependent splits, underscoring the importance of strict subject-wise validation.

\subsection{Methodological Contributions}

The multi-dataset replication framework addresses a critical gap in EEG workload research.
Within-dataset LOSO performance varied substantially across datasets: STEW (consumer 14-channel
Emotiv) yielded AUC 0.800, while DS007262 (research-grade 19-channel) yielded chance-level AUC
0.467 with the 8-channel intersection. This counterintuitive result suggests that dataset
characteristics---task design, class balance, and recording consistency---affect classification
performance more strongly than hardware quality. STEW's superior performance likely reflects its
balanced classes (50\% workload) and within-subject trial structure, which reduces subject-identity
leakage compared to MAT's blocked design (25\% workload class).

Cross-dataset near-transfer for full models was at chance levels (AUC 0.47--0.51),
demonstrating that the 8-channel common feature set is insufficient for dataset-independent
classification by standard classifiers. However, SNWA significantly generalized to DS007262
(AUC 0.618, p=0.005), indicating that subject-normalized feature selection captures
individual-level workload modulation that survives cross-dataset recording differences. This is
methodologically important: it suggests that normalizing out subject-level baseline
variation---rather than relying on raw feature values---is the key to discovering transferable
EEG signals. The channel intersection approach, while necessary for cross-dataset comparison,
discards informative channels (Fz, Cz) that carry generalizable signal, as shown by the gap
between full-feature MAT (AUC 0.796) and intersection-reduced MAT (AUC 0.709). Multiple
transfer confounders contribute: electrode configurations, task designs, populations, and
instruction differences. Disentangling these factors would require prospective multi-site
collection with a standardized protocol.

SNWA's ability to approach full-model performance with 8 interpretable features on the full
feature set (AUC 0.761 vs 0.796) and to generalize to an unseen dataset has practical
implications. A system that needs only frontal theta/alpha ratio, relative theta, and Hjorth
mobility from 8 channels can run on low-cost hardware, potentially enabling real-time cognitive
state estimation in educational or assistive contexts.

\subsection{Limitations}

Eight limitations merit explicit discussion. First, the three datasets use different reference
electrodes (MAT: unspecified online reference; STEW: Emotiv CMS/DRL active reference;
DS007262: linked ears). Ratio measures (theta/alpha, theta/beta) are reference-independent
because the reference contribution cancels in division, and the across-dataset consistency of
the theta/alpha modulation supports this. However, absolute bandpower values are
reference-dependent \cite{klimesch1999}. Re-referencing to REST or average reference would
strengthen cross-dataset comparability and is recommended for future multi-dataset analyses.

Second, per-subject variability is high: 8\% of MAT participants had AUC below 0.6, and
per-subject AUC ranged from 0.34 to 1.00. Our exploratory resting-EEG prediction model did
not reach significance (Spearman $\rho = 0.32$, p = 0.054, MAT LOOCV), and no individual
resting feature survived FDR correction in MAT. Only one feature survived FDR correction in
STEW (occipital theta, q = 0.005). The consistent predictor direction across datasets is
suggestive but does not constitute formal replication. A properly powered prospective study
(N$\ge$200) with pre-registered predictors is needed before clinical screening recommendations
can be made. The 8--10\% resistant-subject rate is consistent with the 20\% estimate from the
BCI literature \cite{lotte2018}, though the lower bound may reflect our lenient AUC $>0.6$
threshold versus typical within-subject criteria.

Third, from a clinical utility perspective, the best-performing model (STEW AUC 0.800) achieves
this at a default probability threshold of 0.5. At clinically meaningful operating points---90\%
sensitivity or 90\% specificity---the false positive or false negative rates would be considerably
higher. For safety-critical applications (e.g., air traffic control monitoring), a false negative
rate above 10\% is typically unacceptable. Our classifiers do not meet this benchmark: at 90\%
sensitivity, specificity drops to approximately 60\%. Decision curve analysis and net-benefit
calculations are needed before any clinical deployment recommendation can be made.

Fourth, cross-dataset transfer for full models was at chance (AUC 0.47--0.51). SNWA achieved
significant transfer (AUC 0.618, p = 0.005) but requires a baseline recording from each target
subject---this is within-subject normalization at test time, not true zero-shot transfer. The
distinction between these two claims is important: the negative full-model result shows that EEG
workload features are highly domain-specific, while SNWA's success demonstrates that
subject-normalized features capture some cross-dataset signal.

Fifth, this is a secondary analysis of existing public data (N$\le$48 per dataset). Prospective
collection with standardized protocols across multiple sites and N$\ge$200 per dataset would
provide stronger evidence. Sixth, overlapping EEG windows (4~s, 50\% overlap) introduce temporal
correlation that inflates the effective sample size; our subject-level bootstrap partially
addresses this by resampling subjects rather than windows \cite{lotte2018}. Seventh, the 8-channel
intersection discards informative channels (Fz, Cz) that carry workload information
\cite{gevins1997}. The gap between full-feature MAT (AUC 0.796) and intersection-reduced MAT
(AUC 0.709) confirms this cost.

Eighth, source localization was not performed. Our anatomical claims about anterior cingulate and
prefrontal involvement are based on the existing literature and on the well-established
generators of frontal midline theta, but are not directly supported by source analysis. Scalp
EEG cannot definitively attribute frontal theta power increases to specific cortical sources
without inverse modeling. Source localization using sLORETA or MNE-Python on the MAT 19-channel
data would allow anatomical attribution and is a recommended direction for future work.

\subsection{Future Directions}

The results suggest several concrete next steps. First, a properly powered prospective study
(N$\ge$200 per dataset, standardized 10-20 montage, average reference, pre-registered analysis
plan) is needed to establish EEG workload classification benchmarks. Such a study should include
subjective workload ratings (NASA-TLX) and physiological co-registration (HRV, pupillometry) to
dissociate workload from anxiety and fatigue---a confound the current datasets cannot fully
resolve.

Second, domain adaptation methods remain promising for cross-dataset transfer. Approaches such
as subspace alignment \cite{jayaram2016} and CORAL should be tested with the full channel sets
rather than the 8-channel intersection used here.

Third, the presence of theta-gamma coupling in both conditions---without task modulation---
warrants further investigation. A preregistered tACS experiment (6~Hz frontal, F3/F4, 2~mA,
sham-controlled crossover, N=40) could test whether induced theta oscillations causally modulate
PAC and arithmetic performance, potentially revealing a task-specific coupling effect that our
correlational analysis was underpowered to detect.

Fourth, source localization using standardized low-resolution electromagnetic tomography
(sLORETA) on the MAT 19-channel data would allow anatomical attribution of the frontal theta
effect to specific cortical sources (e.g., anterior cingulate, prefrontal cortex).

Fifth, closed-loop experiments where task difficulty adapts to real-time theta power could
establish causal relationships between theta oscillations and cognitive performance. Finally,
clinical populations with attentional deficits (ADHD, mild cognitive impairment) may show altered
theta dynamics, providing a physiological biomarker.

\subsection{Ethics Statement}

This work is a secondary analysis of publicly available, de-identified EEG datasets. No new
human-subject data were collected. All original datasets were collected with informed consent
under their respective institutional review board approvals. The PhysioNet EEGMAT dataset
(doi:10.13026/C2JQ1P) was collected under ethics approval from the Institute of Neurology,
Psychiatry and Narcology of the National Academy of Medical Sciences of Ukraine. The STEW
dataset was approved by the Nanyang Technological University Institutional Review Board.
OpenNeuro DS007262 (doi:10.18112/openneuro.ds007262.v1.0.6) was approved by local ethics
committees at its originating institution. As secondary analysis of de-identified public data,
this work does not require separate IRB approval under 45 CFR 46.104(d)(4). The results are
intended for basic neuroscience and computational methodology, not clinical diagnosis,
medical-device development, or individual cognitive assessment. Data use complies with the
original datasets' terms of use.

\subsection{Data and Code Availability}

The original raw datasets are publicly available from: PhysioNet EEGMAT at
\url{https://doi.org/10.13026/C2JQ1P}, STEW at \url{https://huggingface.co/datasets/stew},
and OpenNeuro DS007262 at \url{https://doi.org/10.18112/openneuro.ds007262.v1.0.6}. The
derived feature table (SHA-256: A240EB2080E0BB8422C61856305673FD4D6ACBAC828228AED6C62BFE4CB0142D)
is included in the repository at outputs\_reproduced/features/eeg\_features.csv. All analysis
code is at \url{https://github.com/agangarapu/eeg-workload-replication}. Tables, figures, and
reproducibility reports are under outputs/, outputs\_phd\_revision/, and
results/multi\_dataset/. Paper source (LaTeX) is in paper/. A requirements.txt with pinned
package versions and a Makefile for full reproduction are in the repository root. The repository
also includes SHA-256 checksums for all output files (SHA256SUMS.txt).

\subsection{Acknowledgments}

The author thanks the creators of the PhysioNet EEGMAT, STEW, and OpenNeuro DS007262 datasets
for making their data publicly available. Computational resources were provided by the author's
personal computing infrastructure. No external funding was received for this work.

\subsection{Author Contributions}

Abhijay Gangarapu designed the analysis, implemented the code, ran all analyses, interpreted
results, and prepared the manuscript.

\subsection{Competing Interests}

The author declares no competing interests. The author has no financial relationship with any
EEG hardware company (Emotiv, g.tec, BrainProducts, etc.), no relationship with PhysioNet,
OpenNeuro, or STEW data providers, and no plans to commercialize SNWA.

\bibliographystyle{ieeetr}
\bibliography{references}

\end{document}
"""

# Write the reconstructed file
with open('paper/tex/main.tex', 'w', encoding='utf-8') as f:
    f.write(truncated + appendix)

print(f'Written {len(truncated + appendix)} chars')
print('Done!')
