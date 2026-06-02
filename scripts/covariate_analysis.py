"""Covariate analysis for anxiety/fatigue confound."""
import pandas as pd, numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

pred = pd.read_csv("C:/Users/abhij/Downloads/bioarxivarjun/outputs_reproduced/models/predictions_loso.csv")
pred = pred[pred["model"] == "svm_rbf"]

aucs = {}
for s in pred["subject_id"].unique():
    sp = pred[pred["subject_id"] == s]
    try:
        aucs[s] = roc_auc_score(sp["true_label"], sp["score_workload"])
    except:
        pass

df_auc = pd.DataFrame([(k, v) for k, v in aucs.items()], columns=["subject_id", "auc"])
print(f"Subjects with AUC: {len(df_auc)}")

si = pd.read_csv("C:/Users/abhij/Downloads/bioarxivarjun/data/raw/eegmat/subject-info.csv")
si["subject_id"] = si["Subject"].str.strip()
df_m = df_auc.merge(si, on="subject_id", how="inner")
print(f"Merged: {len(df_m)}")

for covar in ["Age", "Number of subtractions"]:
    r, p = stats.spearmanr(df_m[covar].values, df_m["auc"].values)
    print(f"  {covar} vs AUC: rho={r:.4f} p={p:.4f}")

m_auc = df_m[df_m["Gender"] == "M"]["auc"].values
f_auc = df_m[df_m["Gender"] == "F"]["auc"].values
print(f"  Gender M={len(m_auc)} F={len(f_auc)}: M_mean={m_auc.mean():.4f} F_mean={f_auc.mean():.4f}")
u, p = stats.mannwhitneyu(m_auc, f_auc)
print(f"  Gender MW: p={p:.4f}")

for q in [0, 1]:
    sub = df_m[df_m["Count quality"] == q]["auc"]
    print(f"  Count quality {q}: mean={sub.mean():.4f} n={len(sub)}")
