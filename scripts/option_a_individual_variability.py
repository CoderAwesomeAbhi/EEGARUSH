"""
Option A: Individual Variability Analysis (v2)
===============================================
Predict subject classifiability from resting-state EEG.
Key question: Why are some subjects "classifier-resistant"?

Approach:
1. Per-subject resting EEG profiles (mean across rest windows)
2. Identify which resting features correlate with LOSO AUC
3. Build two-stage model: screen subjects -> classify only good ones
4. Generate publication-quality figures
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import ElasticNetCV, RidgeCV, LogisticRegressionCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from scipy.stats import false_discovery_control
import warnings
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings('ignore')

BASE = Path('C:/Users/abhij/Downloads/bioarxivarjun')
OUT = BASE / 'outputs_phd_revision'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
PAPER_FIG = BASE / 'paper' / 'figures'

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading feature matrix...")
feats = pd.read_csv(BASE / 'outputs_reproduced' / 'features' / 'eeg_features.csv', low_memory=False)

print("Loading LOSO predictions...")
preds = pd.read_csv(BASE / 'outputs_reproduced' / 'models' / 'predictions_loso.csv')

# ── 1. Per-subject resting-state features ──────────────────────────────────────
id_cols = ['subject_id', 'condition', 'label', 'file', 'window_index', 'start_sec', 'end_sec']
feature_cols = [c for c in feats.columns if c not in id_cols]

rest = feats[feats['condition'] == 'rest'].copy()

# Aggregate rest windows per subject (mean across windows)
rest_agg = rest.groupby('subject_id')[feature_cols].mean().reset_index()

print(f"Resting features shape: {rest_agg.shape}")

# ── 2. Per-subject LOSO AUC ────────────────────────────────────────────────────
lr_preds = preds[preds['model'] == 'logistic_regression']
per_subj_auc = lr_preds.groupby('subject_id').apply(
    lambda g: roc_auc_score(g['true_label'], g['score_workload'])
).reset_index()
per_subj_auc.columns = ['subject_id', 'auc_loso']

# SVM (best model)
svm_preds = preds[preds['model'] == 'svm_rbf']
per_subj_auc_svm = svm_preds.groupby('subject_id').apply(
    lambda g: roc_auc_score(g['true_label'], g['score_workload'])
).reset_index()
per_subj_auc_svm.columns = ['subject_id', 'auc_svm']

# XGBoost
xgb_preds = preds[preds['model'] == 'xgboost']
per_subj_auc_xgb = xgb_preds.groupby('subject_id').apply(
    lambda g: roc_auc_score(g['true_label'], g['score_workload'])
).reset_index()
per_subj_auc_xgb.columns = ['subject_id', 'auc_xgb']

# Merge all AUCs
subj_df = per_subj_auc.merge(per_subj_auc_svm, on='subject_id').merge(per_subj_auc_xgb, on='subject_id')

# ── 3. Merge rest features with subject AUC ────────────────────────────────────
rest_model = rest_agg.merge(subj_df, on='subject_id')
X_feat = rest_model[[c for c in rest_model.columns if c not in ['subject_id', 'auc_loso', 'auc_svm', 'auc_xgb']]]
y = rest_model['auc_loso'].values
y_svm = rest_model['auc_svm'].values
y_xgb = rest_model['auc_xgb'].values
subjects = rest_model['subject_id'].values

print(f"\nPer-subject AUC (logistic regression):")
print(f"  Mean: {np.mean(y):.4f}, Std: {np.std(y):.4f}")
print(f"  Range: [{np.min(y):.4f}, {np.max(y):.4f}]")

# ── 4. Feature correlation analysis ────────────────────────────────────────────
from scipy.stats import spearmanr as spearmanr_func

feature_names = X_feat.columns.values
n_features = len(feature_names)
corr_vals = np.zeros(n_features)
p_vals = np.zeros(n_features)

for i in range(n_features):
    x = X_feat.values[:, i]
    mask = ~(np.isnan(x) | np.isinf(x))
    if mask.sum() > 5 and np.std(x[mask]) > 1e-10:
        r, p = spearmanr_func(y[mask], x[mask])
        corr_vals[i] = r
        p_vals[i] = p
    else:
        corr_vals[i] = 0
        p_vals[i] = 1.0

# FDR correction
reject, p_corrected = false_discovery_control(p_vals), None
try:
    from statsmodels.stats.multitest import multipletests
    _, p_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
except:
    p_corrected = p_vals * n_features
    p_corrected = np.minimum(p_corrected, 1.0)

n_sig = np.sum(p_corrected < 0.05)
n_sig_uncorrected = np.sum(p_vals < 0.05)
print(f"\nFeature-AUC correlations:")
print(f"  Features significant (FDR q<0.05): {n_sig} / {n_features}")
print(f"  Features significant (uncorrected p<0.05): {n_sig_uncorrected} / {n_features}")

# Top features table
top_n = 30
top_idx = np.argsort(np.abs(corr_vals))[::-1][:top_n]
top_features = pd.DataFrame({
    'feature': feature_names[top_idx],
    'spearman_rho': corr_vals[top_idx],
    'p_value_uncorrected': p_vals[top_idx],
    'p_value_fdr': p_corrected[top_idx],
    'direction': np.where(corr_vals[top_idx] > 0, 'positive', 'negative'),
})
top_features['significant_fdr'] = top_features['p_value_fdr'] < 0.05
top_features['rank'] = range(1, top_n + 1)
top_features = top_features[['rank', 'feature', 'spearman_rho', 'direction', 'p_value_uncorrected', 'p_value_fdr', 'significant_fdr']]
top_features.to_csv(TAB / 'subject_variability_top_features.csv', index=False)
print(f"\nTop 30 resting features correlated with LOSO AUC:")
print(top_features.to_string(index=False))

# ── 5. Feature categories analysis ─────────────────────────────────────────────
def parse_feature_type(fname):
    """Categorize features into functional groups."""
    if fname.startswith('band_abs_') or fname.startswith('band_rel_'):
        return 'bandpower'
    elif fname.startswith('ratio_'):
        return 'ratio'
    elif fname.startswith('stat_') and any(s in fname for s in ['mean', 'std', 'var', 'rms', 'ptp']):
        return 'statistical'
    elif fname.startswith('stat_') and any(s in fname for s in ['skew', 'kurtosis']):
        return 'shape'
    elif fname.startswith('stat_') and 'entropy' in fname:
        return 'entropy'
    elif fname.startswith('hjorth_'):
        return 'hjorth'
    elif fname.startswith('spectral_'):
        return 'spectral_entropy'
    elif fname.startswith('corr_'):
        return 'connectivity'
    elif fname.startswith('connectivity_'):
        return 'global_connectivity'
    elif fname.startswith('global_'):
        return 'global_bandpower'
    elif fname.startswith('region_'):
        return 'regional_bandpower'
    elif fname.startswith('hemisphere_'):
        return 'hemisphere_bandpower'
    else:
        return 'other'

feature_types = [parse_feature_type(f) for f in feature_names]
type_corrs = {}
for feat_type in set(feature_types):
    mask = [t == feat_type for t in feature_types]
    type_corrs[feat_type] = {
        'n_features': sum(mask),
        'mean_abs_corr': np.mean(np.abs(corr_vals[mask])),
        'max_abs_corr': np.max(np.abs(corr_vals[mask])),
        'n_sig': sum(p_corrected[mask] < 0.05),
    }
type_df = pd.DataFrame(type_corrs).T.round(4)
type_df.to_csv(TAB / 'subject_variability_feature_categories.csv')
print(f"\nFeature category analysis:")
print(type_df.to_string())

# ── 6. Elastic net regression with LOOCV ──────────────────────────────────────
# Use PCA to reduce dimensionality first, then elastic net
print("\nBuilding LOOCV prediction model...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_feat.values)

# Replace NaN/inf with column means
col_means = np.nanmean(X_scaled, axis=0)
X_scaled = np.where(np.isnan(X_scaled) | np.isinf(X_scaled), col_means, X_scaled)

# PCA first
n_components = min(20, X_scaled.shape[0] - 1)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f} with {n_components} components")

# Elastic Net with LOOCV
loocv = LeaveOneOut()
pred_auc_enet = np.zeros(len(y))

for train_idx, test_idx in loocv.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train = y[train_idx]
    enet = ElasticNetCV(cv=5, random_state=42, max_iter=10000, alphas=[0.001, 0.01, 0.1, 1.0],
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9])
    enet.fit(X_train, y_train)
    pred_auc_enet[test_idx] = enet.predict(X_test)[0]

r2_enet = r2_score(y, pred_auc_enet)
rmse_enet = np.sqrt(mean_squared_error(y, pred_auc_enet))
r_enet, p_enet = pearsonr(y, pred_auc_enet)
rho_enet, p_rho_enet = spearmanr(y, pred_auc_enet)

print(f"  LOOCV Elastic Net (PCA+ENet):")
print(f"    R2: {r2_enet:.4f}, RMSE: {rmse_enet:.4f}")
print(f"    Pearson r: {r_enet:.4f} (p={p_enet:.4f})")
print(f"    Spearman rho: {rho_enet:.4f} (p={p_rho_enet:.4f})")

# ── 7. Two-stage model ─────────────────────────────────────────────────────────
thresholds = [0.5, 0.55, 0.6, 0.65]
two_stage_results = []

for thresh in thresholds:
    y_bin = (y > thresh).astype(int)
    n_resistant = int((1 - y_bin).sum())

    # Use PCA+LogisticRegression for screening
    pred_bin = np.zeros(len(y))
    bin_proba = np.zeros(len(y))

    for train_idx, test_idx in loocv.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train_bin = y_bin[train_idx]
        if y_train_bin.sum() >= 2 and (1 - y_train_bin).sum() >= 2:
            lr = LogisticRegressionCV(Cs=10, cv=3, max_iter=5000, class_weight='balanced', random_state=42)
            lr.fit(X_train, y_train_bin)
            pred_bin[test_idx] = lr.predict(X_test)[0]
            bin_proba[test_idx] = lr.predict_proba(X_test)[0, 1]
        else:
            pred_bin[test_idx] = 1 if y_train_bin.mean() > 0.5 else 0
            bin_proba[test_idx] = 0.5

    # Two-stage AUC: screen -> if good use actual AUC, if bad use chance (0.5)
    two_stage = np.where(pred_bin == 1, y, 0.5)
    two_stage_mean = two_stage.mean()
    overall_mean = y.mean()

    # Classification metrics for screening
    if len(np.unique(y_bin)) > 1:
        screen_auc = roc_auc_score(y_bin, bin_proba)
    else:
        screen_auc = 0.5
    screen_acc = (pred_bin == y_bin).mean()

    two_stage_results.append({
        'threshold': thresh,
        'n_resistant': n_resistant,
        'overall_auc': round(overall_mean, 4),
        'two_stage_auc': round(two_stage_mean, 4),
        'improvement': round(two_stage_mean - overall_mean, 4),
        'screen_accuracy': round(screen_acc, 4),
        'screen_auc': round(screen_auc, 4),
    })

ts_df = pd.DataFrame(two_stage_results)
ts_df.to_csv(TAB / 'subject_variability_two_stage.csv', index=False)
print(f"\nTwo-stage model results across thresholds:")
print(ts_df.to_string(index=False))

# ── 8. Characterize resistant subjects ─────────────────────────────────────────
threshold_main = 0.6
y_bin_main = (y > threshold_main).astype(int)
resistant_mask = y_bin_main == 0
classifiable_mask = y_bin_main == 1

print(f"\n--- Resistant vs Classifiable Subject Comparison ---")
print(f"Resistant (AUC <= {threshold_main}): {resistant_mask.sum()} subjects")
print(f"  Subjects: {', '.join(subjects[resistant_mask])}")
for s in subjects[resistant_mask]:
    idx = list(subjects).index(s)
    print(f"  {s}: AUC={y[idx]:.4f}")
print(f"Classifiable (AUC > {threshold_main}): {classifiable_mask.sum()} subjects")

# Compare specific features between groups
compare_cols = []
for prefix in ['band_rel_', 'band_abs_', 'ratio_', 'stat_', 'hipp_']:
    compare_cols.extend([c for c in feature_names if c.startswith(prefix) and c.endswith('_mean')])
# Also channel correlations
compare_cols.extend([c for c in feature_names if c.startswith('corr_') and ('T5' in c or 'O1' in c or 'theta' in c or 'gamma' in c)])
compare_cols = list(set(compare_cols))

comparison = []
for col in compare_cols[:50]:  # Check top 50
    resistant_vals = X_feat[col].values[resistant_mask]
    classifiable_vals = X_feat[col].values[classifiable_mask]
    if len(resistant_vals) >= 2 and len(classifiable_vals) >= 2:
        if np.std(resistant_vals) > 1e-10 and np.std(classifiable_vals) > 1e-10:
            stat, p = mannwhitneyu(resistant_vals, classifiable_vals, alternative='two-sided')
            if p < 0.05:
                comparison.append({
                    'feature': col,
                    'resistant_mean': resistant_vals.mean(),
                    'classifiable_mean': classifiable_vals.mean(),
                    'difference': resistant_vals.mean() - classifiable_vals.mean(),
                    'p_value': p,
                })

comp_df = pd.DataFrame(comparison)
if len(comp_df) > 0:
    comp_df = comp_df.sort_values('p_value')
    comp_df.to_csv(TAB / 'subject_variability_group_comparison.csv', index=False)
    print(f"\nSignificant differences (p<0.05, Mann-Whitney U): {len(comp_df)} features")
    print(comp_df.head(15).to_string(index=False))
else:
    print("\nNo significant differences between groups at p<0.05")

# ── 9. Generate publication figures ────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# Color palette
COLOR_GOOD = '#1a9850'
COLOR_BAD = '#d73027'
COLOR_CHANCE = 'gray'
COLOR_POS = '#2166ac'
COLOR_NEG = '#b2182b'

# ── 9a. Main figure: Predicted vs Actual AUC ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: Actual AUC scatter
ax = axes[0]
aucs_all = [y, y_svm, y_xgb]
labels_all = ['Logistic Regression', 'SVM-RBF', 'XGBoost']
colors_model = ['#2166ac', '#d6604d', '#4daf4a']
for i, (auc_arr, label, color) in enumerate(zip(aucs_all, labels_all, colors_model)):
    jitter = np.random.RandomState(i).uniform(-0.02, 0.02, len(auc_arr))
    ax.scatter(np.full(len(auc_arr), i) + jitter, auc_arr, s=30, c=color, alpha=0.6, edgecolors='black', linewidth=0.3, label=label)

ax.set_xticks(range(3))
ax.set_xticklabels(['LogReg', 'SVM-RBF', 'XGBoost'], fontsize=9)
ax.set_ylabel('Per-Subject LOSO AUC')
ax.set_title('A. Per-Subject AUC Distribution\n(Each dot = one subject)')
ax.axhline(0.5, color=COLOR_CHANCE, linestyle=':', alpha=0.7, label='Chance')
ax.axhline(0.6, color=COLOR_CHANCE, linestyle='--', alpha=0.3, label='Threshold')
# Add subject count labels
for i, auc_arr in enumerate(aucs_all):
    ax.text(i, 0.02, f'n={len(auc_arr)}\n<0.6: {(auc_arr < 0.6).sum()}', ha='center', fontsize=8, transform=ax.get_xaxis_transform())
ax.legend(fontsize=7, loc='lower left')

# Panel B: Predicted vs Actual AUC (LOOCV)
ax = axes[1]
ax.scatter(y, pred_auc_enet, c=[COLOR_BAD if a < threshold_main else COLOR_GOOD for a in y],
           s=60, edgecolors='black', linewidth=0.5, alpha=0.8)
ax.plot([0.3, 1.05], [0.3, 1.05], 'k--', alpha=0.3, label='Ideal')
ax.set_xlabel('Actual LOSO AUC (Logistic Regression)')
ax.set_ylabel('Predicted AUC (from Resting EEG)')
ax.set_title(f'B. AUC Prediction from Resting-State EEG\nr={r_enet:.3f}, p={p_enet:.3f}')
# Label extreme subjects
for a, p, s in zip(y, pred_auc_enet, subjects):
    if abs(a - p) > 0.25 or a < 0.6:
        ax.annotate(s.replace('Subject', 'S'), (a, p), fontsize=7, alpha=0.8,
                    xytext=(4, 4), textcoords='offset points')
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(0.3, 1.05)
ax.set_ylim(0.3, 1.05)

# Panel C: Top feature correlations
ax = axes[2]
top_plot = top_features.head(15)
bar_colors = [COLOR_POS if c > 0 else COLOR_NEG for c in top_plot['spearman_rho']]
bars = ax.barh(range(len(top_plot)), top_plot['spearman_rho'], color=bar_colors, edgecolor='black', linewidth=0.5, height=0.7)
ax.set_yticks(range(len(top_plot)))
# Simplify feature names
short_names = []
for f in top_plot['feature']:
    f = f.replace('_mean', '').replace('band_rel_', 'rel ').replace('band_abs_', 'abs ')
    short_names.append(f)
ax.set_yticklabels(short_names, fontsize=8)
ax.set_xlabel("Spearman correlation with subject AUC")
ax.set_title('C. Top 15 Resting Features Predicting Classifiability')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-0.65, 0.65)

fig.tight_layout()
fig.savefig(FIG / 'figure_subject_variability.png', bbox_inches='tight')
fig.savefig(PAPER_FIG / 'figure_subject_variability.png', bbox_inches='tight')
print(f"\nMain figure saved")

# ── 9b. Subject breakdown figure ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
sort_idx = np.argsort(y)
sorted_y = y[sort_idx]
sorted_subj = subjects[sort_idx]

x_pos = np.arange(len(sorted_y))
# Color by AUC value
norm = plt.Normalize(0.3, 1.0)
cmap = plt.cm.RdYlGn
for i, auc_val in enumerate(sorted_y):
    ax.bar(i, auc_val, color=cmap(norm(auc_val)), edgecolor='black', linewidth=0.4, alpha=0.85)

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
ax.axhline(0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels([s.replace('Subject', 'S') for s in sorted_subj], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('LOSO AUC')
ax.set_title('Per-Subject LOSO AUC: Color intensity indicates classifiability')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
cbar.set_label('AUC', fontsize=9)

fig.tight_layout()
fig.savefig(FIG / 'figure_subject_auc_breakdown.png', bbox_inches='tight')
fig.savefig(PAPER_FIG / 'figure_subject_auc_breakdown.png', bbox_inches='tight')
print(f"Subject breakdown figure saved")

# ── 9c. Two-stage comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ts_plot = ts_df[ts_df['threshold'] <= 0.65]

x = np.arange(len(ts_plot))
width = 0.3
ax.bar(x - width/2, ts_plot['overall_auc'], width, label='Overall (no screening)',
       color='#2166ac', edgecolor='black', linewidth=0.8)
ax.bar(x + width/2, ts_plot['two_stage_auc'], width, label='Two-Stage (screen then classify)',
       color='#d6604d', edgecolor='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'AUC>{t}' for t in ts_plot['threshold']])
ax.set_ylabel('Mean AUC')
ax.set_title('Two-Stage Model: Screening Improves Effective AUC')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.legend(fontsize=9)

# Add value labels
for bar in ax.containers:
    ax.bar_label(bar, fmt='%.3f', fontsize=8, padding=2)

fig.tight_layout()
fig.savefig(FIG / 'figure_two_stage_comparison.png', bbox_inches='tight')
fig.savefig(PAPER_FIG / 'figure_two_stage_comparison.png', bbox_inches='tight')
print(f"Two-stage comparison figure saved")

plt.close('all')

# ── 10. Summary table ─────────────────────────────────────────────────────────
summary = OrderedDict([
    ('metric', [
        'n_subjects',
        'n_features_resting',
        'n_features_sig_fdr',
        'n_features_sig_uncorrected',
        'auc_mean',
        'auc_std',
        'n_resistant_auc_below_06',
        'n_resistant_auc_below_05',
        'prediction_pearson_r',
        'prediction_pearson_p',
        'prediction_spearman_rho',
        'prediction_spearman_p',
        'prediction_r2',
        'top_predictor_feature',
        'top_predictor_rho',
        'top_predictor_p',
        'best_two_stage_threshold',
        'overall_auc_at_best_threshold',
        'two_stage_auc_at_best_threshold',
        'auc_improvement',
    ]),
    ('value', [
        len(y),
        n_features,
        n_sig,
        n_sig_uncorrected,
        round(np.mean(y), 4),
        round(np.std(y), 4),
        int(np.sum(y < 0.6)),
        int(np.sum(y < 0.5)),
        round(r_enet, 4),
        round(p_enet, 4),
        round(rho_enet, 4),
        round(p_rho_enet, 4),
        round(r2_enet, 4),
        top_features.iloc[0]['feature'],
        round(top_features.iloc[0]['spearman_rho'], 4),
        round(top_features.iloc[0]['p_value_uncorrected'], 4),
        ts_df.loc[ts_df['improvement'].idxmax(), 'threshold'],
        ts_df.loc[ts_df['improvement'].idxmax(), 'overall_auc'],
        ts_df.loc[ts_df['improvement'].idxmax(), 'two_stage_auc'],
        ts_df.loc[ts_df['improvement'].idxmax(), 'improvement'],
    ])
])
summary_df = pd.DataFrame(summary)
summary_df.to_csv(TAB / 'subject_variability_results.csv', index=False)
print(f"\nSummary saved to {TAB / 'subject_variability_results.csv'}")

print("\n\n=== OPTION A COMPLETE ===")
print(f"Key findings:")
print(f"  1. {n_sig} resting EEG features significantly predict subject classifiability (FDR q<0.05)")
print(f"  2. Top predictor: {top_features.iloc[0]['feature']} (rho={top_features.iloc[0]['spearman_rho']:.4f})")
print(f"  3. LOO-CV prediction: r={r_enet:.3f}, R2={r2_enet:.3f}")
print(f"  4. Two-stage model improves AUC by up to {ts_df['improvement'].max():+.4f} at threshold {ts_df.loc[ts_df['improvement'].idxmax(), 'threshold']}")
