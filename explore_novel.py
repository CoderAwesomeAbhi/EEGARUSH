"""Look for graded difficulty effect in DS007262"""
import pandas as pd
import numpy as np
from scipy import stats

feat = pd.read_csv('outputs_reproduced/features/eeg_features.csv')

# Find DS007262
ds = feat[feat['file'].str.contains('sub-', na=False)].copy()
print(f'DS007262: {len(ds)} rows')
print(f'Conditions: {ds["condition"].unique()}')
print(f'Subjects: {ds["subject_id"].nunique()}')

# Check if condition is numeric (graded difficulty)
ds['cond_num'] = pd.to_numeric(ds['condition'], errors='coerce')
print(f'Numeric conditions: {ds["cond_num"].nunique()}')

# For each subject, check theta/alpha ratio by difficulty level
subjects = ds.groupby('subject_id')
results = []
for subj, grp in subjects:
    for cond, cgrp in grp.groupby('condition'):
        ta = cgrp['ratio_F3_theta_alpha'].mean()
        results.append({'subject': subj, 'condition': cond, 'ratio': ta})

res = pd.DataFrame(results)
print(f'\n=== Theta/alpha ratio by condition (DS007262) ===')
for cond in sorted(res['condition'].unique()):
    vals = res[res['condition'] == cond]['ratio'].dropna()
    if len(vals) > 0:
        print(f'  Condition {cond}: mean={vals.mean():.4f}, std={vals.std():.4f}')

# Linear trend test
# Group conditions into low, medium, high
res['cond_num'] = pd.to_numeric(res['condition'], errors='coerce')
res = res.dropna(subset=['cond_num'])
print(f'\n=== Linear trend: correlation between difficulty and ratio ===')
for subj in res['subject'].unique():
    s = res[res['subject'] == subj]
    if len(s) >= 3:
        r, p = stats.pearsonr(s['cond_num'], s['ratio'])
        print(f'  {subj}: r={r:.2f}, p={p:.3f}')
