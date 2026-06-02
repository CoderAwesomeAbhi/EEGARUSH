import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, '.')
feat = pd.read_csv('outputs_reproduced/features/eeg_features.csv')
corr_cols = [c for c in feat.columns if c.startswith('corr_') and 'ECG' not in c and 'A2A1' not in c]
print(f'EEG correlation features (excluding ECG/A2A1): {len(corr_cols)}')

channels = set()
for c in corr_cols:
    parts = c.replace('corr_', '').split('_')
    channels.update(parts)
eeg_channels = sorted([ch for ch in channels if ch not in ('ECGECG', 'A2A1')])
print(f'EEG channels: {eeg_channels}')
print(f'N EEG channels: {len(eeg_channels)}')
print(f'Expected pairs: {len(eeg_channels)*(len(eeg_channels)-1)//2}')

# Count all corr features including ECG/A2A1
all_corr = [c for c in feat.columns if c.startswith('corr_')]
print(f'All correlation features: {len(all_corr)}')

# Check MAT
mat = feat[feat['file'].str.contains('Subject', na=False)]
print(f'MAT subjects: {mat["subject_id"].nunique()}')
print(f'MAT conditions: {mat["condition"].unique()}')

# Check Fz/Cz features
print(f'\nFz/Cz features:')
for ch in ['Fz', 'Cz']:
    cols = [c for c in feat.columns if ch in c and ('abs' in c or 'rel' in c or 'ratio' in c)]
    print(f'  {ch}: {cols[:6]}')

# Inspect PAC data
pac_path = 'outputs_phd_revision/pac_analysis_v2/pac_data_v2.csv'
if os.path.exists(pac_path):
    pac = pd.read_csv(pac_path)
    print(f'\nPAC data: {pac.shape}')
    print(f'Columns: {list(pac.columns)}')
    print(pac.head(3).to_string())
