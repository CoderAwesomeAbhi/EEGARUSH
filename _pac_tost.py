import pandas as pd
import numpy as np
from scipy import stats

pac = pd.read_csv('outputs_phd_revision/pac_analysis_v2/pac_statistics_v2.csv')

print('Channel-level PAC task modulation:')
for _, row in pac.iterrows():
    d = row['cohen_d']
    n = int(row['n_subjects'])
    se_d = 1.0 / np.sqrt(n - 1)
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d
    print(f'{row["channel"]:>8}: d={d:.3f} [{ci_low:.3f}, {ci_high:.3f}], p={row["p_fdr"]:.3f}')

ds = pac['cohen_d'].values
meta_d = ds.mean()
print(f'\nMeta-analytic mean d = {meta_d:.3f}')
print(f'Range: [{ds.min():.3f}, {ds.max():.3f}]')
print(f'All p_fdr > 0.05: {(pac["p_fdr"] > 0.05).all()}')

se_meta = np.std(ds, ddof=1) / np.sqrt(len(ds))
ci90_low = meta_d - 1.645 * se_meta
ci90_high = meta_d + 1.645 * se_meta
print(f'\nMeta-analytic 90% CI: [{ci90_low:.3f}, {ci90_high:.3f}]')
print(f'Within d=±0.20: {ci90_low > -0.20 and ci90_high < 0.20}')
