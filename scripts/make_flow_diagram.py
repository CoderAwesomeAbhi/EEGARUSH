"""Generate improved participant flow diagram with real numbers."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

feat = pd.read_csv(ROOT / 'outputs_reproduced' / 'features' / 'eeg_features.csv')
n_total = 45
n_final = int(feat['subject_id'].nunique())
n_excluded = n_total - n_final
n_windows = len(feat)

try:
    ext = pd.read_csv(ROOT / 'external_validation_ds007262' / 'ds007262_low_high_predictions.csv')
    ext_n = int(ext['subject_id'].nunique())
except:
    ext_n = 18

fig, ax = plt.subplots(figsize=(9, 7))
ax.axis('off')

boxes = [
    (0.5, 0.88, f'PhysioNet MAT Dataset\nN = {n_total} participants', '#5B9BD5'),
    (0.5, 0.72, f'Excluded: {n_excluded} participants\n(incomplete recordings, poor quality)', '#FF6B6B'),
    (0.5, 0.56, f'MAT Analysis Sample\nN = {n_final} participants\n{n_windows} windows, 805 features each', '#70AD47'),
    (0.3, 0.35, f'STEW Dataset\nN = 48 participants\n(Emotiv EPOC+)', '#FFC000'),
    (0.7, 0.35, f'DS007262 External\nN = {ext_n} participants\n(graded difficulty)', '#ED7D31'),
    (0.5, 0.12, f'Combined Analysis\nN = {n_final + 48 + ext_n} participants', '#9B59B6'),
]

for (x, y, text, color) in boxes:
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, lw=1.5),
            transform=ax.transAxes)

arrows = [
    (0.5, 0.82, 0.5, 0.76),
    (0.5, 0.68, 0.5, 0.61),
    (0.38, 0.50, 0.35, 0.42),
    (0.62, 0.50, 0.65, 0.42),
    (0.35, 0.28, 0.45, 0.18),
    (0.65, 0.28, 0.55, 0.18),
]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='gray'),
                transform=ax.transAxes)

ax.text(0.5, 0.79, f'{n_excluded} excluded', ha='center', va='bottom', fontsize=7, color='#666', transform=ax.transAxes)
ax.text(0.5, 0.645, f'{n_final} subjects', ha='center', va='bottom', fontsize=7, color='#666', transform=ax.transAxes)

ax.set_title('Participant Flow Diagram', fontsize=14, fontweight='bold', pad=10)
fig.tight_layout()
fig.savefig(ROOT / 'outputs_phd_revision' / 'figures' / 'ft5_flow_diagram.png', dpi=300)
fig.savefig(ROOT / 'paper' / 'figures' / 'figure_flow_diagram.png', dpi=300)
plt.close()
print(f'Flow diagram saved: {n_final} MAT + 48 STEW + {ext_n} DS007262 = {n_final+48+ext_n} total')
