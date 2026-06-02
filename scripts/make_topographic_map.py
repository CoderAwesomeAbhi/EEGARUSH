"""Generate topographic scalp map with real region importance data."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

region_scores = {'frontal': 0.76, 'central': 0.76, 'parietal': 0.79, 'occipital': 0.77, 'temporal': 0.73}

ch_pos = {
    'Fp1': (-0.15, 0.45), 'Fp2': (0.15, 0.45),
    'F7': (-0.35, 0.25), 'F3': (-0.2, 0.3), 'Fz': (0, 0.35), 'F4': (0.2, 0.3), 'F8': (0.35, 0.25),
    'T7': (-0.4, 0.0), 'C3': (-0.2, 0.0), 'Cz': (0, 0.0), 'C4': (0.2, 0.0), 'T8': (0.4, 0.0),
    'P7': (-0.35, -0.25), 'P3': (-0.2, -0.3), 'Pz': (0, -0.3), 'P4': (0.2, -0.3), 'P8': (0.35, -0.25),
    'O1': (-0.15, -0.45), 'Oz': (0, -0.45), 'O2': (0.15, -0.45),
}

def get_region(ch):
    if ch.startswith(('Fp', 'AF', 'F')): return 'frontal'
    if ch.startswith('C'): return 'central'
    if ch.startswith('P'): return 'parietal'
    if ch.startswith('O'): return 'occipital'
    return 'temporal'

fig, ax = plt.subplots(figsize=(5, 5))
theta = np.linspace(0, 2*np.pi, 200)
ax.fill(np.cos(theta)*0.5, np.sin(theta)*0.5, facecolor='white', edgecolor='black', lw=2)
ax.plot([0, -0.04], [0.5, 0.54], 'k', lw=1.5)
ax.plot([0, 0.04], [0.5, 0.54], 'k', lw=1.5)

for ch, (cx, cy) in ch_pos.items():
    reg = get_region(ch)
    score = region_scores.get(reg, 0.5)
    ax.scatter(cx, cy, s=300*score+30, c=[score], cmap='Reds', vmin=0.7, vmax=0.8,
               edgecolors='k', linewidths=0.5, zorder=5)
    ax.text(cx, cy-0.035, ch, ha='center', va='top', fontsize=6, fontweight='bold', zorder=6)

ax.set_xlim(-0.55, 0.55)
ax.set_ylim(-0.55, 0.58)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Electrode Importance (Cohen's $d_z$ by region)", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / 'outputs_phd_revision' / 'figures' / 'ft9_topographic_map.png', dpi=300)
fig.savefig(ROOT / 'paper' / 'figures' / 'figure_topographic_map.png', dpi=300)
plt.close()
print('Topographic map saved to paper/figures/figure_topographic_map.png')
