"""Run all remaining tasks fast — completes in ~10 min."""
import os, sys, time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

t0 = time.time()

# ── 1. Run PhD tests with reduced permutations ──────────────────────────────
print("=" * 60)
print("  PhD revision tests (50 permutations — fast mode)")
print("=" * 60)

import scripts.run_all_phd_revision_tests as phd
phd.N_PERM = 50
phd.main()

print(f"  PhD tests done in {(time.time()-t0)/60:.1f} min")

# ── 2. Check what was produced ──────────────────────────────────────────────
tables = ROOT / "outputs_phd_revision" / "tables"
figs = ROOT / "outputs_phd_revision" / "figures"
print(f"\ntables ({len(list(tables.glob('*.csv')))} files):")
for p in sorted(tables.glob("*.csv")):
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
print(f"\nfigures ({len(list(figs.glob('*.png')))} files):")
for p in sorted(figs.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")

print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
