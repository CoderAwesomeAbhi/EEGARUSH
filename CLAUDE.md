# EEGARUSH — Project Guardrails

**Before any scientific work, read [`docs/EEGARUSH_CHECKPOINT.md`](docs/EEGARUSH_CHECKPOINT.md) in full.**
That file holds the validated scientific checkpoint; this file is the short operating contract.

## Safety Rules (binding)

- Work only on branch `claude/stew-ieee-source-audit`.
- Do **not** edit `main.tex` or any manuscript PDF.
- Do **not** analyze DS007262.
- Do **not** use the Hugging Face / MONSTER STEW bundle for primary modeling.
- Do **not** search for a final confirmation dataset.
- Do **not** add raw data under `data/raw/` to Git.
- Do **not** commit, push, or open a PR unless explicitly told to.
- Do **not** run new scientific models until the STEW provenance audit gate (below) passes.

## Rejected / Do-Not-Revive Claims

Do not reintroduce any claim about:

- an invariant / universal workload axis;
- z-scoring superiority;
- PAC (phase-amplitude coupling);
- gamma mechanisms;
- source-localized neural mechanisms;
- clinical readiness;
- confirmed cross-dataset transfer.

The old z-scoring-centered theory did **not** survive the corrected raw reconstruction.

## Locked MAT Finding (valid)

- Official raw MAT EDF files audited: 36 subjects, 72 EDF files, 500 Hz.
- Condition mapping: `_1 = rest/background`, `_2 = arithmetic`.
- Corrected balanced no-gamma protocol; equal scored-rest and scored-task windows
  per subject; no calibration/scoring overlap.
- Candidate method: **subject resting-baseline mean subtraction**.
- Locked candidate model: **L2 logistic regression**, fixed existing settings.
- Primary metric: **macro subject-level ROC-AUC**.
- MAT mean-subtraction macro subject AUC = **0.880102**.
- Full-pipeline macro subject-level null test: observed = 0.880102,
  null mean = 0.500600, null 95% = [0.441057, 0.553729], empirical p = 0.004975 (200 perms).

### MAT limits

- Mean subtraction **not** proven superior to absolute features:
  paired (mean-sub − absolute) 95% CI crossed zero: [-0.018566, 0.093112].
- MAT is **within-dataset evidence only**. No cross-dataset transfer claim is justified.

## STEW Blocker

- MONSTER bundle audited: `STEW_X.npy` shape `(28512, 14, 256)`, 48 subjects,
  binary low/high workload labels.
- It lacks: genuine resting baseline, temporal/start-time provenance, and
  channel labels sufficient for defensible MAT/STEW harmonization.
- Therefore the MONSTER bundle is **insufficient** for the baseline-relative
  replication test. Official IEEE DataPort STEW source/archive data are required.

## Next Gate — Official IEEE DataPort STEW Audit

Manually download and place the official archive under:

```
data/raw/stew/ieee_dataport_stew/
```

Once present, the next scientific task is **provenance audit only** (no modeling):
verify source files; verify true rest/baseline condition; verify task/workload
condition; verify subject IDs; verify sampling rate and channel names; determine
whether a non-overlapping calibration/scored-rest design is possible; determine
whether MAT/STEW channel harmonization is defensible.

**No STEW models may run until that audit passes.**

## Raw-Data Git Restriction

`data/raw/` is git-ignored and must stay that way. Never `git add -f` raw data.

## Current Research Question

Does subtracting each subject's resting EEG baseline produce a reproducible and
transferable workload-decoding signal across provenance-valid EEG datasets?

## Pivot After Failed Centering Transfer (current status)

See `docs/EEGARUSH_CHECKPOINT.md` "Pivot After Failed Centering Transfer" for full
detail. Summary:

- **Proven:** mean-subtraction decodes workload above chance **within MAT**
  (AUC 0.880102, p=0.004975) and **within STEW** (AUC 0.839498, p=0.004975).
- **Rejected:** mean subtraction as a cross-device **transfer** method (MAT→STEW
  transport failed, AUC 0.447598, below chance, not better than absolute);
  any confirmed generalizable transfer; any invariant/universal axis.
- **New hypothesis (post-hoc, unconfirmed):** under severe cross-device/task unit
  & baseline-scale shift, **z-scoring may transport better than mean subtraction**
  (MAT→STEW z-scoring AUC 0.682823, a secondary diagnostic). Audited valid for
  hypothesis generation only; **must** be tested on an **untouched** dataset.
  MAT/STEW are development-only.
- **Do not** describe the z-scoring transfer as confirmed/replicated/successful.
- A **metadata-only** untouched-dataset eligibility screen is now sanctioned
  (this supersedes the earlier blanket "do not search" rule for *metadata
  screening only*). **No signal/label download or modeling** of any new candidate
  until it is selected and a fresh provenance audit passes.
- Still binding: do not edit `main.tex`; do not re-analyze DS007262; do not commit
  anything under `data/raw/`; do not use the MONSTER/HF STEW bundle for modeling.

## Automatic GitHub Sync Policy

- After a completed and scientifically valid task, automatically sync safe outputs
  to a **draft** PR — unless the user says `local-only`.
- Always work on a feature branch; never commit directly to `main`.
- Never automatically merge a PR.
- Never commit raw/source data or anything under `data/raw/`.
- Never commit or modify `main.tex` or manuscript PDFs unless a later explicit
  manuscript-rewrite instruction unlocks them. (`main.tex` edits are blocked during
  the current analysis-only phase; the sync script enforces this, and
  `.claude/settings.local.json` denies it where supported.)
- Never commit obsolete DS007262 confirmation rewrites or old
  invariant-axis / z-scoring-centered manuscript outputs.
- Before syncing: run safety checks, show the staged files, run the relevant tests,
  and stop if any unsafe file is detected.
- Use the guarded helper `scripts/safe_sync_to_github.sh "<commit message>"` for all
  automatic syncs; it performs the safety gates, commits, pushes the feature branch,
  and opens/locates a draft PR into `main`.
- Final responses for a sync must include: branch, test results, committed files,
  excluded files, and the PR URL.
