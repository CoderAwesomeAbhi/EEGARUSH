# COG-BCI Final Project Decision

Based on the single, consumed one-shot prospective test
(`COG_BCI_ONE_SHOT_PROSPECTIVE_RESULTS.md`;
`results/cog_bci_one_shot/cog_bci_primary_results.json`).

## What was tested

- **Hypothesis (post-hoc):** under cross-device/cross-task domain shift, subject
  resting-baseline **z-scoring** transports a MAT-trained workload decoder better
  than mean subtraction or absolute features.
- **Development evidence only:** MAT and STEW (both used to *generate* the
  hypothesis; neither can validate it).
- **Untouched prospective test:** COG-BCI (Hinss et al. 2023), one shot.
- **Prior status:** mean-subtraction cross-device transfer had **already failed**
  (MAT→STEW 0.447598, below chance). z-scoring was **elevated only after** the STEW
  result (MAT→STEW z-scoring 0.682823, secondary diagnostic) and required a fresh
  prospective test — performed here exactly once.
- **No post-outcome degrees of freedom:** no alternative target/task/session/model/
  feature set/channel set/resampling/endpoint/comparator was tried after COG-BCI
  outcomes were accessed. N-back and alternative sessions were not run.

## Observed result (exactly as produced)

| Method | Macro subject AUC | Above chance? |
|---|---|---|
| z-scoring (primary) | **0.435961** | **No** |
| mean subtraction (comparator) | 0.392153 | No |
| absolute (comparator) | 0.367875 | No |

Primary superiority (paired subject-bootstrap, z-scoring − mean subtraction):
mean Δ = +0.043807, 95 % CI **[−0.076359, +0.175233]** (includes zero).

## Decision logic

- The run produced valid predictive metrics with all 29 subjects and no pre-result
  software/input failure → the `PROSPECTIVE_RUN_INVALID_*` verdict does **not** apply.
- z-scoring is **below chance** (0.435961 < 0.5) → the above-chance precondition for
  *full success* and *partial* is **not** met → neither
  `..._VALIDATED_...` nor `..._PARTIAL_...` applies.
- Per the locked failure rule (z-scoring at/below chance, and does not significantly
  outperform mean subtraction), the outcome is **failure**.

## Honest conclusion

The post-hoc z-scoring transport hypothesis **failed its one and only prospective
test.** On the untouched COG-BCI dataset, MAT-trained workload decoding transports
**below chance** under every calibration strategy, and the z-scoring advantage over
mean subtraction is statistically indistinguishable from zero. The MAT→STEW z-scoring
elevation did **not** replicate. Combined with the earlier mean-subtraction transfer
failure, cross-dataset/cross-task **transfer** of this decoder is **not supported**.

Within-dataset findings remain valid and unchanged (MAT mean-subtraction macro
subject AUC 0.880102, p = 0.004975; within-STEW 0.839498, p = 0.004975) — these are
within-dataset evidence only. No rejected claim is revived. The appropriate output is
an honest **negative / methods** paper: baseline calibration (mean subtraction or
z-scoring) enables within-dataset workload decoding but does **not** confer
cross-device/cross-task transfer, even when restricted to scale/offset-invariant
features at a harmonized sampling representation.

## VERDICT

```
PROSPECTIVE_ZSCORE_TRANSPORT_FAILED_WRITE_NEGATIVE_METHODS_PAPER
```
