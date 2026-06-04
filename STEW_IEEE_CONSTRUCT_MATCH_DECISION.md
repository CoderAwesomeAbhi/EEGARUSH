# STEW (Official IEEE DataPort) Construct-Match Decision

**Decision type:** Provenance/construct-match gate. No models were run. This
decision governs whether the official IEEE STEW source may be used for the
**locked MAT replication** (subject resting-baseline mean subtraction → L2
logistic regression → macro subject-level ROC-AUC).

Inputs: `STEW_IEEE_SOURCE_PROVENANCE_AUDIT.md` and the three CSVs under
`results/stew_ieee_provenance/`.

---

## VERDICT

```
STEW_IEEE_USABLE_ONLY_AS_NONCOMPARABLE_SENSITIVITY
```

The official IEEE DataPort STEW archive is **usable**, but **only as a
non-comparable sensitivity / exploratory dataset** — it does **not** qualify as a
like-for-like confirmatory replication of the locked MAT design, and it cannot by
itself justify a cross-dataset transfer claim.

---

## Why not `STEW_IEEE_NOT_USABLE_REPLACE_EXPLORATORY_DATASET`

The data are genuinely usable for the baseline-subtraction hypothesis:

- 48 subjects, all with complete rest + workload recordings.
- A **genuine resting/no-task baseline** (`_lo`) exists in the official source —
  the exact element the MONSTER bundle lacked.
- Temporal provenance is intact (uniform 150 s / 19200-sample retained segments).
- Subject IDs are explicit (`sub01`–`sub48`).
- Subjective ratings corroborate the condition labels (lo<hi for 45/45 rated subjects).
- A **non-overlapping calibration / scored-rest / scored-task** segmentation is
  constructable per subject.
- The **no-gamma** band-power feature family is computable at 128 Hz
  (Nyquist 64 Hz covers δ/θ/α/β).

So the dataset is not unusable — replacing it is unwarranted.

## Why not `STEW_IEEE_VALID_FOR_LOCKED_REPLICATION`

The locked design is defined on MAT's specific acquisition. STEW differs on
multiple axes that jointly produce substantial, non-ignorable domain shift:

1. **Sampling rate:** 128 Hz (STEW) vs 500 Hz (MAT). Not embedded in the STEW
   source — inferred from documentation + sample count. Requires
   resampling/rate-robust feature handling, so the pipeline is not bit-for-bit
   identical.
2. **Montage / channel set:** 14-channel Emotiv EPOC vs 19 scalp 10-20. Only
   **10 of 14** STEW channels harmonize to MAT; `AF3, AF4, FC5, FC6` have no MAT
   counterpart. A locked replication would run on a *reduced, harmonized* montage,
   not the montage the locked design was fixed on.
3. **Channel labels not embedded in source:** harmonization relies on the
   documented Emotiv EPOC order plus documented 10-20 name equivalences
   (T7≡T3, T8≡T4, P7≡T5, P8≡T6), not on labels verified from the files. This is the
   same structural weakness previously cited against the MONSTER bundle (here
   partially mitigated by the hardware standard, but not eliminated).
4. **Referencing scheme** differs (MAT A2-A1 montage vs Emotiv CMS/DRL), affecting
   cross-dataset feature comparability.
5. **Workload task construct** differs: serial **mental arithmetic** (MAT) vs
   **SIMKAP simultaneous multitasking** (STEW). Both manipulate workload, but they
   are not the same cognitive task.

Any one of these would weaken a "locked replication" claim; together they make a
strict, comparable replication indefensible. This is consistent with the project
checkpoint: *"No cross-dataset transfer claim is yet justified."*

---

## Permitted / not permitted use

**Permitted (later, not in this task):**
- Treat STEW as an independent **exploratory / sensitivity** test of the
  baseline-subtraction hypothesis on a harmonized 10-channel, 128 Hz, no-gamma
  feature set — reported as **non-comparable** to the MAT primary result.

**Not permitted:**
- Presenting STEW as a confirmatory replication of the locked MAT result.
- Using STEW to assert cross-dataset transfer, invariance, or a universal
  workload axis.
- Pooling STEW and MAT into a single "harmonized" primary metric as if comparable.

---

## Gate outcome

Because the verdict is **not** `STEW_IEEE_VALID_FOR_LOCKED_REPLICATION`,
`STEW_LOCKED_REPLICATION_PROTOCOL_BEFORE_MODELING.md` is **intentionally not
created**. No STEW modeling is authorized by this decision. Any future STEW
analysis must be framed and pre-registered as a non-comparable sensitivity study.
