# DS007262 Success/Failure Interpretation Rules

## Primary Success Criterion

The DS007262 confirmatory test is a success only if all of the following are true:

- Raw BrainVision EEG files are materialized and readable for at least 12 released DS007262 subjects.
- At least 6 ordered non-tutorial difficulty levels are available after deterministic event filtering.
- The primary subject-level Spearman correlation between frozen model score and ordinal workload level is positive.
- The primary one-sided Spearman p-value for monotonic increase is `< 0.05`.
- The point estimate is at least `rho >= 0.20`.

## Failure Criterion

The confirmatory test is a failure if any of the following occur:

- The primary Spearman rho is `<= 0`.
- The one-sided p-value is `>= 0.05`.
- The point estimate is positive but smaller than `rho = 0.20`.
- Fewer than 6 ordered non-tutorial difficulty levels are available.
- Fewer than 12 readable subjects remain after deterministic extraction.

## Dataset-Limitation Status

If raw EEG payloads are unavailable because only git-annex/LFS pointer stubs are present, the result is not interpretable and must be reported as `not_run_raw_data_unavailable`, not success or failure.

If the local DS007262 checkout contains seven difficulty levels rather than the requested eight, the script still computes the primary monotonicity statistic over the available levels, but the result must be reported with `available_level_count = 7`. This is a dataset/snapshot limitation and cannot be hidden by relabeling the levels.

## Reporting Rule

Report the exact rho, p-value, number of subjects, number of subject-level rows, number of trials, and available difficulty levels. Do not round a non-significant result into significance, do not remove subjects post hoc, and do not tune preprocessing or model hyperparameters after viewing DS007262 scores.
