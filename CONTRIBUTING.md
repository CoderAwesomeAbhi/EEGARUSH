# Contributing

This repository is intended to support a transparent EEG machine-learning
analysis, not clinical deployment.

## Development workflow

1. Create a clean Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python scripts/smoke_test_synthetic.py` before changing modeling code.
4. For changes that affect results, rerun the full PhysioNet pipeline and
   update the manuscript tables and figures from the regenerated outputs.

## Scientific guardrails

- Do not describe the model as diagnostic or clinically validated.
- Keep subject-wise evaluation as the primary result.
- Report all pipeline changes that can affect extracted features, train/test
  splits, or model hyperparameters.
- Keep raw EDF files out of Git; users should obtain them from PhysioNet under
  the original dataset license.
