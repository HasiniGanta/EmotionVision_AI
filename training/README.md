# Optional Advanced Training Module

This folder is **not required** to run EmotionVision AI. The application ships
with, and automatically downloads, a pretrained emotion recognition model
(see the main README and `config.py`). This module is only for users who
want to explore training their own model locally, on their own machine.

## Scripts

- `prepare_dataset.py` — Attempts to locate/prepare a legitimate public
  emotion dataset (e.g., FER2013 in CSV form) on your local machine. It will
  **not** fabricate a dataset. If it cannot find or automatically obtain one
  because of licensing/authentication requirements (FER2013 is typically
  distributed via Kaggle, which requires an account), it prints clear
  instructions instead of pretending to succeed.
- `train_model.py` — Trains a small CNN on the prepared dataset. Must be run
  explicitly by the user; never runs automatically.
- `evaluate_model.py` — Evaluates a trained model on a held-out split and
  reports real, measured accuracy (never fabricated).

## Usage (run locally, not in a cloud/agent sandbox)

```bash
python training/prepare_dataset.py
python training/train_model.py
python training/evaluate_model.py
```

## Important

- This module is designed for a local machine with a reasonable CPU (a GPU
  will speed it up but is not required).
- Do **not** attempt to run large-scale training here inside a cloud agent
  environment — it is meant for your own computer, on your own time.
- No step in this module invents or hardcodes accuracy numbers. If a metric
  cannot be honestly computed, the scripts say so explicitly.
