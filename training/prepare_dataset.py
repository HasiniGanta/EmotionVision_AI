"""
training/prepare_dataset.py

OPTIONAL. Not run automatically by the main application.

Looks for a local FER2013-style CSV dataset (columns: emotion, pixels, Usage).
FER2013 is commonly distributed through Kaggle, which requires an account/
authentication, so this script cannot legally auto-download it without
credentials. It will never fabricate a dataset.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_CANDIDATES = [
    os.path.join("training", "data", "fer2013.csv"),
    os.path.join("data", "fer2013.csv"),
]


def find_local_dataset():
    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def main():
    print("=" * 60)
    print("EmotionVision AI - Optional Dataset Preparation")
    print("=" * 60)

    path = find_local_dataset()
    if path:
        print(f"Found local dataset at: {path}")
        print("You can proceed to run train_model.py")
        return

    print("No local FER2013 dataset was found.")
    print()
    print("This project does NOT include or fabricate a training dataset.")
    print("FER2013 is commonly distributed via Kaggle, which requires a free")
    print("account and cannot be auto-downloaded without your credentials.")
    print()
    print("To use this optional training module:")
    print("  1. Create a free Kaggle account: https://www.kaggle.com")
    print("  2. Download the FER2013 dataset (search 'fer2013' on Kaggle)")
    print("  3. Place the CSV file at: training/data/fer2013.csv")
    print("  4. Re-run this script, then train_model.py")
    print()
    print("The main application does NOT need this — it already uses an")
    print("automatically-downloaded pretrained model.")


if __name__ == "__main__":
    main()
