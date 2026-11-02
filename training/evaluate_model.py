"""
training/evaluate_model.py

OPTIONAL, local-only. Evaluates a locally trained model on the held-out
FER2013 test split and reports real, measured accuracy. Never fabricates
a number.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_PATH = os.path.join("training", "data", "fer2013.csv")
MODEL_PATH = os.path.join("training", "output", "custom_emotion_model.h5")


def main():
    print("=" * 60)
    print("EmotionVision AI - Optional Local Model Evaluation")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH}. Run train_model.py first.")
        return
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run prepare_dataset.py first.")
        return

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    model = tf.keras.models.load_model(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)
    df = df[df["Usage"] == "PrivateTest"]

    pixels = df["pixels"].apply(lambda s: np.array(s.split(), dtype="float32").reshape(48, 48, 1) / 255.0)
    X_test = np.stack(pixels.values)
    y_test = to_categorical(df["emotion"].values)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Measured test accuracy: {accuracy * 100:.2f}%  (loss: {loss:.4f})")
    print("This is a real, measured value from your local evaluation run —")
    print("not a fabricated or copied number.")


if __name__ == "__main__":
    main()
