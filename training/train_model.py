"""
training/train_model.py

OPTIONAL, advanced, local-only. Trains a small CNN from a prepared
FER2013-style CSV dataset. Must be run explicitly by the user.
Not executed automatically at any point by the main application.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_PATH = os.path.join("training", "data", "fer2013.csv")
OUTPUT_MODEL_PATH = os.path.join("training", "output", "custom_emotion_model.h5")


def build_model(num_classes: int = 7):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_fer2013_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    pixels = df["pixels"].apply(lambda s: np.array(s.split(), dtype="float32").reshape(48, 48, 1) / 255.0)
    X = np.stack(pixels.values)
    y = df["emotion"].values
    usage = df["Usage"].values
    return X, y, usage


def main():
    print("=" * 60)
    print("EmotionVision AI - Optional Local Training")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}.")
        print("Run training/prepare_dataset.py first for instructions.")
        return

    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    print("Loading dataset...")
    X, y, usage = load_fer2013_csv(DATASET_PATH)
    y_cat = to_categorical(y, num_classes=int(y.max()) + 1)

    train_mask = usage == "Training"
    val_mask = usage == "PublicTest"

    X_train, y_train = X[train_mask], y_cat[train_mask]
    X_val, y_val = X[val_mask], y_cat[val_mask]

    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")

    model = build_model(num_classes=y_cat.shape[1])
    print("Training... (this may take a while on CPU; reduce epochs if needed)")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=64)

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    model.save(OUTPUT_MODEL_PATH)
    print(f"Model saved to {OUTPUT_MODEL_PATH}")
    print("To use this model in the main app, update MODEL_PATH / MODEL_SOURCES in config.py")
    print("Run evaluate_model.py next to see real, measured accuracy.")


if __name__ == "__main__":
    main()
