"""
Train an LSTM model on preprocessed MIDI sequences.

Run from the project root:
    python src/train.py
"""

from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential

from utils import MODELS_DIR, PROCESSED_DIR, ensure_project_dirs, load_json, load_pickle


def build_model(vocab_size: int, sequence_length: int, embedding_dim: int = 100) -> tf.keras.Model:
    """Build a beginner-friendly LSTM music model."""
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(256),
            Dropout(0.3),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(vocab_size, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train_model(epochs: int, batch_size: int) -> None:
    """Load processed data, train the LSTM, and save model files."""
    ensure_project_dirs()

    input_path = PROCESSED_DIR / "network_input.pkl"
    output_path = PROCESSED_DIR / "network_output.pkl"
    mapping_path = PROCESSED_DIR / "note_to_int.json"
    metadata_path = PROCESSED_DIR / "metadata.json"

    missing = [path for path in [input_path, output_path, mapping_path, metadata_path] if not path.exists()]
    if missing:
        print("Processed files are missing. Run preprocessing first:")
        print("python src/preprocess.py")
        for path in missing:
            print(f"Missing: {path}")
        return

    network_input = load_pickle(input_path)
    network_output = load_pickle(output_path)
    note_to_int = load_json(mapping_path)
    metadata = load_json(metadata_path)

    sequence_length = int(metadata["sequence_length"])
    vocab_size = len(note_to_int)

    if len(network_input) == 0:
        print("No training sequences found. Add more MIDI data and preprocess again.")
        return

    network_input = np.asarray(network_input, dtype=np.int32)
    network_output = np.asarray(network_output, dtype=np.int32)

    model = build_model(vocab_size=vocab_size, sequence_length=sequence_length)
    model.summary()

    checkpoint_path = MODELS_DIR / "checkpoint_epoch_{epoch:02d}.keras"
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    print("Starting training...")
    model.fit(
        network_input,
        network_output,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
    )

    final_model_path = MODELS_DIR / "music_lstm_model.keras"
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LSTM music generation model.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size)
