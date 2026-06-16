import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.utils import to_categorical

from utils import load_json, load_pickle


def build_model(sequence_length, vocab_size):
    model = Sequential(
        [
            Input(shape=(sequence_length,)),
            Embedding(vocab_size, 128, input_length=sequence_length),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(256),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(vocab_size, activation="softmax"),
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def load_training_data(processed_folder):
    note_to_int = load_json(processed_folder / "note_to_int.json")
    metadata = load_json(processed_folder / "metadata.json")
    input_sequences = load_pickle(processed_folder / "input_sequences.pkl")
    output_sequences = load_pickle(processed_folder / "output_sequences.pkl")

    X = np.array(input_sequences, dtype=np.int32)
    y = to_categorical(output_sequences, num_classes=len(note_to_int))

    return X, y, metadata


def main():
    parser = argparse.ArgumentParser(description="Train a music generation model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    processed_folder = project_root / "data" / "processed"
    model_folder = project_root / "models"
    checkpoint_folder = model_folder / "checkpoints"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    if not processed_folder.exists():
        print("Processed data not found. Run python src/preprocess.py first.")
        return

    X, y, metadata = load_training_data(processed_folder)
    sequence_length = metadata.get("sequence_length", X.shape[1])
    vocab_size = metadata.get("vocab_size", y.shape[1])

    print(f"Training data loaded: {X.shape[0]} sequences")
    print(f"Sequence length: {sequence_length}")
    print(f"Vocabulary size: {vocab_size}")

    model = build_model(sequence_length, vocab_size)
    model.summary()

    checkpoint_path = checkpoint_folder / "weights.{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="loss",
        save_best_only=False,
        save_weights_only=False,
        verbose=1,
    )

    model.fit(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint],
    )

    model_path = model_folder / "music_model.h5"
    model.save(model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
