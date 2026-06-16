"""
Generate a new MIDI file from a trained LSTM model.

Run from the project root:
    python src/generate.py
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import tensorflow as tf
from music21 import chord, instrument, note, stream

from utils import MODELS_DIR, OUTPUT_DIR, PROCESSED_DIR, ensure_project_dirs, load_json, load_pickle


def sample_with_temperature(predictions: np.ndarray, temperature: float) -> int:
    """
    Pick the next note index from model probabilities.

    Lower temperature is more conservative. Higher temperature is more random.
    """
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(np.maximum(predictions, 1e-8)) / temperature
    exp_predictions = np.exp(predictions)
    probabilities = exp_predictions / np.sum(exp_predictions)
    return int(np.random.choice(len(probabilities), p=probabilities))


def create_midi(prediction_output: list[str], output_path) -> None:
    """Convert generated note/chord strings into a MIDI file."""
    offset = 0.0
    output_notes = []

    for pattern in prediction_output:
        if "." in pattern or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=str(output_path))


def generate_music(num_notes: int, temperature: float) -> None:
    """Load the trained model and generate a MIDI sequence."""
    ensure_project_dirs()

    model_path = MODELS_DIR / "music_lstm_model.keras"
    notes_path = PROCESSED_DIR / "notes.pkl"
    int_to_note_path = PROCESSED_DIR / "int_to_note.json"
    metadata_path = PROCESSED_DIR / "metadata.json"

    missing = [path for path in [model_path, notes_path, int_to_note_path, metadata_path] if not path.exists()]
    if missing:
        print("Required files are missing. Run these commands first:")
        print("python src/preprocess.py")
        print("python src/train.py")
        for path in missing:
            print(f"Missing: {path}")
        return

    model = tf.keras.models.load_model(model_path)
    notes = load_pickle(notes_path)
    int_to_note_raw = load_json(int_to_note_path)
    int_to_note = {int(key): value for key, value in int_to_note_raw.items()}
    metadata = load_json(metadata_path)

    sequence_length = int(metadata["sequence_length"])
    note_to_int = {value: key for key, value in int_to_note.items()}

    if len(notes) <= sequence_length:
        print("Not enough processed notes to create a seed pattern.")
        return

    start = random.randint(0, len(notes) - sequence_length - 1)
    seed_notes = notes[start : start + sequence_length]
    pattern = [note_to_int[item] for item in seed_notes]

    prediction_output: list[str] = []

    print("Generating music...")
    for _ in range(num_notes):
        prediction_input = np.array([pattern], dtype=np.int32)
        prediction = model.predict(prediction_input, verbose=0)[0]
        index = sample_with_temperature(prediction, temperature)
        result = int_to_note[index]

        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]

    output_path = OUTPUT_DIR / "generated_music.mid"
    create_midi(prediction_output, output_path)
    print(f"Generated MIDI saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIDI music from a trained LSTM model.")
    parser.add_argument("--num-notes", type=int, default=300, help="Number of notes/chords to generate.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling randomness. Try 0.5 for safer output or 1.2 for more variety.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_music(num_notes=args.num_notes, temperature=args.temperature)
