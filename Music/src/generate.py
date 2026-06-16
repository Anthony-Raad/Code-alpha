import argparse
import random
from pathlib import Path

import numpy as np
from music21 import chord, instrument, note, stream
import tensorflow as tf

from utils import load_json, load_pickle


def generate_notes(model, seed_sequence, int_to_note, sequence_length, num_steps=300):
    pattern = list(seed_sequence)
    generated = []

    for _ in range(num_steps):
        input_data = np.array([pattern[-sequence_length:]], dtype=np.int32)
        prediction = model.predict(input_data, verbose=0)[0]
        index = int(np.argmax(prediction))
        result = int_to_note[str(index)]

        generated.append(result)
        pattern.append(index)

    return generated


def create_midi(prediction_output, output_path):
    output_notes = []

    for pattern in prediction_output:
        if "." in pattern or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            chord_notes = [note.Note(int(p)) for p in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.quarterLength = 0.5
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.quarterLength = 0.5
            output_notes.append(new_note)

    midi_stream = stream.Stream(output_notes)
    midi_stream.insert(0, instrument.Piano())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_stream.write("midi", fp=str(output_path))
    print(f"Generated MIDI saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI music from a trained model.")
    parser.add_argument(
        "--output",
        type=str,
        default="output/generated_music.mid",
        help="Path to save the generated MIDI file.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=300,
        help="How many notes/chords to generate.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    processed_folder = project_root / "data" / "processed"
    model_path = project_root / "models" / "music_model.h5"
    output_path = project_root / args.output

    if not processed_folder.exists():
        print("Processed data not found. Run python src/preprocess.py first.")
        return

    if not model_path.exists():
        print("Trained model not found. Run python src/train.py first.")
        return

    note_to_int = load_json(processed_folder / "note_to_int.json")
    int_to_note = load_json(processed_folder / "int_to_note.json")
    metadata = load_json(processed_folder / "metadata.json")
    sequence_length = metadata.get("sequence_length", 100)

    seed_data = load_pickle(processed_folder / "input_sequences.pkl")
    seed_sequence = random.choice(seed_data)

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    generated_notes = generate_notes(
        model,
        seed_sequence,
        int_to_note,
        sequence_length,
        num_steps=args.num_steps,
    )

    create_midi(generated_notes, output_path)


if __name__ == "__main__":
    main()
