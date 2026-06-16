import argparse
from pathlib import Path

from music21 import chord, converter, instrument, note

from utils import load_json, save_json, save_pickle


def find_midi_files(data_folder):
    return sorted(Path(data_folder).glob("**/*.mid")) + sorted(Path(data_folder).glob("**/*.midi"))


def extract_notes_from_score(score):
    notes = []
    parts = instrument.partitionByInstrument(score)
    elements = parts.parts if parts else [score.flat]

    for part in elements:
        for element in part.recurse():
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(p.midi) for p in element.pitches))

    return notes


def parse_midi_file(midi_path):
    try:
        score = converter.parse(str(midi_path))
        return extract_notes_from_score(score)
    except Exception as exc:
        print(f"Skipping '{midi_path.name}' because it could not be parsed: {exc}")
        return []


def build_sequences(notes, sequence_length=100):
    note_names = sorted(set(notes))
    note_to_int = {note_name: number for number, note_name in enumerate(note_names)}
    int_to_note = {number: note_name for note_name, number in note_to_int.items()}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[note_name] for note_name in sequence_in])
        network_output.append(note_to_int[sequence_out])

    return network_input, network_output, note_to_int, int_to_note


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIDI files into training sequences.")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=100,
        help="Length of each training sequence.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    midi_folder = project_root / "data" / "midi"
    processed_folder = project_root / "data" / "processed"
    processed_folder.mkdir(parents=True, exist_ok=True)

    midi_files = find_midi_files(midi_folder)
    print(f"Found {len(midi_files)} MIDI files in {midi_folder}")

    if not midi_files:
        print("No MIDI files found. Place .mid files in data/midi/ and rerun this script.")
        return

    all_notes = []
    for midi_file in midi_files:
        file_notes = parse_midi_file(midi_file)
        if file_notes:
            all_notes.extend(file_notes)

    if len(all_notes) < args.sequence_length + 1:
        print(
            "Not enough notes to build training sequences. Add more MIDI files or lower the sequence length."
        )
        return

    network_input, network_output, note_to_int, int_to_note = build_sequences(
        all_notes, sequence_length=args.sequence_length
    )

    save_json(note_to_int, processed_folder / "note_to_int.json")
    save_json(int_to_note, processed_folder / "int_to_note.json")
    save_pickle(network_input, processed_folder / "input_sequences.pkl")
    save_pickle(network_output, processed_folder / "output_sequences.pkl")
    save_json(
        {
            "sequence_length": args.sequence_length,
            "vocab_size": len(note_to_int),
            "num_sequences": len(network_input),
        },
        processed_folder / "metadata.json",
    )

    print(f"Saved processed data to {processed_folder}")
    print(f"Vocabulary size: {len(note_to_int)}")
    print(f"Training sequences: {len(network_input)}")


if __name__ == "__main__":
    main()
