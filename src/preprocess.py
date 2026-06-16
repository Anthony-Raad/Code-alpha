"""
Preprocess MIDI files into integer training sequences.

Run from the project root:
    python src/preprocess.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from music21 import chord, converter, instrument, note

from utils import MIDI_DIR, PROCESSED_DIR, ensure_project_dirs, find_midi_files, save_json, save_pickle


def extract_notes_from_midi(midi_path: Path) -> list[str]:
    """
    Extract notes and chords from one MIDI file.

    Notes are stored as names such as C4 or F#5.
    Chords are stored as dot-separated pitch classes such as 0.4.7.
    """
    midi = converter.parse(str(midi_path))

    try:
        parts = instrument.partitionByInstrument(midi)
    except Exception:
        parts = None

    if parts:
        elements = []
        for part in parts.parts:
            elements.extend(part.recurse())
    else:
        elements = midi.flat.notes

    extracted: list[str] = []
    for element in elements:
        if isinstance(element, note.Note):
            extracted.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            extracted.append(".".join(str(pitch) for pitch in element.normalOrder))

    return extracted


def create_sequences(notes: list[str], sequence_length: int) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[int, str]]:
    """Convert note/chord strings into integer input and target arrays."""
    pitchnames = sorted(set(notes))
    note_to_int = {pitch: number for number, pitch in enumerate(pitchnames)}
    int_to_note = {number: pitch for pitch, number in note_to_int.items()}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[item] for item in sequence_in])
        network_output.append(note_to_int[sequence_out])

    return (
        np.array(network_input, dtype=np.int32),
        np.array(network_output, dtype=np.int32),
        note_to_int,
        int_to_note,
    )


def preprocess_dataset(sequence_length: int, max_files: int | None = None) -> None:
    """Read MIDI files, extract notes/chords, and save training artifacts."""
    ensure_project_dirs()
    midi_files = find_midi_files(MIDI_DIR)

    if not midi_files:
        print(f"No MIDI files found in {MIDI_DIR}")
        print("Add .mid or .midi files to data/midi/ and run preprocessing again.")
        return

    if max_files is not None:
        midi_files = midi_files[:max_files]
        print(f"Using the first {len(midi_files)} MIDI file(s) because --max-files was set.")

    all_notes: list[str] = []
    failed_files: list[str] = []

    print(f"Found {len(midi_files)} MIDI file(s). Extracting notes and chords...")
    for midi_file in midi_files:
        try:
            file_notes = extract_notes_from_midi(midi_file)
            if file_notes:
                all_notes.extend(file_notes)
                print(f"Parsed {midi_file.name}: {len(file_notes)} events")
            else:
                print(f"Skipped {midi_file.name}: no notes or chords found")
        except Exception as exc:
            failed_files.append(str(midi_file))
            print(f"Could not parse {midi_file.name}: {exc}")

    if len(all_notes) <= sequence_length:
        print("Not enough musical events to create training sequences.")
        print(f"Need more than {sequence_length} notes/chords, found {len(all_notes)}.")
        return

    network_input, network_output, note_to_int, int_to_note = create_sequences(all_notes, sequence_length)

    save_pickle(all_notes, PROCESSED_DIR / "notes.pkl")
    save_pickle(network_input, PROCESSED_DIR / "network_input.pkl")
    save_pickle(network_output, PROCESSED_DIR / "network_output.pkl")
    save_json(note_to_int, PROCESSED_DIR / "note_to_int.json")
    save_json({str(key): value for key, value in int_to_note.items()}, PROCESSED_DIR / "int_to_note.json")
    save_json(
        {
            "sequence_length": sequence_length,
            "total_events": len(all_notes),
            "unique_events": len(note_to_int),
            "training_sequences": int(len(network_input)),
            "failed_files": failed_files,
        },
        PROCESSED_DIR / "metadata.json",
    )

    print("Preprocessing complete.")
    print(f"Saved processed data to {PROCESSED_DIR}")
    print(f"Training sequences: {len(network_input)}")
    print(f"Unique notes/chords: {len(note_to_int)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MIDI files for LSTM training.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Number of previous notes/chords used to predict the next one.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick tests. Example: --max-files 10",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(sequence_length=args.sequence_length, max_files=args.max_files)
