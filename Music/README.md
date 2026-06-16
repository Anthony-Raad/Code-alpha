# Music Generation with AI

This project trains an LSTM neural network on MIDI files and generates new MIDI music. It uses `music21` to read MIDI notes and chords, TensorFlow/Keras to train the model, and `music21` again to write generated music to `output/generated_music.mid`.

## Project Structure

```text
music-generation-ai/
├── data/
│   └── midi/
├── models/
├── output/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
├── requirements.txt
└── README.md
```

The scripts also create `data/processed/` automatically for saved training data and mappings.

## Install Dependencies

Create and activate a virtual environment if you want to keep dependencies separate:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Add MIDI Files

Put your MIDI files in:

```text
data/midi/
```

The project supports `.mid` and `.midi` files. You can use classical, jazz, pop, game music, or any other MIDI dataset.

Sample MIDI dataset sources:

- Classical MIDI Archives: https://www.classicalarchives.com/midi.html
- Kunst der Fuge MIDI files: https://www.kunstderfuge.com/
- Lakh MIDI Dataset: https://colinraffel.com/projects/lmd/
- Jazz MIDI files: https://bushgrafts.com/midi/

Only use files that you are allowed to download and use for your project.

## Preprocess Data

Run:

```bash
python src/preprocess.py
```

This script:

- Loads MIDI files from `data/midi/`
- Extracts notes such as `C4` and chords such as `0.4.7`
- Converts them to integer sequences
- Saves processed files in `data/processed/`

The default sequence length is `100`. To change it:

```bash
python src/preprocess.py --sequence-length 50
```

## Train the Model

Run:

```bash
python src/train.py
```

This script:

- Loads processed sequences
- Builds an LSTM model with an embedding layer and dropout
- Saves checkpoints in `models/`
- Saves the final model as `models/music_lstm_model.keras`

Default training settings:

- Epochs: `30`
- Batch size: `64`

You can change them from the command line:

```bash
python src/train.py --epochs 50 --batch-size 32
```

For a small student project or a slow computer, start with fewer epochs:

```bash
python src/train.py --epochs 5
```

## Generate Music

After training, run:

```bash
python src/generate.py
```

This creates:

```text
output/generated_music.mid
```

Optional settings:

```bash
python src/generate.py --num-notes 400 --temperature 0.8
```

Temperature controls randomness:

- `0.5` makes safer, more predictable music
- `0.8` is a balanced default
- `1.2` makes more surprising music

## Open or Play the Generated MIDI

You can open `output/generated_music.mid` with:

- MuseScore
- GarageBand
- Logic Pro
- FL Studio
- Ableton Live
- VLC Media Player
- Any MIDI player or digital audio workstation

If the file opens silently, choose a piano or another instrument in your MIDI player.

## Full Workflow

From inside the `music-generation-ai/` folder:

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/generate.py
```

## Notes

- More MIDI files usually produce better results.
- Training can take a long time without a GPU.
- If a MIDI file cannot be parsed, preprocessing prints the error and continues with the next file.
- Generated music quality depends heavily on the dataset size, style consistency, and number of training epochs.
