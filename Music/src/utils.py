import json
import pickle
from pathlib import Path


def ensure_dir(path):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data, file_path):
    ensure_dir(Path(file_path).parent)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(data, file_path):
    ensure_dir(Path(file_path).parent)
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle)


def load_pickle(file_path):
    with open(file_path, "rb") as handle:
        return pickle.load(handle)
