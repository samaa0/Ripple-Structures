from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("RIPPLE_ML_MODEL_DIR", BASE_DIR / "models")).expanduser()
DATA_DIR = Path(os.getenv("RIPPLE_ML_DATA_DIR", BASE_DIR / "data")).expanduser()


def model_path(filename):
    return MODEL_DIR / filename


def data_path(filename):
    return DATA_DIR / filename


def study_storage(filename):
    return f"sqlite:///{model_path(filename)}"
