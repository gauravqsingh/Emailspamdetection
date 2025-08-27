import os
import pandas as pd
from pathlib import Path
from typing import Union

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "spam.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Union[str, os.PathLike, None] = None) -> pd.DataFrame:
    """Load dataset with required columns: label, message."""
    csv_path = Path(path) if path else DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place spam.csv with columns ['label','message'] in data/."
        )
    # Many public spam datasets contain trailing commas which create ragged rows.
    # Read robustly by selecting the first two columns with the Python engine.
    try:
        df = pd.read_csv(
            csv_path,
            encoding='ISO-8859-1',
            engine='python',
            usecols=[0, 1],
            header=0,
            names=['label', 'message'],
        )
        return df[['label', 'message']].copy()
    except Exception:
        # Fallback: try to infer and then map common header variants
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        if {'label', 'message'}.issubset(df.columns):
            return df[['label', 'message']].copy()
        if {'Category', 'Message'}.issubset(df.columns):
            df = df.rename(columns={'Category': 'label', 'Message': 'message'})
            return df[['label', 'message']].copy()
        raise ValueError("CSV must contain columns: 'label' and 'message' (or 'Category'/'Message').")