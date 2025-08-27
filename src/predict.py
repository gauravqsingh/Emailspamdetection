from __future__ import annotations
from joblib import load
from pathlib import Path

from utils import MODELS_DIR

MODEL_PATH = MODELS_DIR / 'spam_classifier.pkl'


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train it first: python -m src.train")
    return load(MODEL_PATH)


def predict(texts: list[str]) -> list[tuple[str, float]]:
    """Returns list of (label, confidence) for each text."""
    pipe = load_model()
    # MultinomialNB has predict_proba
    probs = pipe.predict_proba(texts)
    preds = pipe.classes_[probs.argmax(axis=1)]
    confs = probs.max(axis=1)
    return list(zip(preds.tolist(), confs.tolist()))


if __name__ == "__main__":
    # Quick manual test
    examples = [
        "Congratulations! You have won a free iPhone. Click here to claim now.",
        "Hi mom, I will be home by 7 pm",
    ]
    for text, (label, conf) in zip(examples, predict(examples)):
        print(f"{label} ({conf:.2%}) :: {text}")