from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .utils import load_data, ensure_dirs, MODELS_DIR, RESULTS_DIR
from .preprocess import basic_clean


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=basic_clean, stop_words="english", ngram_range=(1, 2),
                                   max_df=0.95, min_df=2)),
        ("clf", MultinomialNB()),
    ])


def plot_confusion(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ensure_dirs()

    df = load_data()
    df = df.dropna(subset=['label', 'message']).copy()
    df['label'] = df['label'].str.strip().str.lower()
    # Keep only ham/spam
    df = df[df['label'].isin(['ham', 'spam'])]

    X = df['message'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

    # Save model
    model_path = MODELS_DIR / 'spam_classifier.pkl'
    dump(pipe, model_path)

    # Save metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'metrics.txt').write_text(
        f"Accuracy: {acc:.4f}\n\nClassification Report:\n{report}\n"
    )

    # Save confusion matrix
    plot_confusion(y_test, y_pred, labels=['ham', 'spam'], out_path=RESULTS_DIR / 'confusion_matrix.png')

    # Also save a small JSON for quick reading (optional)
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps({"accuracy": acc}, indent=2))

    print(f"Model saved to: {model_path}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()