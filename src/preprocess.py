import re
from typing import Iterable

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s]")


def basic_clean(text: str) -> str:
    """Lightweight cleaner; TF-IDF handles tokenization & stopwords later."""
    text = text or ""
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_batch(texts: Iterable[str]) -> list[str]:
    return [basic_clean(t) for t in texts]