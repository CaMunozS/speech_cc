from __future__ import annotations

from typing import List
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Path to the locally stored model inside the project.
_MODEL_PATH = Path(__file__).resolve().parent / "model" / "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(str(_MODEL_PATH))
    return _model


def embed_texts(texts: List[str]):
    """Return embeddings for a list of texts using the sentence-transformers model.

    Args:
        texts: list of strings to encode.

    Returns:
        A 2D numpy array containing embeddings for each input text.
    """
    model = _get_model()
    return model.encode(texts)


if __name__ == "__main__":
    sentences = [
        "Hola mundo",
        "Esto es un ejemplo",
    ]
    embeddings = embed_texts(sentences)
    print(embeddings.shape)
