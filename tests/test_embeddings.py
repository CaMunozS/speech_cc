import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from embedding_model import embed_texts


def test_embeddings_shape():
    texts = ["Hola", "Adios"]
    try:
        emb = embed_texts(texts)
    except Exception as exc:  # pragma: no cover - network issues
        pytest.skip(f"Model not available: {exc}")
    assert emb.shape[0] == len(texts)
