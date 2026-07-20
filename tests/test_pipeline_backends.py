from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import ForwardRepairPipeline
from retriever import BM25Retriever, DenseRetriever


CORPUS = [
    {"id": "alpha", "title": "Alpha", "text": "First letter."},
    {"id": "beta", "title": "Beta", "text": "Second letter."},
    {"id": "gamma", "title": "Gamma", "text": "Third letter."},
]


def _dense_encoder(texts):
    vectors = {
        "Alpha First letter.": [1.0, 0.0],
        "Beta Second letter.": [0.0, 1.0],
        "Gamma Third letter.": [-1.0, 0.0],
        "Alpha": [1.0, 0.0],
    }
    return np.array([vectors[text] for text in texts], dtype=np.float32)


@pytest.mark.parametrize(
    "retriever",
    [
        BM25Retriever(CORPUS, top_k=1),
        DenseRetriever(CORPUS, top_k=1, encoder=_dense_encoder),
    ],
)
def test_pipeline_forward_path_is_backend_agnostic(retriever):
    pipeline = ForwardRepairPipeline(retriever)
    pipeline.baseline_query_generator = lambda question: SimpleNamespace(query="Alpha")
    pipeline.answer_generator = lambda question, context: SimpleNamespace(answer="alpha")

    result = pipeline.run_baseline("What is the first Greek letter?")

    assert result["docs"][0]["id"] == "alpha"
    assert "Title: Alpha" in result["context"]
    assert result["answer"] == "alpha"
