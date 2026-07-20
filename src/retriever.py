import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Retriever(ABC):
    """Backend-neutral document retriever used by the repair pipeline."""

    def __init__(self, corpus: list[dict], top_k: int = 3):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.corpus = corpus
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return ranked documents with a numeric ``score`` field."""

    def retrieve_union(self, queries: list[str], top_k_total: int) -> list[dict]:
        """Merge query results by best rank while deduplicating document IDs."""
        if top_k_total <= 0 or not queries:
            return []

        best: dict[str, tuple[int, int, dict]] = {}
        seen_order = 0
        for query in queries:
            for rank, doc in enumerate(self.retrieve(query, top_k=top_k_total)):
                doc_id = str(doc["id"])
                if doc_id not in best:
                    best[doc_id] = (rank, seen_order, doc)
                    seen_order += 1
                elif rank < best[doc_id][0]:
                    _, first_seen, _ = best[doc_id]
                    best[doc_id] = (rank, first_seen, doc)

        ranked = sorted(best.values(), key=lambda item: (item[0], item[1]))
        return [dict(doc) for _, _, doc in ranked[:top_k_total]]


class BM25Retriever(Retriever):
    def __init__(self, corpus: list[dict], top_k: int = 3):
        super().__init__(corpus=corpus, top_k=top_k)
        self.tokenized_docs = [
            tokenize(doc["title"] + " " + doc["text"]) for doc in corpus
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        limit = self.top_k if top_k is None else top_k
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []
        for index in ranked_indices[:limit]:
            doc = dict(self.corpus[index])
            doc["score"] = float(scores[index])
            results.append(doc)
        return results


class DenseRetriever(Retriever):
    """Cosine-similarity retrieval over sentence-transformer embeddings."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        corpus: list[dict],
        top_k: int = 3,
        model_name: str = DEFAULT_MODEL,
        encoder: Callable[[list[str]], Any] | None = None,
    ):
        super().__init__(corpus=corpus, top_k=top_k)
        self.model_name = model_name
        self._encoder = encoder or self._load_encoder(model_name)
        documents = [f"{doc['title']} {doc['text']}" for doc in corpus]
        self.embeddings = self._normalize(self._encode(documents))

    @staticmethod
    def _load_encoder(model_name: str) -> Callable[[list[str]], Any]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires sentence-transformers. "
                "Install it with `pip install -r requirements-dense.txt`."
            ) from exc

        model = SentenceTransformer(model_name)
        return lambda texts: model.encode(texts, show_progress_bar=False)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._encoder(texts), dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim != 2:
            raise ValueError("encoder must return a two-dimensional embedding matrix")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, np.finfo(np.float32).eps)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        limit = self.top_k if top_k is None else top_k
        query_vector = self._normalize(self._encode([query]))[0]
        scores = self.embeddings @ query_vector
        ranked_indices = np.argsort(-scores, kind="stable")[:limit]

        results = []
        for index in ranked_indices:
            doc = dict(self.corpus[int(index)])
            doc["score"] = float(scores[index])
            results.append(doc)
        return results


def build_retriever(
    backend: str,
    corpus: list[dict],
    top_k: int,
    dense_model: str = DenseRetriever.DEFAULT_MODEL,
) -> Retriever:
    if backend == "bm25":
        return BM25Retriever(corpus=corpus, top_k=top_k)
    if backend == "dense":
        return DenseRetriever(corpus=corpus, top_k=top_k, model_name=dense_model)
    raise ValueError(f"Unknown retriever backend: {backend!r}")
