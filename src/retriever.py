import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, corpus: list[dict], top_k: int = 3):
        self.corpus = corpus
        self.top_k = top_k
        self.tokenized_docs = [
            tokenize(doc["title"] + " " + doc["text"]) for doc in corpus
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str) -> list[dict]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []
        for idx in ranked_indices[: self.top_k]:
            doc = dict(self.corpus[idx])
            doc["score"] = float(scores[idx])
            results.append(doc)

        return results