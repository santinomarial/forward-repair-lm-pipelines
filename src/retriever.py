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

    def retrieve_union(self, queries: list[str], top_k_total: int) -> list[dict]:
        """Run BM25 for each query over the full corpus, merge by best rank, return top_k_total docs."""
        best_rank: dict[int, int] = {}    # corpus_idx -> minimum rank achieved across queries
        best_score: dict[int, float] = {} # corpus_idx -> BM25 score at that best rank

        for query in queries:
            tokenized_query = tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            for rank, idx in enumerate(ranked_indices):
                if idx not in best_rank or rank < best_rank[idx]:
                    best_rank[idx] = rank
                    best_score[idx] = float(scores[idx])

        sorted_indices = sorted(best_rank, key=lambda i: best_rank[i])

        results = []
        for idx in sorted_indices[:top_k_total]:
            doc = dict(self.corpus[idx])
            doc["score"] = best_score[idx]
            results.append(doc)

        return results