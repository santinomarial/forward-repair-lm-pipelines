from retriever import BM25Retriever, tokenize


def _corpus() -> list[dict]:
    return [
        {"id": "python", "title": "Python", "text": "A programming language."},
        {"id": "snake", "title": "Pythonidae", "text": "A family of snakes."},
        {"id": "java", "title": "Java", "text": "A programming language."},
        {"id": "coffee", "title": "Coffee", "text": "A brewed drink."},
    ]


def test_tokenize_is_case_insensitive_and_removes_punctuation():
    assert tokenize("Python's LANGUAGE!") == ["python", "s", "language"]


def test_bm25_retrieve_ranks_matching_document_first_and_respects_top_k():
    retriever = BM25Retriever(_corpus(), top_k=2)

    results = retriever.retrieve("Java")

    assert [doc["id"] for doc in results] == ["java", "python"]
    assert len(results) == 2
    assert results[0]["score"] > results[1]["score"]
    assert "score" not in _corpus()[0]


class _StubBM25:
    def __init__(self, scores_by_query):
        self.scores_by_query = scores_by_query

    def get_scores(self, tokenized_query):
        return self.scores_by_query[tuple(tokenized_query)]


def test_retrieve_union_deduplicates_and_orders_by_best_rank():
    retriever = BM25Retriever(_corpus(), top_k=2)
    retriever.bm25 = _StubBM25(
        {
            ("first",): [9.0, 8.0, 7.0, 6.0],
            ("second",): [1.0, 3.0, 10.0, 2.0],
        }
    )

    results = retriever.retrieve_union(["first", "second"], top_k_total=3)

    assert [doc["id"] for doc in results] == ["python", "java", "snake"]
    assert len({doc["id"] for doc in results}) == 3
    assert [doc["score"] for doc in results] == [9.0, 10.0, 8.0]


def test_retrieve_union_keeps_score_from_query_with_best_rank():
    retriever = BM25Retriever(_corpus(), top_k=2)
    retriever.bm25 = _StubBM25(
        {
            ("first",): [5.0, 4.0, 3.0, 2.0],
            ("second",): [100.0, 2.0, 3.0, 4.0],
        }
    )

    results = retriever.retrieve_union(["first", "second"], top_k_total=1)

    assert results[0]["id"] == "python"
    assert results[0]["score"] == 5.0
