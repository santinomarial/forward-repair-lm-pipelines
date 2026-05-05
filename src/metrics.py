import re


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize(prediction) == normalize(gold))


def contains_answer(prediction: str, gold: str) -> int:
    return int(normalize(gold) in normalize(prediction))


def recall_at_k(retrieved_docs: list[dict], support_doc_ids: list[str]) -> int:
    retrieved_ids = {doc["id"] for doc in retrieved_docs}
    support_ids = set(support_doc_ids)
    return int(len(retrieved_ids & support_ids) > 0)


def all_support_recall_at_k(retrieved_docs: list[dict], support_doc_ids: list[str]) -> int:
    retrieved_ids = {doc["id"] for doc in retrieved_docs}
    support_ids = set(support_doc_ids)
    return int(support_ids.issubset(retrieved_ids))