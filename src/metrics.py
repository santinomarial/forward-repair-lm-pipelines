import re
import string


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_em(text: str) -> str:
    text = text.lower()
    text = text.strip(string.punctuation + " ")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> int:
    norm_gold = _normalize_em(gold)
    norm_pred = _normalize_em(prediction)
    if norm_gold in ("yes", "no"):
        words = norm_pred.split()
        first = words[0].strip(string.punctuation) if words else ""
        if first in ("yes", "no"):
            norm_pred = first
    return int(norm_pred == norm_gold)


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