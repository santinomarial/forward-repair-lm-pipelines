import re
import string


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_em(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
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


def recovery_rate(
    rows: list[dict],
    corrupted_mode: str = "corrupted",
    repaired_mode: str = "repaired",
) -> dict[str, int | float]:
    """Measure repair success among examples broken by corruption.

    A row is eligible when the corrupted condition has exact match equal to zero.
    The recovery rate is the fraction of those rows whose repaired condition has
    exact match equal to one. An empty eligible set has rate ``0.0`` so summaries
    remain JSON-safe.
    """
    broken = [
        row
        for row in rows
        if row[corrupted_mode]["metrics"]["exact_match"] == 0
    ]
    fixed = [
        row
        for row in broken
        if row[repaired_mode]["metrics"]["exact_match"] == 1
    ]
    return {
        "corrupted_broken_count": len(broken),
        "repaired_fixed_count": len(fixed),
        "recovery_rate": len(fixed) / len(broken) if broken else 0.0,
    }
