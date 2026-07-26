"""Gold-free failure diagnostics and repair-routing policies."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
import re


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower())
        if token not in _STOP_WORDS
    }


def _overlap(source: set[str], target: set[str]) -> float:
    return len(source & target) / len(source) if source else 0.0


class RepairAction(str, Enum):
    ACCEPT = "accept"
    REPAIR_QUERY = "repair_query"
    REPAIR_ANSWER = "repair_answer"
    REPAIR_ITERATIVE = "repair_iterative"


@dataclass(frozen=True)
class FailureSignals:
    """Runtime evidence available without a gold answer or support labels."""

    retrieval_score_margin: float
    positive_score_fraction: float
    question_document_overlap: float
    query_document_overlap: float
    answer_context_overlap: float
    answer_in_context: bool
    answer_is_unknown: bool
    answer_is_binary: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


class FailureDetector(ABC):
    @abstractmethod
    def detect(
        self,
        *,
        question: str,
        query: str,
        docs: list[dict],
        answer: str,
    ) -> FailureSignals:
        """Extract gold-free diagnostic signals from one completed RAG run."""


class LexicalFailureDetector(FailureDetector):
    """Dependency-free diagnostics suitable for BM25 and dense retrieval."""

    def detect(
        self,
        *,
        question: str,
        query: str,
        docs: list[dict],
        answer: str,
    ) -> FailureSignals:
        document_text = " ".join(
            f"{doc.get('title', '')} {doc.get('text', '')}" for doc in docs
        )
        document_tokens = _tokens(document_text)
        question_tokens = _tokens(question)
        query_tokens = _tokens(query)
        answer_tokens = _tokens(answer)

        scores = [float(doc.get("score", 0.0)) for doc in docs]
        if len(scores) >= 2:
            denominator = abs(scores[0]) + abs(scores[1])
            margin = max(0.0, scores[0] - scores[1]) / (
                denominator if denominator > 0 else 1.0
            )
        else:
            margin = 0.0

        normalized_answer = " ".join(re.findall(r"\w+", answer.lower()))
        normalized_context = " ".join(re.findall(r"\w+", document_text.lower()))
        answer_is_unknown = normalized_answer in {
            "",
            "unknown",
            "insufficient information",
            "not enough information",
        }
        answer_is_binary = normalized_answer in {"yes", "no"}
        answer_in_context = bool(
            normalized_answer
            and not answer_is_unknown
            and not answer_is_binary
            and normalized_answer in normalized_context
        )

        return FailureSignals(
            retrieval_score_margin=margin,
            positive_score_fraction=(
                sum(score > 0 for score in scores) / len(scores) if scores else 0.0
            ),
            question_document_overlap=_overlap(question_tokens, document_tokens),
            query_document_overlap=_overlap(query_tokens, document_tokens),
            answer_context_overlap=_overlap(answer_tokens, document_tokens),
            answer_in_context=answer_in_context,
            answer_is_unknown=answer_is_unknown,
            answer_is_binary=answer_is_binary,
        )


class RepairPolicy(ABC):
    name = "abstract"

    @abstractmethod
    def decide(self, signals: FailureSignals) -> RepairAction:
        """Choose whether and where to repair one completed RAG run."""


class HeuristicRepairPolicy(RepairPolicy):
    """Transparent first policy for benchmarking future learned routers."""

    name = "heuristic_v1"

    def __init__(
        self,
        *,
        question_overlap_threshold: float = 0.25,
        severe_overlap_threshold: float = 0.10,
        score_margin_threshold: float = 0.10,
        answer_overlap_threshold: float = 0.50,
    ):
        self.question_overlap_threshold = question_overlap_threshold
        self.severe_overlap_threshold = severe_overlap_threshold
        self.score_margin_threshold = score_margin_threshold
        self.answer_overlap_threshold = answer_overlap_threshold

    def decide(self, signals: FailureSignals) -> RepairAction:
        weak_retrieval = (
            signals.question_document_overlap < self.question_overlap_threshold
            and (
                signals.retrieval_score_margin < self.score_margin_threshold
                or signals.positive_score_fraction < 0.5
            )
        )

        if weak_retrieval:
            if signals.question_document_overlap < self.severe_overlap_threshold:
                return RepairAction.REPAIR_ITERATIVE
            return RepairAction.REPAIR_QUERY

        if signals.answer_is_unknown:
            return RepairAction.REPAIR_ANSWER

        # A lexical detector cannot establish whether "yes" or "no" is entailed.
        # Defer semantic binary-answer handling to a future groundedness detector.
        if signals.answer_is_binary:
            return RepairAction.ACCEPT

        unsupported_answer = (
            not signals.answer_in_context
            and signals.answer_context_overlap < self.answer_overlap_threshold
        )
        if unsupported_answer:
            return RepairAction.REPAIR_ANSWER

        return RepairAction.ACCEPT
