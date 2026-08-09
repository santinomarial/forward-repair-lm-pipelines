"""Gold-free failure diagnostics and repair-routing policies."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
import re


FEATURE_NAMES = (
    "retrieval_score_margin",
    "positive_score_fraction",
    "question_document_overlap",
    "query_document_overlap",
    "answer_context_overlap",
    "answer_in_context",
    "answer_is_unknown",
    "answer_is_binary",
)


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

    def to_vector(self) -> list[float]:
        values = self.to_dict()
        return [float(values[name]) for name in FEATURE_NAMES]


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

    def decision_metadata(self, signals: FailureSignals) -> dict:
        """Return optional policy-specific evidence for audit logs."""
        return {}


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


class LearnedRepairPolicy(RepairPolicy):
    """Serialized multiclass linear policy trained on controlled failures."""

    def __init__(
        self,
        *,
        actions: list[RepairAction],
        coefficients: list[list[float]],
        intercepts: list[float],
        feature_means: list[float],
        feature_scales: list[float],
        name: str = "softmax_v1",
    ):
        feature_count = len(FEATURE_NAMES)
        if not actions:
            raise ValueError("actions must not be empty")
        if len(coefficients) != len(actions) or len(intercepts) != len(actions):
            raise ValueError("one coefficient row and intercept are required per action")
        if any(len(row) != feature_count for row in coefficients):
            raise ValueError(f"each coefficient row must have {feature_count} values")
        if len(feature_means) != feature_count or len(feature_scales) != feature_count:
            raise ValueError(f"feature statistics must have {feature_count} values")
        if any(scale <= 0 for scale in feature_scales):
            raise ValueError("feature scales must be positive")

        self.actions = actions
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.feature_means = feature_means
        self.feature_scales = feature_scales
        self.name = name

    def _scores(self, signals: FailureSignals) -> list[float]:
        features = [
            (value - mean) / scale
            for value, mean, scale in zip(
                signals.to_vector(),
                self.feature_means,
                self.feature_scales,
                strict=True,
            )
        ]
        return [
            intercept
            + sum(
                coefficient * feature
                for coefficient, feature in zip(row, features, strict=True)
            )
            for row, intercept in zip(
                self.coefficients,
                self.intercepts,
                strict=True,
            )
        ]

    def predict_proba(self, signals: FailureSignals) -> dict[str, float]:
        scores = self._scores(signals)
        maximum = max(scores)
        exponentials = [math.exp(score - maximum) for score in scores]
        denominator = sum(exponentials)
        return {
            action.value: value / denominator
            for action, value in zip(self.actions, exponentials, strict=True)
        }

    def decide(self, signals: FailureSignals) -> RepairAction:
        scores = self._scores(signals)
        best_index = max(range(len(scores)), key=scores.__getitem__)
        return self.actions[best_index]

    def decision_metadata(self, signals: FailureSignals) -> dict:
        probabilities = self.predict_proba(signals)
        return {
            "probabilities": probabilities,
            "confidence": max(probabilities.values()),
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "policy_type": "multiclass_linear",
            "name": self.name,
            "feature_names": list(FEATURE_NAMES),
            "actions": [action.value for action in self.actions],
            "coefficients": self.coefficients,
            "intercepts": self.intercepts,
            "feature_means": self.feature_means,
            "feature_scales": self.feature_scales,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LearnedRepairPolicy":
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported router schema: {payload.get('schema_version')!r}"
            )
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Router feature schema does not match this code version")
        return cls(
            actions=[RepairAction(value) for value in payload["actions"]],
            coefficients=[
                [float(value) for value in row]
                for row in payload["coefficients"]
            ],
            intercepts=[float(value) for value in payload["intercepts"]],
            feature_means=[float(value) for value in payload["feature_means"]],
            feature_scales=[float(value) for value in payload["feature_scales"]],
            name=str(payload.get("name", "softmax_v1")),
        )

    @classmethod
    def load(cls, path: Path) -> "LearnedRepairPolicy":
        with path.open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))
