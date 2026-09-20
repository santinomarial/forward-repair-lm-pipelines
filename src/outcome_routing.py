"""Cost-sensitive repair policy learned from full observed action outcomes."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from routing import FEATURE_NAMES, LexicalFailureDetector


ACTIONS = ("accept", "rewrite_query", "expand_context", "regenerate")
OUTCOME_FEATURES = (*FEATURE_NAMES, "answer_words", "query_words", "document_count")
SEEN_FAMILIES = ("vague_query", "ignore_context")
UNSEEN_FAMILIES = ("query_term_substitution", "retrieval_rank_dropout")


def runtime_features(question: str, initial: dict) -> list[float]:
    """Explicit whitelist: no gold, family, source ID, or action outcomes."""
    signals = LexicalFailureDetector().detect(
        question=question, query=initial["query"],
        docs=initial["docs"], answer=initial["answer"],
    )
    return signals.to_vector() + [
        float(len(initial["answer"].split())),
        float(len(initial["query"].split())),
        float(len(initial["docs"])),
    ]


@dataclass
class OutcomeRouter:
    means: list[float]
    scales: list[float]
    coefficients: list[list[float]]
    cost_penalty: float
    ridge: float
    min_gain: float = 0.0

    def utilities(self, features: list[float]) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.shape != (len(OUTCOME_FEATURES),) or not np.isfinite(x).all():
            raise ValueError("invalid runtime features")
        x = (x - self.means) / self.scales
        return np.append(x, 1.0) @ np.asarray(self.coefficients)

    def decide_features(self, features: list[float]) -> str:
        utilities = self.utilities(features)
        choice = int(np.argmax(utilities))
        if utilities[choice] <= utilities[0] + self.min_gain:
            return "accept"
        return ACTIONS[choice]

    def decide(self, *, question: str, initial: dict) -> str:
        return self.decide_features(runtime_features(question, initial))

    def save(self, path: Path, metadata: dict) -> None:
        payload = {
            "schema_version": 1, "actions": list(ACTIONS),
            "features": list(OUTCOME_FEATURES), "policy": asdict(self), "metadata": metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "OutcomeRouter":
        payload = json.loads(path.read_text())
        if (payload.get("schema_version") != 1 or payload.get("actions") != list(ACTIONS)
                or payload.get("features") != list(OUTCOME_FEATURES)):
            raise ValueError("incompatible outcome-router artifact")
        policy = cls(**payload["policy"])
        if (len(policy.means) != len(OUTCOME_FEATURES)
                or len(policy.scales) != len(OUTCOME_FEATURES)
                or np.asarray(policy.coefficients).shape != (len(OUTCOME_FEATURES) + 1, len(ACTIONS))
                or not np.isfinite(policy.means + policy.scales).all()
                or any(scale <= 0 for scale in policy.scales)
                or not np.isfinite(policy.coefficients).all()
                or min(policy.cost_penalty, policy.min_gain) < 0 or policy.ridge <= 0):
            raise ValueError("invalid outcome-router parameters")
        return policy


def observed_utility(row: dict, action: str, cost_penalty: float) -> float:
    outcome = row["outcomes"][action]
    return (float(outcome["metrics"]["exact_match"])
            - cost_penalty * float(outcome["telemetry"]["estimated_cost_usd"]))


def fit_outcome_router(rows: list[dict], *, ridge: float, cost_penalty: float) -> OutcomeRouter:
    if not rows or ridge <= 0 or cost_penalty < 0:
        raise ValueError("invalid training data or hyperparameters")
    if any(row["split"] != "train" or row["family"] not in SEEN_FAMILIES for row in rows):
        raise ValueError("fit accepts only training questions and seen families")
    x = np.asarray([runtime_features(row["question"], row["initial"]) for row in rows])
    y = np.asarray([[observed_utility(row, action, cost_penalty) for action in ACTIONS] for row in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite training observations")
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    design = np.column_stack(((x - means) / scales, np.ones(len(rows))))
    regularizer = np.eye(design.shape[1]) * ridge
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return OutcomeRouter(means.tolist(), scales.tolist(), coefficients.tolist(), cost_penalty, ridge)


def select_router(train: list[dict], validation: list[dict], *, cost_penalty: float = 100.0):
    """Fixed validation grid; no refitting or selection on held-out outcomes."""
    if not validation or any(
        row["split"] != "validation" or row["family"] not in SEEN_FAMILIES for row in validation
    ):
        raise ValueError("selection requires seen-family validation cases")
    if {r["question_id"] for r in train} & {r["question_id"] for r in validation}:
        raise ValueError("training and validation question IDs overlap")
    candidates = []
    policies = []
    for ridge in (0.1, 1.0, 10.0, 100.0):
        fitted = fit_outcome_router(train, ridge=ridge, cost_penalty=cost_penalty)
        for min_gain in (0.0, 0.05, 0.10):
            policy = OutcomeRouter(**{**asdict(fitted), "min_gain": min_gain})
            choices = [policy.decide(question=r["question"], initial=r["initial"]) for r in validation]
            utility = float(np.mean([
                observed_utility(r, a, cost_penalty) for r, a in zip(validation, choices)
            ]))
            cost = float(np.mean([
                r["outcomes"][a]["telemetry"]["estimated_cost_usd"]
                for r, a in zip(validation, choices)
            ]))
            candidates.append({"ridge": ridge, "min_gain": min_gain, "utility": utility, "cost": cost})
            policies.append(policy)
    best = max(range(len(candidates)), key=lambda i: (
        candidates[i]["utility"], -candidates[i]["cost"], candidates[i]["min_gain"],
    ))
    return policies[best], {"selected_index": best, "candidates": candidates}
