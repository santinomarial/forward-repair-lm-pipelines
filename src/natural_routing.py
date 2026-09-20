"""Versioned natural-error policies; historical synthetic routers stay unchanged."""

from dataclasses import asdict, dataclass

import numpy as np

from outcome_routing import ACTIONS, OUTCOME_FEATURES, runtime_features


NATURAL_ACTIONS = (*ACTIONS, "rewrite_query_10", "preserve_evidence")


@dataclass
class NaturalRouter:
    actions: list[str]
    means: list[float]
    scales: list[float]
    coefficients: list[list[float]]
    ridge: float
    min_gain: float
    damage_penalty: float
    cost_penalty: float = 100.0

    def utilities(self, features: list[float]) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.shape != (len(OUTCOME_FEATURES),) or not np.isfinite(x).all():
            raise ValueError("invalid inference features")
        return np.append((x - self.means) / self.scales, 1) @ np.asarray(self.coefficients)

    def decide(self, row: dict) -> str:
        values = self.utilities(runtime_features(row["question"], row["initial"]))
        choice = int(np.argmax(values))
        return self.actions[choice] if values[choice] > values[0] + self.min_gain else "accept"


def utility(row: dict, action: str, damage_penalty: float) -> float:
    after = row["outcomes"][action]
    em = after["metrics"]["exact_match"]
    damage = row["initial"]["metrics"]["exact_match"] * (1 - em)
    return float(em - 100 * after["telemetry"]["estimated_cost_usd"] - damage_penalty * damage)


def select_natural_router(train: list[dict], validation: list[dict], *,
                          actions: tuple[str, ...], damage_penalty: float) -> tuple[dict, dict]:
    if actions not in (ACTIONS, NATURAL_ACTIONS) or damage_penalty not in (0.0, 1.0):
        raise ValueError("use preregistered action sets and damage penalties")
    for rows, split in ((train, "train"), (validation, "validation")):
        if not rows or any(r["split"] != split or r["family"] != "natural" for r in rows):
            raise ValueError("natural development data only; held-out rows are forbidden")
        if len({r["question_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate development question")
    if {r["question_id"] for r in train} & {r["question_id"] for r in validation}:
        raise ValueError("training/validation overlap")
    x = np.asarray([runtime_features(r["question"], r["initial"]) for r in train])
    y = np.asarray([[utility(r, a, damage_penalty) for a in actions] for r in train])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite observations")
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    design = np.column_stack(((x - means) / scales, np.ones(len(train))))
    candidates, policies = [], []
    for ridge in (0.1, 1.0, 10.0, 100.0):
        regularizer = np.eye(design.shape[1]) * ridge
        regularizer[-1, -1] = 0
        coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        for min_gain in (0.0, 0.05, 0.1):
            policy = NaturalRouter(list(actions), means.tolist(), scales.tolist(),
                                   coefficients.tolist(), ridge, min_gain, damage_penalty)
            choices = [policy.decide(r) for r in validation]
            candidates.append({
                "ridge": ridge, "min_gain": min_gain,
                "utility": float(np.mean([utility(r, a, damage_penalty) for r, a in zip(validation, choices)])),
                "cost": float(np.mean([r["outcomes"][a]["telemetry"]["estimated_cost_usd"]
                                       for r, a in zip(validation, choices)])),
            })
            policies.append(policy)
    best = max(range(len(candidates)), key=lambda i: (
        candidates[i]["utility"], -candidates[i]["cost"], candidates[i]["min_gain"],
    ))
    return asdict(policies[best]), {"selected_index": best, "candidates": candidates}
