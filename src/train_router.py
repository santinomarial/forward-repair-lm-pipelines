"""Train and evaluate a leakage-safe adaptive repair router."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from config import (
    FIGURE_ANSWER_PATH,
    FIGURE_ITERATIVE_PATH,
    ROOT_DIR,
    ROUTER_METRICS_PATH,
    ROUTER_MODEL_PATH,
)
from data_loader import load_jsonl
from routing import (
    FEATURE_NAMES,
    FailureSignals,
    LearnedRepairPolicy,
    LexicalFailureDetector,
    RepairAction,
)


STAGE_ACTIONS = [
    RepairAction.ACCEPT,
    RepairAction.REPAIR_QUERY,
    RepairAction.REPAIR_ANSWER,
]


def _artifact_source(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        display_path = str(resolved)

    digest = hashlib.sha256()
    with resolved.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": display_path, "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class RouterExample:
    question_id: str
    source: str
    signals: FailureSignals
    target: RepairAction


def _signals(detector: LexicalFailureDetector, row: dict, mode: str) -> FailureSignals:
    run = row[mode]
    return detector.detect(
        question=row["question"],
        query=run["query"],
        docs=run["docs"],
        answer=run["answer"],
    )


def build_training_examples(
    query_rows: list[dict],
    answer_rows: list[dict],
) -> list[RouterExample]:
    """Build stage labels from controlled interventions, without runtime gold features."""
    detector = LexicalFailureDetector()
    examples = []

    for row in query_rows:
        examples.append(
            RouterExample(
                question_id=str(row["id"]),
                source="baseline",
                signals=_signals(detector, row, "baseline"),
                target=RepairAction.ACCEPT,
            )
        )

        examples.append(
            RouterExample(
                question_id=str(row["id"]),
                source="query_corrupted",
                signals=_signals(detector, row, "corrupted"),
                target=RepairAction.REPAIR_QUERY,
            )
        )

    for row in answer_rows:
        examples.append(
            RouterExample(
                question_id=str(row["id"]),
                source="answer_corrupted",
                signals=_signals(detector, row, "corrupted"),
                target=RepairAction.REPAIR_ANSWER,
            )
        )

    return examples


def split_by_question(
    examples: list[RouterExample],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[RouterExample], list[RouterExample], list[str]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")

    question_ids = sorted({example.question_id for example in examples})
    random.Random(seed).shuffle(question_ids)
    test_count = max(1, round(len(question_ids) * test_fraction))
    test_ids = set(question_ids[:test_count])
    train = [example for example in examples if example.question_id not in test_ids]
    test = [example for example in examples if example.question_id in test_ids]
    return train, test, sorted(test_ids)


def _arrays(
    examples: list[RouterExample],
) -> tuple[np.ndarray, np.ndarray]:
    action_indices = {
        action: index for index, action in enumerate(STAGE_ACTIONS)
    }
    features = np.asarray(
        [example.signals.to_vector() for example in examples],
        dtype=np.float64,
    )
    targets = np.asarray(
        [action_indices[example.target] for example in examples],
        dtype=np.int64,
    )
    return features, targets


def fit_policy(
    examples: list[RouterExample],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    class_weight_power: float = 0.5,
) -> LearnedRepairPolicy:
    if not examples:
        raise ValueError("training examples must not be empty")
    if (
        epochs <= 0
        or learning_rate <= 0
        or l2 < 0
        or not 0 <= class_weight_power <= 1
    ):
        raise ValueError("invalid optimizer configuration")

    features, targets = _arrays(examples)
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-8] = 1.0
    normalized = (features - means) / scales

    class_counts = np.bincount(targets, minlength=len(STAGE_ACTIONS))
    if np.any(class_counts == 0):
        missing = [
            action.value
            for action, count in zip(STAGE_ACTIONS, class_counts, strict=True)
            if count == 0
        ]
        raise ValueError(f"training data is missing action classes: {missing}")

    sample_weights = np.asarray(
        [
            (
                len(targets)
                / (len(STAGE_ACTIONS) * class_counts[target])
            )
            ** class_weight_power
            for target in targets
        ]
    )
    sample_weights /= sample_weights.mean()

    coefficients = np.zeros(
        (len(STAGE_ACTIONS), len(FEATURE_NAMES)),
        dtype=np.float64,
    )
    intercepts = np.zeros(len(STAGE_ACTIONS), dtype=np.float64)
    one_hot = np.eye(len(STAGE_ACTIONS), dtype=np.float64)[targets]

    for _ in range(epochs):
        logits = normalized @ coefficients.T + intercepts
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        error = (probabilities - one_hot) * sample_weights[:, None]
        error /= sample_weights.sum()

        coefficient_gradient = error.T @ normalized + l2 * coefficients
        intercept_gradient = error.sum(axis=0)
        coefficients -= learning_rate * coefficient_gradient
        intercepts -= learning_rate * intercept_gradient

    return LearnedRepairPolicy(
        actions=STAGE_ACTIONS,
        coefficients=coefficients.tolist(),
        intercepts=intercepts.tolist(),
        feature_means=means.tolist(),
        feature_scales=scales.tolist(),
        name="softmax_stage_router_v1",
    )


def classification_metrics(
    policy: LearnedRepairPolicy,
    examples: list[RouterExample],
) -> dict:
    predictions = [policy.decide(example.signals) for example in examples]
    targets = [example.target for example in examples]
    confusion = {
        action.value: {
            predicted.value: 0
            for predicted in STAGE_ACTIONS
        }
        for action in STAGE_ACTIONS
    }
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target.value][prediction.value] += 1

    per_action = {}
    for action in STAGE_ACTIONS:
        true_positive = confusion[action.value][action.value]
        false_positive = sum(
            confusion[other.value][action.value]
            for other in STAGE_ACTIONS
            if other != action
        )
        false_negative = sum(
            confusion[action.value][other.value]
            for other in STAGE_ACTIONS
            if other != action
        )
        support = sum(confusion[action.value].values())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_action[action.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    correct = sum(target == prediction for target, prediction in zip(targets, predictions))
    accept_count = sum(
        prediction == RepairAction.ACCEPT
        for prediction in predictions
    )
    return {
        "examples": len(examples),
        "accuracy": correct / len(examples),
        "macro_f1": (
            sum(item["f1"] for item in per_action.values())
            / len(STAGE_ACTIONS)
        ),
        "predicted_repair_rate": 1 - accept_count / len(examples),
        "per_action": per_action,
        "confusion_matrix": confusion,
        "predicted_action_counts": dict(
            Counter(prediction.value for prediction in predictions)
        ),
    }


def repeated_holdout_metrics(
    examples: list[RouterExample],
    *,
    seeds: list[int],
    test_fraction: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    class_weight_power: float,
) -> dict:
    if not seeds:
        raise ValueError("at least one evaluation seed is required")

    runs = []
    for seed in seeds:
        train, test, _ = split_by_question(
            examples,
            test_fraction=test_fraction,
            seed=seed,
        )
        policy = fit_policy(
            train,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            class_weight_power=class_weight_power,
        )
        metrics = classification_metrics(policy, test)
        runs.append(
            {
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "predicted_repair_rate": metrics["predicted_repair_rate"],
            }
        )

    def aggregate(key: str) -> dict[str, float]:
        values = np.asarray([run[key] for run in runs], dtype=np.float64)
        return {"mean": float(values.mean()), "std": float(values.std())}

    return {
        "seeds": seeds,
        "accuracy": aggregate("accuracy"),
        "macro_f1": aggregate("macro_f1"),
        "predicted_repair_rate": aggregate("predicted_repair_rate"),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-results", type=Path, default=FIGURE_ITERATIVE_PATH)
    parser.add_argument("--answer-results", type=Path, default=FIGURE_ANSWER_PATH)
    parser.add_argument("--model-output", type=Path, default=ROUTER_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=ROUTER_METRICS_PATH)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Tempered inverse-frequency weighting from 0 (none) to 1 (full).",
    )
    parser.add_argument(
        "--evaluation-seeds",
        default="0,1,2,3,4",
        help="Comma-separated grouped-holdout seeds used for stability reporting.",
    )
    args = parser.parse_args()

    examples = build_training_examples(
        load_jsonl(args.query_results),
        load_jsonl(args.answer_results),
    )
    train, test, test_ids = split_by_question(
        examples,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    policy = fit_policy(
        train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        class_weight_power=args.class_weight_power,
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open("w", encoding="utf-8") as file:
        json.dump(
            {
                **policy.to_dict(),
                "training": {
                    "query_results": _artifact_source(args.query_results),
                    "answer_results": _artifact_source(args.answer_results),
                    "seed": args.seed,
                    "test_fraction": args.test_fraction,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "l2": args.l2,
                    "class_weight_power": args.class_weight_power,
                    "train_examples": len(train),
                    "test_examples": len(test),
                    "train_question_ids": len(
                        {example.question_id for example in train}
                    ),
                    "test_question_ids": len(test_ids),
                    "train_action_counts": dict(
                        Counter(example.target.value for example in train)
                    ),
                },
            },
            file,
            indent=2,
        )

    metrics = {
        "schema_version": 1,
        "policy": policy.name,
        "split": {
            "grouped_by": "question_id",
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "test_question_ids": test_ids,
        },
        "train": classification_metrics(policy, train),
        "test": classification_metrics(policy, test),
        "repeated_holdout": repeated_holdout_metrics(
            examples,
            seeds=[
                int(value)
                for value in args.evaluation_seeds.split(",")
                if value.strip()
            ],
            test_fraction=args.test_fraction,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            class_weight_power=args.class_weight_power,
        ),
    }
    with args.metrics_output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
