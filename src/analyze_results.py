import json
from pathlib import Path


RESULTS_PATH = Path("outputs/hotpot_50_final_results.jsonl")
SUMMARY_PATH = Path("outputs/hotpot_50_final_summary.json")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    rows = load_rows(RESULTS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text())

    print("\n=== Quantitative Summary ===")
    for mode in ["baseline", "corrupted", "repaired"]:
        metrics = summary[mode]
        print(
            f"{mode:10s} | "
            f"EM={metrics['exact_match']:.2f} | "
            f"Contains={metrics['contains_answer']:.2f} | "
            f"Recall@K={metrics['recall_at_k']:.2f} | "
            f"AllSupport@K={metrics['all_support_recall_at_k']:.2f}"
        )

    recovery = summary["recovery"]
    print("\n=== Exact-Match Recovery ===")
    print(f"Corrupted broken: {recovery['corrupted_broken_count']}")
    print(f"Repaired fixed:   {recovery['repaired_fixed_count']}")
    print(f"Recovery rate:    {recovery['recovery_rate']:.2%}")

    retrieval_recovered = 0
    retrieval_broken = 0

    for row in rows:
        corrupted_all = row["corrupted"]["metrics"]["all_support_recall_at_k"] == 1
        repaired_all = row["repaired"]["metrics"]["all_support_recall_at_k"] == 1

        if not corrupted_all:
            retrieval_broken += 1
            if repaired_all:
                retrieval_recovered += 1

    print("\n=== Retrieval Recovery ===")
    print(f"Corrupted missing all support: {retrieval_broken}")
    print(f"Repaired recovered all support: {retrieval_recovered}")
    print(
        "Retrieval recovery rate: "
        f"{retrieval_recovered / retrieval_broken:.2%}"
        if retrieval_broken
        else "Retrieval recovery rate: N/A"
    )

    print("\n=== Clean Repaired Examples ===")
    shown = 0
    for row in rows:
        corrupted_em = row["corrupted"]["metrics"]["exact_match"]
        repaired_em = row["repaired"]["metrics"]["exact_match"]

        if corrupted_em == 0 and repaired_em == 1:
            print("\n" + "-" * 80)
            print("Question:", row["question"])
            print("Gold:", row["gold"])
            print("Corrupted query:", row["corrupted"]["query"])
            print("Corrupted answer:", row["corrupted"]["answer"])
            print("Repaired query:", row["repaired"]["query"])
            print("Repaired answer:", row["repaired"]["answer"])

            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()