import argparse
import json
from pathlib import Path

from datasets import load_dataset

from config import CORPUS_PATH, DATA_DIR, EXAMPLES_PATH


def safe_doc_id(title: str) -> str:
    return title.replace(" ", "_").replace("/", "_").lower()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    examples = []
    corpus = []
    seen_doc_ids = set()

    for item in dataset:
        if len(examples) >= args.n:
            break

        question = item["question"]
        answer = item["answer"]

        context_titles = item["context"]["title"]
        context_sentences = item["context"]["sentences"]

        support_titles = set(item["supporting_facts"]["title"])
        support_doc_ids = []

        for title, sentences in zip(context_titles, context_sentences):
            doc_id = safe_doc_id(title)

            if title in support_titles:
                support_doc_ids.append(doc_id)

            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                corpus.append(
                    {
                        "id": doc_id,
                        "title": title,
                        "text": " ".join(sentences),
                    }
                )

        if support_doc_ids:
            examples.append(
                {
                    "id": item["id"],
                    "question": question,
                    "answer": answer,
                    "support_doc_ids": support_doc_ids,
                }
            )

    write_jsonl(EXAMPLES_PATH, examples)
    write_jsonl(CORPUS_PATH, corpus)

    print(f"Wrote {len(examples)} examples to {EXAMPLES_PATH}")
    print(f"Wrote {len(corpus)} documents to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
