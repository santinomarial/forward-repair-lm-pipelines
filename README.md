cat > README.md <<'EOF'
# Localized Forward Repair for Multi-Stage LM Pipelines

This project studies a controlled single-stage failure setting in a multi-stage LM pipeline. Instead of solving full failure attribution, we intentionally corrupt query generation, repair only that stage, and measure whether the fix propagates through retrieval and answer generation.

## Pipeline

Question → DSPy Query Generator → BM25 Retriever → DSPy Answer Generator

## Conditions

- Baseline: normal query generation
- Corrupted: vague query generation
- Repaired: repaired query generation, followed by rerunning retrieval and answer generation

## Dataset

The main experiment uses 50 HotpotQA validation examples.

## Final Results

| Condition | Exact Match | Contains Answer | Recall@K | All Support Recall@K |
|---|---:|---:|---:|---:|
| Baseline | 0.34 | 0.58 | 1.00 | 0.68 |
| Corrupted | 0.18 | 0.34 | 0.54 | 0.10 |
| Repaired | 0.32 | 0.56 | 0.98 | 0.56 |

## Recovery

- Exact-match recovery: 8 / 41 = 19.51%
- Retrieval recovery: 24 / 45 = 53.33%

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/build_hotpot_subset.py
python src/experiment.py
python src/analyze_results.py