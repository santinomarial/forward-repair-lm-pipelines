# Localized Forward Repair for Multi-Stage LM Pipelines

This project studies localized failure injection and repair in a multi-stage LM pipeline built on HotpotQA. We intentionally corrupt a single pipeline stage (query generation or answer generation), repair only that stage, and measure whether the fix propagates through retrieval and final answer quality.

## Pipeline

```
Question → DSPy Query Generator → BM25 Retriever → DSPy Answer Generator
```

## Conditions

| Condition | Description |
|---|---|
| Baseline | Normal deterministic query generation (temp=0) |
| Corrupted (query) | Vague/underspecified query (temp=0.7, seeded) |
| Repaired single-shot | One corrected query, re-retrieves and re-answers |
| Repaired iterative | Two targeted sub-queries merged via `retrieve_union` |
| Corrupted (answer) | Good retrieval, corrupted answer generation |
| Repaired (answer) | Same retrieval, repaired answer generation |

## Dataset

300 HotpotQA validation examples; 2964-document BM25 corpus. Examples are stratified post-hoc into:
- **Single-hop-sufficient** — gold answer appears in exactly one support document
- **Genuinely-multi-hop** — yes/no comparison questions or gold answer requires both documents
- **Answer-not-in-support** — excluded from stratified analysis

## Main Results (n=300, 3 seeds, mean ± std)

| Condition | Exact Match | Contains Answer | Recall@K | AllSupport@K | Recovery Rate |
|---|---|---|---|---|---|
| Baseline | 0.317 | 0.477 | 0.957 | 0.520 | — |
| Corrupted (query) | 0.113 ± 0.003 | 0.201 ± 0.012 | 0.367 ± 0.025 | 0.058 ± 0.010 | — |
| Repaired single-shot | 0.300 ± 0.006 | 0.464 ± 0.010 | 0.937 ± 0.003 | 0.483 ± 0.015 | 24.6% ± 0.4% |
| Repaired iterative | 0.333 | 0.477 | 0.967 | 0.533 | 27.3% |
| Corrupted (answer) | 0.077 | 0.650 | 0.957 | 0.520 | — |
| Repaired (answer) | 0.070 | 0.467 | 0.957 | 0.520 | 2.2% |

## Key Findings

**1. Query repair works; answer repair does not.**
Single-shot query repair recovers 24.6% of corrupted-broken examples. Answer-stage repair recovers only 2.2% — the LM anchors on its confident hallucination rather than re-reasoning from context.

**2. Iterative repair helps most on genuinely-multi-hop questions.**
Issuing two targeted sub-queries and merging results pushes multi-hop EM from 0.494 (single-shot) to 0.610, exceeding the baseline (0.571). Yes/No comparison questions fully recover (0.762 = baseline exactly). Single-hop shows no gain (0.238 = identical).

**3. Partial-to-full retrieval lift.**
Of 140 examples where single-shot repair retrieved one support doc but missed the other (Recall@K=1, AllSupport@K=0), iterative repair completes retrieval in 44 cases (31.4%); 23 of those also gain EM (52.3%).

## Stratified Results (query corruption, seed 0)

| Stratum | n | Baseline EM | Corrupted EM | Repaired EM | Iterative EM |
|---|---|---|---|---|---|
| Single-hop | 223 | 0.229 | 0.094 | 0.238 | 0.238 |
| Multi-hop | 77 | 0.571 | 0.156 | 0.494 | **0.610** |
| Yes/No | 21 | 0.714 | 0.000 | 0.476 | **0.762** |
| Bridge | 56 | 0.518 | 0.214 | 0.500 | 0.554 |

## Project Structure

```
src/
  build_hotpot_subset.py   # Build HotpotQA corpus + example subsets
  experiment.py            # Main experiment runner (--seed, --corrupt-stage, --include-iterative)
  pipeline.py              # ForwardRepairPipeline (baseline / corrupted / repaired / iterative)
  dspy_modules.py          # DSPy signatures and modules
  retriever.py             # BM25Retriever with retrieve_union for iterative repair
  metrics.py               # HotpotQA-style EM normalization + retrieval metrics
  rescore.py               # Re-score stored results without re-running LMs
  stratified_analysis.py   # Hop-stratified analysis, single- and multi-seed
  make_final_figures.py    # Generate all figures and tables for the report
  compare_stages.py        # Compare query vs answer corruption stages
  aggregate_seeds.py       # Aggregate multi-seed results
data/
  hotpot_corpus.jsonl
  hotpot_examples.jsonl
outputs/
  figures/                 # main_results.png, asymmetry_*.png, hop_*.png, ...
  tables/                  # main_table.md, stratified_table.md, ...
  hotpot_300_seed{0,1,2}_results_renormalized.jsonl
  hotpot_300_iterative_seed0_results_renormalized.jsonl
  hotpot_300_answer_corruption_seed0_results_renormalized.jsonl
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
```

## Running Experiments

```bash
# Build dataset (n=300)
python src/build_hotpot_subset.py --n 300

# Run query corruption experiment (3 seeds)
python src/experiment.py --seed 0 --output-suffix hotpot_300_seed0
python src/experiment.py --seed 1 --output-suffix hotpot_300_seed1
python src/experiment.py --seed 2 --output-suffix hotpot_300_seed2

# Run iterative repair experiment
python src/experiment.py --seed 0 --output-suffix hotpot_300_iterative_seed0 --include-iterative

# Run answer corruption experiment
python src/experiment.py --seed 0 --output-suffix hotpot_300_answer_corruption_seed0 --corrupt-stage answer

# Re-score with updated EM normalization (no LM calls)
python src/rescore.py --input outputs/hotpot_300_seed0_results.jsonl

# Stratified analysis (single seed)
python src/stratified_analysis.py --input outputs/hotpot_300_iterative_seed0_results_renormalized.jsonl

# Stratified analysis (multi-seed, mean ± std)
python src/stratified_analysis.py --inputs outputs/hotpot_300_seed0_results_renormalized.jsonl,outputs/hotpot_300_seed1_results_renormalized.jsonl,outputs/hotpot_300_seed2_results_renormalized.jsonl

# Generate all figures and tables
python src/make_final_figures.py
```

## Metrics

- **Exact Match (EM)** — HotpotQA-style: lowercase, strip punctuation, remove articles, collapse whitespace; yes/no answers matched on first word only
- **Contains Answer** — normalized gold is a substring of normalized prediction
- **Recall@K** — at least one support document appears in the top-K retrieved
- **AllSupport@K** — all support documents appear in the top-K retrieved
- **Recovery Rate** — fraction of corrupted-broken examples where repair restores EM=1
