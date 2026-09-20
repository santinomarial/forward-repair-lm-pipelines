# Answer repair versus regeneration

## Question

Does showing a model its previous ungrounded answer reduce the effectiveness of
answer-stage repair? The existing study found low recovery, but did not separate
the effect of seeing the previous answer from the effect of revision instructions.

## Fixed protocol

- Use all 300 questions in the saved, re-normalized seed-0 answer-corruption run.
  Do not select only failed examples: repair can also damage correct answers.
- Freeze each question, query, ordered documents, context, and corrupted answer.
  No retrieval or corruption is rerun, and historical repaired answers are not
  reused as the live control condition.
- Run three contemporaneous conditions: existing revision, matched blind revision,
  and fresh generation. Blind revision retains the revision instructions and
  output description, removing only the `bad_answer` input field.
- Share the model, temperature 0, provider seed, and 300-token output limit.
  Use identical evidence guards. Realized prompt lengths differ because the
  previous answer is absent in two conditions; record those costs rather than
  padding prompts with potentially influential text.
- Disable DSPy's response cache. Seeded per-question randomization changes arm
  order to reduce systematic latency/order effects. Provider-side prompt caching,
  nondeterminism, and model updates can still occur.
- Neither gold answers nor support labels enter model prompts. Stored answer
  metrics are recomputed with the current normalization before analysis.

The primary endpoint is the paired exact-match difference **blind revision minus
revision**. A positive difference supports a detrimental effect of exposing the
previous answer in this setup; it does not establish a universal psychological
mechanism or show that all answer repair fails. Fresh generation versus revision
is secondary because both the instruction and previous-answer exposure change.

## Reporting

Report exact match, contains-answer, and explicit `UNKNOWN` abstention per arm.
Recovery is wrong-to-right transitions divided by initially wrong answers;
damage is right-to-wrong transitions divided by initially correct answers. Empty
denominators are undefined (`null`), not zero. Paired bootstrap confidence
intervals resample questions together; recovery comparisons use the same initially
wrong subset. Intervals are pointwise, with no multiple-comparison adjustment;
secondary analyses are exploratory.

Telemetry covers incremental answer generation only, excluding historical query,
retrieval, and corruption costs. Calls count entries in DSPy's LM history; a
formatting retry may make more than one call per condition. Missing provider cost
is not interpreted as free inference. Report priced-call coverage alongside cost.
Latency is environment-dependent; p95 from a 10-question smoke run is not a stable
benchmark.

The JSONL records evidence once per example, all generated answers, execution
order, usage, and a run configuration with input/prompt hashes and DSPy version.
The summary is reproducible without API access. Offline reports mark incomplete
runs with `complete: false`; do not present those as the full planned evaluation.

## Run

From the repository root, with the environment configured as in the README:

```bash
python src/answer_ablation.py --max-examples 10 --output-suffix answer_ablation_smoke
python src/answer_ablation.py --max-examples 300 --output-suffix answer_ablation_seed0
```

Existing result files are never overwritten. To resume, pass the same arguments
and add `--resume`. Only complete, matching records are reused; a partially
completed three-arm example may incur extra calls when retried. If a partial
JSONL write is detected, preserve it and start a new output suffix.

```bash
python src/answer_ablation.py --max-examples 300 --resume
python src/answer_ablation.py \
  --analyze-only outputs/answer_ablation_seed0_results.jsonl \
  --output-suffix answer_ablation_reanalysis --resamples 20000
```

For local inference, use `--llm ollama --model llama3.2:3b` and a separate output
suffix. This is a different model experiment, not a reproduction of OpenAI's
numbers. Change provider seeds with `--seed`; use separate suffixes and avoid
treating repeated questions across seeds as independent observations.

## Scope and limitations

This first evaluation concerns one synthetic corruption family and one dataset.
It measures recovery conditional on the saved evidence, not generalization to
natural production failures. The corruption prompt encouraged long answers;
exact match can therefore reflect response format as well as factual correctness.
Inspect contains-answer and individual examples before interpreting large EM
changes as factuality improvements. Historical and fresh runs can also differ
because model aliases change over time; compare the three new arms primarily
against one another.

The next extension is replication across seeds and natural failures, with
evidence-sufficiency strata. Do not train or tune a router on this evaluation set
and then report its performance on the same questions.
