# Saved-case audit and review protocol

This is a post-hoc audit of the **300 held-out study states**, not another
evaluation or a new test set. It uses the frozen policy and saved generations;
no LLM calls, retraining, or paid inference are needed. Do not tune on these
observations and then describe the same cases as an untouched test set.

## Reproduce

```bash
python src/error_audit.py
```

The command regenerates `outputs/error_audit/report.md`, `summary.json`, and
`review_queue.jsonl`. The queue is a generated template: save completed reviews
in a **separate file**. The CLI does not read or overwrite reviewer-owned files.
Input hashes cover the model, manifest, outcomes, corpus, and benchmark
annotations. Replay verifies the frozen model, recomputes runtime features and
metrics, and verifies document order and context hashes for every action.

## What the automatic audit establishes

- **57 EM recoveries and 4 EM regressions** under the frozen policy.
- **21 of the 57 recoveries** already contained the complete normalized gold
  phrase before repair, excluding yes/no. These warrant checking whether the
  apparent improvement is formatting rather than factual correction.
- **15 missed recoveries:** a different saved action achieved exact match while
  the chosen action did not. This is a hindsight diagnostic, not a runtime oracle.
- **34 initially correct outputs were kept without repair.**

Flags are deliberately narrow. Gold-phrase presence is not correctness: a
negated or contradictory response can contain the reference answer. Missing
annotated support does not rule out sufficient alternative evidence. Having all
annotated support does not prove the model used it correctly. Gold answers and
support IDs are used only for auditing, never for routing decisions.

## Review sample

The deterministic 20-case queue includes all four damaged cases first. Remaining
slots are filled round-robin across family/EM-transition strata, with cases
ordered by SHA256 of the case ID. This prioritizes coverage and harmful changes;
**it is not a representative sample for estimating error prevalence**.

No cases are claimed to be human-reviewed yet. Labels begin empty. A reviewer
must inspect the question, before/after answers, the reference, and the actual
evidence received by each action, then annotate both answers separately:

| Label | Review criterion |
|:--|:--|
| `format_only` | Semantically answers the question correctly; EM fails because of wording, aliases, or extra explanation. |
| `missing_evidence` | The supplied evidence lacks a necessary fact; identify what is missing. |
| `reasoning_error` | Sufficient evidence is present, but the response draws an incorrect conclusion; cite the relevant passages. |
| `unsupported_claim` | A material claim is not supported by the supplied evidence; it may still be true outside that evidence. |
| `reference_ambiguity` | The reference, question, or accepted-answer scope is ambiguous; explain rather than forcing a verdict. |
| `no_error` | The response answers the question and is supported by the supplied evidence. |
| `uncertain` | The reviewer cannot establish the cause or correctness from the available record. |

Multiple labels are allowed, but avoid contradictory labels such as `no_error`
and `reasoning_error` for the same answer. Record a reviewer identifier, evidence
document IDs, and a rationale. Reference-support documents absent from a run may
explain missing information but must not be treated as evidence the model saw.
The explorer exports reviews locally; it never rewrites published outcomes or
metrics. Session annotations are lost on reload unless downloaded.

For a defensible semantic result, have a second person independently review the
sample and resolve disagreements. Report reviewer counts and disagreements;
do not present this automated shortlist or an assistant's annotations as an
independent human assessment.
