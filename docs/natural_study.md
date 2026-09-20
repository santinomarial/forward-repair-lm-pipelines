# Natural errors: final v1 study

This is a bounded follow-up, not an attempt to tune until repair wins. The earlier
synthetic-trained router had no net gain on 60 unmodified outputs. This study asks
whether natural-outcome training and evidence-preserving repair change that result.

## Locked design

- 600 fresh HotpotQA distractor validation questions, seed 41. Exclude all 300
  historical IDs and duplicate question strings. Split once: 240 train, 60
  validation, 300 test. The first 60 training questions form the pilot; they are
  not extra observations and never enter the test set.
- BM25 over a new pooled corpus assembled from the selected questions' distractor
  passages. Corpus text is available to retrieval, but questions, gold answers,
  and support labels never train retrieval. Documents are bounded to 256 serialized
  tokens; contexts to 3,000 tokens. This corpus differs from the historical one;
  absolute EM is not directly comparable to the historical baseline.
- Existing DSPy query/answer prompts, gpt-4o-mini, temperature 0, provider seed 0,
  output limit 300, caching off. No injected corruption, gold-selected subset,
  answer-confidence oracle, or test-time fitting.
- All questions retain their initial outputs, including correct answers, to
  measure both recovery and damage.

## Action comparisons

| Action | Evidence | Incremental model calls |
|:--|:--|--:|
| Accept | Original answer | 0 |
| Fresh answer | Original top 5 | 1 |
| Rewrite query | Rewritten-query top 5 | 2 |
| Expand context | Original-query top 10 | 1 |
| Rewrite query, 10 docs | Rewritten-query top 10 | 2 |
| Preserve evidence | Original top 5, then new rewritten-query docs up to 10 | 2 |

The three rewrite arms share exactly one query generation. Thus the top-10
replacement versus preservation contrast changes evidence construction, not the
query. Deduplication uses exact-title IDs; original documents retain their order,
then new documents follow rewritten-query BM25 rank. Equal document/token ceilings
are not exact realized token matching. Every arm uses the same answer prompt and
per-document truncation. An original document is never silently dropped by the
preserving arm. Actual collection normally takes eight calls per question;
selected-policy telemetry charges the shared query to each selected rewrite arm.

## Controllers and selection

Keep the historical synthetic-trained four-action controller frozen. Train three
new ridge controllers: natural data/original four actions, natural data/six
actions, and natural data/six actions with an explicit damage penalty. The targets
are EM − 100 × incremental USD, with an additional penalty of 1 for right→wrong
transitions in the damage-aware policy. This penalty is fixed, not selected on test.

Use the existing 11 runtime-only features. Gold answers, correctness, support
annotations, family, and IDs are available only to offline targets/evaluation.
Normalize on training rows only. Validation selects ridge {0.1, 1, 10, 100} and
minimum predicted gain {0, 0.05, 0.1}; utility ties prefer lower cost, then higher
threshold. No refit after selection. Comparing natural_original to synthetic_original
keeps the action set fixed; comparing augmented controllers changes that action set.

## Evaluation and stopping

The sole primary comparison is damage-aware natural six-action policy versus
accept, on held-out EM. Report recovery among initially wrong outputs, damage
among initially correct outputs, abstentions, token F1, cost, calls, and stage
latency. Use 10,000 paired question bootstrap resamples, seed 41, 95% percentile
intervals. Other contrasts are exploratory, with pointwise rather than
multiplicity-adjusted intervals. A small sample or inconclusive interval is not
evidence of equivalence or safety.

Inspect the pilot for collection validity, expense, and approximate paired
precision; keep the declared 300-question holdout regardless of observed gains.
Lock models and analysis hashes before any held-out generation. Never expand the
sample based on significance or retrain after opening the test set. Finish 600
questions once, unless the cap or provider failure stops collection.

The new study has one immutable **$5 conservative reservation cap**, shared across
pilot, development, holdout, retries, and resumes. This is a local request guard,
not an account-wide provider billing limit. Failed/discarded calls retain their
reservations. Actual completed-response costs and reservations are reported
separately. Older studies' ledgers and caps are not changed.

Audit all primary-policy EM transitions with the existing [semantic rubric](error_audit.md).
Export a review queue, clearly mark human review pending, and never treat gold
phrase containment as factual correctness. This is natural benchmark error repair,
not production generalization. No additional dataset, backend, or tuning phase is
part of the v1 finish line.

## Reproduce

```bash
python src/natural_study.py prepare --directory outputs/my_natural_study
python src/natural_study.py collect --directory outputs/my_natural_study --phase pilot
python src/natural_study.py pilot-report --directory outputs/my_natural_study
python src/natural_study.py collect --directory outputs/my_natural_study --phase development --resume
python src/natural_study.py fit --directory outputs/my_natural_study
python src/natural_study.py collect --directory outputs/my_natural_study --phase test
python src/natural_study.py report --directory outputs/my_natural_study
```

Collection requires an API key; preparation downloads HotpotQA if it is not cached.
Saved data and report generation need no API calls. Use a new directory for a new
study. `--resume` never refunds reservations or overwrites completed observations.
The generated review queue is a template; save human annotations in a different
file because report regeneration replaces derived outputs.
