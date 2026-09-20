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

Recovery and damage rates also receive descriptive Wilson binomial intervals,
which retain an upper uncertainty bound when there are zero observed events.
These supplement, rather than replace, the preregistered paired-bootstrap contrasts.

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

For the published research runtime, use Python 3.11+ and
`pip install -r requirements-dev.txt -c constraints-research.txt`. Collection used
CPython 3.14.0 on macOS. The constraints pin direct runtime dependencies, not every
transitive package or the remote model snapshot. Cached provider behavior and API
latency are not reproducible guarantees. DSPy version and collection source hashes
are checked before resuming or analyzing a frozen run.

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

## Pilot observations (development only)

The 60-question pilot used 480 calls, approximately $0.070 in completed-response
costs and $0.215 in conservative reservations. The sample size and action set were
not changed after inspecting it. Full pilot counts are in
[pilot_report.json](../outputs/natural_study/pilot_report.json).

An assistant-inspected illustration of metric sensitivity: in case
`5a7796e05542992a6e59df0f`, preservation changed “The Netherlands” to a full sentence
that still answered “the Netherlands,” producing an EM regression. In
`5abce3015542993a06baf995`, it changed “Approximately 85 nations.” to “85,” also
failing EM. These were the two pilot preservation regressions. The relevant
reference passages remained present. This is not an independent human review,
but it shows why **EM damage must not be equated with factual harm**.

## Frozen controllers

Development completed before held-out collection: 240 training and 60 validation
questions, 2,400 calls, $0.354752 estimated completed-response cost, and $1.082351
reserved. The validation-selected settings are saved in
[models.json](../outputs/natural_study/models.json):

| Controller | Ridge | Required predicted gain |
|:--|--:|--:|
| Natural, original actions | 0.1 | 0.05 |
| Natural, augmented actions | 0.1 | 0.05 |
| Natural, augmented + damage penalty | 100 | 0.00 |

The models and completed development outcomes were committed before the first
held-out request. The test lock also hashes the analysis code. No controller is
refit using test outcomes.

## Final findings

The held-out evaluation finished all 300 questions, with no extra tuning or sample
expansion. The full 600-question collection used 4,800 calls and 4,553,463 tokens;
completed-response cost was $0.70783845 and reservations were $2.158097625 under
the unchanged $5 cap. All 4,800 reservations correspond to completed calls.

1. **The primary selective policy improves natural-output EM.** It rises from
   88/300 (29.3%) to 103/300 (34.3%): +5.0 percentage points, paired 95% CI
   [2.3, 7.7]. It recovers 16/212 initially wrong outputs and regresses 1/88
   initially correct outputs. Token F1 rises by 6.54 points [3.79, 9.53].
2. **A strong fixed strategy remains competitive.** Always expanding context
   reaches 33.7% EM. The primary router's +0.7-point advantage has CI [−2.0, 3.3],
   so superiority is unestablished. Its recorded incremental cost is 40.3% lower
   ($0.000136925 versus $0.000229361/question), while mean recorded wall time is
   1.02s versus 0.99s. Lower token cost is not a demonstrated latency win.
3. **Preservation alone is not the breakthrough.** Under the shared-query
   top-10 control, both replacement and preservation reach 33.0% EM. Their paired
   difference is 0.0 points [−2.7, 2.7], with eight questions favoring each arm.
4. **Natural training alone is not enough.** With the original action set, the
   natural-trained policy reaches 30.0%, versus 32.3% for the frozen synthetic
   controller. Expanding the natural policy's action set yields an exploratory
   +3.7 points [1.3, 6.0]. This does not isolate the value of preservation from the
   additional top-10 replacement option; both were added together.
5. **A damage penalty is not a safety guarantee.** The unpenalized augmented
   policy recovers 13 errors with zero observed regressions; the damage-penalized
   policy recovers 16 with one regression. Their EM difference is inconclusive,
   +0.7 points [−0.7, 2.0]. Zero regressions among 88 correct outputs still has a
   Wilson upper 95% bound of 4.2%.
6. **The action set still limits recovery.** Only 34/212 initially wrong outputs
   become exact matches under any recorded action. A hindsight selector including
   accept reaches 122/300 (40.7%), not near-perfect accuracy. This is an action-set
   ceiling for these sampled generations, not a deployable oracle or universal
   limit on repair.

### Illustrative evidence audit

These examples were inspected by the assistant after the locked evaluation.
They are not independent human annotations or a semantic accuracy estimate.
All 17 primary-policy EM transitions remain pending in the exported review queue.

- **Missing evidence recovered:** `5a7c04c85542996dd594b881` asks about the tribe of
  the woman associated with Arizona State Route 51. Original evidence names Lori
  Piestewa but not her tribe. Preservation adds the Lori Piestewa passage, which
  identifies her as Hopi; the answer changes from UNKNOWN to Hopi.
- **Formatting counted as recovery:** `5a80ae105542992bc0c4a7a2` changes a sentence
  saying both people are writers and poets to the reference's singular “poet.”
  Both original biography passages already support the shared occupation. The
  exact-phrase diagnostic misses this plural/singular variation, so it cannot
  establish that unflagged gains are factual improvements.
- **Abstention counted as damage:** `5a89761d5542995153361310` changes Australian
  to UNKNOWN. The original five passages name the song's performers but do not
  establish the relevant nationality; the reference biography was not retrieved.
  A lost exact match is not automatically a newly false claim, and the initial
  correct string was not evidence-grounded by those passages.

The defensible conclusion is a **positive, bounded lexical reliability result**:
selective repair improves this held-out benchmark run and reduces observed repair
token cost relative to always expanding context. It does not establish factuality,
production safety, a benefit from preservation alone, or accuracy superiority over
the strongest fixed baseline. The planned v1 research phase is complete.
