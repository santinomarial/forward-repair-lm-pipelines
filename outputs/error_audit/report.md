# Saved-case error audit

Post-hoc descriptive signals, not human judgments or estimates of semantic error prevalence. Support annotations are audit-only, never router inputs. Review sample is intentionally nonrepresentative.

| Family | Cases | Recovered (EM) | Harmed (EM) | Correct answer kept without repair | Missed recovery |
|:--|--:|--:|--:|--:|--:|
| all | 300 | 57 | 4 | 34 | 15 |
| ignore_context | 60 | 21 | 1 | 0 | 3 |
| natural | 60 | 1 | 1 | 12 | 3 |
| query_term_substitution | 60 | 7 | 1 | 8 | 4 |
| retrieval_rank_dropout | 60 | 12 | 0 | 8 | 4 |
| vague_query | 60 | 16 | 1 | 6 | 1 |

Missed recovery means another saved action achieved EM=1 when the selected action did not. It is a hindsight diagnostic, not an available runtime oracle.

## Automatic signals

- after: EM failure with all annotated support: 103
- after: annotated support missing: 119
- after: gold phrase present despite EM failure: 56
- before: EM failure with all annotated support: 72
- before: annotated support missing: 195
- before: gold phrase present despite EM failure: 63

Of 57 EM recoveries, 21 already contained the whole normalized gold phrase before repair (excluding yes/no). This is a formatting-review candidate, not proof the original answer was correct.

## Human review

The queue contains 20 cases; **0 are human-reviewed**. All labels start empty. Damage cases are prioritized, then family/transition strata are sampled deterministically. This sample must not be used to estimate population error rates.

Use the explorer to inspect before/after evidence and export explicit reviewer annotations. Missing annotated support does not prove insufficient evidence; complete support does not prove correct reasoning. Unsupported claims and reasoning errors require evidence review, not substring rules.
