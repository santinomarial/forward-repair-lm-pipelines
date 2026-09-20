# Data attribution

The repository's MIT license applies to its original code and documentation, not
to third-party benchmark content.

HotpotQA questions, reference answers, supporting-fact annotations, and Wikipedia
passages are from **Zhilin Yang et al., HotpotQA (EMNLP 2018)**. The
[official dataset card](https://huggingface.co/datasets/hotpotqa/hotpot_qa#licensing-information)
identifies the dataset license as
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
See the [project homepage](https://hotpotqa.github.io/) for the original dataset,
paper, authors, and citation information. Wikipedia passages remain attributable
to their original contributors; document titles are retained in the saved corpus.

This notice covers the HotpotQA-derived fields in `data/hotpot_*.jsonl` and saved
experiment artifacts under `outputs/`, including `outputs/natural_study/`.
Those dataset contents and adaptations retain CC BY-SA 4.0 rather than being
relicensed under MIT. New model responses and numeric evaluation results are
identified separately from benchmark reference fields.

Changes made here: selecting question subsets, joining passage sentences,
deduplicating documents by title, assigning local document IDs, and storing
reference fields beside generated model outputs. The natural-error study also
truncates passages to a fixed token ceiling; its corpus records a `truncated` flag.
The study manifest records the dataset fingerprint, input hashes, and split IDs.

No endorsement by the dataset authors or Wikipedia contributors is implied.
