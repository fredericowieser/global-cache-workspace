# KOR-Bench attribution

The scoring logic in `src/gcw/evaluation/korbench_scoring.py` is adapted from
KOR-Bench's `eval/eval_utils.py`, revision
`bb8194dff8e51e2e2247cf62aa9932c3d57d53d7`:
https://github.com/KOR-Bench/KOR-Bench

Licensed under Apache-2.0; see KOR-Bench-LICENSE. GCW changes include restricted
arithmetic evaluation, explicit malformed-answer rejection, and refactoring the
single-question grading branches. Mixed and multi-round protocols are excluded.
Dataset preparation downloads and verifies the original zero-shot templates and
data from the same revision; prompts retain their original wording.

Regression fixtures in `tests/fixtures/korbench/scoring.json` contain one released
Qwen2.5-7B-Instruct response per rule from the same repository revision. Expected
labels were recomputed using the pinned scorer and data, rather than copied from
historical result labels. These fixtures retain the same Apache-2.0 attribution.
