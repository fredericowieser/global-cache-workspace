# Reasoning Gym truncation audit

All 700,032 train rows, authenticated against the source manifest. Token counts include EOS; standard deviations are population values conditional on truncation. Verified 42 task/cap/outcome strata with the pinned tokenizer and actual `sft_batch`.

Reproduce the training audit: `python scripts/audit_reasoning_gym.py`. Use `--split validation` or `--split test` with a separate `--out-dir` for holdouts. This reads the prepared data without changing it.

| Cap | Truncated rows | Original tokens, mean +/- std | Removed tokens, mean +/- std | Prompt-only rows | Answer-token mass lost |
|---:|---:|---:|---:|---:|---:|
| 512 | 361,200 (51.598%) | 796.3 +/- 278.1 | 284.3 +/- 278.1 | 0 | 38.177% |
| 1024 | 60,725 (8.675%) | 1283.4 +/- 265.1 | 259.4 +/- 265.1 | 0 | 5.855% |
| 1536 | 8,589 (1.227%) | 1809.5 +/- 260.3 | 273.5 +/- 260.3 | 0 | 0.873% |
| 2048 | 1,204 (0.172%) | 2303.6 +/- 283.0 | 255.6 +/- 283.0 | 0 | 0.114% |
| 4096 | 0 (0.000%) | N/A | N/A | 0 | 0.000% |

| Task | At 512 | At 1024 | At 1536 | At 2048 | At 4096 |
|---|---:|---:|---:|---:|---:|
| ab | 99.771% | 33.700% | 3.296% | 0.433% | 0.000% |
| base_conversion | 45.335% | 0.000% | 0.000% | 0.000% | 0.000% |
| color_cube_rotation | 28.737% | 0.000% | 0.000% | 0.000% | 0.000% |
| gcd | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| manipulate_matrix | 41.849% | 0.047% | 0.000% | 0.000% | 0.000% |
| string_manipulation | 93.894% | 18.300% | 4.066% | 0.599% | 0.000% |

[Detailed statistics](report.json) include retained targets, EOS-only truncations, and per-task results. [Distribution counts](quality.json) include matrix degeneracy, GCD labels, base-conversion bases, and the cube initial-colour baseline.

Source SHA-256: `3dbf7c3b5364f144214a5a665f8c5f1143a3105e742c679a67868e2f08fed7a2`.
