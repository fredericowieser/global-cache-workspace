# Reasoning Gym truncation audit

All 6,000 validation rows, authenticated against the source manifest. Token counts include EOS; standard deviations are population values conditional on truncation. Verified 42 task/cap/outcome strata with the pinned tokenizer and actual `sft_batch`.

Reproduce the training audit: `python scripts/audit_reasoning_gym.py`. Use `--split validation` or `--split test` with a separate `--out-dir` for holdouts. This reads the prepared data without changing it.

| Cap | Truncated rows | Original tokens, mean +/- std | Removed tokens, mean +/- std | Prompt-only rows | Answer-token mass lost |
|---:|---:|---:|---:|---:|---:|
| 512 | 3,571 (59.517%) | 847.0 +/- 256.1 | 335.0 +/- 256.1 | 0 | 51.269% |
| 1024 | 548 (9.133%) | 1280.3 +/- 269.0 | 256.3 +/- 269.0 | 0 | 6.021% |
| 1536 | 72 (1.200%) | 1830.7 +/- 286.5 | 294.7 +/- 286.5 | 0 | 0.910% |
| 2048 | 11 (0.183%) | 2372.8 +/- 250.7 | 324.8 +/- 250.7 | 0 | 0.153% |
| 4096 | 0 (0.000%) | N/A | N/A | 0 | 0.000% |

| Task | At 512 | At 1024 | At 1536 | At 2048 | At 4096 |
|---|---:|---:|---:|---:|---:|
| ab | 99.800% | 35.000% | 3.100% | 0.700% | 0.000% |
| color_cube_rotation | 30.300% | 0.000% | 0.000% | 0.000% | 0.000% |
| gcd | 2.100% | 0.000% | 0.000% | 0.000% | 0.000% |
| manipulate_matrix | 31.000% | 0.100% | 0.000% | 0.000% | 0.000% |
| string_manipulation | 93.900% | 19.700% | 4.100% | 0.400% | 0.000% |
| string_splitting | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% |

[Detailed statistics](report.json) include retained targets, EOS-only truncations, and per-task results. [Distribution counts](quality.json) include matrix degeneracy, GCD labels, dismantling outputs, and the cube initial-colour baseline.

Source SHA-256: `d38cf250d98881fdc2bbac559ee16afcf328b10f7b77191f2fb7c3849bdbe686`.
