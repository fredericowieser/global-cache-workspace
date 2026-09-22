# Reasoning Gym truncation audit

Reproduce with `python scripts/audit_reasoning_gym.py` using the prepared dataset and pinned local tokenizer. [Exact distribution counts](quality.json) accompany the truncation findings.

All 700,032 training examples; Qwen3.5-2B tokenizer with the prepared chat template, thinking disabled, empty response prefix and prompt suffix, and answer-only loss. Token counts include EOS. No data or training configuration was changed.

Lengths were read from every checksum-verified training record and checked against preparation-manifest aggregates. Actual sft_batch outputs were checked on 42 available task/cap/outcome combinations. Standard deviations are population standard deviations conditional on truncation at the given cap.

| Cap | Truncated rows | Original tokens, mean +/- std | Removed tokens, mean +/- std | Median removed | Prompt-only rows | Answer-token mass lost |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 414,458 (59.206%) | 844.5 +/- 252.0 | 332.5 +/- 252.0 | 343.0 | 0 | 50.880% |
| 1024 | 60,701 (8.671%) | 1283.5 +/- 265.1 | 259.5 +/- 265.1 | 175.0 | 0 | 5.815% |
| 1536 | 8,589 (1.227%) | 1809.5 +/- 260.3 | 273.5 +/- 260.3 | 204.0 | 0 | 0.867% |
| 2048 | 1,204 (0.172%) | 2303.6 +/- 283.0 | 255.6 +/- 283.0 | 152.5 | 0 | 0.114% |
| 4096 | 0 (0.000%) | N/A | N/A | N/A | 0 | 0.000% |

| Task | Truncated at 512 | At 1024 | At 1536 | At 2048 |
|---|---:|---:|---:|---:|
| ab | 99.771% | 33.700% | 3.296% | 0.433% |
| color_cube_rotation | 28.737% | 0.000% | 0.000% | 0.000% |
| gcd | 1.772% | 0.000% | 0.000% | 0.000% |
| manipulate_matrix | 31.059% | 0.027% | 0.000% | 0.000% |
| string_manipulation | 93.894% | 18.300% | 4.066% | 0.599% |
| string_splitting | 100.000% | 0.000% | 0.000% | 0.000% |

All prompts fit below 512 tokens, so there are no prompt-only examples at any tested cap. The 512-token cap removes part of every machine-dismantling solution and nearly every AB solution. At 2048, 99.828% of training examples remain complete and 99.8864% of answer tokens are retained. The prepared 4096-token setup retains every example completely.

EOS-only truncation counts are 557, 241, 26, and 2 at caps 512, 1024, 1536, and 2048 respectively. These are included in truncated row counts. Different caps condition statistics on different sets of examples.

[Detailed JSON report](report.json) includes the retained-answer mean/std and full per-task statistics. The source dataset SHA-256 is `0d951f835c9eb6e326bb286bdc2ec1ac0129b3958eb17b0c3fc3b86ad34dbbaa`.
