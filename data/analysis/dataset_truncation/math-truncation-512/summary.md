# Historical math SFT truncation audit

Measured on 2026-09-22, using the current cluster file at
`lrc-alpha:/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl`.
The source contains 1,000,000 nonblank rows and 1,392,578,835 bytes.
Only the requested selection was copied locally; no remote file or training
configuration was changed.

## Settings reproduced

- `sft_skip_rows: 128000`
- `sft_take_rows: 128000`
- Zero-based nonblank-row indices 128000 through 255999, inclusive.
- `sequence_len: 512`
- Qwen/Qwen3.5-2B tokenizer, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- `sft_chat_template: false`, empty prompt suffix, newline response prefix.
- Prompt and newline-prefixed answer encoded separately; one EOS token appended.
- `sft_loss_scope: answer_only`.

These settings come from
`configs/train/bottleneck_llm/qwen35_2b_instruct_longshot_from_sft_step2000_frozen_shared.yaml`.
The results apply to checkpoints using this tokenizer and formatting. No
individual checkpoint provenance was supplied or inspected to establish that
every historical checkpoint used identical tokenizer bytes and settings.

## Results

| Measurement | Result |
|---|---:|
| Selected examples | 128,000 |
| Truncated examples | 46,532 (36.3531%) |
| Complete examples | 81,468 (63.6469%) |
| Mean original total length, among truncated examples | 755.39 tokens |
| Mean removed tail length, among truncated examples | 243.39 tokens |
| Median removed tail length, among truncated examples | 199 tokens |
| 95th percentile removed tail length, among truncated examples | 582 tokens |
| Prompt-only examples, with zero supervised answer tokens | 574 (0.4484% of all examples; 1.2336% of truncated examples) |
| Mean original answer length, among truncated examples | 658.80 tokens |
| Mean retained answer length, among truncated examples | 421.12 tokens |
| Mean removed answer length, among truncated examples | 237.68 tokens |
| Original answer tokens across the selection | 50,849,953 |
| Retained answer tokens across the selection | 39,790,105 |
| Answer tokens removed across the selection | 11,059,848 (21.7500%) |

All lengths include the newline response prefix and EOS in the answer where
applicable. `Truncated` means the full encoded pair exceeds 512 tokens. Of the
46,532 truncated examples, 126 lose only EOS; 46,406 lose tokens before EOS too.
`Prompt-only` means the encoded prompt alone is at least 512 tokens. These
574 rows have no unmasked targets under answer-only loss.

The primary effect is partial solution supervision: most affected examples
retain some working but lose the end of the target. This audit counts token
retention; it does not establish a causal effect on checkpoint accuracy or
independently count whether the final mathematical answer remains visible.
Repeating the same rows over another epoch repeats the same truncation; it does
not expose a later window of the omitted solution.

## Verification and artifacts

Every selected row was tokenized without truncation. Counts were cross-checked
against the actual `sft_batch` outputs on 405 checks spanning complete,
truncated, prompt-only, EOS-only, and prompt-length boundary cases. All checks
passed. The transferred slice checksum matched the checksum computed remotely.

- Full source SHA-256: `576f26f134bb20d326c7812a44347a81699c4698dfaeacca1bea3a352abc696a`
- Selected slice SHA-256: `df62346bf8689c1e08055157354fd41bbc1f44116511b07859ec5f8fe13a4c44`
- Detailed statistics and tokenizer hashes: [report.json](report.json)
- Per-row token lengths: [row-lengths.csv.gz](row-lengths.csv.gz)
- Local audit program: `artifacts/audit_math_truncation.py`
- Local compressed input: `artifacts/math-truncation-512/slice.jsonl.gz`

The audit took approximately 27 seconds after tokenizer initialization and
transfer of the selected data.
