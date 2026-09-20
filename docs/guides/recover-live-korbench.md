# Recovering a live KOR-Bench evaluation

`scripts/analysis/recover_live_korbench.py` saves completed responses from the
September 2026 GCW KOR-Bench jobs that started before incremental sample logging
was implemented. It supports Linux, little-endian 64-bit CPython 3.12, and the
native `GCWEvalLM.generate_until` implementation. It needs `py-spy` and permission
to read the target process, normally sudo on these workers.

Run it on the **same worker node** as the evaluation. The campaign manifest
restricts discovery to matching `gcw eval korbench` processes, checkpoint paths,
thinking settings, and output destinations. `--list` only lists matching targets
and does not require sudo. Without `--pid`, capture visits every matching process
on the local node, one at a time.

```bash
python scripts/analysis/recover_live_korbench.py capture \
  --manifest /path/to/campaign/manifest.json \
  --out-dir /shared/path/snapshots \
  --py-spy "$HOME/.local/bin/py-spy" --list

sudo /absolute/path/to/python scripts/analysis/recover_live_korbench.py capture \
  --manifest /path/to/campaign/manifest.json \
  --out-dir /shared/path/snapshots \
  --py-spy "$HOME/.local/bin/py-spy"
```

Create the destination as your normal user first. Use shared storage so the files
survive node shutdown. Files have unique names and are flushed and synced before
publication; the tool does not overwrite existing snapshots. Under sudo, saved
file publication temporarily uses the invoking user's UID/GID, so files are
user-owned and shared storage does not require root write access.

## How capture works

For each process, the tool sends SIGSTOP, obtains local-variable addresses using
`py-spy dump --json --locals --nonblocking`, and reads selected Python objects
through `/proc/PID/mem`, opened read-only. It resumes the process in a `finally`
block before writing the snapshot. It does not execute Python in the target,
change its tensors, or restart the evaluation.

An independent watchdog resumes the process if the recovery process dies or
stalls. The read budget is 25 seconds; the watchdog resumes after 30 seconds.
SIGINT and SIGTERM are ignored by the watchdog. Do not kill the watchdog itself.
If needed, `sudo kill -CONT PID` explicitly resumes an evaluation. Targets that
are already stopped or traced are rejected rather than changing their state.

The saved population includes completed groups in `outs` and completed batches
still held in the current `_run_batches.results`. A batch still generating is
excluded. A rare capture between function return and output-list copying can
recover fewer outputs than the runtime counter reports; another capture can
recover them later. The tool reads documents, prompt arguments, task/doc IDs,
responses, and available per-request diffusion metric aggregates directly from
the live objects. Metric request keys are derived from each live request's task, doc
ID, option index, and repeat count, because py-spy can omit closure-only locals
such as `metric_keys`. If that local is exposed, the derived keys are checked
against it. No reconstruction of request ordering from datasets is required.

Malformed or unsupported object layouts fail closed. Failure to recover diffusion
metrics is reported separately and does not discard successfully recovered
responses. The snapshot states how many responses have metrics; metric values
use the existing collector's raw sums and application counts, not final
per-example averages. Multiple-choice/repeated requests are not part of this
KOR-Bench rescue protocol.

The JSON contains the original campaign manifest, live command line, PID, worker,
capture timestamp, total/completed counts, and a `samples` list. Each sample has
`task`, `doc_id`, `example_id`, `request_id`, `doc`, `arguments`, `response`, and
optional `cache_update_metrics`. Files are explicitly marked `partial: true`.
They are analysis artifacts, not automatic generation-resume checkpoints.

## Score saved responses without sudo

Use the repository's evaluation environment. The underlying pinned comparison
rules must match the campaign; the current GCW response preprocessing is applied
and recorded separately:

```bash
.venv/bin/python scripts/analysis/recover_live_korbench.py score \
  /shared/path/snapshots/*.partial.json
```

Scoring now writes a sibling `.partial.scored-v2.json` with per-example scores,
per-category means/counts, and a `scoring_protocol` block. It strips everything
through the first `</think>` if present, otherwise scoring the response unchanged.
The original responses, campaign manifest, and any older `.partial.scored.json`
files are preserved. Existing files are reused only if both their source hash and
scoring protocol match. The worker wrapper applies this default on every run.

Repeated captures overlap: select one snapshot per
model/thinking condition, normally the latest or the one with most completed
responses. Do not concatenate snapshots or average their scores together.
Partial populations depend on prompt-length grouping, so report their counts
and category coverage and do not present them as full benchmark results.

The memory offsets are checked against the installed CPython 3.12 headers.
The JSON local-address mechanism follows [py-spy's serialized local-variable
format](https://github.com/benfred/py-spy/blob/master/src/stack_trace.rs).
Tests cover built-in and managed-dictionary decoding, a large response population,
real py-spy capture of a controlled child without sudo, response/metric joins,
and resumption after errors, timeouts, and recovery-process SIGKILL.

## Preserve evaluations that finish between captures

The prepared worker wrapper also runs `archive-completed` without sudo. This
checks manifest output paths and saves complete result JSON and its metric
sidecar together under the shared bundle's `completed/` directory. Incomplete
JSON writes are skipped and can be retried on the next invocation.
