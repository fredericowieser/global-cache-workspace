# TODO list 

## Evaluation orchestration refactor

- [ ] Unify the evaluation loops in `src/gcw/evaluation/paper7.py` and
  `src/gcw/evaluation/core.py`. Express Paper7's chat/plain prompting and protocol
  metadata as per-task settings, preserving corrected Paper7 scores and explicit
  `paper7-legacy` compatibility. Share sample logging, diffusion metrics, and
  resume handling across suites; verify protocol parity before switching Paper7.

## TODO 1: Fred's checkpoints that need to be (re-)evaluated
* all following checkpoints need to be (re-)evaluated on paper7 suite with the latest evaluation settings and metrics.
* max new tokens for generation should be set to 2048 without thinking and 4096 with thinking
* make sure to have merged with fred-dev / dev before re-evaluating the checkpoints
* If multiple checkpoints exist for the same model, ensure only step 2000 is evaluated
* new results should then be entered into the paper table (`src/Global-Cache-Workspace-Theory-ICLR-2027-/sections/5_Results_and_Discussion.tex`) once complete and checked for accuracy


### Qwen3. 5-2B 
 
 ```text 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_2b-sft-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_2b-lora-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_2b-bt-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_2b-bgt-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_2b-diffusion-seed24-legacy/checkpoint 
 ``` 
 
 ### Qwen3. 5-4B 
 
 ```text 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_4b-sft-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_4b-lora-seed24-legacy/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_4b-bt-seed24-legacy/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_4b-bgt-seed24-legacy/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/qwen35_4b-diffusion-seed24-legacy/checkpoint 
 ``` 
 
 ### Llama 3. 2-1B 
 
 ```text 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_1b-sft-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_1b-lora-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_1b-bt-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_1b-bgt-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_1b-diffusion-seed24-legacy/checkpoint 
 ``` 
 
 ### Llama 3. 2-3B 
 
 ```text 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_3b-sft-seed24/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_3b-lora-seed24-legacy/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_3b-bt-seed24-legacy/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_3b-bgt-seed24-legacy/checkpoint 
 /shared/gwt/checkpoints/fred/fair-campaign-v1/llama32_3b-diffusion-seed24-legacy/checkpoint 
 ```

 UPDATE: All cache processors and LoRA checkpoints for Qwen3.5-4B and Llama 3.2-3B need to be re-trained with the appropriate sizing of the cache processor / LoRA rank such that the param count approximately matches the target 1.5% of backbone param count. Currently all of these checkpoints were run with a fixed 22M parameter count for the adapters.

 ## TODO 2: all of the models in TODO 1 need to be trained and evaluated with longshot (diffusion-longshot already covered)

 Wait until we have identified the best longshot settings before running this.

## Node worker-0 maintenance: approved restart queued

Approved restart resubmitted on 2026-09-18 at 18:03 UTC after the user restored Slurm defaults. The training pool was reduced to two workers at the user's request at 18:16 UTC. Current arrays are **435**, **426**, and **427**. The tables below record the submission baseline; update completion columns as results arrive.

### Submitted arrays

| Array | Workers × GPUs per worker | Work | Dependency |
| --- | --- | --- | --- |
| 427 (`gcw-restart-1gpu`) | 6 × 1 | 19 tasks: three training-plus-off tasks and 16 evaluation-only mode tasks | none |
| 435 (`gcw-restart-27b-train`) | 2 × 2 | Four Qwen-27B training tasks; no evaluation | none |
| 426 (`gcw-restart-27b-eval`) | 4 × 1 | Four resulting checkpoints, each evaluated off then on (eight evaluations) | `afterok:435` |

The original corrected-default manifests, submission receipts, controller snapshot, and job records are in `slurm_out/restart-20260918-defaults-fixed/`. The manifests are byte-identical to the approved originals: `one-gpu.tsv`, `two-gpu-training.txt`, and `qwen27-eval.tsv`. `submissions.json` maps jobs to fresh queue state; `manifest-sha256.json` records manifest hashes.

Verified live defaults: `DefCpuPerGPU=24`, `DefMemPerGPU=318545`, `JobSubmitPlugins=lua`. All three actual job records show `CpusPerTres=gpu:24`, `MemPerTres=gpu:318545`, and unlimited time. Training workers request **637090 MiB** each; evaluation and combined-pool workers request **318545 MiB** each. Submission arguments still specify only GPUs as resource overrides, leaving CPU, memory, partition, and time at defaults. Evaluation dependency is confirmed as `afterok:435`.

The latest training manifest and receipt are in `slurm_out/restart-20260918-two-training-workers/`. Array 435 has two workers and runs the same four training configs; array 426 was held, detached from 425, updated to `afterok:435`, and released. Array 427 was left running. Maximum concurrent GPU demand before dependent evaluation is now **10 GPUs** (four training GPUs plus six in 427), subject to availability. The four-worker evaluation array starts only after training succeeds.

### Replaced submissions

- Original held arrays 409/411 were canceled after the first replacements were accepted.
- Initial replacements 417/419/420 were submitted at 17:52 UTC under reset defaults, which allocated a node's full memory per worker. Their records remain in `slurm_out/restart-20260918-approved/`.
- After the user repaired the defaults, pending tasks were held and all three arrays were canceled at 18:02:15 UTC. Two training workers had claimed the LoRA batch16 and diffusion rank1 retries, but no checkpoints or new evaluation outputs existed before or after cancellation. No work had completed in those arrays; all approved tasks remain necessary.
- New submissions 425/426/427 were accepted at 18:03 UTC. At verification they were pending node availability/priority and the intended dependency; canceled 417 workers were finishing Slurm cleanup. Do not resubmit them merely because they are pending.

- At 18:16 UTC, canceled training array 425 to replace its three workers with two in 435. None of the four training configs had a saved checkpoint at cancellation; the manifest is unchanged. Canceled workers may remain `COMPLETING` temporarily during cleanup. The six-worker 1-GPU array 427 was not stopped.

### Approved stop sequence and evidence

- At 16:47 UTC, held pending arrays 409 and 411 and verified all eight tasks had `Reason=JobHeldUser`, `Priority=0`.
- Canceled `351_1`, `351_2`, and `386_0` through `386_5`. Accounting records EndTime 16:47:02. Do not repeat the stop sequence or release the held arrays.
- Last training steps: 351_1 diffusion rank1 retry 185/2000; 351_2 LoRA batch16 retry 311/2000; 386_0 Llama-1B diffusion head128 retry 417/2000. None saved a checkpoint, so these restart from step 0.
- 386_1–3 were evaluating Qwen-4B LoRA/BT/BGT with thinking on after completing off. 386_4–5 were evaluating Qwen-4B Longshot/diffusion with thinking off; neither mode has a completed result. All five final checkpoints contain step 2000.
- Original queue records: 351 had 11 claimed, 5 completed, 3 failed; 386 had 16 claimed, 3 completed, 7 failed, and two unclaimed absolute-path Llama-3B retries. These counters omit interrupted work and are not evidence that every evaluation is missing.

Preserve these source directories and the interrupted `slurm_out/gcw-qwen35-27b-351_*.out` and `slurm_out/gcw-fair-v1-train-eval-386_*.out` logs:

- 351: `slurm_out/gcw-qwen35-27b_20260916_145948_535860/`
- 409: `slurm_out/gcw-qwen35-27b-fair-gate-off_20260918_140305_2885901/`
- 386: `slurm_out/gcw-fair-v1-train-eval_20260917_194535_1982275/`
- 411: `slurm_out/gcw-fair-instruct-paper7_20260918_153537_3050043/`

### Resource settings for all three replacement arrays

Set only the GPU count per worker (`GPUS=1` or `GPUS=2`) as an explicit resource request. Leave CPU count, memory, time limit, and partition unset so Slurm applies its defaults. Do not inherit `CPUS_PER_TASK`, `MEM`, `ARRAY_TIME`, `PARTITION`, or equivalent `SBATCH_*` overrides from the submission environment. Worker counts and the evaluation dependency are specified below.

### Reading the tables

Config archive: on 2026-09-18, moved the 31 config files not named in these tables to `configs/experiments/archived/`, preserving subdirectories and contents. All 26 table-listed YAML files remain at their original paths, including completed and superseded configurations. Active array config paths are unchanged.

- YAML filenames are relative to `configs/experiments/`, including all previously described “absolute-path” entries. Relative and absolute entries for the same config are deduplicated here.
- **Outstanding evaluation modes lists only work still needed. `none` means omit the configuration from the replacement queue. Completed evaluations must not be repeated.**
- Training completion was checked against checkpoint `train_state.json` at step 2000. Evaluation completion was checked against parseable result JSON and its task coverage for the current output directory.
- **All Llama models use thinking off only**, including native baselines. Missing or interrupted Llama thinking-on evaluations are deliberately excluded.
- Retain the original Paper7 mixed-prompt protocol, off/on generation caps of 2048/4096, and automatic evaluation batching with maximum 64. Existing 386 results cover all six tasks; existing 351 results omit TheoremQA. Pending-array environment variables were not retained by Slurm. The submitted manifests explicitly use the full six-task suite for the 386/411 pool and omit TheoremQA for all four 351/409-derived checkpoints, matching the existing 351 suite. These choices are explicit in each manifest row; recovered settings for the old pending arrays are not claimed.

### One combined 1-GPU array: former 386 and 411 tasks

Proposed shared queue: up to six workers, one GPU each; all other resource settings use Slurm defaults. Training tasks evaluate their missing mode after training. Completed-training tasks use the existing final checkpoint; separate missing evaluation modes may be claimed independently. No dependency on the 2-GPU array.

| Config or pretrained checkpoint | Training complete? Re-run training? | Outstanding evaluation modes | Modes already complete |
| --- | --- | --- | --- |
| llama32_1b_fair_campaign_v1_longshot_adapter22m_seed24.yaml (head128) | Yes; no | none | off |
| llama32_1b_fair_campaign_v1_diffusion_adapter22m_seed24.yaml (head128) | No; restart from step 0 | off | none |
| qwen35_2b_fair_campaign_v1_longshot_adapter22m_seed24.yaml | Yes; no | none | off, on |
| qwen35_2b_fair_campaign_v1_diffusion_adapter22m_seed24.yaml | Yes; no | on | off |
| llama32_3b_fair_campaign_v1_bt_pct015_seed24.yaml | Yes; no | none | off |
| llama32_3b_fair_campaign_v1_bgt_pct015_seed24.yaml | Yes; no | none | off |
| llama32_3b_fair_campaign_v1_lora_pct015_seed24.yaml | Yes; no | none | off |
| llama32_3b_fair_campaign_v1_longshot_pct015_seed24.yaml (head128 retry) | No; train from step 0 | off | none |
| llama32_3b_fair_campaign_v1_diffusion_pct015_seed24.yaml (head128 retry) | No; train from step 0 | off | none |
| qwen35_4b_fair_campaign_v1_bt_pct015_seed24.yaml | Yes; no | on | off |
| qwen35_4b_fair_campaign_v1_bgt_pct015_seed24.yaml | Yes; no | on | off |
| qwen35_4b_fair_campaign_v1_lora_pct015_seed24.yaml | Yes; no | on | off |
| qwen35_4b_fair_campaign_v1_longshot_pct015_seed24.yaml | Yes; no | off, on | none |
| qwen35_4b_fair_campaign_v1_diffusion_pct015_seed24.yaml | Yes; no | off, on | none |
| /shared/gwt/models/llama32-1b-native-92131767 | Pretrained; no training | off | none |
| /shared/gwt/models/llama32-3b-native-0cb88a4f | Pretrained; no training | off | none |
| /shared/gwt/models/qwen35-2b-native-15852e8c | Pretrained; no training | off, on | none |
| /shared/gwt/models/qwen35_4b/tf_base | Pretrained; no training | off, on | none |
| /shared/gwt/models/qwen35_27b/tf_base | Pretrained; no training | off, on | none |

**Remaining work: three training runs followed by off evaluation, plus 16 evaluation-only mode tasks (eight trained-model evaluations and eight native-baseline evaluations).** Five completed trained configurations require no work. The table retains them for reconciliation.

Checkpoint paths are `<output_dir>/checkpoint`, with step 2000 in `train_state.json`. Result paths are `results/<output_dir-name>_final_paper7_thinking-{off,on}.json`. The four Llama Longshot/diffusion configs use output directories ending `_head128`; older `_superseded_head224` results do not count. The two unclaimed Llama-3B retries use the same config files in `configs/experiments/`, not additional experiment configs.

The last five rows expand to the eight exact checkpoint/output/mode entries in old 411 `configs.txt`, with outputs `results/<model>_pretrained_instruct_native_paper7_thinking-{off,on}.json`. None of these outputs exists. The Qwen-27B **native baseline remains in this 1-GPU pool with both modes**, as in 411. The four new trained Qwen-27B checkpoints also receive both modes, in the separate dependent evaluation array below.

Qwen-2B diffusion training and off evaluation are complete. Its on evaluation failed with a CUDA peer-memory access error; only on evaluation needs retrying. Failed/interrupted evaluation entries never imply that its completed training should be repeated.

### One combined 2-GPU training-only array: former 351 and 409 tasks

Shared queue: two workers, two GPUs per worker; all other resource settings use Slurm defaults. The four remaining 351 and 409 configs are ready independently; remove the old dependency between those training groups. **Run training only in this array (`EVAL_AFTER_TRAIN=0`); do not run any evaluation or thinking-mode watcher here.** Evaluate the four resulting checkpoints in the dependent 1-GPU array described below. This replaces the earlier off-only evaluation plan for 351/409.

The outstanding modes column below describes work for the **dependent evaluation array**, not work to run inside the 2-GPU allocation.

| Config | Training complete? Re-run training? | Outstanding evaluation modes | Modes already complete |
| --- | --- | --- | --- |
| qwen35_27b_instruct_sft_seed24.yaml | Yes; no | none | off, on |
| qwen35_27b_instruct_lora_seed24.yaml | Yes; no | none | off, on |
| qwen35_27b_instruct_longshot_contextual_scalar_outer_product_magnitude_seed24.yaml | Yes; no | none | off, on |
| qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_seed24.yaml | Yes; no | none | off, on |
| qwen35_27b_instruct_bt_seed24.yaml | Yes; no | none | off, on |
| qwen35_27b_instruct_bgt_seed24.yaml | Yes; no | none | off, on |
| qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_lora_seed24.yaml | No; no, superseded by batch16 retry | none | none |
| qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_diffusion_seed24.yaml | No; no, superseded by rank1 retry | none | none |
| qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_lora_batch16_retry_seed24.yaml | No; restart from step 0 | off, on (dependent 1-GPU array) | none |
| qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_diffusion_rank1_retry_seed24.yaml | No; restart from step 0 | off, on (dependent 1-GPU array) | none |
| qwen35_27b_instruct_longshot_fair_gate_rank1_seed24.yaml | No; train from step 0 (old 409 pending) | off, on (dependent 1-GPU array) | none |
| qwen35_27b_instruct_diffusion_longshot_fair_gate_rank1_seed24.yaml | No; train from step 0 (old 409 pending) | off, on (dependent 1-GPU array) | none |

**Remaining work: exactly four training runs in the 2-GPU array, followed by eight evaluations (off and on for each checkpoint) in the dependent 1-GPU array.** Six configurations already have final checkpoints and both evaluation modes complete and require no work. Two failed original configs are superseded and must not be queued alongside their replacements.

SFT seed 25 was stopped intentionally, as confirmed by the user. It is removed from the restart table and must not be trained or evaluated.

Checkpoint paths for this pool are `<output_dir>/checkpoint-2000`; results are `results/<output_dir-name>_step2000_paper7_thinking-{off,on}.json`. All unfinished configs lack checkpoints. Use the explicit batch16 LoRA and rank1 diffusion retry configs; `resume: null` means starting from step 0.

The contextual-vector Longshot seed-24 task in 351_0 was canceled after writing both results, so it lacks a `completed_configs` entry despite completed training and evaluation. Its checkpoint and off/on results were verified: **do not repeat it**.

### Dependent 1-GPU evaluation array: four new Qwen-27B checkpoints

Use **four workers with one GPU each**, with other resource settings left at Slurm defaults. This array depends on successful completion of the new 2-GPU training array (`afterok:<new-training-array-id>`); it is independent of the combined 386/411 pool. Use evaluation-only tasks and verify that all four final checkpoints contain step 2000 before evaluation. If training fails or a checkpoint is missing, resolve that failure before releasing evaluation; do not evaluate a partial checkpoint.

Each row below is one checkpoint task: run thinking off and thinking on sequentially on a single GPU. The four workers can evaluate all four checkpoints concurrently. Set `EVAL_GPUS=1` and explicitly request both thinking modes (`false,true`). Preserve the existing output naming and 2048/4096 generation caps; no old watcher is needed. None of these eight results is already complete.

Paths below are relative to `runs/experiments/`; each checkpoint is `<stem>/checkpoint-2000`, and output files are `results/<stem>_step2000_paper7_thinking-{off,on}.json`.

| Checkpoint stem | Evaluation modes to run | Modes already complete |
| --- | --- | --- |
| qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_lora_batch16_retry_seed24 | off, on | none |
| qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_diffusion_rank1_retry_seed24 | off, on | none |
| qwen35_27b_instruct_longshot_fair_gate_rank1_seed24 | off, on | none |
| qwen35_27b_instruct_diffusion_longshot_fair_gate_rank1_seed24 | off, on | none |

### Submission audit and follow-up

1. Completed: audited tables, original manifests, checkpoint states, existing result JSON, and interrupted-log records. Excluded intentionally stopped SFT seed 25, two superseded configurations, and every completed evaluation. All eight native baseline tasks from 411 and both training configs from 409 are represented.
2. Completed: fresh manifests contain 19 mixed tasks, four training tasks, and four dependent checkpoint tasks, with 27 unique missing evaluation outputs. No Llama thinking-on task exists. All seven training configs load successfully, target step 2000, and start from step 0 without resume.
3. Completed: 21 launcher tests pass. Mixed tasks preserve per-row modes and suite exclusions and reject duplicate/existing outputs. Missing checkpoints may be queued only with a Slurm `afterok` dependency and are required at execution. Worker failures propagate to Slurm, so a failed training array will block 426.
4. Completed: dry-run and actual Slurm records verified worker counts, GPU counts, default resources, and `afterok:435`. Both old held arrays were canceled after all replacements were accepted.
5. Follow up: inspect training failures before resolving any blocked evaluation dependency; verify all four final checkpoints are at step 2000. Check result JSON task coverage and modes before marking this restart complete. A queued job is not evidence of completed training/evaluation.

## Restore diffusion update logging to opt-in after re-run evaluations

Temporary CLI default enabled for native Longshot diffusion checkpoint evaluations
with a results destination and active processor. Future processes in arrays 427
and 426 pick it up without restarting arrays; existing processes are unaffected.
Sidecars use `<results-stem>.cache-update-metrics.jsonl`. No training logging is added.

After the desired evaluations finish, remove the automatic-default block marked
TODO in `_resolve_cache_update_metrics` in `src/gcw/cli.py`, adjust its default-selection
tests, and restore opt-in wording in `docs/guides/evaluate.md`. Keep the existing
explicit `--cache-update-metrics` support. Do not rerun completed evaluations just
for logging.

## Additional small-model diffusion metric evaluations in array 427

Appended four evaluation-only tasks to the end of the live shared queue on
2026-09-18, at the user's request. They repeat evaluation to collect diffusion
update metrics, using separate outputs and the existing step-2000 checkpoints.
No training is repeated, no workers are restarted, and no new array is submitted.

| Diffusion checkpoint | Evaluation modes | New output base under `results/` |
| --- | --- | --- |
| Llama-1B head128 | off | `llama32_1b_fair_campaign_v1_diffusion_adapter22m_seed24_head128_final_paper7_update_metrics.json` |
| Qwen-2B | off, on (separate tasks) | `qwen35_2b_fair_campaign_v1_diffusion_adapter22m_seed24_final_paper7_update_metrics.json` |
| Llama-3B head128 | off | `llama32_3b_fair_campaign_v1_diffusion_pct015_seed24_head128_final_paper7_update_metrics.json` |

Output bases gain `_thinking-off.json` or `_thinking-on.json`; each evaluation
also writes the corresponding `.cache-update-metrics.jsonl` sidecar. All four
use the full six-task Paper7 suite, auto batch size with maximum 64, and the
existing 2048/4096 generation caps. Keep the temporary diffusion logging default
until these extra evaluations finish.

Array 427 now has **23 tasks** (the original 19 plus these four). Its live manifest
is `slurm_out/gcw-restart-1gpu_20260918_180337_1744858/configs.txt`; the same directory
contains `configs.before-metrics-reruns.txt`, `metrics-reruns-added.tsv`, and
`metrics-reruns-audit.json`. Original submission manifests remain historical
snapshots. The appended tasks are claimed after the original tasks have been
claimed; they may overlap with slower original evaluations still running.

## Paper performance table update — approved and entered (2026-09-20)

**Approved by the user and entered into `src/Global-Cache-Workspace-Theory-ICLR-2027-/sections/5_Results_and_Discussion.tex`.** The proposal and source tables below are retained as the review record. All 33 previously populated row/mode token entries were independently recomputed and matched at displayed precision. The 20 approved row mappings (29 new mode evaluations) are now entered; original populated scores are preserved. The two missing 27B on-mode final JSONs were rechecked and remain absent. No additional modes were added. Earlier restart completion columns above are historical submission snapshots; this section audits the results now available.

### Proposed scope

- 20 affected model/method rows: fill or extend 15 existing rows and add five missing baseline/variant rows. There are **29 newly reportable mode evaluations**, plus three already-entered off evaluations shown for context (Qwen-2B Longshot/diffusion and Qwen-4B BT).
- **No populated performance cells need replacement under this proposal.** In particular, the current Llama-3B LoRA/BT/BGT and Qwen-4B BT-off cells already match the resized `pct015` runs exactly. Earlier fixed-22M checkpoints are not their current table sources.
- Preserve the existing 27B contextual-scalar and contextual-vector Longshot rows. The new static-scalar/rank1 and Longshot+LoRA experiments differ in architecture/training; add explicitly labelled rows rather than silently replacing them.
- Keep the original performance evaluations for the small diffusion models. List the four additional logging evaluations separately below; do not count repeated evaluation of one checkpoint as another model.
- Two mode evaluations still lack final score JSON: **27B native thinking on** and **27B contextual-vector diffusion rank1 thinking on**. Leave those modes blank; sidecars/progress are not completed benchmark results.

### Source and validation conventions

`A` = `/shared/gwt/checkpoints/martin/results/` (relocated completed results). `W` = `/opt/soperator-home/martin/global-cache-workspace/results/`. Each source below identifies the exact final JSON. Config names are relative to `configs/experiments/`. Trained checkpoints are the config output directory plus `/checkpoint` for small models or `/checkpoint-2000` for 27B; moved small-model runs live beneath `/shared/gwt/checkpoints/martin/runs/experiments/`. All 24 active trained configurations in the restart tables have saved step 2000 checkpoints (including rows already reported). Failed superseded configs are excluded.

Every selected result was checked for `paper7_mixed_prompt_v2`, all expected score fields and sample counts, thinking mode, chat generation / plain option-likelihood settings, and generation caps 2048 off / 4096 on. Full suite: GSM8K 1319, SVAMP 300, GSM8K-Hard 1319, TheoremQA 800, LogiQA 651, Gaokao 351. Trained 27B omits TheoremQA. Mean tokens = sum of response tokens / sum of generated responses, not a mean of task means. Scores/averages use unrounded source values before display rounding.

### Exact proposed row mapping

| ID | Model | Method | Completed modes | Proposed table action | Training config |
| --- | --- | --- | --- | --- | --- |
| R01 | llama32_1b | Pretrained instruct | off | Fill blank row | Native pretrained checkpoint; no training |
| R02 | llama32_1b | Longshot | off | Fill blank row | `llama32_1b_fair_campaign_v1_longshot_adapter22m_seed24.yaml` |
| R03 | llama32_1b | Diffusion Longshot | off | Fill blank row | `llama32_1b_fair_campaign_v1_diffusion_adapter22m_seed24.yaml` |
| R04 | llama32_3b | Pretrained instruct | off | Fill blank row | Native pretrained checkpoint; no training |
| R05 | llama32_3b | Longshot | off | Fill blank row | `llama32_3b_fair_campaign_v1_longshot_pct015_seed24.yaml` |
| R06 | llama32_3b | Diffusion Longshot | off | Fill blank row | `llama32_3b_fair_campaign_v1_diffusion_pct015_seed24.yaml` |
| R07 | qwen35_2b | Pretrained instruct | off, on | Add missing baseline row | Native pretrained checkpoint; no training |
| R08 | qwen35_2b | Longshot | off, on | Add on; retain existing off | `qwen35_2b_fair_campaign_v1_longshot_adapter22m_seed24.yaml` |
| R09 | qwen35_2b | Diffusion Longshot | off, on | Add on; retain existing off | `qwen35_2b_fair_campaign_v1_diffusion_adapter22m_seed24.yaml` |
| R10 | qwen35_4b | Pretrained instruct | off, on | Fill blank row | Native pretrained checkpoint; no training |
| R11 | qwen35_4b | LoRA | off, on | Fill blank row | `qwen35_4b_fair_campaign_v1_lora_pct015_seed24.yaml` |
| R12 | qwen35_4b | BT | off, on | Add on; retain existing off | `qwen35_4b_fair_campaign_v1_bt_pct015_seed24.yaml` |
| R13 | qwen35_4b | BGT | off, on | Fill blank row | `qwen35_4b_fair_campaign_v1_bgt_pct015_seed24.yaml` |
| R14 | qwen35_4b | Longshot | off, on | Fill blank row | `qwen35_4b_fair_campaign_v1_longshot_pct015_seed24.yaml` |
| R15 | qwen35_4b | Diffusion Longshot | off, on | Fill blank row | `qwen35_4b_fair_campaign_v1_diffusion_pct015_seed24.yaml` |
| R16 | qwen35_27b | Pretrained instruct | off | Add missing baseline row; on incomplete | Native pretrained checkpoint; no training |
| R17 | qwen35_27b | Longshot (static scalar, rank1) | off, on | Add separate variant row; retain scalar/vector rows | `qwen35_27b_instruct_longshot_fair_gate_rank1_seed24.yaml` |
| R18 | qwen35_27b | Diffusion Longshot (static scalar, rank1) | off, on | Fill blank diffusion row; label variant explicitly | `qwen35_27b_instruct_diffusion_longshot_fair_gate_rank1_seed24.yaml` |
| R19 | qwen35_27b | Longshot (vector) + LoRA | off, on | Add separate combined-method row | `qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_lora_batch16_retry_seed24.yaml` |
| R20 | qwen35_27b | Diffusion Longshot (contextual vector, rank1) | off | Add separate variant row; on incomplete | `qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_diffusion_rank1_retry_seed24.yaml` |

Native checkpoint identities: Llama-1B `/shared/gwt/models/llama32-1b-native-92131767`; Llama-3B `/shared/gwt/models/llama32-3b-native-0cb88a4f`; Qwen-2B `/shared/gwt/models/qwen35-2b-native-15852e8c`; Qwen-4B `/shared/gwt/models/qwen35_4b/tf_base`; Qwen-27B `/shared/gwt/models/qwen35_27b/tf_base`.

### Scores ready for entry

Accuracy and math average are percentages. This review table shows the mean over **all available tasks**, so the native 27B row below is initially six-task; see its explicit comparability decision below. Token totals are millions.

| ID / mode | GSM8K | SVAMP | GSM-Hd | ThmQA | LogiQA | Gaokao | Math avg. | Mean tokens | Total (M) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 off | 43.37 | 63.33 | 12.81 | 11.25 | 30.41 | 27.07 | 31.37 | 260.9 | 0.975 |
| R02 off | 38.82 | 60.67 | 10.61 | 8.75 | 29.19 | 25.93 | 28.99 | 337.8 | 1.263 |
| R03 off | 42.53 | 59.67 | 10.99 | 12.25 | 30.11 | 27.92 | 30.58 | 426.6 | 1.595 |
| R04 off | 69.90 | 78.00 | 24.26 | 15.38 | 29.34 | 31.05 | 41.32 | 265.1 | 0.991 |
| R05 off | 67.85 | 82.00 | 22.67 | 11.00 | 29.49 | 28.21 | 40.20 | 307.8 | 1.150 |
| R06 off | 70.13 | 83.67 | 23.73 | 21.50 | 29.03 | 29.06 | 42.85 | 308.0 | 1.151 |
| R07 off | 66.94 | 84.33 | 38.21 | 21.38 | 31.03 | 33.62 | 45.92 | 623.3 | 2.330 |
| R07 on | 34.50 | 53.67 | 11.14 | 14.75 | 31.03 | 33.62 | 29.78 | 3643.9 | 13.621 |
| R08 off | 65.73 | 82.67 | 39.35 | 26.75 | 28.73 | 31.34 | 45.76 | 351.7 | 1.315 |
| R08 on | 73.69 | 84.00 | 41.62 | 26.62 | 28.73 | 31.34 | 47.67 | 1202.9 | 4.496 |
| R09 off | 67.32 | 80.67 | 39.27 | 25.50 | 28.57 | 32.48 | 45.64 | 340.9 | 1.274 |
| R09 on | 68.54 | 78.33 | 37.98 | 24.88 | 28.57 | 32.48 | 45.13 | 1023.3 | 3.825 |
| R10 off | 80.21 | 88.67 | 51.86 | 29.50 | 35.33 | 34.19 | 53.29 | 629.0 | 2.351 |
| R10 on | 64.52 | 73.67 | 30.55 | 14.75 | 35.33 | 34.19 | 42.17 | 2979.2 | 11.136 |
| R11 off | 86.43 | 89.67 | 52.08 | 39.25 | 36.41 | 39.03 | 57.14 | 275.2 | 1.029 |
| R11 on | 86.88 | 91.00 | 53.30 | 42.12 | 36.41 | 39.03 | 58.12 | 425.9 | 1.592 |
| R12 off | 73.84 | 89.00 | 45.34 | 35.88 | 34.10 | 31.62 | 51.63 | 364.6 | 1.363 |
| R12 on | 47.84 | 50.00 | 24.26 | 20.12 | 34.10 | 31.62 | 34.66 | 3268.4 | 12.217 |
| R13 off | 80.36 | 86.67 | 48.07 | 35.12 | 33.79 | 32.19 | 52.70 | 320.8 | 1.199 |
| R13 on | 57.39 | 67.00 | 31.01 | 31.00 | 33.79 | 32.19 | 42.06 | 2359.9 | 8.821 |
| R14 off | 83.40 | 86.67 | 51.55 | 35.12 | 34.10 | 37.61 | 54.74 | 280.9 | 1.050 |
| R14 on | 86.88 | 89.67 | 51.63 | 36.00 | 34.10 | 37.61 | 55.98 | 771.5 | 2.884 |
| R15 off | 82.18 | 87.67 | 49.81 | 35.00 | 33.49 | 39.03 | 54.53 | 310.1 | 1.159 |
| R15 on | 83.93 | 89.33 | 50.27 | 33.88 | 33.49 | 39.03 | 54.99 | 522.3 | 1.952 |
| R16 off | 84.23 | 91.33 | 54.74 | 32.12 | 39.94 | 59.54 | 60.32 | 594.2 | 2.221 |
| R17 off | 87.72 | 90.00 | 54.13 | — | 45.62 | 55.56 | 66.61 | 217.0 | 0.638 |
| R17 on | 90.14 | 93.67 | 55.72 | — | 45.62 | 55.56 | 68.14 | 348.9 | 1.025 |
| R18 off | 86.73 | 91.00 | 54.74 | — | 43.63 | 52.71 | 65.76 | 211.5 | 0.621 |
| R18 on | 88.63 | 92.00 | 55.65 | — | 43.63 | 52.71 | 66.52 | 268.8 | 0.790 |
| R19 off | 89.92 | 91.00 | 55.34 | — | 43.78 | 58.12 | 67.63 | 197.6 | 0.581 |
| R19 on | 90.60 | 92.67 | 56.03 | — | 43.78 | 58.12 | 68.24 | 222.1 | 0.653 |
| R20 off | 88.02 | 93.00 | 55.04 | — | 40.86 | 50.71 | 65.53 | 209.0 | 0.614 |

**27B native comparability decision:** its off evaluation includes TheoremQA (32.12%), unlike every trained 27B row. Six-task average / four-task token statistics are 60.32%, 594.2, 2.221M as above. For comparisons with trained 27B, the five-task average excluding TheoremQA is **65.96%**, and the three-task token mean/total are **465.8 / 1.369M**. Proposed paper treatment: report its available TheoremQA score, but use the common five-task average and three-task token statistics throughout the 27B block, with an explicit caption exception. This requires approval along with the rows; do not silently mix six-task and five-task averages.

### Exact result files

- R01 off: `A/llama32_1b_pretrained_instruct_native_paper7_thinking-off.json`
- R02 off: `A/llama32_1b_fair_campaign_v1_longshot_adapter22m_seed24_head128_final_paper7_thinking-off.json`
- R03 off: `A/llama32_1b_fair_campaign_v1_diffusion_adapter22m_seed24_head128_final_paper7_thinking-off.json`
- R04 off: `A/llama32_3b_pretrained_instruct_native_paper7_thinking-off.json`
- R05 off: `A/llama32_3b_fair_campaign_v1_longshot_pct015_seed24_head128_final_paper7_thinking-off.json`
- R06 off: `A/llama32_3b_fair_campaign_v1_diffusion_pct015_seed24_head128_final_paper7_thinking-off.json`
- R07 off: `A/qwen35_2b_pretrained_instruct_native_paper7_thinking-off.json`
- R07 on: `W/qwen35_2b_pretrained_instruct_native_paper7_thinking-on.json`
- R08 off: `A/qwen35_2b_fair_campaign_v1_longshot_adapter22m_seed24_final_paper7_thinking-off.json`
- R08 on: `A/qwen35_2b_fair_campaign_v1_longshot_adapter22m_seed24_final_paper7_thinking-on.json`
- R09 off: `A/qwen35_2b_fair_campaign_v1_diffusion_adapter22m_seed24_final_paper7_thinking-off.json`
- R09 on: `A/qwen35_2b_fair_campaign_v1_diffusion_adapter22m_seed24_final_paper7_thinking-on.json`
- R10 off: `A/qwen35_4b_pretrained_instruct_native_paper7_thinking-off.json`
- R10 on: `W/qwen35_4b_pretrained_instruct_native_paper7_thinking-on.json`
- R11 off: `A/qwen35_4b_fair_campaign_v1_lora_pct015_seed24_final_paper7_thinking-off.json`
- R11 on: `A/qwen35_4b_fair_campaign_v1_lora_pct015_seed24_final_paper7_thinking-on.json`
- R12 off: `A/qwen35_4b_fair_campaign_v1_bt_pct015_seed24_final_paper7_thinking-off.json`
- R12 on: `A/qwen35_4b_fair_campaign_v1_bt_pct015_seed24_final_paper7_thinking-on.json`
- R13 off: `A/qwen35_4b_fair_campaign_v1_bgt_pct015_seed24_final_paper7_thinking-off.json`
- R13 on: `A/qwen35_4b_fair_campaign_v1_bgt_pct015_seed24_final_paper7_thinking-on.json`
- R14 off: `A/qwen35_4b_fair_campaign_v1_longshot_pct015_seed24_final_paper7_thinking-off.json`
- R14 on: `A/qwen35_4b_fair_campaign_v1_longshot_pct015_seed24_final_paper7_thinking-on.json`
- R15 off: `A/qwen35_4b_fair_campaign_v1_diffusion_pct015_seed24_final_paper7_thinking-off.json`
- R15 on: `A/qwen35_4b_fair_campaign_v1_diffusion_pct015_seed24_final_paper7_thinking-on.json`
- R16 off: `W/qwen35_27b_pretrained_instruct_native_paper7_thinking-off.json`
- R17 off: `W/qwen35_27b_instruct_longshot_fair_gate_rank1_seed24_step2000_paper7_thinking-off.json`
- R17 on: `W/qwen35_27b_instruct_longshot_fair_gate_rank1_seed24_step2000_paper7_thinking-on.json`
- R18 off: `W/qwen35_27b_instruct_diffusion_longshot_fair_gate_rank1_seed24_step2000_paper7_thinking-off.json`
- R18 on: `W/qwen35_27b_instruct_diffusion_longshot_fair_gate_rank1_seed24_step2000_paper7_thinking-on.json`
- R19 off: `W/qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_lora_batch16_retry_seed24_step2000_paper7_thinking-off.json`
- R19 on: `W/qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_lora_batch16_retry_seed24_step2000_paper7_thinking-on.json`
- R20 off: `W/qwen35_27b_instruct_longshot_contextual_vector_outer_product_magnitude_diffusion_rank1_retry_seed24_step2000_paper7_thinking-off.json`

### Old versus new configuration comparison

#### Resized small models: what has already been replaced, and what remains blank

Old model configs were read from `/shared/gwt/checkpoints/fred/fair-campaign-v1/<model>-<method>-seed24-legacy/checkpoint/config.json`; new configs were checked against their saved checkpoint `config.json`. Below, widths refer to the cache processor, not the backbone. Parameter counts are new-config expected adapter counts, not fresh measurements.

| Model / method | Exact old → new model changes | New adapter count | Current paper status |
| --- | --- | --- | --- |
| Llama-3B LoRA | rank/alpha 15/15 → 28/28; same seven projection targets | 42,549,248 | Already entered; retain |
| Llama-3B BT | hidden 96 → 144; intermediate 320 → 512 | 42,245,084 | Already entered; retain |
| Llama-3B BGT | hidden 168 → 288; intermediate 1600 → 3136 | 42,273,569 | Already entered; retain |
| Llama-3B diffusion | hidden 672 → 896; intermediate 2688 → 3984; processor attention/KV heads 3 → 7; selected cache heads (`topk_heads`) 3 → 224 | 42,301,056 | Blank; fill with R06 |
| Qwen-4B LoRA | rank/alpha 11/11 → 26/26; same 12 projection targets | 52,755,456 | Blank; fill with R11 |
| Qwen-4B BT | hidden 192 → 288; intermediate 896 → 1728 | 53,303,560 | Off already entered; add on (R12) |
| Qwen-4B BGT | hidden 384 → 672; intermediate 2240 → 4320 | 53,610,657 | Blank; fill with R13 |
| Qwen-4B diffusion | hidden 192 → 384; intermediate 1248 → 3776; selected attention heads 3 → 32; selected linear heads 72 → 768; gate init -5 → -2 | 53,527,680 | Blank; fill with R15 |

For the BT/BGT/diffusion Llama-3B and Qwen-4B rows above, saved model configs also change `processor_generation_mode: windowed → post_prefill_relative` and `legacy_bottleneck_generate: null → true`. These are not size-only changes. All target approximately 1.5% of **non-embedding** backbone parameters. The new configs retain seed 24, 2000 steps, 512-token training sequences, 65,536 tokens per optimizer step, learning rate 1e-4, answer-only training with thinking disabled, and the historical 128K OpenMath dataset. Physical training batch settings can differ; they do not change the configured token batch.

Further diffusion/routing comparisons:

- Llama-1B legacy diffusion → R03: hidden 672 → 640, intermediate 2720 → 2936, processor attention/KV heads 3 → 5, selected cache heads 3 → 128. New expected adapter count 22,018,048. The current paper cell is blank; this does not replace an existing score.
- Qwen-2B legacy diffusion → R09: selected attention heads 3 → 12 and selected linear heads 72 → 288; other saved model fields match. The current paper off scores already come from the new configuration, so only on is added.
- The `head128` suffix denotes processor attention head dimension: 640/5 = 128 for Llama-1B and 896/7 = 128 for Llama-3B. It does **not** mean 128 selected cache heads for Llama-3B: its `topk_heads` is 224. The old Llama-1B `_superseded_head224` runs use a different processor shape and remain excluded.
- New Longshot expected adapter counts: Llama-1B 22,018,049; Llama-3B 42,301,057; Qwen-2B 22,131,841; Qwen-4B 53,527,681. The corresponding diffusion configs use four deterministic inference solver steps and scheduled four-step training (noise scale 0.1, horizon 50, refinement gate init -4). Longshot uses learned output scale; diffusion uses fixed-one output scale. Small Longshot checkpoints omit the explicit `output_scale_mode` field, which resolves to the existing `learned` default; this is not a changed checkpoint.

#### 27B variants: do not conflate the gates or replace existing rows

- **R17 Longshot static-scalar/rank1 vs the currently entered Longshot (vector):** `update_gate_type: contextual_vector → static_scalar`; `live_rank1_linear_writer: absent/false → true`; `out_proj_init: absent/normal → zero`; `output_scale_mode` is explicitly `learned` (same default as the old config). Both use the same 896-wide / 3584-intermediate / seven-layer processor, seven attention/KV heads, 16 selected attention heads and 576 selected linear heads, process every 64 tokens, topk 32, magnitude routing, and outer-product updates. Dataset, optimizer and training batch settings match. The existing Longshot (scalar) uses **contextual_scalar**, also distinct from static_scalar. Proposed: retain both contextual rows and add R17.
- **R18 fair-gate diffusion vs R17 fair-gate Longshot:** same static-scalar gate and live rank1 writer; switches `output_scale_mode: learned → fixed_one` and adds diffusion: four scheduled training / deterministic inference solver steps, time conditioning, eta/conditioning scale 1, noise 0.1, refinement gate init -4, horizon 50. Fill the currently blank diffusion row with an explicit static-scalar/rank1 label.
- **R19 Longshot (vector) + LoRA vs current Longshot (vector):** adds rank-32/alpha-32/dropout-0 LoRA on `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`; `trainable: cache_processor → lora_and_cache_processor`, `freeze_backbone: true → false`, `save_cache_processor_only: true → false`, physical batch 32 → 16. Base weights remain outside the selected LoRA+processor trainable set. This is a combined method, not a replacement for standalone LoRA or processor-only Longshot. Compared to the failed original **combined** config, its only execution-setting change is batch 32 → 16 (plus output/W&B name); total batch remains 65,536 tokens.
- **R20 contextual-vector diffusion rank1 retry vs failed original diffusion:** only model-setting change is `live_rank1_linear_writer: absent/false → true` (plus output/W&B identity). Both retain contextual-vector gating and zero output projection initialization, four scheduled/deterministic diffusion steps and the same training settings. Compared with R18, R20 uses contextual-vector instead of static-scalar gating; R18 explicitly fixes output scale to one. These are two distinct trained checkpoints. R20 off is complete; on is not yet reportable.
- The four new 27B configs all train from the same native backbone, seed 24, step 2000, sequence length 512, LR 1e-4, total batch 65,536 tokens, and OpenMathInstruct-2 rows 128000–255999. All four saved checkpoints were checked at step 2000. Retry labels do not indicate resumed training; restart notes specify fresh training.

### Repeated diffusion evaluations for logging — explicit source choice

Proposed: keep the original performance JSONs listed above. These extra evaluations reused the same step-2000 checkpoint and requested Paper7/mode settings with auto batching up to 64; no training config changed. The later evaluation processes used newer code and collected update metrics. Small score changes are observed; their cause has not been established by this audit. Do not attribute them to improved training or select individual cells from whichever run scores higher.

| Model / mode | Original → logging rerun scores (GSM8K, SVAMP, GSM-Hd, ThmQA, LogiQA, Gaokao) | Original → rerun avg.; mean tokens; total M |
| --- | --- | --- |
| llama32_1b off | 42.53, 59.67, 10.99, 12.25, 30.11, 27.92 → 42.61, 60.00, 11.22, 13.00, 30.11, 27.92 | 30.58; 426.6; 1.595 → 30.81; 427.7; 1.599 |
| qwen35_2b off | 67.32, 80.67, 39.27, 25.50, 28.57, 32.48 → 67.32, 80.67, 39.27, 25.50, 28.57, 32.48 | 45.64; 340.9; 1.274 → 45.64; 340.9; 1.274 |
| qwen35_2b on | 68.54, 78.33, 37.98, 24.88, 28.57, 32.48 → 68.54, 78.33, 37.98, 24.88, 28.57, 32.48 | 45.13; 1023.3; 3.825 → 45.13; 1023.3; 3.825 |
| llama32_3b off | 70.13, 83.67, 23.73, 21.50, 29.03, 29.06 → 70.43, 83.67, 23.81, 21.12, 29.03, 29.06 | 42.85; 308.0; 1.151 → 42.85; 306.1; 1.144 |

Logging rerun files are the corresponding original filenames with `_update_metrics` inserted immediately before `_thinking-...`. Llama-1B is under `A`; Qwen-2B off/on and Llama-3B are under `W`. Qwen-2B scores and displayed token figures match exactly; the Llama figures differ as shown. If the user prefers the newest logging evaluations as the performance sources, replace each entire evaluation consistently, including token columns; do not mix cells.

### Reconciliation and approval checklist

- [x] Approve the 20 row mappings and score sources above; retain original small diffusion performance evaluations.
- [x] Approve separate 27B static-scalar, contextual-vector diffusion and Longshot+LoRA labels/rows, keeping existing contextual-scalar/vector Longshot rows.
- [x] Approve the common five-task / three-generation-task 27B aggregation with native TheoremQA shown separately, and update the caption accordingly.
- [x] Once approved, recheck the two missing final JSONs before editing; report any newly completed mode for review rather than silently extending the approved list.

Already represented and retained: Llama-1B SFT/LoRA/BT/BGT, Llama-3B SFT and resized LoRA/BT/BGT, Qwen-2B SFT/LoRA/BT/BGT and Longshot/diffusion off, Qwen-4B SFT and resized BT off, and 27B SFT/LoRA/BT/BGT/contextual-scalar/contextual-vector Longshot off/on. No Llama thinking-on results, intentionally stopped seed 25, failed superseded 27B configs, old head224 Llama runs, or historical 32K-cap evaluations are proposed. All active restart-table configs and eight native-mode tasks are accounted for by proposed, retained, or still-incomplete entries.

### Final paper table selection (2026-09-20)

At the user's request, removed the exploratory average-excluding-TheoremQA
column and retained the original average definition. For 27B, retain only
static-scalar/rank1 Longshot (R17) and Diffusion Longshot (R18), alongside
pretrained instruct, SFT, LoRA, BT, and BGT. Removed contextual-scalar
Longshot, contextual-vector Longshot, Longshot+LoRA (R19), and contextual-vector
rank1 diffusion (R20) from the paper table. The earlier review remains a
historical record; its recommendation to include those variants is superseded.
Result files and configs are retained.


## KOR-Bench evaluation: paper-table checkpoint inventory

Audited **2026-09-21 13:26 UTC** against all **35 paper-table rows** in
`src/Global-Cache-Workspace-Theory-ICLR-2027-/sections/5_Results_and_Discussion.tex`.
**All 35 are available: 30 fully imported trained checkpoints (20 Martin, 10 Fred),
plus five verified local native backbones. None are missing or partially imported.**

This supersedes the September 20 inventory of 15 available, one incomplete, and
19 missing. Newly verified imports needing KOR-Bench evaluation are the ten Fred
checkpoints, Qwen3.5-2B Diffusion-Longshot, and Qwen3.5-27B BT, BGT, fair-gate
Longshot, and fair-gate Diffusion-Longshot: **15 newly available trained checkpoints**.
The existing evaluation queue and its completed results are tracked in the next
section; availability here does not mean that an evaluation must be repeated.
Qwen3.5-27B SFT thinking off remains excluded as already evaluated elsewhere.

### Import verification

- Fred copy job **27037** completed successfully at **2026-09-21 10:29:45 UTC**
  (`slurm_out/copy-fred-nb-27037.log`). Its script uses `set -euo pipefail` and
  prints the completion marker only after rsync succeeds.
- Martin follow-up copy job **27070** completed successfully at
  **2026-09-21 13:03:56 UTC** (`slurm_out/copy-martin-nb-27070.log`).
- All 30 trained checkpoints have parseable config/training-state JSON and
  recorded step **2000**. All checkpoint safetensors headers and contiguous data
  offsets account for the entire file size; no transfer temporary files remain
  inside these checkpoint directories.
- All 20 Martin checkpoints match every expected file size in the three copied
  migration manifests under `martin_nb/data/migrations/` (`20260919T120745Z`,
  `20260920T171050Z`, `copy-20260920T205432Z`). Every supplied SHA256 was verified,
  including the four newly imported 27B weight files. Files without a supplied
  checksum were checked by size and safetensors boundaries, not a full weight hash.
- Fred availability is established by successful rsync completion plus local
  config/state/weight checks; no source checksum comparison is claimed.
- The five native backbones are under `/mnt/jfs-01/shared/gcw/checkpoints/hf_native/`,
  outside the two import roots. Their configs and safetensors boundaries were
  rechecked. Local tokenizer/backbone mappings remain in
  `slurm_out/korbench-v3-20260920/backbones.json`.

Exact table variants are retained: Llama Longshot/diffusion use `head128`, resized
3B/4B adapters use `pct015`, and 27B Longshot/diffusion use `fair_gate_rank1`.
Older fixed-size adapters and contextual-vector/scalar alternatives are not substitutes.

### Available checkpoint paths

“Needs evaluation” identifies the 15 newly available checkpoints only. Other rows
retain the existing campaign/result tracking; this audit does not enqueue work.

| KOR-Bench task | Model | Method | Checkpoint path | Import status |
| --- | --- | --- | --- | --- |
| Existing campaign/results | Llama 3.2-1B | Pretrained instruct | `/mnt/jfs-01/shared/gcw/checkpoints/hf_native/llama32_1b/tf_base` | Local native backbone verified |
| [ ] Needs evaluation | Llama 3.2-1B | SFT | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/llama32_1b-sft-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| [ ] Needs evaluation | Llama 3.2-1B | LoRA | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/llama32_1b-lora-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| [ ] Needs evaluation | Llama 3.2-1B | BT | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/llama32_1b-bt-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| [ ] Needs evaluation | Llama 3.2-1B | BGT | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/llama32_1b-bgt-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| Existing campaign/results | Llama 3.2-1B | Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/llama32_1b_fair_campaign_v1_longshot_adapter22m_seed24_head128/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Llama 3.2-1B | Diffusion-Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/llama32_1b_fair_campaign_v1_diffusion_adapter22m_seed24_head128/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Llama 3.2-3B | Pretrained instruct | `/mnt/jfs-01/shared/gcw/checkpoints/hf_native/llama32_3b/tf_base` | Local native backbone verified |
| [ ] Needs evaluation | Llama 3.2-3B | SFT | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/llama32_3b-sft-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| Existing campaign/results | Llama 3.2-3B | LoRA | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/llama32_3b_fair_campaign_v1_lora_pct015_seed24/checkpoint` | Fully imported; step 2000; 6/6 manifest files verified |
| Existing campaign/results | Llama 3.2-3B | BT | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/llama32_3b_fair_campaign_v1_bt_pct015_seed24/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Llama 3.2-3B | BGT | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/llama32_3b_fair_campaign_v1_bgt_pct015_seed24/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Llama 3.2-3B | Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/llama32_3b_fair_campaign_v1_longshot_pct015_seed24_head128/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Llama 3.2-3B | Diffusion-Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/llama32_3b_fair_campaign_v1_diffusion_pct015_seed24_head128/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Qwen3.5-2B | Pretrained instruct | `/mnt/jfs-01/shared/gcw/checkpoints/hf_native/qwen35_2b/tf_base` | Local native backbone verified |
| [ ] Needs evaluation | Qwen3.5-2B | SFT | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/qwen35_2b-sft-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| [ ] Needs evaluation | Qwen3.5-2B | LoRA | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/qwen35_2b-lora-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| [ ] Needs evaluation | Qwen3.5-2B | BT | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/qwen35_2b-bt-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| [ ] Needs evaluation | Qwen3.5-2B | BGT | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/qwen35_2b-bgt-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| Existing campaign/results | Qwen3.5-2B | Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_2b_fair_campaign_v1_longshot_adapter22m_seed24/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| [ ] Needs evaluation | Qwen3.5-2B | Diffusion-Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_2b_fair_campaign_v1_diffusion_adapter22m_seed24/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Qwen3.5-4B | Pretrained instruct | `/mnt/jfs-01/shared/gcw/checkpoints/hf_native/qwen35_4b/tf_base` | Local native backbone verified |
| [ ] Needs evaluation | Qwen3.5-4B | SFT | `/mnt/jfs-01/shared/gcw/fred_nb/fair-campaign-v1/qwen35_4b-sft-seed24/checkpoint` | Fully imported; step 2000; successful copy and local checks verified |
| Existing campaign/results | Qwen3.5-4B | LoRA | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_4b_fair_campaign_v1_lora_pct015_seed24/checkpoint` | Fully imported; step 2000; 6/6 manifest files verified |
| Existing campaign/results | Qwen3.5-4B | BT | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_4b_fair_campaign_v1_bt_pct015_seed24/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Qwen3.5-4B | BGT | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_4b_fair_campaign_v1_bgt_pct015_seed24/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Qwen3.5-4B | Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_4b_fair_campaign_v1_longshot_pct015_seed24/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Qwen3.5-4B | Diffusion-Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_4b_fair_campaign_v1_diffusion_pct015_seed24/checkpoint` | Fully imported; step 2000; 7/7 manifest files verified |
| Existing campaign/results | Qwen3.5-27B | Pretrained instruct | `/mnt/jfs-01/shared/gcw/checkpoints/hf_native/qwen35_27b/tf_base` | Local native backbone verified |
| Existing campaign/results | Qwen3.5-27B | SFT | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_27b_instruct_sft_seed24/checkpoint-2000` | Fully imported; step 2000; 3/3 manifest files verified |
| Existing campaign/results | Qwen3.5-27B | LoRA | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_27b_instruct_lora_seed24/checkpoint-2000` | Fully imported; step 2000; 3/3 manifest files verified |
| [ ] Needs evaluation | Qwen3.5-27B | BT | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_27b_instruct_bt_seed24/checkpoint-2000` | Fully imported; step 2000; 3/3 manifest files verified |
| [ ] Needs evaluation | Qwen3.5-27B | BGT | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_27b_instruct_bgt_seed24/checkpoint-2000` | Fully imported; step 2000; 3/3 manifest files verified |
| [ ] Needs evaluation | Qwen3.5-27B | Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_27b_instruct_longshot_fair_gate_rank1_seed24/checkpoint-2000` | Fully imported; step 2000; 3/3 manifest files verified |
| [ ] Needs evaluation | Qwen3.5-27B | Diffusion-Longshot | `/mnt/jfs-01/shared/gcw/martin_nb/runs/experiments/qwen35_27b_instruct_diffusion_longshot_fair_gate_rank1_seed24/checkpoint-2000` | Fully imported; step 2000; 3/3 manifest files verified |

### Missing or incomplete imports

None among the 35 paper-table checkpoints.

## KOR-Bench v3 array: approved queue and native backbones

Status: On 2026-09-21, transferred all 20 unstarted valid evaluations to **array 27071**, submitted through `scripts/cluster/submit_array.sh` with four workers and one GPU each. Old workers 0/2 exited after queue closure (cancellation was also requested); workers 1/3 retain only their active Qwen4B LoRA/native thinking-on evaluations and exit afterward. The old counter is 38, preventing further claims. Completed and active evaluations are excluded from the new manifest. Two new workers started immediately; two were pending `AssocGrpGRES` at verification. Order and separate adjacent thinking modes are preserved; v3, 2048/4096 token caps, local dependencies, and gpu/eval/qwen35 extras with --no-sync are retained. Manifest and receipt: `slurm_out/korbench-v3-20260920/submit-array-migration/`.

User-approved scope: the 14 previously selected, fully imported trained checkpoints plus five native instruct backbones, with a later queue override made before the affected task was claimed: **skip Qwen3.5-27B LoRA thinking off and run Qwen3.5-27B SFT thinking on instead**. No additional newly imported trained checkpoints are added to this queue.

Llama 3.2 uses thinking off only. Completed invalid Llama thinking-on results remain excluded. The former custom queue and its terminal guard rows are retired; pending work now belongs exclusively to array 27071.

Protocol: `gcw.korbench.zero_shot.v3`, full 1,250-question suite, chat formatting, automatic batching up to 64, and sample logging. Maximum new tokens: **2048 off / 4096 on**, including reasoning and final answer. Runtime: `uv run --locked --no-sync --extra gpu --extra eval --extra qwen35`; conversion additionally selects `--extra convert`. Evaluation partition `agent-xlong` is explicit because the cluster has no default partition; its maximum/default job limit is five days. Each worker receives eight CPUs and 250000 MiB via the partition GPU defaults. Preparation used `agent-long`.

Native backbones live in `/mnt/jfs-01/shared/gcw/checkpoints/hf_native/<model>/tf_base`. Existing Qwen-2B/4B models are reused; Llama-1B/3B and Qwen-27B were downloaded/converted from Hugging Face and verified complete before submission. Llama revisions are pinned to the imported checkpoints’ native sources. The original imported checkpoint files are preserved; evaluation views remap only backbone/tokenizer locations. Code and task definitions are snapshotted for this run.

Run artifacts: `slurm_out/korbench-v3-20260920/`. `queue-order.json` records task order, `backbones.json` records native sources, `checkpoint-views.json` maps evaluation paths to original checkpoints, and `code-sha256.json` identifies the code snapshot. Final results and incremental sample logs go under its `results/` directory.

| Live tasks | Model | Method |
| --- | --- | --- |
| 0 (off only; 1 invalid) | llama32_3b | lora |
| 2 (off only; 3 invalid) | llama32_3b | pretrained_instruct |
| 4 (off only; 5 invalid) | llama32_3b | longshot |
| 6 (off only; 7 invalid) | llama32_3b | diffusion |
| 8, 9 | qwen35_4b | lora |
| 10, 11 | qwen35_4b | pretrained_instruct |
| 12, 13 | qwen35_4b | longshot |
| 14, 15 | qwen35_4b | diffusion |
| 16 (off only) | llama32_3b | bt |
| 17 (off only) | llama32_3b | bgt |
| 18, 19 | qwen35_4b | bt |
| 20, 21 | qwen35_4b | bgt |
| 22 (off only) | llama32_1b | pretrained_instruct |
| 23 (off only) | llama32_1b | longshot |
| 24 (off only) | llama32_1b | diffusion |
| 25, 26 | qwen35_2b | pretrained_instruct |
| 27, 28 | qwen35_2b | longshot |
| 29 (on only) | qwen35_27b | sft |
| 30 (on only) | qwen35_27b | lora |
| 31, 32 | qwen35_27b | pretrained_instruct |


### Updated live KOR-Bench queue: 2026-09-21

Exclude Qwen3.5-27B native, Longshot, and Diffusion-Longshot thinking OFF:
these are already reported in the paper KOR-Bench table (averages 69.12,
49.84, and 46.96). Native OFF was removed from the live array 27071 manifest
under its queue lock after verifying it was unclaimed; the other two OFF runs
were only proposed and were never added. Running evaluations remain untouched.

Applied to live array **27071** at **2026-09-21 13:47 UTC**: 40 unstarted evaluations, with 15 newly added checkpoint views using the existing local backbone/tokenizer mappings. Updated `configs.txt` atomically under `queue.lock`, preserving both claimed entries and all running jobs. No array restart or resubmission.

| Positions | Backbone | Ordered remaining evaluations |
| --- | --- | --- |
| 1–3 | Llama-3B | SFT OFF; BT OFF; BGT OFF |
| 4–10 | Qwen-4B | SFT OFF/ON; diffusion ON; BT OFF/ON; BGT OFF/ON |
| 11–17 | Llama-1B | Native; SFT; LoRA; Longshot; diffusion; BT; BGT — all OFF |
| 18–31 | Qwen-2B | Native; SFT; LoRA; Longshot; diffusion; BT; BGT — each OFF/ON |
| 32–40 | Qwen-27B | SFT ON; LoRA ON; native ON; Longshot ON; diffusion ON; BT OFF/ON; BGT OFF/ON |

OFF/ON means separate adjacent queue entries, without hard dependencies.
The additions and backbone regrouping above are now live. The previous manifest, pending task list, and update receipt are saved under `slurm_out/korbench-v3-20260920/submit-array-migration/expanded-queue-20260921/`.
Retain v3, 2048/4096 OFF/ON token caps, local dependencies and existing extras.


### Additional shared-queue workers: 2026-09-21

Added `submit_array.sh --attach-eval-queue STATE_DIR` for Slurm evaluation queues;
24 launcher tests pass. It reuses the existing task list, claim records and lock.
The attempted four-worker addition was rejected with `QOSMaxSubmitJobPerUserLimit`:
normal QoS has MaxSubmitJobsPU=6, with five jobs already running. Submitted one
additional 1-GPU worker as **27077**, sharing array 27071's queue and evaluation
environment. Three further workers remain blocked by the submitted-job limit.
Request a limit of at least 9 to accommodate eight shared-queue workers plus the
old live Qwen-4B native ON worker temporarily (8 after that worker exits).
No existing jobs were stopped or restarted. Submission wrapper and receipts:
`slurm_out/korbench-v3-20260920/submit-array-migration/add-workers.sh`,
`add-workers-submission.log`, and `add-one-worker-submission.log`.
