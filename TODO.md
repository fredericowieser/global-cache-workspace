# TODO list 

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
