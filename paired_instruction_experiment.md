# Matched-seed instruction experiment

Implementation includes preparation, matched-seed execution, anonymous media export, blinded
review orchestration and paired descriptive aggregation. Do not interpret unit tests or process
startup as evidence of instruction effectiveness; inspect each task's summary and diagnostics.

## Pilot execution

Run `RoboLab/.venv/bin/python run_paired_pilot.py --session diagnostics/NEW_PAIRED_SESSION`.
This freezes the three historical checkpoint configurations before launching any inference.
Dedicated model ports are 18960, 18961 and 18964 on physical GPUs 0, 1 and 4. Existing model
services and training are not stopped. Check available capacity and ports before launching.
Each task runs two 12-candidate pools sequentially; tasks run concurrently. Fruits and organic
run A then B; tools runs B then A. This is partial order counterbalancing, not complete balance.
Each task's first arm also repeats its first exact request once; the repeat action is not executed.
There are 72 executed candidates and three extra diagnostic inference requests.

User-authorized consistency policy: small numerical and image differences do not block runs.
`consistency.json` records initial-scene maximum absolute differences and per-pair input-image MAE;
the first candidate's `repeatability.json` records same-input/seed raw-action differences.
No numeric equality threshold is used. Checkpoint identity, seed acknowledgment, finite shapes,
camera validity and executed-action integrity checks remain mandatory.

After each task completes both pools, its dedicated model exits and a fresh gpt-6-astra/low
reviewer receives only its anonymous media directory. It must write all24 scored reviews before
the orchestrator unblinds and writes summary.json. Best-of-four is a secondary score-ranked
analysis, prioritizing constraint compliance; groups with uncertain scores/safety are unresolved.
Failures are recorded in status.json and are not automatically rerun. Task completion is tracked
in completion.json; SAS backup can use sync_instruction_results.py on the session directory.

## Frozen design

Pilot checkpoints: the last retained fruits step80, tools step48 and organic step112 checkpoints
in diagnostics/multi_env_six_20260918T065328Z. No failed-pool endpoint selection.
For each checkpoint, A repeats one baseline full prompt twelve times; B repeats one preregistered
alternative twelve times. The twelve seeds match across arms, with sixteen executed steps each.
Keep the target, completion criteria and avoidance constraints unchanged. Total: 72 candidates.
Do not chain selected endpoints, adapt prompts after outcomes, or add retries for bad task outcomes.

Prepare arm files with paired_instruction_experiment.py prepare --task TASK --checkpoint CHECKPOINT
--instructions-a A.json --instructions-b B.json --prompt-a A.txt --prompt-b B.txt
--seeds seeds.json --output NEW_DIRECTORY. Each instruction file contains twelve identical objects;
prompt files contain the exact full model prompts, including overall task and constraints.
Freeze and externally record the SHA256 of both resulting arm files before inference.

Run sampling_multi_env.py with its usual task/checkpoint/instructions/output flags and
--experiment-arm NEW_DIRECTORY/arm_A.json (or arm_B.json). Use separate output directories and
--execute-steps 16. Do not reuse initialization-only example prompts for real inference.
The service must be restarted from the modified source under an explicitly scoped experiment
process: existing running services do not acquire code changes automatically. An old service
without matching actual_seed fails before candidate action execution.

## Required gates before the pilot

- Record code/model identity and frozen prompt/checkpoint/seed manifest hashes.
- Repeat identical actual model input and seed; measure raw action equality and explicitly
  report any nondeterminism. Matching seeds alone do not prove deterministic inference.
- Compare restored A/B initial states and model input images diagnostically;
  report camera rendering noise and cross-environment variation as limitations, not rejection gates.
- Confirm server seed receipts, raw 32x8 action shapes and HDF5 equality of executed prefixes.
- Counterbalance arm order across tasks and retain every trial, including failures.
- Keep old experiment files immutable and use the existing SAS completion-marker backup.

## Evaluation

Anonymous review packages must hide condition, instruction and seed, while exposing the same
target/goal and actual start/end views. Use gpt-6-astra low without the prompt-author history;
retain uncertain outcomes, with user adjudication for ambiguous contact or grasp evidence.
Freeze reviews before unblinding. Primary outcome: per-seed improved/tied/worse, and unsafe
contact separately, reported by task. Twelve samples of one checkpoint are not twelve tasks.
Secondary exploratory best-of4 uses three fixed disjoint groups (0..3, 4..7, 8..11), with
identical budgets and blind selection on both sides. Do not claim full-task success from alignment.

Export verified anonymous media using paired_instruction_experiment.py blind --arms RUN_A RUN_B
--output NEW_BLIND_DIRECTORY --mapping PRIVATE_MAPPING.json --protocol COMMON_PROTOCOL.txt.
The mapping must stay outside the review directory and must not be given to the reviewer.
Protocol text is supplied by the experiment owner and must not reveal condition or instructions.
The exporter validates completed pools, seed receipts and media hashes, and exports no original
result/selection files. Review sheets with source labels are intentionally omitted.

This tests one fixed instruction rewrite, not adaptive twelve-instruction search. It uses one
automated blinded reviewer per task; independent human or second-reviewer confirmation remains
necessary before treating ambiguous visual evidence as a strong result.