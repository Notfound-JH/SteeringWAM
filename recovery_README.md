# Live WAM + Coding Agent recovery

For trajectory ingestion, field meanings, time-step alignment and real examples,
see [Recovery trajectory data contract](recovery_DATA_FORMAT.md).

## Hammer task scope: all four hammers

The user's complete hammer task is **take all four hammers out of the bin and
place them on the table**. Use
[`ToolsPickingAllHammersTask`](RoboLab/robolab/tasks/benchmark/tools_picking_all_hammers.py),
whose overall instruction is `Take out all the hammers and put them on the table`.
Its native success predicate requires **all** of `husky_hammer`, `blue_hammer`,
`red_hammer`, and `wood_hammer` to satisfy the existing outside-bin, table-support,
and gripper-detached checks.

The current Condition is a subgoal, not a replacement for this overall task.
There is no required blue-first order: choosing another unfinished hammer can
advance the full task. Reviews should consider completed/remaining hammers and
actual acquisition, extraction, and placement progress. Placing one hammer is
partial progress; only all four passing the native predicate is task success.
Do not reset the environment or replenish the episode's correction budget
between hammers.

`ToolsPickingHammerTask` is a **single-blue-hammer diagnostic**, not the full
four-hammer task. In particular, the three `recovery_hammer_condition_*_seed3407_20261015`
episodes used that single-object task and cannot be presented as full-task
comparisons. Preserve their original prompts, native checker, and recordings.
The 600-step override in examples below remains an experiment budget; hitting
that limit without all four placements is a timeout, not completion.

## Research contract: task-progress-first recovery (2026-10-15)

The objective is **completion of the long-horizon task**, not the accumulation of
locally successful corrections. These requirements are implemented by the opt-in
`wam-recovery-v3` runtime (`TASK_PROGRESS_RECOVERY=1`). The default V2 runtime
remains unchanged: it runs 16 WAM steps after every correction before accepting
a local assessment. Historical episodes retain their original semantics.

### Condition priority and unknown decision points

The working hypothesis is that Condition primarily influences task-level choices
at important decision points: which object to approach, which subgoal to pursue,
and when to transition between stages. Those points are task-dependent and are
not assumed to occur only in the first chunk. This is a hypothesis to investigate,
not a result established by the existing hammer episodes.

At WAM chunk boundaries, first compare actual behavior with the overall goal and
the expected task stage. Wrong target selection, incorrect stage/order, or an
off-track task plan should prioritize **Condition over Action**. Do not substitute
increasingly detailed gripper-language instructions for this task-level diagnosis.
Action remains appropriate when the target and task stage are correct but physical
execution prevents progress. Its proposal must explain why changing Condition
alone does not address the observed problem. Priority is a decision rule, not a
requirement to waste a Condition trial on every clearly physical failure.

### Separate correction and recovery-judgment roles

- The **correction Agent** diagnoses the WAM failure against the task goal,
  identifies the recovery objective, and proposes Condition, Action, or Joint.
- The **recovery-judgment Agent** evaluates the actual response. A correction
  proposal is not its own success verdict. Judge inputs must include the overall
  task, expected/observed stage, target object, relevant history, correction intent,
  and before/after actual observations.
- Record the two roles and their actual model/caller identities when available.
  Separate role names alone do not establish independent model evaluation.

| Mode | Judgment boundary | Required question |
|---|---|---|
| Condition | After the next executed WAM chunk under the revised condition | Did actual WAM behavior return to the correct target and task progression? |
| Action | Immediately after the bounded controller program, before any further WAM action | What physical effect actually occurred, and does it enable the intended next task stage? |
| Joint | Preserve both applicable boundaries; explicitly record the stage order | What was the Action effect, and what was the subsequent/preceding Condition-driven WAM effect? |

In particular, do not execute an extra WAM chunk before recording the Action
effect: otherwise a later WAM release can be confused with failed controller
execution. Predictions and command compliance are supporting signals, not proof
of the actual effect.

### Evaluate task progress, not only local motion

Keep these outcomes separate: correction execution/effect, task-progress
alignment or restoration, readiness for WAM continuation, and final native task
success. Reaching a Cartesian target, issuing close commands, or holding the wrong
object cannot by itself establish recovery. Conversely, a valid recovery can
restore a prerequisite for the next stage without completing the whole task.
Evidence that is insufficient should remain uncertain, not become success.

A positive recovery judgment permits continuation; it is not final task success.
Continue monitoring subsequent WAM chunks for renewed task-progress deviation.
Preserve failures, the no-reset rule, frozen WAM parameters, and the configured
attempt/horizon/fallback policy. Do not relabel historical V1/V2 assessments as
task-progress judgments or independent-judge results.

### Running V3

```bash
MODEL_GPU=5 SIM_GPU=4 TASK_PROGRESS_RECOVERY=1 \
RECOVERY_MAX_ATTEMPTS=5 CONTINUE_WAM_AFTER_BUDGET=1 RECOVERY_MAX_STEPS=600 \
bash launch_recovery_experiment.sh diagnostics/NEW_HAMMER_RUN ToolsPickingAllHammersTask
```

Add the existing `RECOVERY_GENESIS` and `RECOVERY_GENESIS_SHA256` variables when
reproducing a saved initialization. Initialization does not permit later resets.

Use `recovery_client.py --socket SOCKET --output RESPONSE.json OPERATION ...`:

1. `wam --condition "..." --subgoal "..." --chunks 1` executes one chunk and
   stops at `awaiting_progress_check`. `review_progress REVIEW.json` records the
   correction Agent's structured task-stage diagnosis. A negative diagnosis
   enters recovery; a positive diagnosis permits the next chunk.
2. `recover PROPOSAL.json` consumes one attempt. A Condition attempt executes
   one 16-step WAM chunk; Action executes only the bounded controller program.
   Each stops at `awaiting_assessment`.
3. Give a **separate read-only Agent context** the current `context` response,
   before/after images, task goal, intent, measured state and relevant history.
   Save that context's actual ID and output; send its judgment with
   `assess ASSESSMENT.json`. The runtime checks distinct caller IDs/roles, but
   those fields alone are not proof that an independent model was invoked.
4. Joint currently uses **Action → judgment → Condition/WAM → judgment**.
   Only aligned target/stage diagnoses can select Action or this Joint order.
   Wrong/uncertain target or stage must first use Condition. A positive Joint
   Action judgment enters `ready_condition_evaluation`; explicitly call
   `evaluate_condition`, then obtain the second judgment. A failed Action stage
   ends that attempt without contaminating its effect with WAM actions.
5. Positive `recovery_success` permits WAM continuation, not final success.
   Ordinary chunks still require progress review. Failed fifth attempts, or a
   new failure after a successful fifth attempt, use `wam_after_budget` when
   fallback is enabled. No sixth attempt or unbudgeted recovery rewrite is
   allowed. Native termination interrupts execution without padding/reset.

V3 decisions use `agent_role` (`correction_agent` or `recovery_judge`), `agent_id`,
and `task_progress`: `target_object`, `expected_stage`, `observed_stage`,
`target_alignment`/`stage_alignment` (`correct`, `wrong`, `uncertain`),
`progress` (`advancing`, `stalled`, `regressing`, `uncertain`) and `evidence`.
Reviews add boolean `needs_recovery`; negative reviews add `failure_type`.
Proposals retain V2 intent/code fields and add `condition_reason`.
Judgments identify `stage` (`action_effect` or `condition_progress`), separate
`correction_effect` (`achieved`, `partial`, `failed`, `uncertain`) from boolean
`recovery_success`, and require `continuation_reason` and `evidence`.
Uncertain effect/alignment/progress cannot authorize successful recovery.

In V3, native success during the Condition-driven WAM chunk can satisfy the
full chain after its positive independent judgment; no redundant WAM actions are
executed after a native terminal goal. Controller-only completion and successful
WAM fallback after failed recovery are not labeled accepted recovery chains.
Use `verify_recovery_episode.py RUN/recovery` to verify the versioned contract.

## Pure-WAM instruction comparisons

For non-intervening instruction comparisons, the existing V2 recording worker can
be used with only `wam`, `status`/`context`, and `finalize` operations. Do not call
`mark_failure`, `recover`, or `control`: annotate errors offline without changing
the trajectory. Record `study_episode.json` with
`control_policy="frozen_wam_only_no_recovery"`; the legacy recording container
still calls its interface `experiment_mode="wam_recovery"`. Verify that every
actual action source is `wam`, the attempt count is zero, and compare native
task success rather than the Recovery-only `acceptance_success` field.

The model launcher accepts `SEED=3407 DETERMINISTIC_SEED=1` to use seed 3407 for
**every inference request**, allowing a shared frozen policy server without
request interleaving changing the generation-seed sequence. The default remains
`DETERMINISTIC_SEED=0`, which advances a NumPy RNG initialized from `SEED`.
Keep the simulator seed, historical genesis, goal checker, horizon and sampler
settings fixed too. Matching seeds does not guarantee bitwise GPU determinism.
One run per instruction gives descriptive trajectories, not statistical failure
probabilities or proof that every difference is caused by language alone.

The [fixed-seed hammer comparison scripts](diagnostics/hammer_condition_comparison_20261015/)
extract measured trajectories, audit seeds/videos, build the visually reviewed
offline annotations, and verify selected Azure artifacts. They require the
matching archived study data and never issue robot-control requests.

## Read-only experiment dashboard

Start the existing dashboard server without launching a model or simulator:

```bash
./RoboLab/.venv/bin/python rollout_dashboard/server.py --host 127.0.0.1 --port 8767
```

On the server, open **http://127.0.0.1:8767/recovery**. From another computer,
use the forwarded address rather than assuming its localhost is the server.

For **VS Code Remote Tunnels** (including vscode.dev) or Remote SSH:

1. In the VS Code window connected to this repository, run **Ports: Focus on
   Ports View** from the Command Palette.
2. Select **Forward a Port**, then enter **8767**. Connecting to the remote
   machine's tunnel does not itself forward the web application's port.
3. Open or copy the **Forwarded Address** shown for that port, and append
   `/recovery` to its origin. Use the actual address VS Code supplies; do not
   append the path to the `vscode.dev/tunnel/...` editor URL.
4. If the forwarded address is HTTPS, keep HTTPS. For a private tunnel, sign in
   with the account that owns the tunnel; this is separate from `gh` Git
   authentication on the server.
5. Keep port visibility **Private**. There is no need to change the server's
   bind address to `0.0.0.0` or expose the port publicly.

For desktop VS Code versions with the integrated browser's remote proxy support,
use these workspace settings:

```json
{
  "workbench.browser.enableRemoteProxy": true,
  "workbench.browser.dataStorage": "workspace",
  "workbench.browser.openLocalhostLinks": true
}
```

Run **Browser: Open Integrated Browser** and enter
`http://127.0.0.1:8767/recovery`. Open a new browser tab after changing the
settings. The remote indicator in its address bar confirms it reaches the remote
machine's localhost over the existing VS Code connection, without a separate
forwarded port or the standalone public Dev Tunnel URL. Do not use an external
browser or the older **Simple Browser** command for this route. Globally scoped
browser storage and `file://` URLs are not remotely proxied. This still requires
the existing VS Code remote connection to be healthy.

The dashboard has no authentication and is not intended to be exposed directly
to the public internet. Keep it bound to localhost.

### Standalone Microsoft Dev Tunnels

As an alternative to the VS Code Ports panel, use Microsoft's `devtunnel` CLI.
It is installed on this machine at `~/.local/bin/devtunnel`. Its login is separate
from both VS Code Remote Tunnels and `gh`:

```bash
~/.local/bin/devtunnel user login --github --use-device-code-auth
~/.local/bin/devtunnel host -p 8767 --protocol http
```

Approve the device code in your own browser. Keep the existing dashboard server
running while hosting the tunnel. Open the HTTPS address printed by `host`, with
`/recovery` appended. The browser must sign in as the account that owns this
tunnel. Do **not** add `--allow-anonymous`: private access is intentional.

The `host` process must remain running for the URL to work. Hosting a new tunnel
without an existing tunnel ID can produce a new address, so use its actual output
rather than a guessed or previously expired URL.

Current private deployment (2026-10-12):
**https://0kvl8z1c-8767.usw3.devtunnels.ms/recovery**.
Sign in as `Notfound-JH` through GitHub. Page, API, stylesheet and video range
requests were verified through an authenticated tunnel client; unauthenticated
HTTPS requests redirect to GitHub sign-in.

If hosting stops, keep the dashboard running on port 8767 and resume the existing
tunnel rather than creating a different address:

```bash
~/.local/bin/devtunnel host peaceful-cat-0kljz4n.usw3
```

The address requires an active host process and an unexpired tunnel. No
anonymous/public access grant was added.

**Availability update (2026-10-15):** the standalone tunnel account exhausted
its 5 GB bandwidth allowance. Authenticated requests return HTTP 429, although
the localhost page/API/video remain healthy. `devtunnel limits` reports a reset
on 2026-11-12. Restarting or creating another tunnel on that account does not
restore the user-level allowance. Do not expose the unauthenticated dashboard
publicly or publish the existing container-wide write-capable Azure SAS.

### Offline review when forwarding is unavailable

Export a finalized episode into one self-contained HTML file:

```bash
./RoboLab/.venv/bin/python -m rollout_dashboard.offline \
  diagnostics/recovery_blue_hammer_v2_20261015T0824 \
  --output diagnostics/recovery_blue_hammer_v2_20261015T0824/offline_review.html
```

Download the file through the remote file explorer and open the downloaded copy
in Chrome or Edge. No localhost forwarding, external assets, credentials or
HTTP server are needed. Existing output files are never overwritten.

The offline snapshot embeds the original actual MP4, execution phases, all
recorded correction proposals/code, evaluation windows, assessments, and
compressed wrist/left-shoulder before/after images. It preserves failed, unknown,
mock and Agent-only labels rather than upgrading them to verified success.
It is **not** a live dashboard or a replacement for the full archive: predicted
futures, all per-step observations and raw trajectories stay in the source
episode. The export validates the finalized event-log checksum and does not
rewrite source recordings. Treat the resulting HTML as private experiment data.

### Online dashboard capabilities

The page uses a viewport-height workspace with one visible tab at a time, not a
vertical stack of sections. The default tab shows only the static execution axis
and error callouts. Instructions/events, video, per-attempt detail, predictions
and downloads have separate tabs. Expanded details scroll within their own tab,
not the whole page. No media is requested until its viewer is selected.
Long evidence and Agent reasons show excerpts; expanding them reveals the
complete original text. Versioned asset URLs prevent mixing old cached scripts
with this layout.

When opened, playback defaults to a low-bitrate, maximum-640px-wide H.264 preview with MP4
metadata at the front (`faststart`). Frame timing is preserved; the quality
selector can switch to the untouched original at the same playback position.
Original-video downloads and offline exports still use the original recording.
New recordings also put their MP4 metadata at the front.

Observation images use small JPEG previews. The optional state inspector updates
only when its control-step slider or camera is selected; video playback does not
update the timeline, descriptions or robot-state display. The original image
remains available through its link. Comparison images load lazily, and predicted
videos do not preload and compete with the main video.

Disposable previews are generated only for finalized recovery episodes and
cached under each run's `.dashboard_cache/`, outside the recorded `recovery/`
artifacts. Source path/size/mtime and the encoding profile identify cache entries.
HTTP byte ranges and private browser caching are supported; failed conversions
are explicit errors rather than silent success/fallback. Preview caches are not
training data and are excluded from the existing Azure backup's hidden paths.

The online page reads finalized episodes from `diagnostics/*/recovery/manifest.json`.
It provides:

- Full actual video and a step-aligned execution timeline with separate WAM,
  Agent-controller and WAM-instruction lanes. Every WAM block represents only
  its executed action prefix, not the full predicted chunk; instruction cards
  show the original condition, full model prompt and all associated step ranges.
  Recovery evaluation and legacy condition-only actions are WAM-generated,
  not Agent-controller steps.
- Every recorded failure/command error has its own step marker and a straight
  arrow to a description box. Failure boxes link the corresponding attempts,
  original Agent mode-selection reasons, diagnoses and assessments; later failures
  do not overwrite earlier ones. Failed assessments are not invented failure triggers.
  The axis is static: no moving playhead, automatic selection or video-seeking
  behavior on its blocks/markers. Video is an independent optional viewer.
  Narrow windows can scroll the axis horizontally. The event list remains expandable.
- Failure evidence, every recorded correction attempt, its original bounded code,
  before/after observations and measured Cartesian tracking error.
- Separate WAM input, predicted future, condition and actual execution length.
- Explicit local/native/full-chain outcomes, including unverified, mock and
  failed records; a native success label alone never earns verified acceptance.
- Agent-input provenance from `experiment.json`, with a prominent warning for
  `debug_ground_truth_assisted` episodes and an explicit unknown label when the
  input mode or exact Agent model version was not recorded.
- Allowlisted downloads of Manifest, JSONL, HDF5, verification and video.

The page does not send robot commands or execute displayed code. It can generate
disposable viewing previews, but never rewrites episode recordings or opens the simulator. Corrupt event
logs and missing files are visible errors rather than success-shaped defaults.
Existing rollout and sampling pages remain available; the old home page links
to Recovery Lab. A specific episode can be bookmarked with
`/recovery?episode=recovery_cube_20261011`.

Focused API tests:

```bash
./RoboLab/.venv/bin/python -m pytest test_recovery_dashboard.py -q
```

## Per-run attempt budget, failed-budget handoff and horizon

Legacy defaults remain three attempts, stop after the last failed attempt, and
the task's original native horizon. For five attempts, WAM fallback after budget
exhaustion, and a 600-control-step task:

```bash
RECOVERY_MAX_ATTEMPTS=5 CONTINUE_WAM_AFTER_BUDGET=1 RECOVERY_MAX_STEPS=600 \
  MODEL_GPU=5 SIM_GPU=4 \
  bash launch_recovery_experiment.sh diagnostics/NEW_HAMMER_RUN ToolsPickingAllHammersTask
```

These map to `--recovery-max-attempts`, `--continue-wam-after-budget`, and
`--recovery-max-steps` on the simulator worker launcher. The horizon is applied
to the native environment configuration before construction, not a shortened
video or a post-hoc label. Native task success may end the run earlier.

After the fifth negative assessment the worker returns to `ready_wam`, retains
`local_recovery_success=false`, and emits `control_returned_to_wam_after_budget`.
Subsequent ordinary `recovery_client.py wam` calls run with source
`wam_after_budget`; no sixth correction is allowed. The external driver must
continue issuing WAM chunks until native termination. This does not claim
successful Recovery, even if WAM later reaches native task success.

## Verified first episode: 2026-10-11

The real Isaac Sim episode
[recovery_cube_20261011](diagnostics/recovery_cube_20261011/) passed the full-chain
verifier. This is real simulator stepping, **not hardware** and not a mock.

| Phase | Actual control steps | Observed outcome |
|---|---:|---|
| Frozen WAM | 1-96 | Fingers closed above cube; cube stayed on table |
| Agent Action recovery | 97-116 | Reopened, moved flange forward/down to an open pre-grasp pose |
| WAM continuation | 117-212 | Grasped and lifted cube, transported it and released it inside bowl |

Native `RubiksCubeTask` success and full-chain `acceptance_success` are both
true; no reset or state restoration occurred. Only **one** correction attempt
was needed. The Agent did not pick/place the cube itself.

Correction target `[0.475, -0.065, 0.195]` m versus measured
`[0.47323275, -0.06489056, 0.19715637]` m: final tracking error **2.79 mm**.
The complete episode contains 212 actual actions, 12 WAM predictions, 96 WAM
continuation steps, and 213 wrist-video frames at 15 Hz. HDF5 has 212 action,
robot-state and EE-pose records, with native `success=True`. Each WAM executed
prefix was checked against its prediction; every actual command was checked
against HDF5.

Useful artifacts:
[verification](diagnostics/recovery_cube_20261011/recovery/verification.json),
[manifest](diagnostics/recovery_cube_20261011/recovery/manifest.json),
[full video](diagnostics/recovery_cube_20261011/recovery/episode_wrist.mp4),
[failure](diagnostics/recovery_cube_20261011/failure_01.json),
[executed proposal](diagnostics/recovery_cube_20261011/proposal_01.json),
[assessment](diagnostics/recovery_cube_20261011/assessment_01.json).

The successful schedule used seed 3407 and 16 executed steps per WAM chunk:

1. Six chunks: `Pick up the rubiks cube and put it in the bowl.`
2. Inspect the real missed grasp; execute the saved Action proposal: reopen for
   4 steps, move to the target over 12 steps, hold for 4 steps.
3. Inspect feedback; hand control back after the saved visual assessment.
4. Three chunks: `Grasp the cube with the open gripper, lift it, and place it in the red bowl.`
5. Three chunks: `Place the grasped rubiks cube inside the red bowl and release it.`

Do not reuse the saved request IDs in a new episode. Read its current token and
images, and diagnose its actual state; the saved assessments are not automatic
success labels for future runs. This is one verified success, not a success-rate
estimate or a claim of bitwise reproducibility. Condition/Joint interfaces are
implemented and protocol-tested, but this full real episode used Action mode.

The original probe's generic `result.json` in this first episode still says
`frame_captured`; it was not rewritten retroactively. Use the recovery manifest
and independent verification as authority. The launcher has since been corrected
to report the recovery manifest and native/acceptance results explicitly.

This is a simulator experiment, not a physical-robot safety controller. It keeps
one RoboLab environment alive. There is no reset, state restore, replay, shell,
Python `exec`, or simulator handle in the recovery tool API. WAM training is not
part of this experiment.

## Six historical tasks: 2026-10-12 diagnostic batch

All six tasks were run through the live recovery pipeline, one new episode per
historical task. **None reached its complete native task goal in this batch.**
The previous successful cube-in-bowl episode remains unchanged and is not
included in these six results.

| Task | Actual steps | WAM continuation steps | Observed partial progress / final problem |
|---|---:|---:|---|
| `tools` | 458 | 160 | Clearer hammer approach, but final closure was empty; no hammer placed |
| `fruits` | 1040 | 768 | WAM placed two lemons on the displaced plate; third-fruit grasp stalled |
| `blocks` | 1030 | 752 | WAM placed blue/red blocks; final approach to green block failed |
| `black_items` | 740 | 448 | Mouse grasp locally recovered, then dropped before reaching the bin |
| `organic` | 514 | 160 | Orange contact was lost; later grasp attempts failed |
| `toys` | 1926 | 1280 | WAM placed all three cubes and blue/red blocks; lizard grasp repeatedly slipped |

Each episode used all three allowed correction attempts. After the budget was
exhausted, an unresolved failure ended the episode; no reset, fourth correction,
or replacement episode was used. These are early stops on unresolved failure,
not native timeouts. Every episode has `native_success=false` and
`acceptance_success=false`, while its independent **recording** verification
passed. Partial progress is not full-task success.

Totals: **5708 actual control steps**, 312 WAM predictions, 18 correction
attempts (5 Action, 13 Joint), and 3568 WAM continuation steps. Of the actual
steps, 716 are bounded controller corrections and 208 are corrected-condition
WAM prefixes. All successful and unsuccessful attempts are retained. The Agent
gave 13 positive and 5 negative local assessments; positive labels include
pre-grasp readiness or short grasp retention and are not native success labels.
Condition-only correction was not exercised in this batch.

**Methodological limitation:** this batch used actual camera images and robot
state **plus archived simulator object poses** for diagnosis, Cartesian targets,
and assessment. Its `experiment.json` files explicitly record
`agent_input_mode=debug_ground_truth_assisted`; the dashboard displays that
warning. It is a privileged-state-assisted debugging baseline, **not a pure
visual GPT benchmark**. The exact hosted Agent model/version was not
independently recorded. Local assessments remain Agent judgments. One episode
per task does not establish a generalization success rate.

Each initial state matched its original SHA-256-verified genesis exactly
(`max_state_error=0.0`, seed 3407). Restoration occurred only at episode step
zero. In fruits, an orange was initially already on the plate, but WAM later
displaced it and the plate; it is not counted as a new placement. Historical
short rollouts and these longer interactive episodes are not paired
success-rate comparisons.

Data directories:
`diagnostics/recovery_six_{tools,fruits,blocks,black_items,organic,toys}_20261012/`.
The [machine-readable batch results](diagnostics/recovery_six_20261012_services/results.json)
list every outcome and its provenance. Each episode contains the standard
JSONL, actual/predicted actions and video, HDF5, all proposals/assessments, and
independent verification. The shared policy log is in
[batch services](diagnostics/recovery_six_20261012_services/).

All episode files were synchronized to
`jinghe/steeringwam/recovery/recovery_six_<task>_20261012/` in the configured Azure
container. Metadata, genesis, manifest, verification, events, video and HDF5 were
downloaded from Azure and SHA-256-compared with the local files for every task.
The dashboard now lists these six recordings alongside the original success.

To start another episode from the same fruits scene, use a **new** output
directory (the launcher starts a frozen policy server unless `REUSE_POLICY=1`):

```bash
RECOVERY_GENESIS=/home/aiscuser/steeringwam-data/replay-dependencies/fruits/genesis.npz \
RECOVERY_GENESIS_SHA256=e3fe11d18035c730cf93511a092e17280c8ba463d0fe48b816afd11ff7d4831c \
SIM_GPU=6 \
bash launch_recovery_experiment.sh diagnostics/recovery_fruits_new FruitsOnPlate3Task
```

Use the printed socket and the interactive tools below. Inspect fresh
observations and tokens; do not replay old request IDs or automatically reuse
the saved local-success assessments. The run reproduces initialization and the
available workflow, not bitwise stochastic-policy or Agent behavior.

### Tools completion follow-up: two additional failed trials

The request to complete all four hammers has **not yet been achieved**. Two
additional independent episodes retained the original three-correction limit,
unchanged physics/native checker, frozen policy and no in-episode resets:

| Episode | Actual steps | WAM continuation | Outcome |
|---|---:|---:|---|
| `recovery_tools_complete_20261012T1050` | 238 | 0 | Wrong-bin approach; corrected blue-handle contact slipped during lift |
| `recovery_tools_isolated_grasp_20261012T1125` | 522 | 320 | Clearance-first approach reached the wooden handle, but WAM did not establish a retained grasp |

Both recordings passed verification, but native/full-chain success is false
and no hammer was extracted. The second episode's positive local assessments
refer only to navigation and handle contact, not grasp retention. All six new
correction attempts are preserved; these episodes are not concatenated with
each other or with the earlier tools failure.

The final 192 WAM steps of the second episode include **57 actual close
commands**, of which **54** still had measured gripper closure below 0.05.
Therefore the failure cannot simply be described as WAM never commanding
closure. Contact obstruction and gripper dynamics need further diagnosis;
these measurements alone do not establish a simulator defect. The
[follow-up results](diagnostics/recovery_tools_isolated_grasp_20261012T1125/followup_results.json)
record the exact metrics and unresolved goal. Inputs remained explicitly
privileged-state-assisted, including archived object poses and static geometry.

## Separate Agent coding control baseline

Cartesian control uses DLS damping `0.1` and uniformly scales oversized joint
increments before applying joint-position limits. Componentwise step clipping
distorted the IK direction during a recorded wrist-singularity failure. The
controller settings are recorded in each new session; physics, gripper force,
task checks, and the existing per-step safety bounds are unchanged.

`AGENT_CONTROL=1` starts a separately labeled experiment without loading WAM.
The task, physics, native horizon and success checker are unchanged. This mode
is **not WAM Recovery**: its `acceptance_success` stays false, and successful
native completion is recorded separately as `agent_control_success`.

```bash
AGENT_CONTROL=1 SIM_GPU=7 \
RECOVERY_GENESIS=/home/aiscuser/steeringwam-data/replay-dependencies/tools/genesis.npz \
RECOVERY_GENESIS_SHA256=f39ad091d69f571bb68e02519f68f5998bd4377ac0e270cc6ea4b761b75f4d4b \
bash launch_recovery_experiment.sh diagnostics/agent_tools_new ToolsPickingAllHammersTask
```

Use `recovery_client.py --socket SOCKET context`, then
`recovery_client.py --socket SOCKET control proposal.json`. Each proposal uses
the existing action-only schema and a fresh request token. Its diagnosis and
goal are retained alongside the literal controller code, compiled targets,
authorized joint commands and measured response. Observe feedback between
programs; do not infer grasp from a close command or successful IK alone.

Alternatively, `control --code 'controller.hold(steps=8)' --diagnosis '...'
--goal '...'` reads the current token and preflights the same bounds before
submission. The worker still records the complete proposal and literal code.

Programs retain the same workspace, translation, rotation, joint-step and
48-step bounds. The three-attempt **Recovery** budget does not apply to this
explicitly separate direct-control baseline; the native task horizon still
terminates execution. Direct control is rejected in ordinary Recovery mode.
No unrestricted code, simulator mutation or reset operation is exposed.
Additional measured joint/body states aid gripper-contact diagnosis.

Finalize and verify with:

```bash
./RoboLab/.venv/bin/python verify_recovery_episode.py RUN/recovery --require-agent-success
```

The verifier recompiles every bounded control program, checks its executed
Cartesian prefix and joint commands against the event log/HDF5, and checks the
native outcome. `--require-success` continues to require the original WAM
failure/recovery/continuation chain and cannot be satisfied by this baseline.
The dashboard distinguishes direct-control programs and their task outcome.

### Strict single-blue-hammer placement

The unmodified single-hammer checker can terminate while the handle touches the
table but the head is still near the gripper. A native success alone therefore
does not establish a fully released, settled placement.

For a separately labeled, stricter experiment, add `VERIFY_BLUE_PLACEMENT=1`
and select `ToolsPickingHammerTask` with `AGENT_CONTROL=1`. The original native
predicate must still pass. Additionally, the entire hammer world bounding box
must clear the bin horizontally by 5 mm, the commanded gripper must be open
with measured closure below 0.1, the flange must be at least 20 cm from the
hammer bounding box, and linear/angular speed must remain at or below
0.01 m/s and 0.05 rad/s for 15 consecutive control steps (1 second).

This strengthens the experiment's termination condition; it is not the
unmodified benchmark. Physics, action limits, the 60-second task horizon and
the no-reset rule remain unchanged. Raw native outcomes and every stability
sample are recorded, and the verifier independently recomputes the consecutive
sample checks. `stable_blue_placement_verified` is separate from an ordinary
native success label.

The first strict episode, `diagnostics/agent_blue_hammer_strict_20261012`,
reached the unchanged 900-step horizon after 22 bounded programs without a
reset. The first 17 programs reproduced the earlier grasp with zero measured
EE/blue-root boundary-position deviation. Subsequent release left the hammer
visibly outside the bin near the table edge, but **strict success is false**.
The recorded conservative bounding box did not fully clear the bin. At steps
833-835, the original native predicate, low-velocity limits and table force
were satisfied; from step 836 onward, measured velocities and reported table
force were zero, and the original native predicate became false. A sleeping
rigid body's contact reporting is a possible explanation, not a verified
diagnosis. Neither the criterion nor recorded outcomes were changed to force
success. This remains an Agent-only episode, not a WAM Recovery success.

Placement samples retain raw normalized finger-joint closure, which can
slightly exceed `[0, 1]`; policy proprioception clips that value. The verifier
checks raw closure against measured joint position and separately checks the
clipped observation instead of treating this difference as corrupted data.

## Minimal integration

- [probe_isaac_camera.py](probe_isaac_camera.py) initializes the registered DROID
  task once, disables RoboLab's early-terminal auto-reset exception, and serves
  the live environment.
- [recovery_worker.py](recovery_worker.py) reuses Cosmos3Client's observations,
  prompt packing and binary gripper conversion. WAM returns 32 joint-position
  commands (7 Panda joints + gripper) and decoded predicted future; at most 16
  commands are executed before yielding control.
- [recovery_protocol.py](recovery_protocol.py) validates Condition / Action /
  Joint proposals. Action programs use the existing IsaacLab differential IK,
  robot-root `base_link` pose, wxyz quaternions, and 0=open / 1=closed gripper.
  The gripper flange is **not** the finger contact point.
- [recovery_client.py](recovery_client.py) is the external Agent's local tool.
  WAM is paused between commands: wall-clock reasoning time does not advance
  physics. Each mutating decision needs the latest request ID.
- [verify_recovery_episode.py](verify_recovery_episode.py) checks actual actions
  against HDF5, predicted prefixes, simulator step continuity and ordered events.
  Native success during Agent correction alone does not pass acceptance.

RoboLab's standard task termination manager remains authoritative. The first
task is `RubiksCubeTask`: put the cube in the bowl, make container contact, and
detach the gripper. The task's 40-second simulation horizon is unchanged.
Failure is manually diagnosed from real observations, not injected by resetting
or teleporting objects. Local recovery is a separate visual Agent assessment,
not a substitute for the native whole-task checker.

## Run

Use the existing local RoboLab and Cosmos virtual environments, checkpoint and
assets described in [rollout_README.md](rollout_README.md). Apply the dependency
patches from [patches/](patches/) when restoring another checkout:

```bash
git -C RoboLab apply ../patches/robolab-recovery-no-reset.patch
git -C packages/cosmos3 apply ../../patches/cosmos3-robolab-runtime.patch
```

Skip a patch already applied; `git apply --check` can detect applicability.
Model execution uses evaluation/inference mode and explicitly disables parameter
gradients. GPU 5 is the model default and GPU 4 the renderer default; check
availability first. No other users' processes are terminated.

```bash
cd /home/aiscuser/SteeringWAM
bash launch_recovery_experiment.sh diagnostics/recovery_$(date -u +%Y%m%dT%H%M%SZ)
```

This foreground launcher prints the socket, starts the services and a 60-second
Azure backup sidecar, and waits for `finalize`. Use `REUSE_POLICY=1` only to reuse
an explicitly selected running model server. `SYNC_BLOB=0` disables upload.

For a new episode using a historical initial scene, supply `RECOVERY_GENESIS`
and `RECOVERY_GENESIS_SHA256` to the launcher. The SHA-256 is checked before
startup; the existing scene-restore helper is used **once, before the first WAM
action**, and the complete state must match within `1e-5` at episode step zero.
The initialization provenance and copied genesis are saved with the new episode.
This is not replay of old actions, and it does not add a restore/reset operation
to the recovery API. During recovery and continuation, the live state remains
the only execution state.

In another terminal / Coding Agent tool session:

```bash
PY=./RoboLab/.venv/bin/python
SOCKET=/tmp/wam-recovery-REPLACE_WITH_PRINTED_PID.sock
$PY recovery_client.py --socket "$SOCKET" --output /tmp/context.json context
$PY recovery_client.py --socket "$SOCKET" --output /tmp/wam.json wam \
  --condition "Pick up the cube and put it in the bowl" \
  --subgoal "approach and grasp the cube" --chunks 3
```

Read the real camera PNGs referenced by `latest_observation`, robot state, and
recent history. For a visual-only experiment, `scene.npz` is archival simulator
ground truth, **not** an Agent perception input. If a debugging run consults its
object poses for diagnosis, controller targets or assessment, record
`"agent_input_mode": "debug_ground_truth_assisted"` in `experiment.json`.
Such a run is a privileged-state-assisted diagnostic baseline, not a visual-only
Agent evaluation. Record the actual Agent model/version when known; do not infer
it from the client name. These provenance fields are metadata declarations, not
proof supplied by the trajectory verifier. Predicted video is never evidence of
actual success.

## GPT Agent configuration and tools

Phase one uses an interactive visual GPT Coding Agent (for example the GPT model
selected in VS Code Copilot). No OpenAI API key is necessary for this mode, and
this repository does not silently choose a model or send observations to another
endpoint. Configure the desired GPT model in the Agent UI, grant access to the
local client and observation images, and record its actual model/version in the
episode's experiment metadata when known. There is no autonomous API-backed
GPT failure detector in this phase.

The Agent must actually invoke these tools and inspect feedback, not stop after
writing a proposal. All operations below use the same socket/environment:

1. `context`: current Observation, Robot State, condition/subgoal, recent events.
2. `mark_failure failure.json`: pause after real WAM execution, with visual evidence.
3. `recover proposal.json`: validate and execute condition/action/joint correction.
4. `context`, then `assess assessment.json`: assess feedback; retry on failure,
   up to three executed correction attempts across the episode.
5. On local success, `wam ...`: resume WAM from the corrected state.
6. `finalize --reason "..."`: export both successful and failed episodes.

New sessions use `"protocol_version": "wam-recovery-v2"` and the current
`"request_id"` returned by the previous tool. Failure JSON also contains
`failure_type`, `evidence`, and `subgoal`. Assessment JSON contains boolean
`local_recovery_success` and nonempty `evidence`.

An Action proposal example (replace coordinates using the current measured pose;
do not submit this blindly):

```json
{
  "protocol_version": "wam-recovery-v2",
  "request_id": "CURRENT_TOKEN",
  "diagnosis": "The fingers closed beside the cube; the cube remains on the table.",
  "recovery_goal": "Reopen and reach a pre-grasp pose so WAM can grasp.",
  "correction_mode": "action",
  "mode_reason": "The instruction is still appropriate; only the open pre-grasp pose needs repair.",
  "control_code": "controller.gripper(value=0, steps=3)\ncontroller.move_ee(position=[0.45, 0.0, 0.35], steps=8)"
}
```

Only literal keyword calls to `controller.move_ee`, `controller.gripper`, and
`controller.hold` are parsed. Imports, assignments, loops, expressions, arbitrary
calls and attribute traversal are rejected; code is never evaluated. JSON
`program` operations are an equivalent alternative to `control_code`, not an
additional program. Condition mode requires `condition`; Joint combines it with a bounded action
program. Both must actually change the current condition. If supplied,
`wam_execute_steps` must be 16. Mode choice is still the Agent's, not an
automatic failure classifier.

### V2: mode rationale, condition provenance, and fixed evaluation

Every `recover` proposal requires a nonempty `mode_reason` explaining why direct
action repair, a condition rewrite, or both are needed. This is distinct from
the failure diagnosis and recovery goal. It is an Agent-provided explanation,
not proof that the selected mode is optimal.

All three modes now run **16 WAM steps before local assessment**:

| Mode | Controller stage | WAM evaluation condition |
|---|---|---|
| Action | Bounded local program | Exactly the pre-correction condition |
| Condition | None | Rewritten condition |
| Joint | Bounded local program | Rewritten condition after action repair |

The worker owns this window; external WAM calls and assessments cannot
interrupt it or change its condition. New Action attempts therefore execute
an additional WAM chunk compared with V1. All window steps have source
`recovery_evaluation`, not Cartesian Agent action. Native termination/horizon
can shorten or prevent the window; actual length and interruption reason are
recorded, with no reset, padding, or fabricated local assessment. Completion
of 16 steps only enables Agent assessment; it does not imply local success.
Native success during evaluation still does not satisfy the separate
full-chain acceptance rule without an explicit handoff and later continuation.

`recovery_evaluation_started/completed` record the attempt, condition, step
boundaries and requested/actual length. `recovery_attempt_completed` and session
state also retain the evaluation object. Assessments identify the evaluated
end step, window length and `label_source="agent_assessment"`.

Outside the window, changing a previously supplied WAM condition requires
both `--condition-change-kind` and `--condition-change-reason`:

```bash
$PY recovery_client.py --socket "$SOCKET" wam \
  --condition "Keep holding the cube and carry it above the bowl." \
  --subgoal "transport" \
  --condition-change-kind subgoal_progression \
  --condition-change-reason "The grasp stage is complete; begin transport."
```

Kinds are `subgoal_progression` (normal stage advancement) and
`recovery_rewrite` (a corrective rewrite). Initial and unchanged conditions
are automatically marked `initial` / `unchanged` and must not receive change
arguments. For `--chunks N`, change metadata applies only to the first chunk.
Every request/prediction records the condition provenance; each actual text
change also emits `condition_changed` with before/after, reason, source and
attempt ID (`null` outside a formal recovery). Ordinary WAM rewrites are not
silently counted as formal `recover` attempts. Action's condition lock ends
only after its fixed window; later subgoal changes are separately auditable.

The verifier independently checks mode rationale, request/condition history,
controller execution, each window and assessment alignment. It reports
`comparison_contract_verified` and complete/incomplete window counts. Legacy
V1 episodes remain readable and retain their original labels; their comparison
contract is **unknown (`null`)**, not retroactively verified. Use V2 requests
with a newly launched V2 worker; do not replay old tokens or edit archived
proposals. Agent-only control retains its separate semantics and does not
acquire a Recovery evaluation window or a required mode-selection rationale.

Limits: 12 cm per translation segment, 30 cm total, 2 cm per target step, 16
steps per operation, 48 control steps per attempt, 45 degrees per rotation
segment, 0.12 rad per rotation step and per-joint command increment. Joint limits
are enforced. These bounds do **not** prove collision freedom or physical-robot
safety. The controller uses measured feedback at each IK step, not open-loop
joint replay. Actual tracking may differ substantially from the Cartesian intent.

On execution exceptions, the session becomes `execution_unknown`, preserves the
error and attempted command, and forbids automatic re-execution. Use `context`
and `finalize` to preserve evidence. Never reset and call the result a recovery.

## Data and verification

`RUN/recovery/` contains:

- `events.jsonl`: timestamped command intent before stepping, actual environment
  commands after stepping, measured robot state, failure, every proposal/error,
  assessment, control handoff and continuation.
- `observations/`: all real control-step camera frames, robot JSON and scene NPZ.
- `segments/`: model input, condition, raw/processed predicted joint chunks,
  predicted future NPY/MP4, actual executed prefixes and results.
- `recovery_attempts/`: diagnosis/goal/mode/mode reason, original control code, normalized
  program, compiled Cartesian intents, and actual IK joint commands.
- `trajectory.hdf5`: existing RoboLab trajectory format, including measured
  states. An environment action is a controller command, not measured motion.
- `episode_wrist.mp4`: initial observation + every actual control step at 15 Hz,
  without duplicated pause/review frames.
- `manifest.json`, `verification.json`: provenance and separate local/native/
  full-chain acceptance results.

```bash
$PY recovery_client.py --socket "$SOCKET" finalize --reason "Native task completed"
$PY verify_recovery_episode.py RUN/recovery --require-success
```

The verifier exits nonzero if the full chain is absent, even with valid files.
Unit tests use protocol fixtures only and are not real execution evidence:

```bash
$PY -m pytest test_recovery_protocol.py test_recovery_evaluation.py -q
```

Azure backup reuses [sync_instruction_results.py](sync_instruction_results.py),
reading the private `~/.blob-sync/temp_sas_url.txt` without logging the SAS.
Destination prefix: `jinghe/steeringwam/recovery/RUN_NAME/` in that configured
container. Failed attempts are included. Check `errors=[]` in
`~/.blob-sync/instruction-results/RUN_NAME/status.json`; an upload process merely
being alive is not proof of synchronization. After finalization, perform a final
`--once` pass.

[sync_recovery_code.py](sync_recovery_code.py) periodically pushes already
committed checkpoints on the recovery branch and verifies the remote commit:

```bash
$PY sync_recovery_code.py --interval 60
# One foreground retry after configuring local GitHub authentication:
$PY sync_recovery_code.py --once
```

It never stages, commits, force-pushes, or modifies unrelated worktree changes.
Authentication or permission failures are recorded as `blocked`, not `synced`,
in `~/.blob-sync/recovery-code/status.json`. GitHub authentication was completed
on 2026-10-12, and the recovery branch was pushed with its remote commit verified
against the local checkpoint. The initial HTTPS/SSH authentication failures and
VS Code task timeout did not alter the verified episode.

For another machine, install GitHub CLI, authorize it in your own browser, then
configure Git's credential helper:

```bash
gh auth login --hostname github.com --web
gh auth setup-git --hostname github.com
gh auth status --hostname github.com
```

These flags also work with Ubuntu's older GitHub CLI package. Do not send tokens
to the Agent or store them in the repository/episode. Check the CLI's credential
storage policy before authenticating on a shared Linux account.

As an additional recoverable backup, an incremental Git bundle of the experiment
commits is stored with the episode on Azure. It is not a substitute claim for
GitHub synchronization; it can be fetched into a checkout containing base commit
`27a8c70` if a separate restore is needed.
