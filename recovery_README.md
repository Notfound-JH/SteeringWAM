# Live WAM + Coding Agent recovery

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
recent history. `scene.npz` is archival simulator ground truth, **not** an Agent
perception input. Predicted video is never evidence of actual success.

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

Each JSON includes `"protocol_version": "wam-recovery-v1"` and the current
`"request_id"` returned by the previous tool. Failure JSON also contains
`failure_type`, `evidence`, and `subgoal`. Assessment JSON contains boolean
`local_recovery_success` and nonempty `evidence`.

An Action proposal example (replace coordinates using the current measured pose;
do not submit this blindly):

```json
{
  "protocol_version": "wam-recovery-v1",
  "request_id": "CURRENT_TOKEN",
  "diagnosis": "The fingers closed beside the cube; the cube remains on the table.",
  "recovery_goal": "Reopen and reach a pre-grasp pose so WAM can grasp.",
  "correction_mode": "action",
  "control_code": "controller.gripper(value=0, steps=3)\ncontroller.move_ee(position=[0.45, 0.0, 0.35], steps=8)"
}
```

Only literal keyword calls to `controller.move_ee`, `controller.gripper`, and
`controller.hold` are parsed. Imports, assignments, loops, expressions, arbitrary
calls and attribute traversal are rejected; code is never evaluated. JSON
`program` operations are an equivalent alternative to `control_code`, not an
additional program. Condition mode requires `condition` and optional
`wam_execute_steps`; Joint combines these fields with a bounded action program.
Joint executes the program then one corrected WAM chunk. Mode choice is the
Agent's, not a fixed recovery recipe.

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
- `recovery_attempts/`: diagnosis/goal/mode, original control code, normalized
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
$PY -m pytest test_recovery_protocol.py -q
```

Azure backup reuses [sync_instruction_results.py](sync_instruction_results.py),
reading the private `~/.blob-sync/temp_sas_url.txt` without logging the SAS.
Destination prefix: `jinghe/steeringwam/recovery/RUN_NAME/` in that configured
container. Failed attempts are included. Check `errors=[]` in
`~/.blob-sync/instruction-results/RUN_NAME/status.json`; an upload process merely
being alive is not proof of synchronization. After finalization, perform a final
`--once` pass. Code is synchronized to the current GitHub branch at explicit
validated checkpoints; do not periodically commit unrelated worktree changes.
