"""Single-task, twelve-environment sampling with serial Cosmos inference."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from instruction_sampling import validate_instruction, write_json
from instruction_sampling_worker import flat_state, hold_restored_robot_pose


NUM_ENVS = 12


def configure_runtime_libraries():
    root = Path(__file__).resolve().parent
    directories = [str(root / "packages/ffmpeg/lib")]
    for pattern in ("libglu-*/lib", "xorg-libxt-*/lib"):
        directories.extend(str(path) for path in sorted((Path.home() / ".conda/pkgs").glob(pattern)))
    directories.extend(filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":")))
    library_path = ":".join(dict.fromkeys(directories))
    if os.environ.get("LD_LIBRARY_PATH") != library_path:
        os.environ["LD_LIBRARY_PATH"] = library_path
        os.execv(sys.executable, [sys.executable, *sys.argv])


def load_instructions(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list) or len(values) != NUM_ENVS:
        raise ValueError("Provide a JSON array of exactly 12 instructions")
    instructions = [validate_instruction(value) for value in values]
    if len({value["subgoal"] for value in instructions}) != 1:
        raise ValueError("All twelve instructions must share the current subgoal")
    return instructions


def tile_checkpoint(expected, template):
    if set(expected) != set(template):
        raise ValueError("Checkpoint scene fields differ")
    result = {}
    for name, target in template.items():
        source = np.asarray(expected[name])
        if source.shape != (1, *target.shape[1:]) or not np.isfinite(source).all():
            raise ValueError(f"Invalid single-environment checkpoint field: {name}")
        result[name] = np.repeat(source, target.shape[0], axis=0)
    return result


def stack_actions(chunks, step, active_ids):
    chunks = np.asarray(chunks, dtype=np.float32)
    if chunks.shape != (NUM_ENVS, 32, 8) or not np.isfinite(chunks).all():
        raise ValueError("Expected twelve finite (32, 8) Cosmos chunks")
    actions = np.zeros((NUM_ENVS, 8), dtype=np.float32)
    actions[active_ids] = chunks[active_ids, step]
    return actions


def checkpoint_history(path, task):
    from instruction_sampling_worker import verified

    history, seen = [], set()
    while path is not None:
        path = Path(path).resolve()
        if path in seen:
            raise ValueError("Checkpoint parent cycle")
        seen.add(path)
        checkpoint = json.loads(path.read_text())
        if checkpoint["task"] != task:
            raise ValueError("Checkpoint task mismatch")
        if checkpoint.get("native_terminated") or checkpoint.get("native_truncated"):
            raise ValueError("Cannot continue a terminated candidate")
        verified(checkpoint["scene"])
        history.append(checkpoint)
        parent = checkpoint.get("parent")
        path = verified(parent) if parent else None
    return list(reversed(history))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--task", required=True)
    value.add_argument("--instructions", required=True, type=Path)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument("--sim-gpu", type=int, default=4)
    value.add_argument("--allow-gpu-sharing", action="store_true", help="Explicitly allow existing GPU compute processes")
    value.add_argument("--policy-port", type=int, default=18745)
    value.add_argument("--seed", type=int, default=3407)
    value.add_argument("--execute-steps", type=int, default=16, choices=range(1, 33))
    value.add_argument("--checkpoint", type=Path, help="Selected candidate checkpoint.json from this runner")
    value.add_argument("--experiment-arm", type=Path, help="Frozen matched-seed experiment arm")
    value.add_argument("--repeatability-probe", action="store_true", help="Repeat the first seeded request; record differences without rejecting numerical drift")
    value.add_argument("--no-videos", action="store_true", help="Keep PNG frames but skip MP4 encoding")
    value.add_argument("--initialization-only", action="store_true", help="Check twelve cameras and initial poses without policy requests")
    return value


def run(env, env_cfg, args, instructions):
    import hashlib
    import time
    import uuid

    import h5py
    import torch
    from PIL import Image
    from policies.cosmos3.client import Cosmos3Client
    from robolab.eval.websocket_transport import MsgPackWebSocketTransport

    from instruction_sampling_worker import artifact, make_sheet, restore_scene_state, verified

    output = args.output_dir
    experiment = None
    if getattr(args, "experiment_arm", None):
        from paired_instruction_experiment import load_arm

        experiment = load_arm(args.experiment_arm, args.task, args.checkpoint, args.instructions)
        if args.execute_steps != experiment["execute_steps"]:
            raise ValueError("Experiment execution budget mismatch")
        write_json(output / "experiment_arm.json", experiment)
    write_json(output / "instructions.json", instructions)
    env.reset_eval_state()
    obs, _ = env.reset(seed=args.seed)
    for _ in range(8):
        env.scene.write_data_to_sim()
        env.sim.step(render=True)
        env.scene.update(env.physics_dt)
    template = flat_state(env.scene.get_state(is_relative=True))
    start_step = 0
    history = checkpoint_history(args.checkpoint, args.task) if args.checkpoint else []
    if args.checkpoint:
        with np.load(verified(history[0]["genesis"]), allow_pickle=False) as source:
            state = tile_checkpoint(source, template)
    else:
        state = tile_checkpoint({name: value[:1] for name, value in template.items()}, template)
    obs = restore_scene_state(env, state)
    genesis_path = output / "genesis.npz"
    np.savez(genesis_path, **{name: value[:1] for name, value in state.items()})
    genesis = artifact(genesis_path)
    for checkpoint in history:
        actions = np.load(verified(checkpoint["executed_actions"]), allow_pickle=False)
        if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
            raise ValueError("Invalid selected action prefix")
        for action in actions:
            replay_action = torch.as_tensor(np.repeat(action[None], NUM_ENVS, axis=0), device=env.device)
            obs, _, terminated, truncated, _ = env.step(replay_action)
            if terminated.any() or truncated.any():
                raise ValueError("Selected history unexpectedly terminated during replay")
            start_step += 1
        if start_step != checkpoint["step"]:
            raise ValueError("Checkpoint history step mismatch")
        with np.load(verified(checkpoint["scene"]), allow_pickle=False) as source:
            obs = restore_scene_state(env, tile_checkpoint(source, template))
    env.episode_length_buf[:] = start_step
    hold_restored_robot_pose(env)
    env.scene.write_data_to_sim()

    def save_scene(directory, env_id):
        scene = flat_state(env.scene.get_state(is_relative=True))
        path = directory / "scene.npz"
        np.savez(path, **{name: value[env_id:env_id + 1] for name, value in scene.items()})
        return artifact(path)

    def capture(observation, directory, env_id, step):
        directory.mkdir(parents=True, exist_ok=False)
        frames = []
        for group in ("image_obs", "viewport_cam"):
            for camera, tensor in observation[group].items():
                rgb = tensor[env_id].detach().cpu().numpy()[..., :3].astype(np.uint8)
                if not rgb.size or float(rgb.std(axis=(0, 1)).max()) < 1:
                    raise ValueError(f"Blank camera: env {env_id}, {camera}")
                path = directory / f"{camera}.png"
                Image.fromarray(rgb).save(path)
                frames.append({**artifact(path), "camera": camera, "step": step,
                               "view": "viewport" if group == "viewport_cam" else "actual",
                               "debug_camera": group == "viewport_cam"})
        return frames

    candidates = []
    for env_id, instruction in enumerate(instructions):
        directory = output / f"candidate_{env_id:02d}"
        directory.mkdir()
        candidates.append({"env_id": env_id, "id": uuid.uuid4().hex, "instruction": instruction,
                           "directory": str(directory), "frames": capture(obs, directory / "frames/000", env_id, 0),
                           "executed": [], "trace": [], "finished": False})
    initial_scene = save_scene(output, 0)
    write_json(output / "initial.json", {"task": args.task, "step": start_step, "scene": initial_scene,
               "joint_positions": obs["proprio_obs"]["arm_joint_pos"].cpu().numpy().tolist()})
    if args.initialization_only:
        write_json(output / "result.json", {"status": "initialized", "num_envs": NUM_ENVS,
                   "candidates": candidates})
        return

    client = Cosmos3Client.__new__(Cosmos3Client)
    client._image_h, client._image_w, client._eval_episode_idx = 360, 640, 0
    chunks = []
    transport = MsgPackWebSocketTransport(f"ws://127.0.0.1:{args.policy_port}", metadata_timeout=15)
    try:
        transport.connect()
        for candidate in candidates:
            directory = Path(candidate["directory"])
            instruction = candidate["instruction"]
            prompt = (f"Overall task: {env_cfg.instruction}\nCurrent subgoal: {instruction['instruction']}\n"
                      f"Avoid: {instruction['avoid']}")
            extracted = client._extract_observation(obs, env_id=candidate["env_id"])
            request = client._pack_request(extracted, prompt)
            if experiment:
                prompt = experiment["prompt"]
                request["prompt"] = prompt
                request["request_seed"] = experiment["seeds"][candidate["env_id"]]
            request["session_id"] = candidate["id"]
            Image.fromarray(request["observation/image"]).save(directory / "input.png")
            write_json(directory / "request.json", {"session_id": candidate["id"], "prompt": prompt,
                       "env_id": candidate["env_id"], "checkpoint": initial_scene,
                       "request_seed": request.get("request_seed")})
            started = time.monotonic()
            response = transport.request(request, timeout=240)
            if experiment:
                actual_seed = response.get("actual_seed") if isinstance(response, dict) else None
                if type(actual_seed) is not int or actual_seed != request["request_seed"]:
                    raise ValueError("Server did not acknowledge the requested experiment seed")
                candidate["experiment"] = {"condition": experiment["condition"],
                                           "pair_id": candidate["env_id"], "actual_seed": actual_seed}
                write_json(directory / "seed_receipt.json", candidate["experiment"])
            raw = client._unpack_response(response)
            if raw.shape != (32, 8) or not np.isfinite(raw).all():
                raise ValueError(f"Invalid Cosmos chunk for env {candidate['env_id']}: {raw.shape}")
            if experiment and getattr(args, "repeatability_probe", False) and candidate["env_id"] == 0:
                repeated_request = {**request, "session_id": candidate["id"] + "-repeat"}
                repeated_response = transport.request(repeated_request, timeout=240)
                if repeated_response.get("actual_seed") != request["request_seed"]:
                    raise ValueError("Repeatability request seed mismatch")
                repeated = client._unpack_response(repeated_response)
                if repeated.shape != raw.shape or not np.isfinite(repeated).all():
                    raise ValueError("Invalid repeatability response")
                np.save(directory / "repeat_raw_actions.npy", repeated)
                write_json(directory / "repeatability.json", {
                    "actual_seed": request["request_seed"], "exact_equal": bool(np.array_equal(raw, repeated)),
                    "max_abs_difference": float(np.max(np.abs(raw - repeated))),
                    "mean_abs_difference": float(np.mean(np.abs(raw - repeated))),
                    "diagnostic_only": True, "repeat_actions_executed": False})
            candidate["inference_seconds"] = time.monotonic() - started
            np.save(directory / "raw_actions.npy", raw)
            chunk = client._postprocess_chunk(raw).astype(np.float32)
            np.save(directory / "actions.npy", chunk)
            chunks.append(chunk)
    finally:
        transport.close()

    recorder = env.recorder_manager
    recorder.clear(env_ids=list(range(NUM_ENVS)))
    recorder.set_hdf5_file(str(output / "rollout.hdf5"))
    for env_id in range(NUM_ENVS):
        recorder.set_episode_index(env_id, env_ids=[env_id])
    for step in range(args.execute_steps):
        active_ids = [candidate["env_id"] for candidate in candidates if not candidate["finished"]]
        actions = stack_actions(chunks, step, active_ids)
        obs, _, terminated, truncated, _ = env.step(torch.as_tensor(actions, device=env.device))
        for env_id in active_ids:
            candidate = candidates[env_id]
            directory = Path(candidate["directory"])
            actual = env.action_manager.action[env_id].detach().cpu().numpy().copy()
            if not np.array_equal(actual, actions[env_id]):
                raise ValueError(f"Recorded action differs from Cosmos prefix: env {env_id}")
            candidate["executed"].append(actual.tolist())
            candidate["native_terminated"] = bool(terminated[env_id])
            candidate["native_truncated"] = bool(truncated[env_id])
            candidate["native_success"] = bool(env.termination_manager.get_term("success")[env_id])
            if candidate["native_terminated"] or candidate["native_truncated"]:
                if not bool(env._frozen_envs[env_id]):
                    raise ValueError(f"Environment reset instead of freezing at termination: {env_id}")
            candidate["frames"].extend(capture(obs, directory / f"frames/{step + 1:03d}", env_id, step + 1))
            state_dir = directory / f"state_{step + 1:03d}"
            state_dir.mkdir()
            candidate["trace"].append({"step": start_step + step + 1, "scene": save_scene(state_dir, env_id)})
            candidate["finished"] = bool(terminated[env_id] or truncated[env_id])
            np.save(directory / "executed_actions.npy", np.asarray(candidate["executed"], dtype=np.float32))
        if all(candidate["finished"] for candidate in candidates):
            break
    remaining = [env_id for env_id in range(NUM_ENVS) if not bool(env._frozen_envs[env_id])]
    if remaining:
        recorder.export_episodes(env_ids=remaining)
    for name in ("_dataset_file_handler", "_failed_episode_dataset_file_handler"):
        handler = getattr(recorder, name, None)
        if handler is not None:
            handler.close()
    with h5py.File(output / "rollout.hdf5", "r") as recording:
        for candidate in candidates:
            env_id = candidate["env_id"]
            directory = Path(candidate["directory"])
            executed = np.asarray(candidate.pop("executed"), dtype=np.float32)
            recorded = recording[f"data/demo_{env_id}/actions"][:]
            if not np.array_equal(recorded, executed):
                raise ValueError(f"HDF5 action mismatch: env {env_id}")
            final_scene = candidate["trace"][-1]["scene"]
            checkpoint = {"task": args.task, "step": start_step + len(executed), "scene": final_scene,
                          "candidate_id": candidate["id"], "native_success": candidate["native_success"],
                          "native_terminated": candidate["native_terminated"],
                          "native_truncated": candidate["native_truncated"], "genesis": genesis,
                          "executed_actions": artifact(directory / "executed_actions.npy"),
                          "parent": artifact(args.checkpoint) if args.checkpoint else None}
            write_json(directory / "checkpoint.json", checkpoint)
            times = {0, len(executed) // 4, len(executed) // 2, 3 * len(executed) // 4, len(executed)}
            frames = [frame for frame in candidate["frames"] if frame["step"] in times and
                      (frame["camera"] == "wrist_cam" or frame["debug_camera"])]
            candidate["sheet"] = make_sheet(frames, directory / "review.png", f"Candidate {env_id:02d}")
            candidate["videos"] = []
            if not args.no_videos:
                ffmpeg = Path(__file__).resolve().parent / "packages/ffmpeg/bin/ffmpeg"
                cameras = sorted({frame["camera"] for frame in frames})
                for camera in cameras:
                    video = directory / f"{camera}.mp4"
                    subprocess.run([str(ffmpeg), "-v", "error", "-n", "-framerate", str(1 / float(env.step_dt)),
                        "-start_number", "0", "-i", str(directory / "frames/%03d" / f"{camera}.png"),
                        "-frames:v", str(len(executed) + 1), "-c:v", "libx264", "-threads", "1",
                        "-pix_fmt", "yuv420p", str(video)], check=True)
                    candidate["videos"].append(artifact(video))
            candidate.update(status="complete", executed_steps=len(executed), final_checkpoint=artifact(directory / "checkpoint.json"))
            write_json(directory / "result.json", candidate)
    write_json(output / "result.json", {"status": "complete", "task": args.task, "num_envs": NUM_ENVS,
               "checkpoint_parent": str(args.checkpoint) if args.checkpoint else None,
               "instructions_sha256": hashlib.sha256((output / "instructions.json").read_bytes()).hexdigest(),
               "candidates": candidates})


def main():
    args = parser().parse_args()
    configure_runtime_libraries()
    instructions = load_instructions(args.instructions)
    args.output_dir = args.output_dir.resolve()
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.sim_gpu), "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi", "-i", str(args.sim_gpu),
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if processes and not args.allow_gpu_sharing:
        raise RuntimeError("GPU has existing compute processes; use --allow-gpu-sharing only after checking capacity")
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, OMNI_KIT_ACCEPT_EULA="YES")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    import sys

    sys.argv = [sys.argv[0]]
    app = SimulationApp({"headless": True, "active_gpu": args.sim_gpu, "physics_gpu": 0,
        "multi_gpu": False, "renderer": "RaytracedLighting", "anti_aliasing": 0,
        "enable_crashreporter": False, "limit_cpu_threads": 4,
        "extra_args": ["--/isaaclab/render/offscreen=true", "--/physics/fabricUpdateTransformations=true",
                       "--/physics/fabricUpdateJointStates=true", "--/physics/fabricUseGPUInterop=false",
                       "--/renderer/multiGpu/autoEnable=false"]},
        experience=str(Path(sys.prefix) / "lib/python3.11/site-packages/isaaclab/apps/isaaclab.python.headless.rendering.kit"))
    env = None
    try:
        from omni.physx import acquire_physx_interface
        from isaaclab.envs.mdp import reset_scene_to_default
        from isaaclab.managers import EventTermCfg
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        acquire_physx_interface().overwrite_gpu_setting(0)
        set_output_dir(str(args.output_dir))
        auto_register_droid_envs(task=[args.task], cameras=WRIST_LEFT_RIGHT_HEAD)
        names = get_envs(task=[args.task])
        if len(names) != 1:
            raise ValueError(f"Expected a single task registration: {names}")
        env, cfg = create_env(names[0], device="cpu", use_fabric=True, num_envs=NUM_ENVS,
            seed=args.seed, events={"reset": EventTermCfg(func=reset_scene_to_default, mode="reset",
                                                         params={"reset_joint_targets": True})})
        run(env, cfg, args, instructions)
    except Exception as error:
        write_json(args.output_dir / "failure.json", {"error": str(error)})
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()