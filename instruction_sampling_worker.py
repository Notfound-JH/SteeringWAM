"""Persistent Isaac worker; imported only after SimulationApp initialization."""

import hashlib
import json
from pathlib import Path
import socket
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from instruction_sampling import digest, identity, write_json


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def verified(reference):
    path = Path(reference["path"])
    if digest(path) != reference["sha256"]:
        raise ValueError(f"Checkpoint/artifact identity mismatch: {path}")
    return path


def flat_state(tree, prefix=""):
    result = {}
    for name, value in tree.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            result.update(flat_state(value, key))
        else:
            result[key] = value.detach().cpu().numpy().copy()
    return result


def state_error(current, expected):
    if not current or set(current) != set(expected):
        raise ValueError("Scene-state fields differ")
    maximum = 0.0
    for name in current:
        if not current[name].size or current[name].shape != expected[name].shape:
            raise ValueError(f"Scene-state shape differs: {name}")
        if not np.isfinite(current[name]).all() or not np.isfinite(expected[name]).all():
            raise ValueError(f"Non-finite scene state: {name}")
        maximum = max(maximum, float(np.max(np.abs(current[name] - expected[name]))))
    return maximum


def image_mae(current, reference):
    if current.shape != reference.shape:
        raise ValueError("Restore image shape differs")
    if not current.size or not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Invalid restore image")
    return float(np.abs(current.astype(float) - reference.astype(float)).mean())


def hold_restored_robot_pose(env):
    robot = env.scene["robot"]
    positions = robot.data.joint_pos.clone()
    zeros = positions * 0
    robot.set_joint_position_target(positions)
    robot.set_joint_velocity_target(zeros)
    robot.set_joint_effort_target(zeros)


def restore_scene_state(env, expected):
    import torch

    current = env.scene.get_state(is_relative=True)
    if set(flat_state(current)) != set(expected):
        raise ValueError("Cannot overlay a different scene")

    def overlay(tree, prefix=""):
        for name, value in tree.items():
            key = f"{prefix}/{name}" if prefix else name
            if isinstance(value, dict):
                overlay(value, key)
            else:
                restored = torch.as_tensor(expected[key], device=env.device)
                if restored.shape != value.shape or not torch.isfinite(restored).all():
                    raise ValueError(f"Invalid restored state: {key}")
                tree[name] = restored

    overlay(current)
    env.scene.reset_to(current, env_ids=None, is_relative=True)
    hold_restored_robot_pose(env)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.render()
    env.scene.update(0.0)
    return env.observation_manager.compute()


def make_sheet(frames, destination, title):
    times = sorted({frame["step"] for frame in frames})
    sheet = Image.new("RGB", (320 * len(times), 412), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), title, fill="black")
    for row, view in enumerate(("actual", "viewport")):
        for column, step in enumerate(times):
            frame = next(item for item in frames if item["step"] == step and item["view"] == view)
            with Image.open(verified(frame)) as image:
                tile = image.convert("RGB")
                tile.thumbnail((320, 180))
                sheet.paste(tile, (column * 320, 32 + row * 190))
            draw.text((column * 320 + 4, 20 + row * 190), f"{view} step {step}", fill="black")
    sheet.save(destination)
    return artifact(destination)


def serve(env, env_cfg, args):
    import h5py
    import torch
    from policies.cosmos3.client import Cosmos3Client
    from robolab.eval.websocket_transport import MsgPackWebSocketTransport

    directory = args.output_dir.resolve()
    startup_identity = identity()
    client = Cosmos3Client.__new__(Cosmos3Client)
    client._image_h, client._image_w = 360, 640
    client._eval_episode_idx = 0
    transport = MsgPackWebSocketTransport(f"ws://127.0.0.1:{args.policy_port}", metadata_timeout=15)
    transport.connect()
    observation = env.observation_manager.compute()

    def scene():
        return flat_state(env.scene.get_state(is_relative=True))

    def save_scene(path):
        np.savez(path, **scene())
        return artifact(path)

    def correct_scene(reference):
        with np.load(verified(reference), allow_pickle=False) as expected:
            return restore_scene_state(env, expected)

    def robot_state(obs, step):
        return {"step": step, "sim_time": step * float(env.step_dt), "captured_at": time.time(),
                "frame": "robot-root", "quaternion_order": "wxyz", "ee_link": "base_link",
                "proprio": {key: obs["proprio_obs"][key][0].cpu().numpy().tolist()
                            for key in ("arm_joint_pos", "gripper_pos", "ee_pos", "ee_quat")}}

    def capture(obs, destination, step):
        destination.mkdir(parents=True, exist_ok=False)
        frames = []
        for group in ("image_obs", "viewport_cam"):
            for name, tensor in obs[group].items():
                rgb = tensor[0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                if not rgb.size or float(rgb.std(axis=(0, 1)).max()) < 1:
                    raise ValueError(f"Blank camera: {name}")
                path = destination / f"{name}.png"
                Image.fromarray(rgb).save(path)
                frames.append({**artifact(path), "camera": name, "step": step,
                               "view": "viewport" if group == "viewport_cam" else "actual",
                               "debug_camera": group == "viewport_cam"})
        return frames

    def checkpoint(obs, destination, step):
        destination.mkdir(parents=True, exist_ok=False)
        value = {"task": args.task, "step": step, "seed": args.seed,
                 "scene": save_scene(destination / "scene.npz"),
                 "frames": capture(obs, destination / "images", step),
                 "robot": robot_state(obs, step), "control_dt": float(env.step_dt),
                 "native_goal": env_cfg.instruction}
        write_json(destination / "checkpoint.json", value)
        return artifact(destination / "checkpoint.json")

    def load_checkpoint(reference):
        value = json.loads(verified(reference).read_text())
        if value["task"] != args.task or value["seed"] != args.seed:
            raise ValueError("Checkpoint task or seed mismatch")
        return value

    def check_images(obs, expected):
        errors = {}
        for frame in expected["frames"]:
            group = "viewport_cam" if frame["debug_camera"] else "image_obs"
            rgb = obs[group][frame["camera"]][0].cpu().numpy()[:, :, :3]
            with Image.open(verified(frame)) as image:
                reference = np.asarray(image)
            errors[frame["camera"]] = image_mae(rgb, reference)
        return errors

    def events(terminated, truncated):
        return {"native_terminated": bool(terminated[0]), "native_truncated": bool(truncated[0]),
                "native_success": bool(env.termination_manager.get_term("success")[0]),
                "episode_step": int(env.episode_length_buf[0])}

    def execute(command):
        candidate = command["candidate"]
        output = Path(command["output"]).resolve()
        output.mkdir(parents=True, exist_ok=False)
        started = time.time()
        write_json(output / "request.json", command)
        try:
            env.recorder_manager.clear(env_ids=[0])
            env.recorder_manager.set_hdf5_file(str(output / "rollout.hdf5"))
            env.recorder_manager.set_episode_index(0, env_ids=[0])
            env.reset_eval_state()
            env.reset(seed=args.seed)
            for _ in range(8):
                env.scene.write_data_to_sim()
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
            genesis = load_checkpoint(command["genesis"])
            obs = correct_scene(genesis["scene"])
            state_error(scene(), np.load(verified(genesis["scene"]), allow_pickle=False))
            replay_count = 0
            replay_checks = []
            replay_event_differences = []
            for segment in command["selected"]:
                for ref in segment["artifacts"]:
                    verified(ref)
                replay_actions = np.load(segment["executed_actions"], allow_pickle=False)
                trace = json.loads(Path(segment["trace"]).read_text())
                if len(trace) != len(replay_actions):
                    raise ValueError("Replay trace length mismatch")
                for action, sample in zip(replay_actions, trace):
                    obs, _, terminated, truncated, _ = env.step(torch.as_tensor(action).unsqueeze(0))
                    maximum = state_error(scene(), np.load(verified(sample["scene"]), allow_pickle=False))
                    actual_events = events(terminated, truncated)
                    if actual_events != sample["events"]:
                        replay_event_differences.append({"step": replay_count + 1,
                                                        "expected": sample["events"], "actual": actual_events})
                    if terminated[0] or truncated[0]:
                        raise ValueError("Replay unexpectedly terminated")
                    replay_count += 1
                    replay_checks.append(maximum)
            expected = load_checkpoint(candidate["checkpoint"])
            if replay_count != expected["step"]:
                raise ValueError("Replay does not reach the committed checkpoint")
            before = state_error(scene(), np.load(verified(expected["scene"]), allow_pickle=False))
            obs = correct_scene(expected["scene"])
            after = state_error(scene(), np.load(verified(expected["scene"]), allow_pickle=False))
            start_frames = capture(obs, output / "frames/000", 0)
            image_errors = check_images(obs, expected)
            write_json(output / "restore.json", {"steps": replay_count, "trace_errors": replay_checks,
                       "before_correction": before, "after_correction": after, "image_mae": image_errors,
                       "event_differences": replay_event_differences,
                       "checkpoint_scene_applied": True,
                       "state_validation": "structural_checks_and_external_visual_review",
                       "state_threshold_enforced": False,
                       "image_validation": "external_visual_review", "pixel_threshold_enforced": False})
            extracted = client._extract_observation(obs)
            instruction = candidate["instruction"]
            prompt = (f"Overall task: {env_cfg.instruction}\nCurrent subgoal: {instruction['instruction']}\n"
                      f"Avoid: {instruction['avoid']}")
            request = client._pack_request(extracted, prompt)
            request["session_id"] = candidate["id"]
            Image.fromarray(request["observation/image"]).save(output / "input.png")
            inference_start = time.monotonic()
            reply = transport.request(request, timeout=240)
            raw = client._unpack_response(reply)
            if raw.shape != (32, 8) or not np.isfinite(raw).all():
                raise ValueError(f"Invalid Cosmos actions: {raw.shape}")
            np.save(output / "raw_actions.npy", raw)
            processed = client._postprocess_chunk(raw)
            np.save(output / "actions.npy", processed)
            inference_seconds = time.monotonic() - inference_start
            executed, trace, frames = [], [], list(start_frames)
            robot = [robot_state(obs, replay_count)]
            final_events = {}
            for index, action in enumerate(processed[:command["execute_steps"]].astype(np.float32), 1):
                obs, _, terminated, truncated, _ = env.step(torch.as_tensor(action).unsqueeze(0))
                actual = env.action_manager.action[0].detach().cpu().numpy().copy()
                if not np.array_equal(actual, action):
                    raise ValueError("Environment command differs from Cosmos prefix")
                executed.append(actual)
                final_events = events(terminated, truncated)
                trace.append({"scene": save_scene(output / f"state_{index:03d}.npz"), "events": final_events})
                robot.append(robot_state(obs, replay_count + index))
                np.save(output / "executed_actions.npy", np.stack(executed))
                write_json(output / "trace.json", trace)
                write_json(output / "robot.json", robot)
                frames.extend(capture(obs, output / f"frames/{index:03d}", index))
                if terminated[0] or truncated[0]:
                    break
            np.save(output / "executed_actions.npy", np.stack(executed))
            write_json(output / "trace.json", trace)
            write_json(output / "robot.json", robot)
            if not env.all_terminated:
                env.recorder_manager.export_episodes(env_ids=[0])
            for name in ("_dataset_file_handler", "_failed_episode_dataset_file_handler"):
                handler = getattr(env.recorder_manager, name, None)
                if handler is not None:
                    handler.close()
            with h5py.File(output / "rollout.hdf5", "r") as recording:
                recorded = recording["data/demo_0/actions"][:]
            if len(recorded) != replay_count + len(executed) or not np.array_equal(recorded[-len(executed):], executed):
                raise ValueError("HDF5 action prefix/length mismatch")
            final_checkpoint = checkpoint(obs, output / "final", replay_count + len(executed))
            times = sorted({0, len(executed) // 4, len(executed) // 2, 3 * len(executed) // 4, len(executed)})
            review_frames = [frame for frame in frames if frame["step"] in times and
                             (frame["camera"] == "wrist_cam" or frame["debug_camera"])]
            sheet = make_sheet(review_frames, output / "review.png", f"Candidate {candidate['attempt']:02d}")
            write_json(output / "completion.json", {"candidate_id": candidate["id"],
                       "status": "complete", "operation_exit_code": 0})
            result = {"status": "complete", "candidate_id": candidate["id"], "checkpoint": candidate["checkpoint"],
                      "execution_receipt": artifact(output / "completion.json"),
                      "action_source": "cosmos", "final_checkpoint": final_checkpoint,
                      "raw_actions": str(output / "raw_actions.npy"), "processed_actions": str(output / "actions.npy"),
                      "executed_actions": str(output / "executed_actions.npy"), "executed_steps": len(executed),
                      "trace": str(output / "trace.json"), "robot": str(output / "robot.json"),
                      "frames": frames, "review_frames": review_frames, "sheet": sheet,
                      "started_at": started, "finished_at": time.time(), "inference_seconds": inference_seconds,
                      "prompt": prompt, "replay_steps": replay_count, "control_dt": float(env.step_dt),
                      "source_identity": startup_identity, **final_events,
                      "artifacts": [artifact(path) for path in sorted(output.rglob("*")) if path.is_file()]}
            write_json(output / "result.json", result)
            return result
        except Exception as error:
            write_json(output / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
                       "candidate_id": candidate["id"], "finished_at": time.time(), "operation_exit_code": 1})
            raise

    initial = checkpoint(observation, directory / "initial", 0)
    address = Path(args.worker_socket)
    if address.exists():
        raise ValueError("Worker socket already exists; refusing a second controller")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(address))
            address.chmod(0o600)
            server.listen(1)
            write_json(directory / "ready.json", {"checkpoint": initial, "gpu": args.sim_gpu,
                       "task": args.task, "code": startup_identity})
            while True:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(1200)
                    try:
                        command = json.loads(connection.makefile("rb").readline(8 * 1024 * 1024))
                        if identity() != startup_identity or command["code"] != startup_identity:
                            raise ValueError("Worker source identity changed")
                        if command["op"] == "shutdown":
                            connection.sendall(b'{"status":"stopped"}\n')
                            break
                        if command["op"] == "initial":
                            reply = {"checkpoint": initial}
                        elif command["op"] == "execute":
                            reply = execute(command)
                        else:
                            raise ValueError("Unknown worker command")
                        connection.sendall(json.dumps(reply, allow_nan=False).encode() + b"\n")
                    except Exception as error:
                        traceback.print_exc()
                        try:
                            connection.sendall(json.dumps({"error": str(error)}).encode() + b"\n")
                        except OSError:
                            pass
    finally:
        transport.close()
        address.unlink(missing_ok=True)