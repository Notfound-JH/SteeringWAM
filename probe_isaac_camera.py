import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--sim-gpu", type=int, default=4, help="Physical renderer GPU index; CUDA_VISIBLE_DEVICES must match its UUID.")
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--initialization-diagnostic", action="store_true", help="Capture reset and warmup robot positions/targets without policy inference.")
parser.add_argument("--worker-socket", type=Path, help="Serve persistent language-sampling commands on a local Unix socket.")
parser.add_argument("--recovery-worker-socket", type=Path, help="Serve a persistent live-state WAM recovery session.")
parser.add_argument("--agent-control", action="store_true", help="Run an explicitly separate bounded Agent-only baseline without WAM.")
parser.add_argument("--verify-blue-placement", action="store_true", help="Require a fully outside, released, stationary blue hammer for 15 steps in addition to the original native goal.")
parser.add_argument("--recovery-genesis", type=Path, help="Initialize a recovery episode from a historical scene before any WAM action.")
parser.add_argument("--recovery-genesis-sha256", help="Required SHA-256 of the historical recovery genesis.")
parser.add_argument("--task", help="Capture a registered RoboLab task instead of the small cube scene.")
parser.add_argument("--rollout-chunks", type=int, default=0, help="Run this many sequential policy chunks; zero only captures cameras.")
parser.add_argument("--execute-steps", type=int, default=16)
parser.add_argument("--policy-port", type=int, default=18745)
parser.add_argument("--instruction", help="Override the task instruction sent to the policy.")
parser.add_argument("--replay-actions", type=Path, help="Replay actions after reset before starting the rollout.")
parser.add_argument("--replay-steps", type=int, default=0)
parser.add_argument("--replay-state", type=Path, help="NPZ state expected at the replay boundary.")
parser.add_argument("--render-width", type=int, default=640)
parser.add_argument("--render-height", type=int, default=480)
parser.add_argument("--rendering-mode", choices=("performance", "balanced", "quality"), default="balanced")
args = parser.parse_args()
if args.agent_control and not args.recovery_worker_socket:
    parser.error("--agent-control requires --recovery-worker-socket")
if args.verify_blue_placement and (not args.agent_control or args.task != "ToolsPickingHammerTask"):
    parser.error("--verify-blue-placement requires Agent-only ToolsPickingHammerTask")
args.recovery_initialization = {"kind": "seeded_reset", "seed": args.seed}
if bool(args.recovery_genesis) != bool(args.recovery_genesis_sha256):
    parser.error("--recovery-genesis and --recovery-genesis-sha256 must be supplied together")
if args.recovery_genesis:
    if not args.recovery_worker_socket:
        parser.error("--recovery-genesis requires --recovery-worker-socket")
    if hashlib.sha256(args.recovery_genesis.read_bytes()).hexdigest() != args.recovery_genesis_sha256:
        parser.error("Historical recovery genesis SHA-256 mismatch")
if args.sim_gpu not in range(8):
    parser.error("--sim-gpu must be a physical GPU index in 0..7")
if args.initialization_diagnostic and (not args.task or args.worker_socket or args.recovery_worker_socket or args.rollout_chunks or args.replay_actions):
    parser.error("--initialization-diagnostic requires a task and cannot execute policy/replay/worker commands")
if args.worker_socket and (not args.task or args.rollout_chunks or args.replay_actions or args.recovery_worker_socket):
    parser.error("--worker-socket requires a task and cannot be combined with sequential rollout/replay")
if args.recovery_worker_socket and (not args.task or args.rollout_chunks or args.replay_actions):
    parser.error("--recovery-worker-socket requires a task and cannot be combined with rollout/replay")
if args.rollout_chunks < 0 or not 1 <= args.execute_steps <= 32:
    parser.error("rollout-chunks must be nonnegative and execute-steps must be in 1..32")
if args.replay_steps < 0 or bool(args.replay_actions) != bool(args.replay_steps):
    parser.error("--replay-actions and a positive --replay-steps must be provided together")
if args.replay_state and not args.replay_actions:
    parser.error("--replay-state requires --replay-actions")
if args.render_width < 320 or args.render_height < 240:
    parser.error("render-width and render-height must be at least 320x240")
if args.rollout_chunks and not args.task:
    parser.error("--rollout-chunks requires --task")
if args.rollout_chunks or args.worker_socket or (args.recovery_worker_socket and not args.agent_control):
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{args.policy_port}/healthz", timeout=5) as response:
        if response.status != 200:
            raise SystemExit("Policy service is not ready.")
sys.argv = [sys.argv[0]]
if (args.output_dir / "result.json").exists():
    raise SystemExit("Use a new output directory; existing results must not be overwritten.")
gpu_uuid = subprocess.check_output([
    "nvidia-smi", "-i", str(args.sim_gpu), "--query-gpu=uuid", "--format=csv,noheader",
], text=True).strip()
if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
    raise SystemExit(f"CUDA_VISIBLE_DEVICES must contain only physical GPU {args.sim_gpu}'s UUID ({gpu_uuid}).")
args.output_dir.mkdir(parents=True, exist_ok=True)

result = {
    "status": "starting",
    "cuda_visible_devices": gpu_uuid,
    "renderer_active_gpu": args.sim_gpu,
    "seed": args.seed,
    "physics_device": "cpu",
    "renderer": "RaytracedLighting",
    "multi_gpu": False,
    "model_loaded": False,
    "task": args.task,
    "rollout_chunks_requested": args.rollout_chunks,
    "replay_actions": str(args.replay_actions) if args.replay_actions else None,
    "replay_steps": args.replay_steps,
    "camera_capture_fps": 15 if args.rollout_chunks else None,
    "renderer_resolution": [args.render_width, args.render_height],
    "rendering_quality": args.rendering_mode,
}


def save_result():
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def set_stage(stage):
    result["status"] = stage
    save_result()
    faulthandler.dump_traceback_later(90, repeat=False)


save_result()
faulthandler.dump_traceback_later(90, repeat=False)
simulation_app = None
env = None
client = None
exit_code = 1
try:
    from isaacsim import SimulationApp

    dlss_mode = {"performance": 0, "balanced": 1, "quality": 2}[args.rendering_mode]
    simulation_app = SimulationApp({
        "headless": True,
        "active_gpu": args.sim_gpu,
        "physics_gpu": 0,
        "multi_gpu": False,
        "width": args.render_width,
        "height": args.render_height,
        "renderer": "RaytracedLighting",
        "anti_aliasing": 0,
        "denoiser": False,
        "enable_crashreporter": False,
        "disable_viewport_updates": False,
        "limit_cpu_threads": 4,
        "extra_args": [
            "--/isaaclab/render/offscreen=true",
            "--/physics/fabricUpdateTransformations=true",
            "--/physics/fabricUpdateJointStates=true",
            "--/physics/fabricUseGPUInterop=false",
            "--/renderer/multiGpu/autoEnable=false",
            "--/rtx/reflections/enabled=false",
            "--/rtx/indirectDiffuse/enabled=false",
            "--/rtx/translucency/enabled=false",
            f"--/rtx/post/dlss/execMode={dlss_mode}",
            "--/rtx/post/aa/op=0",
            "--/rtx-transient/dldenoiser/enabled=false",
            "--/rtx-transient/dlssg/enabled=false",
            "--/rtx/directLighting/sampledLighting/enabled=false",
            "--/rtx/directLighting/sampledLighting/denoisingTechnique=0",
        ],
    }, experience=str(
        Path(sys.prefix) / "lib/python3.11/site-packages/isaaclab/apps/isaaclab.python.headless.rendering.kit"
    ))
    set_stage("app_initialized")

    import numpy as np
    import omni.replicator.core as rep
    from omni.physx import acquire_physx_interface
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from PIL import Image

    def save_frame(frame, name):
        if frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
            raise RuntimeError(f"No valid image for {name}: shape={frame.shape}")
        rgb = frame[:, :, :3]
        spatial_std = float(rgb.astype(np.float32).std(axis=(0, 1)).max())
        if not np.isfinite(rgb).all() or spatial_std < 1.0:
            raise RuntimeError(f"Image {name} is invalid or spatially uniform.")
        path = args.output_dir / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb.astype(np.uint8)).save(path)
        return {
            "path": str(path.relative_to(args.output_dir)), "shape": list(rgb.shape),
            "spatial_std": spatial_std, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def capture_observation(observation, prefix):
        frames = []
        for group in ("image_obs", "viewport_cam"):
            for name, tensor in observation[group].items():
                frames.append(save_frame(tensor[0].detach().cpu().numpy(), f"{prefix}{name}"))
        return frames

    def save_prediction_video(video, relative_path):
        video = np.asarray(video)
        if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] < 2:
            raise ValueError(f"Invalid predicted video: shape={video.shape}, dtype={video.dtype}")
        future = np.ascontiguousarray(video[1:])
        path = args.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.mp4")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to save predicted videos")
        try:
            subprocess.run([
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-video_size", f"{future.shape[2]}x{future.shape[1]}", "-framerate", "15",
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ], input=future.tobytes(), check=True)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "path": str(path.relative_to(args.output_dir)),
            "source_shape": list(video.shape),
            "future_shape": list(future.shape),
            "fps": 15,
        }

    acquire_physx_interface().overwrite_gpu_setting(0)
    if args.task:
        import robolab.constants
        from isaaclab.envs.mdp import reset_scene_to_default
        from isaaclab.managers import EventTermCfg
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        robolab.constants.set_output_dir(str(args.output_dir))
        auto_register_droid_envs(task=[args.task], cameras=WRIST_LEFT_RIGHT_HEAD)
        task_envs = get_envs(task=[args.task])
        if len(task_envs) != 1:
            raise RuntimeError(f"Expected one registered task, got {task_envs}")
        env, env_cfg = create_env(
            task_envs[0], device="cpu", use_fabric=True, num_envs=1,
            seed=args.seed, rendering_mode=args.rendering_mode,
            events={"reset": EventTermCfg(func=reset_scene_to_default, mode="reset",
                                         params={"reset_joint_targets": True})},
        )
        if args.recovery_worker_socket:
            env.cfg.reset_early_termination = False
        if args.verify_blue_placement:
            from isaaclab.managers import TerminationTermCfg
            from blue_placement_verification import strict_blue_placement_success

            env._blue_native_success_cfg = env.termination_manager.get_term_cfg("success")
            env.termination_manager.set_term_cfg(
                "success", TerminationTermCfg(func=strict_blue_placement_success)
            )
        if env.device != "cpu":
            raise RuntimeError(f"Unexpected physics device: {env.device}")
        result["render_mode"] = env.sim.render_mode.name
        result["fabric_sync"] = env.sim.is_fabric_enabled()
        if env.sim.render_mode == env.sim.RenderMode.NO_GUI_OR_RENDERING:
            raise RuntimeError("Offscreen rendering is not enabled for RoboLab.")
        if args.rollout_chunks:
            env.recorder_manager.set_hdf5_file("rollout.hdf5")
            env.recorder_manager.set_episode_index(0, env_ids=[0])
        set_stage("resetting_robolab")
        def record_initialization(stage):
            if not args.initialization_diagnostic:
                return
            robot = env.scene["robot"]
            data = robot.data
            values = {"stage": stage, "joint_names": list(data.joint_names)}
            for name in ("joint_pos", "joint_vel", "default_joint_pos", "joint_pos_target", "joint_vel_target", "joint_effort_target"):
                value = getattr(data, name, None)
                values[name] = value.detach().cpu().numpy().tolist() if value is not None else None
            result.setdefault("initialization_trace", []).append(values)
            save_result()

        record_initialization("before_reset")
        env.reset()
        record_initialization("after_reset")
        for step in range(8):
            set_stage(f"stepping_robolab_{step}")
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(env.physics_dt)
            record_initialization(f"after_physics_step_{step + 1}")
        set_stage("reading_robolab_observations")
        observation = env.observation_manager.compute()
        if args.recovery_genesis:
            from instruction_sampling_worker import flat_state, restore_scene_state, state_error

            set_stage("initializing_recovery_genesis")
            with np.load(args.recovery_genesis, allow_pickle=False) as saved:
                genesis = {name: saved[name].copy() for name in saved.files}
            observation = restore_scene_state(env, genesis)
            error = state_error(flat_state(env.scene.get_state(is_relative=True)), genesis)
            if error > 1e-5 or int(env.episode_length_buf[0]) != 0:
                raise RuntimeError(f"Recovery genesis initialization failed: state error={error}")
            np.savez(args.output_dir / "initial_genesis.npz", **genesis)
            args.recovery_initialization = {
                "kind": "historical_genesis_before_execution",
                "seed": args.seed,
                "source": str(args.recovery_genesis.resolve()),
                "sha256": args.recovery_genesis_sha256,
                "max_state_error": error,
                "episode_step": 0,
            }
            result["recovery_initialization"] = args.recovery_initialization
        result["frames"] = capture_observation(observation, "")
        if not result["frames"]:
            raise RuntimeError("RoboLab returned no camera images.")
        result["physics_steps"] = 8
        if args.initialization_diagnostic:
            from instruction_sampling_worker import flat_state, restore_scene_state

            saved_scene = flat_state(env.scene.get_state(is_relative=True))
            robot = env.scene["robot"]
            robot.set_joint_position_target(robot.data.default_joint_pos * 0)
            env.reset_eval_state()
            env.reset(seed=args.seed)
            record_initialization("repeated_reset")
            for step in range(8):
                env.scene.write_data_to_sim()
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
                record_initialization(f"repeated_reset_step_{step + 1}")
            robot.set_joint_position_target(robot.data.default_joint_pos * 0)
            observation = restore_scene_state(env, saved_scene)
            record_initialization("restored_checkpoint")
            for step in range(8):
                env.scene.write_data_to_sim()
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
                record_initialization(f"restored_checkpoint_step_{step + 1}")
            observation = env.observation_manager.compute()
            result["restored_frames"] = capture_observation(observation, "restored/")
        if args.worker_socket:
            from instruction_sampling_worker import serve

            set_stage("sampling_worker")
            faulthandler.cancel_dump_traceback_later()
            serve(env, env_cfg, args)
        if args.recovery_worker_socket:
            from recovery_worker import serve

            set_stage("recovery_worker")
            faulthandler.cancel_dump_traceback_later()
            serve(env, env_cfg, args, observation)
            manifest = json.loads((args.output_dir / "recovery/manifest.json").read_text())
            result.update(
                model_loaded=not args.agent_control,
                native_success=manifest["native_success"],
                acceptance_success=manifest["acceptance_success"],
                agent_control_success=manifest["agent_control_success"],
                recovery_manifest="recovery/manifest.json",
                steps_executed=manifest["step"],
            )
        if args.rollout_chunks:
            import time
            import torch
            from policies.cosmos3.client import Cosmos3Client

            if args.replay_actions:
                replay_actions = np.load(args.replay_actions)
                if replay_actions.ndim != 2 or replay_actions.shape[1] != 8:
                    raise ValueError(f"Replay actions must have shape (N, 8), got {replay_actions.shape}")
                if len(replay_actions) < args.replay_steps or not np.isfinite(replay_actions[:args.replay_steps]).all():
                    raise ValueError("Replay action prefix is missing or contains non-finite values")
                set_stage("replaying_actions")
                for replay_step, replay_action in enumerate(replay_actions[:args.replay_steps]):
                    observation, _, terminated, truncated, _ = env.step(
                        torch.as_tensor(replay_action, dtype=torch.float32, device=env.device).unsqueeze(0))
                    if terminated[0] or truncated[0]:
                        raise RuntimeError(f"Replay terminated unexpectedly at step {replay_step + 1}")
                result["replay_complete"] = True
                if args.replay_state:
                    expected = np.load(args.replay_state)
                    actual_joint = observation["proprio_obs"]["arm_joint_pos"][0].cpu().numpy()
                    actual_gripper = observation["proprio_obs"]["gripper_pos"][0].cpu().numpy()
                    joint_error = float(np.max(np.abs(actual_joint - expected["joint_position"])))
                    gripper_error = float(np.max(np.abs(actual_gripper - expected["gripper_position"])))
                    result["replay_state_check"] = {
                        "expected": str(args.replay_state),
                        "joint_max_abs_error": joint_error,
                        "gripper_max_abs_error": gripper_error,
                    }
                    if joint_error > 1e-5 or gripper_error > 1e-5:
                        raise RuntimeError(f"Replay state mismatch: joint={joint_error}, gripper={gripper_error}")
                result["replay_frames"] = capture_observation(observation, "replay/")

            instruction = args.instruction or env_cfg.instruction

            result.update({"mode": "sequential_closed_loop", "chunks": [], "steps_executed": 0,
                           "instruction": instruction, "native_success": False,
                           "native_terminated": False, "native_truncated": False})

            class RecordingCosmosClient(Cosmos3Client):
                def _query_server(self, request):
                    chunk_index = len(result["chunks"])
                    directory = args.output_dir / "chunks" / f"{chunk_index:03d}"
                    directory.mkdir(parents=True, exist_ok=False)
                    request["session_id"] = args.output_dir.name
                    input_frame = save_frame(request["observation/image"], f"chunks/{chunk_index:03d}/input")
                    np.savez(directory / "state.npz", joint_position=request["observation/joint_position"],
                             gripper_position=request["observation/gripper_position"])
                    set_stage(f"policy_request_{chunk_index}")
                    started = time.monotonic()
                    reply = self.client.request(request, timeout=240)
                    if not isinstance(reply, dict) or "video" not in reply:
                        raise ValueError("Policy response did not contain a predicted video")
                    prediction_video = save_prediction_video(
                        reply["video"], f"chunks/{chunk_index:03d}/prediction.mp4")
                    raw = self._unpack_response(reply)
                    if raw.shape != (32, 8) or not np.isfinite(raw).all():
                        raise ValueError(f"Invalid model action chunk: {raw.shape}")
                    np.save(directory / "raw_actions.npy", raw)
                    result["chunks"].append({"index": chunk_index,
                        "input": input_frame, "start_step": result["steps_executed"],
                        "prompt": request["prompt"], "generated_shape": list(raw.shape),
                        "prediction_video": prediction_video,
                        "inference_seconds": time.monotonic() - started, "executed_steps": 0})
                    save_result()
                    return reply

                def _postprocess_chunk(self, chunk):
                    processed = super()._postprocess_chunk(chunk)
                    directory = args.output_dir / "chunks" / f"{len(result['chunks']) - 1:03d}"
                    np.save(directory / "actions.npy", processed)
                    return processed

            client = RecordingCosmosClient(remote_host="127.0.0.1", remote_port=args.policy_port)
            client.open_loop_horizon = args.execute_steps
            client.begin_episode(0)
            executed = []
            proprioception = [observation["proprio_obs"]["arm_joint_pos"][0].cpu().numpy().copy()]
            for step in range(args.rollout_chunks * args.execute_steps):
                prediction = client.infer(observation, instruction, env_id=0)
                action = np.asarray(prediction["action"], dtype=np.float32)
                if action.shape != (8,) or not np.isfinite(action).all():
                    raise ValueError("Invalid executable action")
                set_stage(f"executing_policy_step_{step}")
                observation, reward, terminated, truncated, info = env.step(
                    torch.as_tensor(action, device=env.device).unsqueeze(0))
                executed.append(action.copy())
                proprioception.append(observation["proprio_obs"]["arm_joint_pos"][0].cpu().numpy().copy())
                result["steps_executed"] = step + 1
                result["chunks"][-1]["executed_steps"] += 1
                result["native_terminated"] = bool(terminated[0].item())
                result["native_truncated"] = bool(truncated[0].item())
                result["native_success"] = bool(env.termination_manager.get_term("success")[0].item())
                np.save(args.output_dir / "executed_actions.npy", np.stack(executed))
                np.save(args.output_dir / "observed_joints.npy", np.stack(proprioception))
                capture_observation(observation, f"frames/step_{step + 1:04d}/")
                save_result()
                if terminated[0] or truncated[0]:
                    break
            for chunk in result["chunks"]:
                directory = args.output_dir / "chunks" / f"{chunk['index']:03d}"
                planned = np.load(directory / "actions.npy")[:chunk["executed_steps"]]
                actual = np.stack(executed)[chunk["start_step"]:chunk["start_step"] + chunk["executed_steps"]]
                chunk["executed_prefix_matches"] = bool(np.array_equal(planned.astype(np.float32), actual))
                if not chunk["executed_prefix_matches"]:
                    raise RuntimeError("Executed actions differ from the model chunk prefix")
            if not env.all_terminated:
                env.recorder_manager.export_episodes(env_ids=[0])
            result["final_frames"] = capture_observation(observation, "final/")
            result["native_results"] = env.get_env_results()
            result["closed_loop_complete"] = (
                len(result["chunks"]) == args.rollout_chunks
                and result["steps_executed"] == args.rollout_chunks * args.execute_steps)
    else:
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        world.scene.add_default_ground_plane()
        world.scene.add(DynamicCuboid(
            prim_path="/World/ProbeCube",
            position=np.array([0.0, 0.0, 0.6]),
            scale=np.array([0.6, 0.6, 0.6]),
            color=np.array([0.8, 0.15, 0.1]),
        ))
        rep.create.light(light_type="Dome", intensity=1000.0)
        camera = rep.create.camera(position=(2.0, 2.0, 1.8), look_at=(0.0, 0.0, 0.3))
        product = rep.create.render_product(camera, (320, 240))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach([product])
        set_stage("resetting_cube_scene")
        world.reset()
        set_stage("rendering_cube_scene")
        for step in range(12):
            world.step(render=True)
        set_stage("capturing_cube_frame")
        frame = np.asarray(annotator.get_data())
        result["frames"] = [save_frame(frame, "camera")]
    result["status"] = ("rollout_complete" if result["closed_loop_complete"] else "task_terminated") if args.rollout_chunks else "frame_captured"
    if args.worker_socket:
        result["status"] = "sampling_worker_stopped"
    if args.recovery_worker_socket:
        result["status"] = "recovery_worker_stopped"
    exit_code = 0
except Exception as error:
    result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    traceback.print_exc()
finally:
    faulthandler.cancel_dump_traceback_later()
    save_result()
    if client is not None:
        client.close()
    if env is not None:
        env.close()
    if simulation_app is not None:
        simulation_app.close(wait_for_replicator=False)
sys.exit(exit_code)