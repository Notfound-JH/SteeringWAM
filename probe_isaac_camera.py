import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--task", help="Capture a registered RoboLab task instead of the small cube scene.")
parser.add_argument("--rollout-chunks", type=int, default=0, help="Run this many sequential policy chunks; zero only captures cameras.")
parser.add_argument("--execute-steps", type=int, default=16)
parser.add_argument("--policy-port", type=int, default=18745)
args = parser.parse_args()
if args.rollout_chunks < 0 or not 1 <= args.execute_steps <= 32:
    parser.error("rollout-chunks must be nonnegative and execute-steps must be in 1..32")
if args.rollout_chunks and not args.task:
    parser.error("--rollout-chunks requires --task")
if args.rollout_chunks:
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{args.policy_port}/healthz", timeout=5) as response:
        if response.status != 200:
            raise SystemExit("Policy service is not ready.")
sys.argv = [sys.argv[0]]
if (args.output_dir / "result.json").exists():
    raise SystemExit("Use a new output directory; existing results must not be overwritten.")
args.output_dir.mkdir(parents=True, exist_ok=True)
gpu_uuid = "GPU-9bda66e3-4891-0eb0-79dd-c3530fa33f8a"
if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
    raise SystemExit("This probe requires CUDA_VISIBLE_DEVICES to contain only physical GPU 4's UUID.")

result = {
    "status": "starting",
    "cuda_visible_devices": gpu_uuid,
    "renderer_active_gpu": 4,
    "physics_device": "cpu",
    "renderer": "RaytracedLighting",
    "multi_gpu": False,
    "model_loaded": False,
    "task": args.task,
    "rollout_chunks_requested": args.rollout_chunks,
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

    simulation_app = SimulationApp({
        "headless": True,
        "active_gpu": 4,
        "physics_gpu": 0,
        "multi_gpu": False,
        "width": 320,
        "height": 240,
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
            "--/rtx/post/dlss/execMode=0",
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

    acquire_physx_interface().overwrite_gpu_setting(0)
    if args.task:
        import robolab.constants
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
            seed=3407, rendering_mode="performance",
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
        env.reset()
        for step in range(8):
            set_stage(f"stepping_robolab_{step}")
            env.sim.step(render=True)
            env.scene.update(env.physics_dt)
        set_stage("reading_robolab_observations")
        observation = env.observation_manager.compute()
        result["frames"] = capture_observation(observation, "")
        if not result["frames"]:
            raise RuntimeError("RoboLab returned no camera images.")
        result["physics_steps"] = 8
        if args.rollout_chunks:
            import time
            import torch
            from policies.cosmos3.client import Cosmos3Client

            result.update({"mode": "sequential_closed_loop", "chunks": [], "steps_executed": 0,
                           "instruction": env_cfg.instruction, "native_success": False,
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
                    raw = self._unpack_response(reply)
                    if raw.shape != (32, 8) or not np.isfinite(raw).all():
                        raise ValueError(f"Invalid model action chunk: {raw.shape}")
                    np.save(directory / "raw_actions.npy", raw)
                    result["chunks"].append({"index": chunk_index,
                        "input": input_frame, "start_step": result["steps_executed"],
                        "prompt": request["prompt"], "generated_shape": list(raw.shape),
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
                prediction = client.infer(observation, env_cfg.instruction, env_id=0)
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
                if (step + 1) % 4 == 0 or terminated[0] or truncated[0]:
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