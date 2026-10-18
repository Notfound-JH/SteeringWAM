"""Persistent live-environment worker for closed-loop WAM recovery experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import traceback
import uuid
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image

from instruction_sampling import digest, write_json
from instruction_sampling_worker import artifact, flat_state
from recovery_protocol import (
    PROTOCOL_VERSION,
    RECOVERY_EVALUATION_STEPS,
    SafetyLimits,
    compile_program,
    validate_assessment,
    validate_condition_change,
    validate_failure_decision,
    validate_recovery_proposal,
)


FORMAT = "wam-agent-recovery-v1"

if TYPE_CHECKING:
    import torch


def _limit_ik_step(delta: torch.Tensor, max_step_rad: float) -> torch.Tensor:
    # Independent joint clipping can reverse Cartesian motion near singularities.
    scale = max_step_rad / delta.abs().amax(dim=-1, keepdim=True).clamp_min(max_step_rad)
    return delta * scale


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, allow_nan=False, default=_jsonable, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _encode_video(frames: list[Path], destination: Path, fps: float) -> dict[str, Any] | None:
    if not frames:
        return None
    frame_dir = destination.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=False)
    for index, source in enumerate(frames):
        target = frame_dir / f"{index:06d}.png"
        target.symlink_to(source.resolve())
    ffmpeg = Path(__file__).resolve().parent / "packages/ffmpeg/bin/ffmpeg"
    executable = str(ffmpeg) if ffmpeg.exists() else shutil.which("ffmpeg")
    if not executable:
        raise FileNotFoundError("ffmpeg is required to encode recovery videos")
    subprocess.run(
        [
            executable, "-v", "error", "-n", "-framerate", str(fps),
            "-i", str(frame_dir / "%06d.png"), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(destination),
        ],
        check=True,
    )
    subprocess.run([executable, "-v", "error", "-i", str(destination), "-f", "null", "-"], check=True)
    for link in frame_dir.iterdir():
        link.unlink()
    frame_dir.rmdir()
    return artifact(destination)


class RecoverySession:
    protocol_version = PROTOCOL_VERSION

    def __init__(self, env, env_cfg, args, observation):
        import h5py  # noqa: F401 - validates the recorder dependency before execution
        from policies.cosmos3.client import Cosmos3Client
        from robolab.eval.websocket_transport import MsgPackWebSocketTransport

        self.env = env
        self.env_cfg = env_cfg
        self.args = args
        self.directory = args.output_dir.resolve() / "recovery"
        self.directory.mkdir(exist_ok=False)
        self.events_path = self.directory / "events.jsonl"
        self.observation = observation
        self.step = 0
        self.attempts = 0
        self.segment_index = 0
        self.closed = False
        self.event_index = 0
        self.continuation_steps = 0
        self.control_programs = 0
        self.agent_control = args.agent_control
        self.actual_video_frames: list[Path] = []
        self.limits = SafetyLimits(max_attempts=args.recovery_max_attempts)
        self.client = Cosmos3Client.__new__(Cosmos3Client)
        self.client._image_h, self.client._image_w, self.client._eval_episode_idx = 360, 640, 0
        self.transport = None
        if not self.agent_control:
            self.transport = MsgPackWebSocketTransport(
                f"ws://127.0.0.1:{args.policy_port}", metadata_timeout=15
            )
            self.transport.connect()
        self.state = {
            "format": FORMAT,
            "protocol_version": self.protocol_version,
            "task": args.task,
            "native_goal": env_cfg.instruction,
            "status": "ready_agent" if self.agent_control else "ready_wam",
            "experiment_mode": "agent_control" if self.agent_control else "wam_recovery",
            "success_criterion": (
                "native_and_stable_blue_placement" if args.verify_blue_placement else "native"
            ),
            "control_programs": 0,
            "request_id": uuid.uuid4().hex,
            "step": 0,
            "attempts": 0,
            "failure": None,
            "local_recovery_success": None,
            "recovery_evaluation_steps": (
                RECOVERY_EVALUATION_STEPS
                if not self.agent_control and self.protocol_version == PROTOCOL_VERSION else None
            ),
            "recovery_evaluation": None,
            **({"evaluation_policy": "task_progress", "task_progress_recovered": None}
               if self.protocol_version != PROTOCOL_VERSION else {}),
            "budget_exhaustion_policy": "continue_wam" if args.continue_wam_after_budget else "stop",
            "budget_fallback": False,
            "wam_after_budget_steps": 0,
            "max_episode_steps": int(env.max_episode_length),
            "native_success": False,
            "native_terminated": False,
            "native_truncated": False,
            "safety_limits": self.limits.to_dict(),
            "cartesian_controller": {
                "method": "dls", "damping": 0.1,
                "joint_step_limit": "uniform_scale_then_soft_limits",
            },
            "control_dt": float(env.step_dt),
            "reset_after_initialization_allowed": False,
            "initialization": args.recovery_initialization,
            "agent": {"interface": "local_socket", "execution": "literal_controller_DSL_no_exec"},
        }
        if args.task == "ToolsPickingAllHammersTask":
            self.state.update(self._events())
        recorder = env.recorder_manager
        recorder.clear(env_ids=[0])
        recorder.set_hdf5_file(str(self.directory / "trajectory.hdf5"))
        recorder.set_episode_index(0, env_ids=[0])
        if env.cfg.reset_early_termination:
            raise ValueError("Recovery requires disabling early-termination resets")
        initial = self._snapshot("initial", video_frame=True)
        self.state["latest_observation"] = initial
        self._event("session_started", observation=initial, state=self._robot_state())
        self._save_state()

    def _save_state(self) -> None:
        self.state.update(step=self.step, attempts=self.attempts)
        write_json(self.directory / "session.json", self.state)

    def _event(self, event_type: str, **payload: Any) -> None:
        _append_jsonl(
            self.events_path,
            {
                "event_index": self.event_index,
                "event_type": event_type,
                "recorded_at": time.time(),
                "episode_step": self.step,
                **payload,
            },
        )
        self.event_index += 1

    def _rotate_token(self) -> None:
        self.state["request_id"] = uuid.uuid4().hex

    def _robot_state(self) -> dict[str, Any]:
        obs = self.observation
        robot = self.env.scene["robot"]
        joint_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
        result = {
            "frame": "robot-root",
            "quaternion_order": "wxyz",
            "ee_link": "base_link",
            "arm_joint_pos": obs["proprio_obs"]["arm_joint_pos"][0].detach().cpu().tolist(),
            "gripper_pos": obs["proprio_obs"]["gripper_pos"][0].detach().cpu().tolist(),
            "ee_pos": obs["proprio_obs"]["ee_pos"][0].detach().cpu().tolist(),
            "ee_quat": obs["proprio_obs"]["ee_quat"][0].detach().cpu().tolist(),
            "measured_arm_joint_pos": robot.data.joint_pos[0, joint_ids].detach().cpu().tolist(),
            "measured_arm_joint_vel": robot.data.joint_vel[0, joint_ids].detach().cpu().tolist(),
            "joint_position_target": robot.data.joint_pos_target[0].detach().cpu().tolist(),
        }
        if self.agent_control:
            result.update(
                joint_names=list(robot.joint_names),
                measured_joint_pos=robot.data.joint_pos[0].detach().cpu().tolist(),
                measured_joint_vel=robot.data.joint_vel[0].detach().cpu().tolist(),
                body_names=list(robot.body_names),
                body_pos_w=robot.data.body_pos_w[0].detach().cpu().tolist(),
                body_quat_w=robot.data.body_quat_w[0].detach().cpu().tolist(),
            )
        return result

    def _events(self, terminated=None, truncated=None) -> dict[str, Any]:
        if terminated is None:
            terminated = [self.state["native_terminated"]]
        if truncated is None:
            truncated = [self.state["native_truncated"]]
        result = {
            "native_success": bool(self.env.termination_manager.get_term("success")[0]),
            "native_terminated": bool(terminated[0]),
            "native_truncated": bool(truncated[0]),
            "episode_step": int(self.env.episode_length_buf[0]),
        }
        if self.args.task == "ToolsPickingAllHammersTask":
            success_cfg = self.env.termination_manager.get_term_cfg("success")
            objects = success_cfg.params["object"]
            expected = {"husky_hammer", "blue_hammer", "red_hammer", "wood_hammer"}
            if set(objects) != expected or success_cfg.params["logical"] != "all":
                raise ValueError("Four-hammer progress requires the unchanged all-hammers checker")
            per_object = {
                name: bool(success_cfg.func(
                    self.env, **{**success_cfg.params, "object": [name]}
                )[0])
                for name in objects
            }
            if all(per_object.values()) != result["native_success"]:
                raise ValueError("Per-hammer completion disagrees with native all-hammers success")
            result["hammer_task_progress"] = {
                "criterion": "native_success_predicate_per_object",
                "per_object": per_object,
                "completed_count": sum(per_object.values()),
                "required_count": len(objects),
                "remaining_objects": [name for name, complete in per_object.items() if not complete],
            }
        if self.args.verify_blue_placement:
            result["placement_verification"] = getattr(self.env, "_blue_placement_sample", None)
        return result

    def _snapshot(self, label: str, *, video_frame: bool = False) -> dict[str, Any]:
        directory = self.directory / "observations" / f"{self.step:05d}_{label}"
        directory.mkdir(parents=True, exist_ok=False)
        frames = []
        for group in ("image_obs", "viewport_cam"):
            for camera, tensor in self.observation[group].items():
                rgb = tensor[0].detach().cpu().numpy()[..., :3].astype(np.uint8)
                if not rgb.size or float(rgb.std(axis=(0, 1)).max()) < 1:
                    raise ValueError(f"Blank camera frame: {camera}")
                path = directory / f"{camera}.png"
                Image.fromarray(rgb).save(path)
                frames.append({
                    **artifact(path),
                    "camera": camera,
                    "view": "viewport" if group == "viewport_cam" else "actual",
                    "debug_camera": group == "viewport_cam",
                })
                if video_frame and group == "image_obs" and camera == "wrist_cam":
                    self.actual_video_frames.append(path)
        state_path = directory / "scene.npz"
        np.savez(state_path, **flat_state(self.env.scene.get_state(is_relative=True)))
        robot_path = directory / "robot.json"
        write_json(robot_path, self._robot_state())
        return {
            "label": label,
            "step": self.step,
            "frames": frames,
            "robot": artifact(robot_path),
            "scene": artifact(state_path),
        }

    def _save_predicted_video(self, video: Any, directory: Path) -> dict[str, Any]:
        array = np.asarray(video)
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3 or len(array) < 2:
            raise ValueError(f"Invalid predicted future video: {array.shape} {array.dtype}")
        np.save(directory / "predicted_future.npy", array)
        frames = []
        frame_dir = directory / "predicted_future_frames"
        frame_dir.mkdir()
        for index, frame in enumerate(array[1:]):
            path = frame_dir / f"{index:06d}.png"
            Image.fromarray(frame).save(path)
            frames.append(path)
        video_ref = _encode_video(
            frames, directory / "predicted_future.mp4", 1.0 / float(self.env.step_dt)
        )
        return {
            "array": artifact(directory / "predicted_future.npy"),
            "video": video_ref,
            "semantics": "model_prediction_not_actual_observation",
        }

    def _query_wam(
        self, condition: str, subgoal: str, execute_steps: int, source: str,
        *, change_kind: str | None = None, change_reason: str | None = None,
    ) -> dict[str, Any]:
        if self.transport is None:
            raise ValueError("WAM is disabled in the agent-control baseline")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError("condition must be nonempty")
        if not isinstance(subgoal, str) or not subgoal.strip():
            raise ValueError("subgoal must be nonempty")
        if type(execute_steps) is not int or not 1 <= execute_steps <= self.limits.max_wam_execute_steps:
            raise ValueError(f"execute_steps must be in [1, {self.limits.max_wam_execute_steps}]")
        change = validate_condition_change(
            self.state.get("condition"), condition, kind=change_kind, reason=change_reason
        )
        condition = change["after"]
        attempt = self.attempts if source == "recovery_evaluation" else None
        directory = self.directory / "segments" / f"{self.segment_index:03d}_{source}"
        self.segment_index += 1
        directory.mkdir(parents=True, exist_ok=False)
        extracted = self.client._extract_observation(self.observation)
        prompt = f"Overall task: {self.env_cfg.instruction}\nCurrent subgoal: {condition}"
        request = self.client._pack_request(extracted, prompt)
        request["session_id"] = self.args.output_dir.name
        Image.fromarray(request["observation/image"]).save(directory / "input.png")
        np.savez(
            directory / "request_state.npz",
            joint_position=request["observation/joint_position"],
            gripper_position=request["observation/gripper_position"],
        )
        write_json(directory / "request.json", {
            "prompt": prompt,
            "condition": condition,
            "subgoal": subgoal,
            "source": source,
            "start_step": self.step,
            "condition_change": change,
            "attempt": attempt,
        })
        started = time.monotonic()
        reply = self.transport.request(request, timeout=240)
        if not isinstance(reply, dict) or "video" not in reply:
            raise ValueError("WAM response must include decoded predicted future video")
        raw = self.client._unpack_response(reply)
        if raw.shape != (32, 8) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid WAM action chunk: {raw.shape}")
        processed = self.client._postprocess_chunk(raw).astype(np.float32)
        np.save(directory / "raw_actions.npy", raw)
        np.save(directory / "predicted_actions.npy", processed)
        predicted_future = self._save_predicted_video(reply["video"], directory)
        prediction = {
            "source": source,
            "prompt": prompt,
            "subgoal": subgoal,
            "raw_actions": artifact(directory / "raw_actions.npy"),
            "predicted_actions": artifact(directory / "predicted_actions.npy"),
            "predicted_future": predicted_future,
            "prediction_length": len(processed),
            "requested_execution_length": execute_steps,
            "inference_seconds": time.monotonic() - started,
            "condition_change": change,
            "attempt": attempt,
        }
        if change["changed"]:
            self._event("condition_changed", source=source, attempt=attempt, **change)
        self._event("wam_prediction", **prediction)
        self.state.update(condition=condition, subgoal=subgoal)
        execution = self._execute_actions(processed[:execute_steps], source, directory)
        write_json(directory / "result.json", {**prediction, **execution})
        return {**prediction, **execution}

    def _execute_actions(
        self, actions: np.ndarray, source: str, directory: Path
    ) -> dict[str, Any]:
        import torch

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("Executed actions must be a nonempty finite (N, 8) array")
        executed = []
        observations = []
        final_events = self._events()
        for index, command in enumerate(actions, 1):
            self._before_step(source, authorized_action=command)
            self.observation, _, terminated, truncated, _ = self.env.step(
                torch.as_tensor(command, device=self.env.device).unsqueeze(0)
            )
            actual = self.env.action_manager.action[0].detach().cpu().numpy().copy()
            if not np.array_equal(actual, command):
                raise ValueError("Environment action differs from the authorized command")
            self.step += 1
            self._check_step()
            if source == "wam_continuation":
                self.continuation_steps += 1
            elif source == "wam_after_budget":
                self.state["wam_after_budget_steps"] += 1
            self.state["last_action_source"] = source
            executed.append(actual)
            final_events = self._events(terminated, truncated)
            observation = self._snapshot(f"{source}_{index:03d}", video_frame=True)
            observations.append(observation)
            self._event(
                "actual_execution_response",
                action_source=source,
                authorized_action=command,
                actual_action=actual,
                robot_state=self._robot_state(),
                observation=observation,
                events=final_events,
            )
            if final_events["native_terminated"] or final_events["native_truncated"]:
                break
        executed_array = np.asarray(executed, dtype=np.float32)
        path = directory / "executed_actions.npy"
        np.save(path, executed_array)
        self.state.update(final_events)
        if final_events["native_success"]:
            self.state["status"] = "task_success"
        elif final_events["native_terminated"] or final_events["native_truncated"]:
            self.state["status"] = "task_failed"
        return {
            "executed_actions": artifact(path),
            "executed_steps": len(executed_array),
            "post_observations": observations,
            **final_events,
        }

    def _execute_cartesian_targets(
        self, targets: np.ndarray, directory: Path, source: str = "agent_correction"
    ) -> dict[str, Any]:
        import torch
        from isaaclab.controllers import DifferentialIKController
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.utils.math import matrix_from_quat, quat_inv

        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim != 2 or targets.shape[1] != 8 or not len(targets):
            raise ValueError("Cartesian targets must have shape (N, 8)")
        controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls",
                ik_params={"lambda_val": 0.1},
            ),
            num_envs=1,
            device=self.env.device,
        )
        robot = self.env.scene["robot"]
        joint_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
        body_ids, _ = robot.find_bodies("base_link", preserve_order=True)
        if len(joint_ids) != 7 or len(body_ids) != 1:
            raise ValueError("Recovery IK requires seven Panda joints and one base_link")
        body_jacobian_index = body_ids[0] - 1 if robot.is_fixed_base else body_ids[0]
        joint_commands = []
        observations = []
        final_events = self._events()
        for index, target in enumerate(targets, 1):
            command = torch.as_tensor(target[:7], device=self.env.device).unsqueeze(0)
            current = self._robot_state()
            ee_pos = torch.as_tensor(current["ee_pos"], device=self.env.device).unsqueeze(0)
            ee_quat = torch.as_tensor(current["ee_quat"], device=self.env.device).unsqueeze(0)
            jacobian = robot.root_physx_view.get_jacobians()[
                :, body_jacobian_index, :, joint_ids
            ].clone()
            inverse_rotation = matrix_from_quat(quat_inv(robot.data.root_quat_w))
            jacobian[:, :3] = inverse_rotation @ jacobian[:, :3]
            jacobian[:, 3:] = inverse_rotation @ jacobian[:, 3:]
            joint_pos = robot.data.joint_pos[:, joint_ids]
            controller.set_command(command)
            joint_target = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
            if not torch.isfinite(joint_target).all():
                raise ValueError("IK returned nonfinite joint targets")
            unconstrained = joint_target.clone()
            joint_target = joint_pos + _limit_ik_step(
                joint_target - joint_pos, self.limits.max_joint_step_rad,
            )
            joint_limits = robot.data.soft_joint_pos_limits[:, joint_ids]
            joint_target = torch.clamp(joint_target, joint_limits[..., 0], joint_limits[..., 1])
            environment_action = torch.cat(
                [
                    joint_target,
                    torch.tensor([[target[-1]]], device=self.env.device),
                ],
                dim=1,
            )
            self._before_step(
                source, cartesian_intent=target,
                unconstrained_ik=unconstrained[0].detach().cpu().numpy(),
                authorized_action=environment_action[0].detach().cpu().numpy(),
            )
            self.observation, _, terminated, truncated, _ = self.env.step(environment_action)
            actual = self.env.action_manager.action[0].detach().cpu().numpy().copy()
            authorized = environment_action[0].detach().cpu().numpy().copy()
            if not np.array_equal(actual, authorized):
                raise ValueError("Environment action differs from the IK-authorized joint command")
            self.step += 1
            self._check_step()
            self.state["last_action_source"] = source
            joint_commands.append(actual)
            final_events = self._events(terminated, truncated)
            observation = self._snapshot(f"{source}_{index:03d}", video_frame=True)
            observations.append(observation)
            self._event(
                "actual_execution_response",
                action_source=source,
                cartesian_intent=target,
                ik_joint_command=authorized,
                actual_action=actual,
                robot_state=self._robot_state(),
                observation=observation,
                events=final_events,
            )
            if final_events["native_terminated"] or final_events["native_truncated"]:
                break
        executed_array = np.asarray(joint_commands, dtype=np.float32)
        path = directory / "executed_joint_actions.npy"
        np.save(path, executed_array)
        self.state.update(final_events)
        if final_events["native_success"]:
            self.state["status"] = "task_success"
        elif final_events["native_terminated"] or final_events["native_truncated"]:
            self.state["status"] = "task_failed"
        return {
            "cartesian_targets": artifact(directory / "compiled_cartesian_targets.npy"),
            "executed_actions": artifact(path),
            "executed_steps": len(executed_array),
            "post_observations": observations,
            **final_events,
        }

    def status(self) -> dict[str, Any]:
        return dict(self.state)

    def control(self, decision: dict[str, Any]) -> dict[str, Any]:
        if not self.agent_control or self.state["status"] != "ready_agent":
            raise ValueError("Direct control requires a ready agent-control baseline")
        if decision.get("correction_mode") != "action":
            raise ValueError("Agent-control programs must use action mode")
        robot = self._robot_state()
        # Reuse all action bounds, but not the separate Recovery attempt budget.
        proposal = validate_recovery_proposal(
            decision, self.state["request_id"], 0, robot["ee_pos"], self.limits
        )
        compiled = compile_program(
            proposal["program"], robot["ee_pos"], robot["ee_quat"],
            robot["gripper_pos"], self.limits,
        )
        self.control_programs += 1
        self.state["control_programs"] = self.control_programs
        directory = self.directory / "control_programs" / f"{self.control_programs:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "proposal.json", proposal)
        (directory / "control_code.txt").write_text(
            proposal.get("control_code", json.dumps(proposal["program"], indent=2)) + "\n"
        )
        np.save(directory / "compiled_cartesian_targets.npy", compiled)
        self._event(
            "agent_control_proposal", program_index=self.control_programs,
            proposal=artifact(directory / "proposal.json"),
        )
        result = self._execute_cartesian_targets(compiled, directory, source="agent_control")
        write_json(directory / "result.json", result)
        self.state["latest_observation"] = self._snapshot(f"control_program_{self.control_programs}")
        self._event("agent_control_completed", program_index=self.control_programs, result=result)
        self._rotate_token()
        self._save_state()
        return {"state": self.status(), "result": result}

    def _before_step(self, source: str, **intent: Any) -> None:
        if self.state["native_terminated"] or self.state["native_truncated"]:
            raise ValueError("Cannot step a terminated environment")
        if self.step >= self.state.get("max_episode_steps", float("inf")):
            raise ValueError("Configured episode step limit exhausted")
        self._check_step()
        self._event("action_authorized", action_source=source, **intent)

    def _check_step(self) -> None:
        if int(self.env.episode_length_buf[0]) != self.step:
            raise RuntimeError("Episode counter discontinuity; possible reset, execution blocked")

    def context(self) -> dict[str, Any]:
        from collections import deque

        with self.events_path.open() as handle:
            recent = [json.loads(line) for line in deque(handle, maxlen=12)]
        return {"state": self.status(), "robot_state": self._robot_state(), "recent_history": recent}

    def run_wam(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.state["status"] != "ready_wam":
            raise ValueError(f"WAM execution is not allowed from {self.state['status']}")
        if decision.get("request_id") != self.state["request_id"]:
            raise ValueError("Stale or incorrect request_id")
        if decision.get("protocol_version") != self.state["protocol_version"]:
            raise ValueError(f"protocol_version must be {self.state['protocol_version']!r}")
        condition = decision.get("condition")
        subgoal = decision.get("subgoal")
        execute_steps = decision.get("execute_steps", self.limits.max_wam_execute_steps)
        source = (
            "wam_after_budget" if self.state.get("budget_fallback")
            else "wam_continuation" if self.state.get(
                "task_progress_recovered", self.state["local_recovery_success"]
            ) else "wam"
        )
        result = self._query_wam(
            condition, subgoal, execute_steps, source,
            change_kind=decision.get("condition_change_kind"),
            change_reason=decision.get("condition_change_reason"),
        )
        self._rotate_token()
        self.state["latest_observation"] = self._snapshot("wam_boundary")
        self._save_state()
        return {"state": self.status(), "result": result}

    def mark_failure(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.state["status"] != "ready_wam":
            raise ValueError(f"Failure can only be marked from ready_wam, not {self.state['status']}")
        if not self.step:
            raise ValueError("Failure requires actual WAM execution first")
        if not self.limits.allows_attempt(self.attempts):
            raise ValueError("Episode recovery attempt limit exhausted")
        validated = validate_failure_decision(
            decision, self.state["request_id"], protocol_version=self.state["protocol_version"]
        )
        self.state.update(
            status="ready_recovery", failure=validated, local_recovery_success=None,
            recovery_evaluation=None,
        )
        self._event(
            "failure_detected",
            failure=validated,
            observation=self.state["latest_observation"],
            robot_state=self._robot_state(),
        )
        self._rotate_token()
        self._save_state()
        return self.status()

    def recover(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.state["status"] != "ready_recovery":
            raise ValueError(f"Recovery is not allowed from {self.state['status']}")
        robot = self._robot_state()
        condition_before = self.state["condition"]
        proposal = validate_recovery_proposal(
            decision,
            self.state["request_id"],
            self.attempts,
            robot["ee_pos"],
            self.limits,
            recovery_condition=condition_before,
        )
        if proposal["correction_mode"] in {"action", "joint"}:
            compiled = compile_program(
                proposal["program"], robot["ee_pos"], robot["ee_quat"],
                robot["gripper_pos"], self.limits,
            )
        self.attempts += 1
        self.state.update(status="recovering", recovery_evaluation=None)
        attempt_dir = self.directory / "recovery_attempts" / f"{self.attempts:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        write_json(attempt_dir / "proposal.json", proposal)
        (attempt_dir / "control_code.txt").write_text(
            proposal.get("control_code", json.dumps(proposal.get("program", []), indent=2)) + "\n"
        )
        self._event(
            "recovery_proposal",
            attempt=self.attempts,
            diagnosis=proposal["diagnosis"],
            recovery_goal=proposal["recovery_goal"],
            correction_mode=proposal["correction_mode"],
            mode_reason=proposal["mode_reason"],
            condition_before=condition_before,
            proposal=artifact(attempt_dir / "proposal.json"),
        )
        results: list[dict[str, Any]] = []
        if proposal["correction_mode"] in {"action", "joint"}:
            compiled_path = attempt_dir / "compiled_actions.npy"
            np.save(compiled_path, compiled)
            cartesian_path = attempt_dir / "compiled_cartesian_targets.npy"
            compiled_path.replace(cartesian_path)
            self._event(
                "correction_program_compiled",
                attempt=self.attempts,
                intent=proposal["program"],
                compiled_cartesian_targets=artifact(cartesian_path),
            )
            results.append(self._execute_cartesian_targets(compiled, attempt_dir))
        condition = proposal.get("condition", condition_before)
        evaluation_start = self.step
        self._event(
            "recovery_evaluation_started", attempt=self.attempts,
            condition_before=condition_before, condition=condition,
            correction_mode=proposal["correction_mode"],
            requested_steps=RECOVERY_EVALUATION_STEPS,
        )
        if not self.state["native_terminated"] and not self.state["native_truncated"]:
            self.state["status"] = "evaluating_recovery"
            results.append(self._query_wam(
                condition,
                self.state["failure"]["subgoal"],
                RECOVERY_EVALUATION_STEPS,
                "recovery_evaluation",
                change_kind="recovery_rewrite" if condition != condition_before else None,
                change_reason=proposal["mode_reason"] if condition != condition_before else None,
            ))
        evaluation = {
            "condition_before": condition_before,
            "condition": condition,
            "start_step": evaluation_start,
            "end_step": self.step,
            "requested_steps": RECOVERY_EVALUATION_STEPS,
            "executed_steps": self.step - evaluation_start,
            "complete": self.step - evaluation_start == RECOVERY_EVALUATION_STEPS,
            "interrupted_by": (
                None if self.step - evaluation_start == RECOVERY_EVALUATION_STEPS
                else "native_success" if self.state["native_success"]
                else "native_horizon" if self.state["native_truncated"]
                else "native_termination" if self.state["native_terminated"] else None
            ),
        }
        self.state["recovery_evaluation"] = evaluation
        self._event("recovery_evaluation_completed", attempt=self.attempts, evaluation=evaluation)
        if not self.state["native_terminated"] and not self.state["native_truncated"]:
            if not evaluation["complete"]:
                raise RuntimeError("Nonterminal recovery evaluation did not execute its full window")
            self.state["status"] = "awaiting_assessment"
        self.state["latest_observation"] = self._snapshot(f"recovery_attempt_{self.attempts}")
        self._rotate_token()
        self._event(
            "recovery_attempt_completed",
            attempt=self.attempts,
            results=results,
            evaluation=evaluation,
            observation=self.state["latest_observation"],
            events=self._events(),
        )
        self._save_state()
        return {"state": self.status(), "results": results}

    def assess(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.state["status"] != "awaiting_assessment":
            raise ValueError(f"Assessment is not allowed from {self.state['status']}")
        evaluation = self.state["recovery_evaluation"]
        if not evaluation or not evaluation["complete"] or evaluation["end_step"] != self.step:
            raise ValueError("Assessment requires the completed, unchanged recovery evaluation window")
        assessment = validate_assessment(decision, self.state["request_id"])
        assessment.update(
            evaluation_steps=evaluation["executed_steps"],
            evaluation_end_step=evaluation["end_step"],
            label_source="agent_assessment",
        )
        self.state["local_recovery_success"] = assessment["local_recovery_success"]
        if assessment["local_recovery_success"]:
            self.state["status"] = "ready_wam"
            self._event("control_handed_back_to_wam", observation=self.state["latest_observation"])
        elif self.limits.allows_attempt(self.attempts):
            self.state["status"] = "ready_recovery"
        elif self.state.get("budget_exhaustion_policy") == "continue_wam":
            self.state.update(status="ready_wam", budget_fallback=True)
            self._event(
                "control_returned_to_wam_after_budget", attempt=self.attempts,
                local_recovery_success=False, observation=self.state["latest_observation"],
            )
        else:
            self.state["status"] = "recovery_failed"
        self._event("recovery_assessment", attempt=self.attempts, assessment=assessment)
        self._rotate_token()
        self._save_state()
        return self.status()

    def _acceptance_success(self) -> bool:
        return bool(
            not self.agent_control and self.state["native_success"] and self.state["failure"]
            and self.state["local_recovery_success"] and self.continuation_steps > 0
            and self.state.get("last_action_source") == "wam_continuation"
            and self.state["status"] == "task_success"
        )

    def finalize(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise ValueError("Session is already finalized")
        if decision.get("request_id") != self.state["request_id"]:
            raise ValueError("Stale or incorrect request_id")
        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A nonempty finalize reason is required")
        recorder = self.env.recorder_manager
        if not self.env.all_terminated:
            recorder.export_episodes(env_ids=[0])
        for name in ("_dataset_file_handler", "_failed_episode_dataset_file_handler"):
            handler = getattr(recorder, name, None)
            if handler is not None:
                handler.close()
        video = _encode_video(
            self.actual_video_frames,
            self.directory / "episode_wrist.mp4",
            1.0 / float(self.env.step_dt),
        )
        self._event("session_finalized", reason=reason, final_state=dict(self.state), video=video)
        manifest = {
            **self.state,
            "finalize_reason": reason,
            "trajectory": artifact(self.directory / "trajectory.hdf5"),
            "events": artifact(self.events_path),
            "episode_video": video,
            "source_identity": {
                "recovery_worker.py": digest(Path(__file__)),
                "recovery_protocol.py": digest(Path(__file__).with_name("recovery_protocol.py")),
                "blue_placement_verification.py": digest(
                    Path(__file__).with_name("blue_placement_verification.py")
                ),
                **({"recovery_progress.py": digest(Path(__file__).with_name("recovery_progress.py"))}
                   if self.protocol_version != PROTOCOL_VERSION else {}),
            },
            "correction_attempts_preserved": self.attempts,
            "task_success_authority": (
                "RoboLab native goal AND strict blue placement verification"
                if self.args.verify_blue_placement else "RoboLab termination_manager.success"
            ),
            "mock_execution": False,
            "wam_continuation_steps": self.continuation_steps,
            "acceptance_success": self._acceptance_success(),
            "agent_control_success": bool(
                self.agent_control and self.state["native_success"]
                and self.state.get("last_action_source") == "agent_control"
                and self.state["status"] == "task_success"
            ),
        }
        write_json(self.directory / "manifest.json", manifest)
        self.state["status"] = "finalized"
        self.state["manifest"] = artifact(self.directory / "manifest.json")
        self._rotate_token()
        self._save_state()
        if self.transport is not None:
            self.transport.close()
        self.closed = True
        return self.status()


def serve(env, env_cfg, args, observation) -> None:
    if args.task_progress_recovery:
        from recovery_progress import ProgressRecoverySession

        session = ProgressRecoverySession(env, env_cfg, args, observation)
    else:
        session = RecoverySession(env, env_cfg, args, observation)
    address = Path(args.recovery_worker_socket)
    if address.exists():
        raise ValueError("Recovery worker socket already exists")
    handlers = {
        "status": lambda _: session.status(),
        "context": lambda _: session.context(),
        "wam": session.run_wam,
        "mark_failure": session.mark_failure,
        "recover": session.recover,
        "control": session.control,
        "assess": session.assess,
        "finalize": session.finalize,
    }
    if args.task_progress_recovery:
        handlers.update(review_progress=session.review_progress,
                        evaluate_condition=session.evaluate_condition)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(address))
            address.chmod(0o600)
            server.listen(1)
            write_json(args.output_dir / "recovery_ready.json", {
                "socket": str(address),
                "task": args.task,
                "status": "ready",
            })
            while not session.closed:
                connection, _ = server.accept()
                with connection:
                    operation = None
                    try:
                        command = json.loads(connection.makefile("rb").readline(8 * 1024 * 1024))
                        operation = command.pop("op", None)
                        if operation not in handlers:
                            raise ValueError(f"Unknown recovery operation: {operation}")
                        if operation not in {"status", "context"}:
                            session._event("command_received", operation=operation, command=command)
                        reply = {"result": handlers[operation](command)}
                    except Exception as error:
                        session._event(
                            "command_failed", operation=operation,
                            error_type=type(error).__name__, error=str(error),
                            traceback=traceback.format_exc(),
                        )
                        if operation in {"wam", "recover", "control", "evaluate_condition"}:
                            session.state["status"] = "execution_unknown"
                            session._rotate_token()
                            session._save_state()
                        reply = {
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    try:
                        connection.sendall(
                            (json.dumps(reply, allow_nan=False, default=_jsonable) + "\n").encode()
                        )
                    except BrokenPipeError:
                        session._event("client_disconnected", operation=operation)
    finally:
        address.unlink(missing_ok=True)
        if not session.closed and session.transport is not None:
            session.transport.close()
