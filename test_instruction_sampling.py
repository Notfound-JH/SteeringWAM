import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import instruction_sampling as sampling
import instruction_sampling_worker as worker
import retry_instruction_sampling as retry
import sampling_multi_env as multi_env


def test_server_explicit_seed_preserves_default_rng():
    import ast
    from types import SimpleNamespace

    path = sampling.ROOT / "packages/cosmos3/cosmos_framework/scripts/action_policy_server_robolab.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_next_seed")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    next_seed = namespace["_next_seed"]
    service = SimpleNamespace(cfg=SimpleNamespace(seed=3407, deterministic_seed=False), _rng=np.random.default_rng(42))
    reference = np.random.default_rng(42)
    assert next_seed(service, 123) == 123
    assert next_seed(service) == int(reference.integers(0, 2**31))
    service.cfg.deterministic_seed = True
    assert next_seed(service) == 3407
    for invalid in (-1, 2**31, True, 1.5, "123"):
        with pytest.raises(ValueError):
            next_seed(service, invalid)


def test_frozen_paired_arms(tmp_path):
    from types import SimpleNamespace
    from paired_instruction_experiment import prepare, load_arm

    scene = tmp_path / "scene.npz"
    np.savez(scene, joints=np.zeros((1, 7)))
    checkpoint = tmp_path / "checkpoint.json"
    sampling.write_json(checkpoint, {"task": "test", "scene": worker.artifact(scene), "parent": None})
    instruction = {"subgoal": "approach", "instruction": "Approach fruit", "avoid": "contact",
                   "completion_condition": "Aligned"}
    for condition in ("a", "b"):
        sampling.write_json(tmp_path / f"{condition}.json", [{**instruction, "instruction": f"Approach fruit {condition}"}] * 12)
        (tmp_path / f"{condition}.txt").write_text(f"Full frozen prompt {condition}")
    sampling.write_json(tmp_path / "seeds.json", list(range(12)))
    args = SimpleNamespace(task="test", checkpoint=checkpoint, instructions_a=tmp_path / "a.json",
                           instructions_b=tmp_path / "b.json", prompt_a=tmp_path / "a.txt",
                           prompt_b=tmp_path / "b.txt", seeds=tmp_path / "seeds.json", output=tmp_path / "plan")
    prepare(args)
    first = load_arm(args.output / "arm_A.json", "test", checkpoint, args.instructions_a)
    second = load_arm(args.output / "arm_B.json", "test", checkpoint, args.instructions_b)
    assert first["seeds"] == second["seeds"]
    assert first["checkpoint"] == second["checkpoint"]
    assert first["best_of_groups"] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    invalid = copy.deepcopy(first)
    invalid["seeds"][0] = True
    sampling.write_json(args.output / "invalid.json", invalid)
    with pytest.raises(ValueError, match="integer seeds"):
        load_arm(args.output / "invalid.json", "test", checkpoint, args.instructions_a)
    args.instructions_a.write_text("[]")
    with pytest.raises(ValueError, match="instructions mismatch"):
        load_arm(args.output / "arm_A.json", "test", checkpoint, args.instructions_a)


def test_blind_export_validates_pairs_and_hides_mapping(tmp_path):
    from types import SimpleNamespace
    from paired_instruction_experiment import export_blind

    arms = []
    for condition in ("A", "B"):
        directory = tmp_path / condition
        directory.mkdir()
        arms.append(directory)
        sampling.write_json(directory / "experiment_arm.json", {"condition": condition,
            "task": "test", "checkpoint": {"sha256": "same"}, "seeds": list(range(12)), "execute_steps": 16})
        candidates = []
        for pair_id in range(12):
            frame = directory / f"frame_{pair_id}.png"
            Image.new("RGB", (8, 8), "green").save(frame)
            candidates.append({"id": f"{condition}-{pair_id}", "experiment": {
                "condition": condition, "pair_id": pair_id, "actual_seed": pair_id},
                "frames": [{**worker.artifact(frame), "step": 0}], "videos": []})
        sampling.write_json(directory / "result.json", {"status": "complete", "candidates": candidates})
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("Judge progress toward the shared target; mark uncertainty.")
    args = SimpleNamespace(arms=arms, output=tmp_path / "blind", mapping=tmp_path / "private.json", protocol=protocol)
    export_blind(args)
    mapping = json.loads(args.mapping.read_text())
    assert len(mapping) == 24
    assert len(list(args.output.glob("sample_*/frames/*/*.png"))) == 24
    assert not list(args.output.rglob("*.json"))
    assert sorted(item["pair_id"] for item in mapping.values() if item["condition"] == "A") == list(range(12))
    from paired_instruction_experiment import summarize

    reviews = tmp_path / "reviews.json"
    sampling.write_json(reviews, {label: {"score": int(entry["condition"] == "B"),
        "unsafe": False, "evidence": "TEST ONLY"} for label, entry in mapping.items()})
    summary = tmp_path / "summary.json"
    summarize(SimpleNamespace(reviews=reviews, mapping=args.mapping, output=summary))
    report = json.loads(summary.read_text())
    assert report["paired_counts"] == {"improved": 12, "tied": 0, "worse": 0, "uncertain": 0}
    assert len(report["best_of_four"]) == 3
    plan_path = arms[1] / "experiment_arm.json"
    plan = json.loads(plan_path.read_text())
    plan["seeds"][0] = 99
    sampling.write_json(plan_path, plan)
    with pytest.raises(ValueError, match="Unmatched arms"):
        export_blind(args)


def test_twelve_env_instructions_state_and_actions(tmp_path):
    instruction = {"subgoal": "approach", "instruction": "Approach fruit", "avoid": "contact",
                   "completion_condition": "Aligned"}
    path = tmp_path / "instructions.json"
    path.write_text(json.dumps([instruction] * 12))
    assert len(multi_env.load_instructions(path)) == 12
    path.write_text(json.dumps([instruction] * 4))
    with pytest.raises(ValueError, match="exactly 12"):
        multi_env.load_instructions(path)
    path.write_text(json.dumps([instruction] * 11 + [{**instruction, "subgoal": "lift"}]))
    with pytest.raises(ValueError, match="subgoal"):
        multi_env.load_instructions(path)
    source = {"robot/joints": np.arange(7).reshape(1, 7)}
    template = {"robot/joints": np.zeros((12, 7))}
    tiled = multi_env.tile_checkpoint(source, template)
    for env_id in range(12):
        np.testing.assert_array_equal(tiled["robot/joints"][env_id], source["robot/joints"][0])
    with pytest.raises(ValueError, match="checkpoint field"):
        multi_env.tile_checkpoint(template, template)
    with pytest.raises(ValueError, match="fields differ"):
        multi_env.tile_checkpoint({}, template)
    chunks = np.arange(12 * 32 * 8, dtype=np.float32).reshape(12, 32, 8)
    actions = multi_env.stack_actions(chunks, 7, [0, 3, 11])
    np.testing.assert_array_equal(actions[[0, 3, 11]], chunks[[0, 3, 11], 7])
    assert not actions[1].any()
    chunks[11, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        multi_env.stack_actions(chunks, 0, [0])


def test_twelve_env_checkpoint_history(tmp_path):
    scene = tmp_path / "scene.npz"
    np.savez(scene, joints=np.zeros((1, 7)))
    first = tmp_path / "first.json"
    sampling.write_json(first, {"task": "test", "scene": worker.artifact(scene), "parent": None})
    second = tmp_path / "second.json"
    sampling.write_json(second, {"task": "test", "scene": worker.artifact(scene), "parent": worker.artifact(first)})
    assert len(multi_env.checkpoint_history(second, "test")) == 2
    with pytest.raises(ValueError, match="task mismatch"):
        multi_env.checkpoint_history(second, "other")
    sampling.write_json(first, {"task": "test", "scene": worker.artifact(scene), "native_terminated": True})
    with pytest.raises(ValueError, match="identity mismatch"):
        multi_env.checkpoint_history(second, "test")
    with pytest.raises(ValueError, match="terminated"):
        multi_env.checkpoint_history(first, "test")


def test_twelve_env_serial_inference_execution_and_resume(tmp_path, monkeypatch):
    import h5py
    import torch
    from types import ModuleType, SimpleNamespace

    requests = []

    class FakeClient:
        def _extract_observation(self, obs, env_id=0):
            return {"env_id": env_id}

        def _pack_request(self, obs, prompt):
            return {**obs, "prompt": prompt, "observation/image": np.full((4, 4, 3), 70, dtype=np.uint8)}

        def _unpack_response(self, response):
            return response["action"] if isinstance(response, dict) else response

        def _postprocess_chunk(self, chunk):
            return chunk

    class FakeTransport:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def close(self):
            pass

        def request(self, request, **kwargs):
            requests.append(request)
            action = np.full((32, 8), request["env_id"] + 1, dtype=np.float32)
            if "request_seed" in request:
                return {"action": action, "actual_seed": request["request_seed"]}
            return action

    for name, attribute, value in (
        ("policies.cosmos3.client", "Cosmos3Client", FakeClient),
        ("robolab.eval.websocket_transport", "MsgPackWebSocketTransport", FakeTransport),
    ):
        module = ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    class FakeEnv:
        device = "cpu"
        physics_dt = 0.01

        def __init__(self):
            self.positions = torch.zeros((12, 7))
            self._frozen_envs = torch.zeros(12, dtype=torch.bool)
            self.episode_length_buf = torch.zeros(12, dtype=torch.long)
            self.sim = SimpleNamespace(step=lambda **kwargs: None)
            self.scene = SimpleNamespace(get_state=lambda **kwargs: {"joints": self.positions.clone()},
                write_data_to_sim=lambda: None, update=lambda dt: None)
            self.action_manager = SimpleNamespace(action=torch.zeros(12, 8))
            self.termination_manager = SimpleNamespace(get_term=lambda name: self._frozen_envs.clone())
            self.buffers = [[] for _ in range(12)]
            self.recording = None
            self.recorder_manager = SimpleNamespace(clear=self.clear, set_hdf5_file=self.open_recording,
                set_episode_index=lambda *args, **kwargs: None, export_episodes=self.export,
                _dataset_file_handler=None, _failed_episode_dataset_file_handler=None)

        def observation(self):
            image = torch.arange(12 * 4 * 4 * 3, dtype=torch.int64).reshape(12, 4, 4, 3).to(torch.uint8)
            return {"image_obs": {"wrist_cam": image}, "viewport_cam": {"viewport": image},
                    "proprio_obs": {"arm_joint_pos": self.positions.clone()}}

        def clear(self, **kwargs):
            self.buffers = [[] for _ in range(12)]

        def open_recording(self, path):
            self.recording = h5py.File(path, "w")
            self.recording.create_group("data")
            self.recorder_manager._dataset_file_handler = self.recording

        def export(self, env_ids):
            for env_id in env_ids:
                group = self.recording.create_group(f"data/demo_{env_id}")
                group.create_dataset("actions", data=np.stack(self.buffers[env_id]))

        def reset_eval_state(self):
            self._frozen_envs[:] = False

        def reset(self, **kwargs):
            self.positions[:] = 0
            self.episode_length_buf[:] = 0
            return self.observation(), {}

        def step(self, actions):
            self.action_manager.action = actions.clone()
            for env_id in range(12):
                if not self._frozen_envs[env_id]:
                    self.positions[env_id] += actions[env_id, :7]
                    if self.recording is not None:
                        self.buffers[env_id].append(actions[env_id].numpy().copy())
            self.episode_length_buf += 1
            terminated = torch.zeros(12, dtype=torch.bool)
            if self.episode_length_buf[0] == 3 and actions[0, 0] == 1:
                terminated[0] = True
                self._frozen_envs[0] = True
                if self.recording is not None:
                    self.export([0])
            return self.observation(), None, terminated, torch.zeros(12, dtype=torch.bool), {}

    def restore(env, expected):
        env.positions = torch.as_tensor(expected["joints"]).clone()
        return env.observation()

    monkeypatch.setattr(worker, "restore_scene_state", restore)
    monkeypatch.setattr(multi_env, "hold_restored_robot_pose", lambda env: None)
    instruction_path = tmp_path / "instructions.json"
    instructions = [{"subgoal": "approach", "instruction": f"Instruction {env_id}", "avoid": "contact",
                     "completion_condition": "aligned"} for env_id in range(12)]
    instruction_path.write_text(json.dumps(instructions))
    output = tmp_path / "round1"
    output.mkdir()
    args = SimpleNamespace(output_dir=output, seed=3407, checkpoint=None, task="test", initialization_only=False,
                           policy_port=8000, execute_steps=4, instructions=instruction_path, no_videos=True)
    multi_env.run(FakeEnv(), SimpleNamespace(instruction="native"), args, instructions)
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "complete" and len(result["candidates"]) == 12
    assert len(requests) == 12 and len({request["session_id"] for request in requests}) == 12
    assert [request["env_id"] for request in requests] == list(range(12))
    assert result["candidates"][0]["executed_steps"] == 3
    assert result["candidates"][1]["executed_steps"] == 4
    assert all(f"Instruction {env_id}" in requests[env_id]["prompt"] for env_id in range(12))
    args.checkpoint = output / "candidate_01/checkpoint.json"
    args.output_dir = tmp_path / "round2"
    args.output_dir.mkdir()
    multi_env.run(FakeEnv(), SimpleNamespace(instruction="native"), args, instructions)
    final = json.loads((args.output_dir / "candidate_11/checkpoint.json").read_text())
    assert final["step"] == 8
    assert len(multi_env.checkpoint_history(args.output_dir / "candidate_11/checkpoint.json", "test")) == 2
    with np.load(final["scene"]["path"]) as scene:
        np.testing.assert_array_equal(scene["joints"], np.full((1, 7), 56))
    from paired_instruction_experiment import sha256

    args.experiment_arm = tmp_path / "arm.json"
    sampling.write_json(args.experiment_arm, {"format": "paired-instruction-v1", "task": "test",
        "condition": "A", "checkpoint": {"sha256": sha256(args.checkpoint)},
        "instructions": {"sha256": sha256(instruction_path)}, "seeds": list(range(12)),
        "execute_steps": 16, "prompt": "Frozen full prompt"})
    args.execute_steps = 16
    args.output_dir = tmp_path / "paired"
    args.output_dir.mkdir()
    multi_env.run(FakeEnv(), SimpleNamespace(instruction="native"), args, instructions)
    assert [request["request_seed"] for request in requests[-12:]] == list(range(12))
    assert all(request["prompt"] == "Frozen full prompt" for request in requests[-12:])
    assert json.loads((args.output_dir / "candidate_11/seed_receipt.json").read_text())["actual_seed"] == 11
    monkeypatch.setattr(FakeTransport, "request", lambda *args, **kwargs: {"actual_seed": -1})
    args.output_dir = tmp_path / "bad_receipt"
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="acknowledge"):
        multi_env.run(FakeEnv(), SimpleNamespace(instruction="native"), args, instructions)
    assert not list(args.output_dir.glob("candidate_*/executed_actions.npy"))


def state_fixture():
    return {"status": "ready", "request_id": "current", "chunk": 0, "step": 0,
            "stop_chunk": 8, "slots": [{"gpu": index} for index in range(4)],
            "checkpoint": {"id": "start"}, "candidates": [], "pending": [],
            "instruction": {"subgoal": "approach", "instruction": "Approach the fruit.",
                            "completion_condition": "Open fingers above fruit.", "avoid": "Do not close."}}


def test_four_way_budget_and_tokens():
    assert list(sampling.TASKS.values()) == ["FruitsOnPlate3Task", "ToolsPickingAllHammersTask",
                                           "BlocksInBinTask", "BlackItemsInBinTask",
                                           "ClearOrganicObjectsTask", "CleanUpToysTask"]
    assert list(sampling.ROUNDS.values()) == [8, 8, 10, 10, 10, 10]
    state = state_fixture()
    for batch_index in range(3):
        batch = sampling.reserve_batch(state, {"request_id": "current"})
        assert len(batch) == 4
        assert len(state["candidates"]) == (batch_index + 1) * 4
        assert all(item["checkpoint"] == {"id": "start"} for item in batch)
        with pytest.raises(ValueError, match="Illegal state"):
            sampling.reserve_batch(state, {"request_id": "current"})
        state["status"] = "ready"
    with pytest.raises(ValueError, match="Twelve"):
        sampling.reserve_batch(state, {"request_id": "current"})
    with pytest.raises(ValueError, match="Stale"):
        sampling.reserve_batch(state_fixture(), {"request_id": "old"})
    for status in ("pending_review", "need_selection", "stopped"):
        state = state_fixture()
        state["status"] = status
        with pytest.raises(ValueError, match="Illegal"):
            sampling.reserve_batch(state, {"request_id": "current"})


def test_instruction_and_stop_boundaries(monkeypatch):
    for value in ({}, {"instruction": "pick"}, None):
        with pytest.raises(ValueError):
            sampling.validate_instruction(value)
    state = state_fixture()
    instructions = [copy.deepcopy(state["instruction"]) for _ in range(4)]
    instructions[2]["subgoal"] = "different"
    with pytest.raises(ValueError, match="subgoal"):
        sampling.reserve_batch(state, {"request_id": "current", "instructions": instructions})
    assert not state["candidates"]
    state["chunk"] = 8
    with pytest.raises(ValueError, match="budget"):
        sampling.reserve_batch(state, {"request_id": "current"})
    original = {"source": "test", "steps": [1, 2]}
    sealed = sampling.seal(copy.deepcopy(original))
    checksum = sealed["integrity"]
    assert sampling.seal(sealed)["integrity"] == checksum
    sealed["steps"].append(3)
    assert sampling.seal(sealed)["integrity"] != checksum
    for script in ("instruction_sampling.py", "probe_isaac_camera.py"):
        help_result = subprocess.run([sys.executable, str(sampling.ROOT / script), "--help"], capture_output=True, text=True)
        assert help_result.returncode == 0
    subprocess.run(["bash", "-n", str(sampling.ROOT / "launch_cosmos3_robolab_server.sh")], check=True)
    monkeypatch.setattr(sampling.subprocess, "check_output", lambda command, **kwargs:
                        "GPU-example, 1883, 20612\n" if "--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in command
                        else "\n".join(f"{index}, GPU-{index}" for index in range(8)))
    with pytest.raises(RuntimeError, match="unrelated"):
        sampling.preflight()
    assert sampling.preflight(allow_gpu_sharing=True) == {index: f"GPU-{index}" for index in range(8)}


def result_fixture(tmp_path, candidate, length=16):
    raw = np.arange(256, dtype=np.float32).reshape(32, 8)
    directory = tmp_path / candidate["id"]
    directory.mkdir()
    planned, executed = directory / "actions.npy", directory / "executed.npy"
    np.save(planned, raw)
    np.save(executed, raw[:length])
    return {"status": "complete", "candidate_id": candidate["id"], "checkpoint": candidate["checkpoint"],
            "artifacts": [{"path": str(path), "sha256": sampling.digest(path)} for path in (planned, executed)],
            "processed_actions": str(planned), "executed_actions": str(executed), "executed_steps": length,
            "review_frames": [{"sha256": "frame"}], "native_success": False,
            "native_terminated": False, "native_truncated": False, "final_checkpoint": {"id": "end"}}


def test_review_and_selection(tmp_path):
    state = state_fixture()
    state.update(selected=[], rounds=[])
    batch = sampling.reserve_batch(state, {"request_id": "current"})
    for candidate in batch:
        candidate["result"] = result_fixture(tmp_path, candidate)
    reviews = [{"candidate_id": item["id"], "verdict": "accept", "subgoal_complete": False,
                "evidence": "TEST FIXTURE ONLY", "reviewed_frame_sha256": ["frame"]} for item in batch]
    state["status"] = "pending_review"
    invalid = copy.deepcopy(reviews)
    invalid[0]["reviewed_frame_sha256"] = []
    with pytest.raises(ValueError, match="Review all"):
        sampling.review_batch(state, {"request_id": "current", "reviews": invalid})
    sampling.review_batch(state, {"request_id": "current", "reviews": reviews})
    assert state["status"] == "ready"
    with pytest.raises(ValueError):
        sampling.select_candidate(state, {"request_id": "current", "candidate_id": batch[0]["id"], "reason": "test"})
    state["status"] = "need_selection"
    batch[0]["result"]["native_success"] = True
    sampling.select_candidate(state, {"request_id": "current", "candidate_id": batch[0]["id"], "reason": "test"})
    assert state["step"] == 16 and state["chunk"] == 1 and state["stop_reason"] == "native_success"
    for candidate in batch:
        receipt_path = tmp_path / candidate["id"] / "completion.json"
        sampling.write_json(receipt_path, {"candidate_id": candidate["id"], "status": "complete", "operation_exit_code": 0})
        candidate["result"]["execution_receipt"] = worker.artifact(receipt_path)
        sampling.validate_result(candidate, candidate["result"])
    receipt_path.write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        sampling.validate_result(batch[-1], batch[-1]["result"])


def test_recorded_prefix_integrity(tmp_path, monkeypatch):
    assert worker.image_mae(np.full((4, 4, 3), 2, dtype=np.uint8), np.zeros((4, 4, 3))) == 2
    with pytest.raises(ValueError, match="shape"):
        worker.image_mae(np.zeros((4, 4, 3)), np.zeros((2, 2, 3)))
    assert sampling.FORMAT == "instruction-sampling-v3-visual-review"
    candidate = sampling.reserve_batch(state_fixture(), {"request_id": "current"})[0]
    result = result_fixture(tmp_path, candidate, length=7)
    assert sampling.validate_result(candidate, result)["executed_steps"] == 7
    np.save(result["executed_actions"], np.zeros((7, 8)))
    with pytest.raises(ValueError, match="Artifact changed"):
        sampling.validate_result(candidate, result)
    expected = {"robot/joints": np.zeros((1, 7))}
    assert worker.state_error(expected, expected) == 0
    assert worker.state_error({"robot/joints": np.full((1, 7), 0.040494386)}, expected) == pytest.approx(0.040494386)
    assert worker.state_error({"robot/joints": np.ones((1, 7))}, expected) == 1
    with pytest.raises(ValueError, match="Non-finite"):
        worker.state_error({"robot/joints": np.full((1, 7), np.nan)}, expected)
    with pytest.raises(ValueError, match="fields"):
        worker.state_error({"other/joints": np.ones((1, 7))}, expected)
    with pytest.raises(ValueError, match="shape"):
        worker.state_error({"robot/joints": np.ones((1, 6))}, expected)
    frames = []
    for view in ("actual", "viewport"):
        for step in (0, 4, 8, 12, 16):
            path = tmp_path / f"{view}_{step}.png"
            Image.new("RGB", (32, 24), (step * 10, 70, 90)).save(path)
            frames.append({**worker.artifact(path), "view": view, "step": step})
    sheet = worker.make_sheet(frames, tmp_path / "review.png", "TEST ONLY")
    with Image.open(sheet["path"]) as image:
        assert image.size == (1600, 412)
    state = state_fixture()
    state.update(task="FruitsOnPlate3Task", stop_reason="budget_reached", selected=[], rounds=[], code={})
    candidate = sampling.reserve_batch(state, {"request_id": "current"})[0]
    result = result_fixture(tmp_path, candidate, length=7)
    result.update(raw_actions=result["processed_actions"], frames=[], replay_steps=0)
    candidate.update(result=result, review={"verdict": "accept"}, selection_reason="TEST ONLY")
    state.update(selected=[candidate], rounds=[], candidates=[], step=7, chunk=1)
    sampling.export(tmp_path, state)
    exported = json.loads((tmp_path / "export/manifest.json").read_text())
    assert exported["segments"][0]["predicted_length"] == 32
    assert exported["segments"][0]["length"] == 7
    assert exported["segments"][0]["action_source"] == "cosmos"
    assert exported["segments"][0]["unexecuted_tail_verified"] is False
    np.testing.assert_array_equal(np.load(tmp_path / "export/actions.npy"), np.load(result["executed_actions"]))
    with pytest.raises(ValueError, match="already exists"):
        sampling.export(tmp_path, state)


def test_multi_env_runtime_library_reexec(monkeypatch, tmp_path):
    import sampling_multi_env as multi

    monkeypatch.setattr(multi.Path, "home", lambda: tmp_path)
    library = tmp_path / ".conda/pkgs/libglu-test/lib"
    library.mkdir(parents=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    calls = []
    monkeypatch.setattr(multi.os, "execv", lambda executable, arguments: calls.append((executable, arguments)))
    multi.configure_runtime_libraries()
    paths = multi.os.environ["LD_LIBRARY_PATH"].split(":")
    assert str(library) in paths and "/existing/lib" in paths
    assert len(calls) == 1
    assert calls[0] == (multi.sys.executable, [multi.sys.executable, *multi.sys.argv])
    multi.configure_runtime_libraries()
    assert len(calls) == 1


def test_restored_robot_targets_follow_measured_pose():
    import torch
    from types import SimpleNamespace

    positions = torch.tensor([[0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0, 0.0]])
    targets = {}
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=positions),
        set_joint_position_target=lambda value: targets.update(position=value),
        set_joint_velocity_target=lambda value: targets.update(velocity=value),
        set_joint_effort_target=lambda value: targets.update(effort=value))
    worker.hold_restored_robot_pose(SimpleNamespace(scene={"robot": robot}))
    assert torch.equal(targets["position"], positions)
    assert targets["position"].data_ptr() != positions.data_ptr()
    assert torch.count_nonzero(targets["velocity"]) == 0
    assert torch.count_nonzero(targets["effort"]) == 0
    positions[0, 3] = -1.5
    worker.hold_restored_robot_pose(SimpleNamespace(scene={"robot": robot}))
    assert targets["position"][0, 3] == -1.5


def test_prompt_seed_audit_uses_actual_log_seed(tmp_path):
    from datetime import datetime, timezone
    from audit_sampling_prompts import audit

    directory = tmp_path / "fruits/candidates/example"
    directory.mkdir(parents=True)
    started = datetime(2026, 9, 18, 4, 20, 0, tzinfo=timezone.utc).timestamp()
    (directory / "result.json").write_text(json.dumps({"status": "complete", "started_at": started,
        "finished_at": started + 30, "prompt": "Actual sent prompt"}))
    (directory / "request.json").write_text(json.dumps({"candidate": {"id": "example", "chunk": 0,
        "attempt": 0, "slot": {"gpu": 0, "seed": 3407},
        "instruction": {"instruction": "Keep the gripper open."}}}))
    log = tmp_path / "services/model_0/worker.log"
    log.parent.mkdir(parents=True)
    log.write_text("[09-18 04:19:00|INFO] [robolab-policy-server] ready seed=3407 deterministic_seed=False\n"
                   "[09-18 04:20:05|INFO] [robolab-policy-server] prompt='Actual sent prompt' seed=1253383803\n")
    report = audit(tmp_path)
    item = report["batches"][0]["candidates"][0]
    assert item["configured_slot_seed"] == 3407 and item["actual_seed"] == 1253383803
    assert report["rounds"][0]["unique_instruction_texts"] == 1
    log.write_text(log.read_text() + "[09-18 04:20:06|INFO] [robolab-policy-server] prompt='another request' seed=42\n")
    ambiguous = audit(tmp_path)["batches"][0]
    assert ambiguous["candidates"][0]["actual_seed"] is None
    assert not ambiguous["all_seeds_log_verified"]


def test_one_extra_pool_preserves_start_and_parent(tmp_path, monkeypatch):
    parent = state_fixture()
    parent.update(status="stopped", stop_reason="no_acceptable_candidate", chunk=2, step=32,
                  checkpoint={"path": "selected/checkpoint.json", "sha256": "same-start"},
                  selected=[{"id": "selected"}], rounds=[], pending=[], request_id="current")
    parent["candidates"] = [{"id": str(index), "checkpoint": copy.deepcopy(parent["checkpoint"]),
                             "review": {"verdict": "reject"}, "result": {"status": "complete"}}
                            for index in range(12)]
    before = copy.deepcopy(parent)
    decision = {"request_id": "current", "user_request": "If twelve have none usable, try once more.",
                "reason": "TEST FIXTURE: one explicitly authorized extra pool"}
    child = retry.retry_state(parent, decision, "parent/fruits")
    assert parent == before
    assert child["checkpoint"] == parent["checkpoint"] and child["selected"] == parent["selected"]
    assert (child["chunk"], child["step"], child["stop_chunk"]) == (2, 32, 8)
    assert len(child["rounds"][-1]) == 12 and not child["candidates"] and not child["pending"]
    child.update(status="stopped", stop_reason="no_acceptable_candidate", candidates=copy.deepcopy(parent["candidates"]))
    with pytest.raises(ValueError, match="no third pool"):
        retry.retry_state(child, decision, "child/fruits")
    for field, value in (("status", "execution_unknown"), ("request_id", "stale"), ("step", 1500)):
        invalid = copy.deepcopy(parent)
        invalid[field] = value
        with pytest.raises(ValueError):
            retry.retry_state(invalid, decision, "parent/fruits")
    invalid = copy.deepcopy(parent)
    invalid["candidates"][0]["review"]["verdict"] = "accept"
    with pytest.raises(ValueError, match="no accept"):
        retry.retry_state(invalid, decision, "parent/fruits")
    origin = tmp_path / "origin"
    (origin / "fruits").mkdir(parents=True)
    checkpoint = origin / "checkpoint.json"
    checkpoint.write_text("{}")
    source = {"test": "source"}
    monkeypatch.setattr(sampling, "identity", lambda: source)
    slots = [{"gpu": index, "uuid": f"GPU-{index}", "port": 18840 + index,
              "master_port": 29840 + index, "seed": 3407 + index, "socket": "unused"} for index in range(4)]
    manifest = {"format": sampling.FORMAT, "code": source, "checkpoint": "test-model",
                "slots": slots, "models": [{"pid": 100 + index, "start_ticks": "test", "directory":
                    str(origin / "services" / f"model_{index}")} for index in range(4)],
                "workers": [], "activated": ["fruits"], "active": ["fruits"], "allow_gpu_sharing": True}
    parent.update(format=sampling.FORMAT, code=source, task="FruitsOnPlate3Task", name="fruits",
                  slots=slots, genesis={"path": str(checkpoint), "sha256": sampling.digest(checkpoint)})
    parent["checkpoint"] = copy.deepcopy(parent["genesis"])
    for candidate in parent["candidates"]:
        candidate["checkpoint"] = copy.deepcopy(parent["checkpoint"])
    sampling.write_json(origin / "experiment.json", sampling.seal(manifest))
    sampling.write_json(origin / "fruits/session.json", sampling.seal(parent))
    decision_path = tmp_path / "retry.json"
    sampling.write_json(decision_path, decision)
    original_bytes = (origin / "fruits/session.json").read_bytes()
    monkeypatch.setattr(sampling, "validate_result", lambda candidate, result: result)
    monkeypatch.setattr(sampling, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(sampling, "healthy", lambda port: True)
    monkeypatch.setattr(sampling, "process_identity", lambda pid: "test")
    monkeypatch.setattr(sampling, "wait_ready", lambda check, process: None)
    monkeypatch.setattr(retry.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "sockets"))
    launched = []

    def fake_spawn(command, environment, directory):
        launched.append(command)
        return {"pid": 1000 + len(launched), "start_ticks": "test", "directory": str(directory), "command": command}

    monkeypatch.setattr(sampling, "spawn", fake_spawn)
    retry.run(origin, "fruits", decision_path)
    destination = next((origin / "retries").iterdir())
    created = sampling.load_state(destination / "fruits")
    assert created["status"] == "need_instruction" and created["checkpoint"] == parent["checkpoint"]
    assert created["selected"] == parent["selected"] and (created["chunk"], created["step"]) == (2, 32)
    assert len(launched) == 4 and all("probe_isaac_camera.py" in command[1] for command in launched)
    assert (origin / "fruits/session.json").read_bytes() == original_bytes
    assert not sampling.load_manifest(destination)["models"]
    with pytest.raises(ValueError, match="already reserved"):
        retry.run(origin, "fruits", decision_path)
    assert len(launched) == 4