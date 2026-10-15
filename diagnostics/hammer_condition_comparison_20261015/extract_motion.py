"""Extract measured, offline evidence; never send simulator or policy requests."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent.parent


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(variant, live):
    run = ROOT / "diagnostics" / variant["episode"]
    directory = run / "recovery"
    text = (directory / "events.jsonl").read_text()
    incomplete_tail = not text.endswith("\n")
    if incomplete_tail:
        if not live:
            raise ValueError("Incomplete final event log")
        text = text.rsplit("\n", 1)[0]
    events = [json.loads(line) for line in text.splitlines()]
    initial = next(e for e in events if e["event_type"] == "session_started")
    actual = [e for e in events if e["event_type"] == "actual_execution_response"]
    if not actual:
        raise ValueError(f"No recorded actions for {variant['id']}")
    assert [e["episode_step"] for e in actual] == list(range(1, len(actual) + 1))
    assert {e["action_source"] for e in actual} == {"wam"}
    assert not any(e["event_type"] in {
        "failure_detected", "recovery_proposal", "agent_control_proposal",
        "recovery_assessed", "progress_review",
    } for e in events)

    observations = [initial["observation"]] + [e["observation"] for e in actual]
    robots = [initial["state"]] + [e["robot_state"] for e in actual]
    ee = np.array([r["ee_pos"] for r in robots])
    grip = np.array([r["gripper_pos"][0] for r in robots])
    hammer = []
    object_positions = {}
    for observation in observations:
        with np.load(observation["scene"]["path"]) as scene:
            np.testing.assert_array_equal(
                scene["articulation/robot/root_pose"][0],
                [0, 0, 0, 1, 0, 0, 0],
            )
            hammer.append(scene["rigid_object/blue_hammer/root_pose"][0, :3])
            for key in scene.files:
                if key.startswith("rigid_object/") and key.endswith("/root_pose"):
                    object_positions.setdefault(key.split("/")[1], []).append(scene[key][0, :3])
    hammer = np.array(hammer)
    command = np.array([e["actual_action"][-1] for e in actual])
    actual_commands = np.array([e["actual_action"] for e in actual])
    measured_joints = np.array([r["measured_arm_joint_pos"] for r in robots[1:]])
    tracking_error = actual_commands[:, :7] - measured_joints
    motion = [{
        "step": i, "ee_x_m": ee[i, 0], "ee_y_m": ee[i, 1], "ee_z_m": ee[i, 2],
        "gripper_measured": grip[i],
        "gripper_command": command[i - 1] if i else "",
        "arm_tracking_error_l2_rad": float(np.linalg.norm(tracking_error[i - 1])) if i else "",
        "arm_tracking_error_max_abs_rad": float(np.abs(tracking_error[i - 1]).max()) if i else "",
        "hammer_x_m": hammer[i, 0], "hammer_y_m": hammer[i, 1],
        "hammer_z_m": hammer[i, 2],
        "native_success": actual[i - 1]["events"]["native_success"] if i else False,
    } for i in range(len(actual) + 1)]
    write_csv(STUDY / f"{variant['id']}_motion.csv", motion)

    chunks = []
    boundaries = [0]
    for result_path in sorted((directory / "segments").glob("*_wam/result.json")):
        request = json.loads((result_path.parent / "request.json").read_text())
        result = json.loads(result_path.read_text())
        assert request["condition"] == variant["condition"]
        start, end = request["start_step"], result["episode_step"]
        assert result["executed_steps"] == end - start
        if end > len(actual):
            if live:
                continue
            raise ValueError("Segment extends beyond recorded actions")
        boundaries.append(end)
        predicted = np.load(result_path.parent / "predicted_actions.npy")
        assert predicted.shape == (32, 8) and np.isfinite(predicted).all()
        np.testing.assert_array_equal(predicted[:end - start], actual_commands[start:end])
        predicted_closes = np.flatnonzero(predicted[:, -1] > 0.5) + 1
        ee_range = float(np.linalg.norm(np.ptp(ee[start:end + 1], axis=0)))
        hammer_range = float(np.linalg.norm(np.ptp(hammer[start:end + 1], axis=0)))
        chunks.append({
            "chunk": len(chunks) + 1, "first_step": start + 1, "last_step": end,
            "actual_steps": end - start,
            "close_commands": int((command[start:end] > 0.5).sum()),
            "predicted_first_16_close_commands": int((predicted[:16, -1] > 0.5).sum()),
            "predicted_last_16_close_commands": int((predicted[16:, -1] > 0.5).sum()),
            "predicted_first_close_offset": int(predicted_closes[0]) if len(predicted_closes) else "",
            "closure_only_in_unexecuted_half": bool(
                end - start == 16
                and not (predicted[:16, -1] > 0.5).any()
                and (predicted[16:, -1] > 0.5).any()
            ),
            "gripper_start": grip[start], "gripper_end": grip[end],
            "ee_dx_m": ee[end, 0] - ee[start, 0],
            "ee_dy_m": ee[end, 1] - ee[start, 1],
            "ee_dz_m": ee[end, 2] - ee[start, 2],
            "ee_range_norm_m": ee_range,
            "arm_tracking_error_mean_l2_rad": float(np.linalg.norm(tracking_error[start:end], axis=1).mean()),
            "arm_tracking_error_max_abs_rad": float(np.abs(tracking_error[start:end]).max()),
            "hammer_range_norm_m": hammer_range,
            "hammer_dz_from_initial_m": hammer[end, 2] - hammer[0, 2],
            "near_static_motion_only": ee_range <= 0.02 and hammer_range <= 0.01,
            "inference_seconds_including_queue": result["inference_seconds"],
            "native_success": result["native_success"],
        })
    if chunks:
        write_csv(STUDY / f"{variant['id']}_chunk_metrics.csv", chunks)

    selected = boundaries
    tile_w, tile_h, columns = 320, 208, 4
    sheet = Image.new("RGB", (tile_w * columns, tile_h * ((len(selected) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, step in enumerate(selected):
        frame = next(f for f in observations[step]["frames"] if f["camera"] == "wrist_cam")
        with Image.open(frame["path"]) as image:
            image = image.resize((320, 180))
            x, y = (index % columns) * tile_w, (index // columns) * tile_h
            sheet.paste(image, (x, y + 28))
            draw.text((x + 4, y + 7), f"{variant['id']} step {step} | measured grip {grip[step]:.3f}", fill="black")
    sheet.save(STUDY / f"{variant['id']}_boundary_contact_sheet.jpg", quality=94)

    close_indices = np.flatnonzero(command > 0.5) + 1
    measured_close = np.flatnonzero(grip > 0.8)
    transitions = np.flatnonzero(np.diff((command > 0.5).astype(int)) != 0) + 2
    summary = {
        "variant": variant["id"], "condition": variant["condition"],
        "live_partial_analysis": live, "incomplete_tail_ignored": incomplete_tail,
        "actual_steps_analyzed": len(actual), "completed_chunks": len(chunks),
        "source_counts": dict(Counter(e["action_source"] for e in actual)),
        "first_close_command": int(close_indices[0]) if len(close_indices) else None,
        "close_commands": len(close_indices),
        "first_measured_closure_above_0_8": int(measured_close[0]) if len(measured_close) else None,
        "gripper_command_transition_steps": transitions.tolist(),
        "ee_initial": ee[0].tolist(), "ee_final": ee[-1].tolist(),
        "hammer_initial": hammer[0].tolist(), "hammer_final": hammer[-1].tolist(),
        "hammer_max_displacement_from_initial_m": float(np.linalg.norm(hammer - hammer[0], axis=1).max()),
        "hammer_max_root_height_gain_m": float((hammer[:, 2] - hammer[0, 2]).max()),
        "object_max_displacement_from_initial_m": {
            name: float(np.linalg.norm(np.array(positions) - positions[0], axis=1).max())
            for name, positions in object_positions.items()
        },
        "native_success": any(e["events"]["native_success"] for e in actual),
        "closure_only_in_unexecuted_half_windows": [
            [c["first_step"], c["last_step"]] for c in chunks if c["closure_only_in_unexecuted_half"]
        ],
        "near_static_motion_windows_not_automatically_errors": [
            [c["first_step"], c["last_step"]] for c in chunks if c["near_static_motion_only"]
        ],
    }
    if not live:
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["step"] == len(actual)
        assert manifest["correction_attempts_preserved"] == manifest["attempts"] == 0
        assert len(actual) <= 600
        assert manifest["native_terminated"] or manifest["native_truncated"]
        assert boundaries[-1] == len(actual)
        authorized = [e for e in events if e["event_type"] == "action_authorized"]
        assert len(authorized) == len(actual)
        for intent, response in zip(authorized, actual):
            assert intent["episode_step"] + 1 == response["episode_step"]
            np.testing.assert_array_equal(intent["authorized_action"], response["actual_action"])
        summary["native_truncated"] = manifest["native_truncated"]
        summary["zero_interventions_verified"] = True
        summary["all_authorized_actions_match_actual"] = True
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    design = json.loads((STUDY / "study_design.json").read_text())
    summaries = [extract(variant, args.live) for variant in design["variants"]]
    (STUDY / "motion_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
