"""Summarize actual trajectories and prepare multi-view checkpoint evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
run = args.run.resolve()
directory = run / "recovery"
raw = (directory / "events.jsonl").read_text()
if not raw.endswith("\n"):
    raw = raw.rsplit("\n", 1)[0]
events = [json.loads(line) for line in raw.splitlines()]
actual = [e for e in events if e["event_type"] == "actual_execution_response"]
initial = next(e for e in events if e["event_type"] == "session_started")
observations = [initial["observation"]] + [e["observation"] for e in actual]
steps = len(actual)
assert [e["episode_step"] for e in actual] == list(range(1, steps + 1))
objects = ["wood_hammer", "husky_hammer", "red_hammer", "blue_hammer"]
positions = {name: [] for name in objects}
for obs in observations:
    with np.load(obs["scene"]["path"]) as scene:
        for name in objects:
            positions[name].append(scene[f"rigid_object/{name}/root_pose"][0, :3])
motion = {}
for name, values in positions.items():
    points = np.array(values)
    motion[name] = {
        "maximum_displacement_m": float(np.linalg.norm(points - points[0], axis=1).max()),
        "maximum_root_height_gain_m": float((points[:, 2] - points[0, 2]).max()),
        "initial_xyz": points[0].tolist(), "final_xyz": points[-1].tolist(),
    }
segments = []
for path in sorted((directory / "segments").glob("*/result.json")):
    result = json.loads(path.read_text())
    req = json.loads((path.parent / "request.json").read_text())
    end, start = result["episode_step"], req["start_step"]
    if end > steps:
        continue
    commands = np.load(path.parent / "predicted_actions.npy")
    responses = actual[start:end]
    ee = np.array([e["robot_state"]["ee_pos"] for e in responses])
    segments.append({
        "start_step": start + 1, "end_step": end, "condition": req["condition"],
        "change_kind": req["condition_change"]["kind"],
        "close_commands_executed": sum(e["actual_action"][-1] > 0.5 for e in responses),
        "close_commands_predicted_tail": int((commands[16:, -1] > 0.5).sum()),
        "ee_range_norm_m": float(np.linalg.norm(np.ptp(ee, axis=0))),
        "progress": result["hammer_task_progress"],
    })
summary = {
    "run": run.name, "actual_steps": steps,
    "source_counts": dict(Counter(e["action_source"] for e in actual)),
    "close_commands": sum(e["actual_action"][-1] > 0.5 for e in actual),
    "max_measured_closure": max((e["robot_state"]["gripper_pos"][0] for e in actual), default=0),
    "progress": actual[-1]["events"]["hammer_task_progress"] if actual else None,
    "objects": motion, "segments": segments,
    "current_native_success": actual[-1]["events"]["native_success"] if actual else False,
}
output = run / "offline_reviews"
output.mkdir(exist_ok=True)
(output / f"step_{steps:04d}.json").write_text(json.dumps(summary, indent=2) + "\n")
selected = sorted({0, *range(32, steps + 1, 32), steps})
width, height = 320, 400
sheet = Image.new("RGB", (4 * width, ((len(selected) + 3) // 4) * height), "white")
draw = ImageDraw.Draw(sheet)
for index, step in enumerate(selected):
    x, y = index % 4 * width, index // 4 * height
    draw.text((x + 5, y + 5), f"{run.name.split('_')[3]} step {step}", fill="black")
    frames = {f["camera"]: Path(f["path"]) for f in observations[step]["frames"]}
    with Image.open(frames["wrist_cam"]) as image:
        sheet.paste(image.resize((320, 180)), (x, y + 25))
    with Image.open(frames["over_shoulder_left_camera"]) as image:
        sheet.paste(image.crop((400, 200, 1060, 580)).resize((320, 184)), (x, y + 208))
sheet.save(output / f"step_{steps:04d}.jpg", quality=94)
print(json.dumps({k: v for k, v in summary.items() if k != "segments"}, indent=2))
print("Recent chunks:", json.dumps(segments[-2:], indent=2))
