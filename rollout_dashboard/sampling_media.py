"""Derived videos and read-only Copilot records, separate from locked sampling artifacts."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("wrist_cam", "egocentric_mirrored_camera")
VERSION = "actual-dual-instruction-v1"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, data):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def experiment_roots(diagnostics):
    return sorted(path for path in diagnostics.glob("instruction_*")
                  if path.is_dir() and (path / "experiment.json").is_file())


def output_directory(experiment, candidate_id):
    if candidate_id != Path(candidate_id).name or not candidate_id:
        raise ValueError("Invalid candidate id")
    return experiment / "media/chunks" / candidate_id


def verified_image(experiment, reference):
    path = Path(reference["path"]).resolve(strict=True)
    path.relative_to(experiment.resolve())
    if digest(path) != reference["sha256"]:
        raise ValueError("Source image checksum differs")
    return path


def render_candidate(experiment, result_path):
    raw = result_path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    result = json.loads(raw)
    if result.get("status") != "complete":
        return None
    candidate_id = result["candidate_id"]
    destination = output_directory(experiment, candidate_id)
    if destination.exists():
        cached = read_json(destination / "manifest.json")
        if (cached.get("source_result_sha256") != source_hash or cached.get("version") != VERSION
                or not (destination / "actual.mp4").is_file()):
            raise ValueError("Existing derived video has a different source; refusing overwrite")
        return destination
    request = read_json(result_path.parent / "request.json")
    candidate = request.get("candidate", {})
    instruction = candidate.get("instruction", {}).get("instruction") or result.get("prompt", "")
    count = result["executed_steps"]
    dt = result["control_dt"]
    if type(count) is not int or not 1 <= count <= 32 or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid actual execution timing")
    images, references = {}, []
    for frame in result["frames"]:
        if frame["camera"] not in CAMERAS:
            continue
        path = verified_image(experiment, frame)
        key = (frame["camera"], frame["step"])
        if key in images:
            raise ValueError("Duplicate camera/time")
        with Image.open(path) as image:
            images[key] = ImageOps.pad(image.convert("RGB"), (640, 360), color=(22, 24, 26))
        references.append(frame)
    if set(images) != {(camera, step) for camera in CAMERAS for step in range(count + 1)}:
        raise ValueError("Missing actual start/intermediate/end frames")
    font_file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_file), 18) if font_file.is_file() else ImageFont.load_default()
    canvas = Image.new("RGB", (1280, 528), (22, 24, 26))
    draw = ImageDraw.Draw(canvas)
    task = result_path.parent.parent.parent.name
    draw.text((14, 8), f"ACTUAL EXECUTION | {task} | chunk {candidate.get('chunk', '?')} / candidate {candidate.get('attempt', '?')} | {candidate_id[:8]}", fill="white", font=font)
    words, lines, line = instruction.split(), [], ""
    for word in words:
        proposal = f"{line} {word}".strip()
        if line and draw.textlength(proposal, font=font) > 1248:
            lines.append(line)
            line = word
        else:
            line = proposal
    if line:
        lines.append(line)
    for index, line in enumerate(lines[:3]):
        if index == 2 and len(lines) > 3:
            line = line[:max(1, len(line) - 3)] + "..."
        draw.text((14, 35 + 22 * index), line, fill=(233, 209, 123), font=font)
    draw.text((14, 109), "Wrist camera", fill="white", font=font)
    draw.text((654, 109), "Viewport (debug camera)", fill="white", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ROOT / "packages/ffmpeg/bin/ffmpeg"
    ffprobe = ROOT / "packages/ffmpeg/bin/ffprobe"
    with tempfile.TemporaryDirectory(prefix=".encoding-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        video = staging / "actual.mp4"
        with (staging / "encode.log").open("wb") as log:
            process = subprocess.Popen([str(ffmpeg), "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-video_size", "1280x528", "-framerate", str(1 / dt), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-threads", "2", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                stdin=subprocess.PIPE, stderr=log)
            try:
                for step in range(count + 1):
                    composed = canvas.copy()
                    for column, camera in enumerate(CAMERAS):
                        composed.paste(images[camera, step], (column * 640, 136))
                    ImageDraw.Draw(composed).text((14, 502), f"Step {step}/{count} | t={step * dt:.3f}s | actual recorded frames; not a success label", font=font, fill="white")
                    if step == 0:
                        composed.save(staging / "poster.jpg", quality=85)
                    process.stdin.write(composed.tobytes())
            finally:
                process.stdin.close()
                returncode = process.wait()
            if returncode:
                raise RuntimeError((staging / "encode.log").read_text())
        probe = json.loads(subprocess.check_output([str(ffprobe), "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_name,width,height,nb_read_frames:format=duration", "-of", "json", str(video)], text=True))
        stream = probe["streams"][0]
        if int(stream["nb_read_frames"]) != count + 1 or stream["codec_name"] != "h264":
            raise ValueError("Video frame count or codec differs")
        subprocess.run([str(ffmpeg), "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"], check=True, capture_output=True)
        if digest(result_path) != source_hash:
            raise ValueError("Candidate changed during encoding")
        atomic_json(staging / "manifest.json", {"version": VERSION, "candidate_id": candidate_id,
            "source_result": str(result_path), "source_result_sha256": source_hash, "sources": references,
            "instruction": instruction, "source": "actual_simulation", "model_prediction_video": False,
            "executed_steps": count, "fps": 1 / dt, "unique_frames": count + 1, "simulation_span": count * dt,
            "display_duration": float(probe["format"]["duration"]), "checkpoint_frame_included": True,
            "unexecuted_tail_included": False, "video_sha256": digest(video), "full_decode": "passed"})
        staging.rename(destination)
    return destination


def scan_once(diagnostics):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from rollout_dashboard.server import _sampling_run

    total = {"complete_candidates": 0, "videos_ready": 0, "errors": []}
    experiments = experiment_roots(diagnostics)
    latest = max(experiments, key=lambda path: read_json(path / "experiment.json").get("created_at", 0), default=None)
    for experiment in experiments:
        status = {"checked_at": time.time(), "complete_candidates": 0, "videos_ready": 0, "errors": []}
        for marker in sorted(experiment.rglob("candidates/*/result.json")):
            if read_json(marker).get("status") != "complete":
                continue
            status["complete_candidates"] += 1
            try:
                if render_candidate(experiment, marker):
                    status["videos_ready"] += 1
            except Exception as error:
                status["errors"].append({"result": str(marker.relative_to(experiment)), "error": str(error)})
        (experiment / "media").mkdir(exist_ok=True)
        atomic_json(experiment / "media/video_status.json", status)
        snapshot = _sampling_run(experiment)
        snapshot_path = experiment / "media/copilot_outputs.json"
        serialized = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
        if not snapshot_path.is_file() or snapshot_path.read_text() != serialized:
            atomic_json(snapshot_path, snapshot)
        web = ROOT / "rollout_dashboard/web"
        local_snapshot = json.loads(json.dumps(snapshot).replace(f"/media/{experiment.name}/", "../"))
        embedded = json.dumps(local_snapshot, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        html = (web / "sampling.html").read_text()
        html = html.replace('<link rel="stylesheet" href="/assets/sampling.css?v=1">', "<style>" + (web / "sampling.css").read_text() + "</style>")
        html = html.replace('<script src="/assets/sampling.js?v=1"></script>',
                            "<script>window.SAMPLING_SNAPSHOT=" + embedded + ";</script><script>" + (web / "sampling.js").read_text() + "</script>")
        html = html.replace('<a href="/">旧版看板</a>', '<span>离线快照</span>')
        html_path = experiment / "media/review.html"
        if not html_path.is_file() or html_path.read_text() != html:
            temporary_html = html_path.with_name(".review.html.tmp")
            temporary_html.write_text(html)
            temporary_html.replace(html_path)
        if experiment == latest and diagnostics.resolve() == (ROOT / "diagnostics").resolve():
            live_embedded = json.dumps(snapshot, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            public_html = html.replace(embedded, live_embedded).replace("离线快照", "实验快照")
            public_path = web / "sampling-latest.html"
            if not public_path.is_file() or public_path.read_text() != public_html:
                public_temporary = web / ".sampling-latest.html.tmp"
                public_temporary.write_text(public_html)
                public_temporary.replace(public_path)
        total["complete_candidates"] += status["complete_candidates"]
        total["videos_ready"] += status["videos_ready"]
        total["errors"].extend(status["errors"])
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "diagnostics")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    with (args.diagnostics / ".sampling_video.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stopped.is_set():
            summary = scan_once(args.diagnostics)
            print(json.dumps(summary), flush=True)
            if not args.watch:
                return 1 if summary["errors"] else 0
            stopped.wait(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())