from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from urllib.parse import quote
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent / "web"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _image_url(run_name: str, relative_path: str) -> str:
    return f"/media/{run_name}/{relative_path}"


def _chunk_camera_frames(run_dir: Path, start_step: int, executed_steps: int, camera: str) -> list[Path]:
    if start_step == 0:
        frames = [run_dir / camera]
    else:
        frames = [run_dir / "frames" / f"step_{start_step:04d}" / camera]
    end_step = start_step + executed_steps
    for step_dir in sorted((run_dir / "frames").glob("step_*")):
        try:
            step = int(step_dir.name.removeprefix("step_"))
        except ValueError:
            continue
        if start_step < step <= end_step:
            frames.append(step_dir / camera)
    return [frame for frame in frames if frame.is_file()]


def _chunk_viewport_frames(run_dir: Path, start_step: int, executed_steps: int) -> list[Path]:
    return _chunk_camera_frames(run_dir, start_step, executed_steps, "egocentric_mirrored_camera.png")


def _make_chunk_video(run_dir: Path, chunk_index: int, start_step: int, executed_steps: int) -> Path:
    run_dir = run_dir.resolve()
    chunk_dir = run_dir / "chunks" / f"{chunk_index:03d}"
    output = chunk_dir / "viewport.mp4"
    metadata = chunk_dir / "viewport.json"
    frames = _chunk_viewport_frames(run_dir, start_step, executed_steps)
    if len(frames) < 2:
        raise FileNotFoundError("Chunk viewport frames are not ready")
    newest_frame = max(frame.stat().st_mtime for frame in frames)
    duration = 1 / 15 if len(frames) >= executed_steps + 1 else 4 / 15
    cache_key = {"frame_count": len(frames), "frame_duration": duration}
    if output.is_file() and output.stat().st_mtime >= newest_frame and _read_json(metadata) == cache_key:
        return output

    ffmpeg = shutil.which("ffmpeg") or str(ROOT / "packages" / "ffmpeg" / "bin" / "ffmpeg")
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError("ffmpeg is not available")
    if any("'" in str(frame) for frame in frames):
        raise ValueError("Unsupported quote in rollout path")

    frame_list = chunk_dir / ".viewport_frames.txt"
    temporary = chunk_dir / "viewport.tmp.mp4"
    lines = []
    for frame in frames:
        lines.extend((f"file '{frame}'", f"duration {duration:.9f}"))
    lines.append(f"file '{frames[-1]}'")
    frame_list.write_text("\n".join(lines) + "\n")
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                "-i", str(frame_list), "-r", "15", "-fps_mode", "cfr",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(temporary),
            ],
            check=True,
        )
        temporary.replace(output)
        metadata.write_text(json.dumps(cache_key, sort_keys=True) + "\n")
    finally:
        frame_list.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return output


def _make_three_view_video(run_dir: Path, chunk_index: int, start_step: int, executed_steps: int) -> Path:
    run_dir = run_dir.resolve()
    chunk_dir = run_dir / "chunks" / f"{chunk_index:03d}"
    output = chunk_dir / "executed_three_view.mp4"
    metadata = chunk_dir / "executed_three_view.json"
    cameras = ("wrist_cam.png", "over_shoulder_left_camera.png", "over_shoulder_right_camera.png")
    frame_sets = [_chunk_camera_frames(run_dir, start_step, executed_steps, camera) for camera in cameras]
    frame_count = len(frame_sets[0])
    if frame_count < 2 or any(len(frames) != frame_count for frames in frame_sets[1:]):
        raise FileNotFoundError("Synchronized three-view frames are not ready")
    newest_frame = max(frame.stat().st_mtime for frames in frame_sets for frame in frames)
    duration = 1 / 15 if frame_count >= executed_steps + 1 else 4 / 15
    cache_key = {"frame_count": frame_count, "frame_duration": duration, "layout": "wrist-left-right-v1"}
    if output.is_file() and output.stat().st_mtime >= newest_frame and _read_json(metadata) == cache_key:
        return output

    ffmpeg = shutil.which("ffmpeg") or str(ROOT / "packages" / "ffmpeg" / "bin" / "ffmpeg")
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError("ffmpeg is not available")
    if any("'" in str(frame) for frames in frame_sets for frame in frames):
        raise ValueError("Unsupported quote in rollout path")

    frame_lists = [chunk_dir / f".three_view_{index}.txt" for index in range(3)]
    temporary = chunk_dir / "executed_three_view.tmp.mp4"
    for frame_list, frames in zip(frame_lists, frame_sets, strict=True):
        lines = []
        for frame in frames:
            lines.extend((f"file '{frame}'", f"duration {duration:.9f}"))
        lines.append(f"file '{frames[-1]}'")
        frame_list.write_text("\n".join(lines) + "\n")
    try:
        command = [ffmpeg, "-y", "-loglevel", "error"]
        for frame_list in frame_lists:
            command.extend(("-f", "concat", "-safe", "0", "-i", str(frame_list)))
        command.extend((
            "-filter_complex",
            "[0:v]scale=640:360:flags=lanczos[wrist];"
            "[1:v]scale=320:180:flags=lanczos[left];"
            "[2:v]scale=320:180:flags=lanczos[right];"
            "[left][right]hstack=inputs=2[shoulders];[wrist][shoulders]vstack=inputs=2[three]",
            "-map", "[three]", "-r", "15", "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(temporary),
        ))
        subprocess.run(command, check=True)
        temporary.replace(output)
        metadata.write_text(json.dumps(cache_key, sort_keys=True) + "\n")
    finally:
        for frame_list in frame_lists:
            frame_list.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return output


def _chunks(run_dir: Path, run_name: str, result: dict) -> list[dict]:
    recorded = {chunk.get("index"): chunk for chunk in result.get("chunks", [])}
    chunks_dir = run_dir / "chunks"
    if not chunks_dir.is_dir():
        return []
    output = []
    for chunk_dir in sorted(path for path in chunks_dir.iterdir() if path.is_dir()):
        try:
            index = int(chunk_dir.name)
        except ValueError:
            continue
        item = recorded.get(index, {})
        start_step = item.get("start_step")
        executed_steps = item.get("executed_steps", 0)
        viewport_frames = _chunk_viewport_frames(run_dir, start_step or 0, executed_steps)
        video_version = f"{len(viewport_frames)}-{int(max((frame.stat().st_mtime for frame in viewport_frames), default=0))}"
        prediction_video = chunk_dir / "prediction.mp4"
        three_view_frames = [
            _chunk_camera_frames(run_dir, start_step or 0, executed_steps, camera)
            for camera in ("wrist_cam.png", "over_shoulder_left_camera.png", "over_shoulder_right_camera.png")
        ]
        three_view_ready = len(three_view_frames[0]) >= 2 and len({len(frames) for frames in three_view_frames}) == 1
        three_view_version = f"{len(three_view_frames[0])}-{int(max((frame.stat().st_mtime for frames in three_view_frames for frame in frames), default=0))}"
        output.append({
            "index": index,
            "start_step": start_step,
            "executed_steps": executed_steps,
            "inference_seconds": item.get("inference_seconds"),
            "prefix_matches": item.get("executed_prefix_matches"),
            "prompt": item.get("prompt", result.get("instruction")),
            "input": _image_url(run_name, f"chunks/{chunk_dir.name}/input.png")
            if (chunk_dir / "input.png").is_file() else None,
            "actions_ready": (chunk_dir / "actions.npy").is_file(),
            "prediction_video": _image_url(run_name, f"chunks/{chunk_dir.name}/prediction.mp4")
            if prediction_video.is_file() else None,
            "three_view_video": f"/api/runs/{run_name}/chunks/{index}/three-view.mp4?v={three_view_version}"
            if three_view_ready else None,
            "viewport_video": f"/api/runs/{run_name}/chunks/{index}/viewport.mp4?v={video_version}"
            if len(viewport_frames) >= 2 else None,
            "viewport_poster": _image_url(run_name, str(viewport_frames[0].relative_to(run_dir)))
            if viewport_frames else None,
        })
    return output


def _summarize(run_dir: Path) -> dict:
    result = _read_json(run_dir / "result.json")
    chunks = _chunks(run_dir, run_dir.name, result)
    stat = run_dir.stat()
    status = result.get("status") or ("running" if (run_dir / "launch.log").exists() else "preparing")
    steps = result.get("steps_executed")
    if steps is None:
        steps = sum(chunk["executed_steps"] for chunk in chunks)
    return {
        "id": run_dir.name,
        "status": status,
        "task": result.get("task", "Unknown task"),
        "updated_at": _timestamp(stat.st_mtime),
        "steps": steps,
        "requested_chunks": result.get("rollout_chunks_requested"),
        "native_success": result.get("native_success"),
        "closed_loop_complete": result.get("closed_loop_complete", False),
        "renderer_gpu": result.get("renderer_active_gpu"),
        "physics_device": result.get("physics_device"),
        "instruction": result.get("instruction"),
        "chunks": chunks,
    }


def _sampling_run(run_dir: Path) -> dict:
    tasks, candidates = [], {}
    run_dir = run_dir.resolve()

    def media_url(path):
        try:
            path = Path(path).resolve()
            relative = path.relative_to(run_dir)
            if not path.is_file():
                return None
            return f"/media/{quote(run_dir.name)}/{quote(relative.as_posix(), safe='/')}"
        except (TypeError, ValueError, OSError):
            return None

    for state_path in sorted(run_dir.rglob("session.json")):
        state = _read_json(state_path)
        if not state.get("task") or not state.get("format", "").startswith("instruction-sampling-"):
            continue
        scope = state_path.parent.relative_to(run_dir).as_posix()
        tasks.append({"task": state["task"], "alias": state.get("name", state_path.parent.name),
                      "scope": scope, "status": state.get("status"), "chunk": state.get("chunk"),
                      "step": state.get("step"), "stop_chunk": state.get("stop_chunk"),
                      "stop_reason": state.get("stop_reason"), "extra_pool": bool(state.get("retry_parent"))})
        records = [item for group in state.get("rounds", []) for item in group] + state.get("candidates", [])
        selected = {item["id"]: item for item in state.get("selected", [])}
        for candidate in records + list(selected.values()):
            candidate_id = candidate.get("id")
            if not candidate_id or Path(candidate_id).name != candidate_id:
                continue
            result = candidate.get("result") or {}
            directory = state_path.parent / "candidates" / candidate_id
            if result.get("executed_actions"):
                inferred = Path(result["executed_actions"]).resolve().parent
                if inferred.is_relative_to(run_dir):
                    directory = inferred
            elif directory.is_dir():
                result = _read_json(directory / "result.json")
            previous = candidates.get(candidate_id)
            instruction = candidate.get("instruction") or {}
            review = candidate.get("review")
            media_dir = run_dir / "media/chunks" / candidate_id
            movie = _read_json(media_dir / "manifest.json")
            chosen = selected.get(candidate_id)
            failure = _read_json(directory / "failure.json")
            candidates[candidate_id] = {
                "id": candidate_id, "task": state["task"], "task_alias": state.get("name", state_path.parent.name),
                "scope": previous["scope"] if previous else directory.parent.parent.relative_to(run_dir).as_posix(),
                "chunk": candidate.get("chunk", 0), "attempt": candidate.get("attempt", 0),
                "batch": candidate.get("attempt", 0) // 4 + 1,
                "status": "complete" if result.get("status") == "complete" else ("failed" if failure else "reserved"),
                "instruction": instruction, "prompt": result.get("prompt"), "review": review,
                "selected": bool(chosen) or bool(previous and previous["selected"]),
                "selection_reason": (chosen or {}).get("selection_reason") or (previous or {}).get("selection_reason"),
                "executed_steps": result.get("executed_steps", 0), "predicted_steps": 32,
                "native_success": result.get("native_success"), "inference_seconds": result.get("inference_seconds"),
                "video": media_url(media_dir / "actual.mp4"), "poster": media_url(media_dir / "poster.jpg"),
                "sheet": media_url((result.get("sheet") or {}).get("path")),
                "video_duration": movie.get("display_duration"), "failure": failure.get("error"),
                "finished_at": result.get("finished_at"), "source": "cosmos",
            }
    ordered = sorted(candidates.values(), key=lambda item: (item["scope"], item["chunk"], item["attempt"]))
    progress = []
    for path in (run_dir / "copilot_control/CLI_PROGRESS.md", run_dir / "CLI_PROGRESS.md"):
        if path.is_file():
            progress.append({"path": path.relative_to(run_dir).as_posix(), "text": path.read_text(errors="replace")[-20000:]})
    lifecycle = run_dir / "copilot_control/lifecycle.log"
    console = run_dir / "copilot_control/console.log"
    console_tail = ""
    if console.is_file():
        with console.open("rb") as handle:
            handle.seek(max(0, console.stat().st_size - 24000))
            console_tail = handle.read().decode("utf-8", errors="replace")
    return {"id": run_dir.name, "tasks": tasks, "candidates": ordered, "progress": progress,
            "lifecycle": lifecycle.read_text(errors="replace")[-8000:] if lifecycle.is_file() else "",
            "console_tail": console_tail, "video_status": _read_json(run_dir / "media/video_status.json"),
            "counts": {"total": len(ordered), "complete": sum(item["status"] == "complete" for item in ordered),
                       "videos": sum(bool(item["video"]) for item in ordered),
                       "reviewed": sum(item["review"] is not None for item in ordered),
                       "selected": sum(item["selected"] for item in ordered)}}


def create_app(diagnostics_dir: Path | None = None) -> FastAPI:
    diagnostics = (diagnostics_dir or ROOT / "diagnostics").resolve()
    app = FastAPI(title="RoboLab Rollout Live")
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    def resolve_run(run_id: str) -> Path:
        if not run_id or run_id != Path(run_id).name:
            raise HTTPException(404, "Run not found")
        path = (diagnostics / run_id).resolve()
        if path.parent != diagnostics or not path.is_dir():
            raise HTTPException(404, "Run not found")
        return path

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/sampling")
    def sampling_index():
        return FileResponse(WEB_DIR / "sampling.html", headers={"Cache-Control": "no-store"})

    @app.get("/six")
    def six_index():
        return FileResponse(WEB_DIR / "six.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/six")
    def six_run():
        import psutil

        sessions = sorted(diagnostics.glob("multi_env_six_*/experiment.json"))
        if not sessions:
            return {"tasks": [], "session": None, "sync": {}}
        directory = sessions[-1].parent
        manifest = _read_json(sessions[-1])
        processes = []
        for process in psutil.process_iter(["cmdline"]):
            processes.append(process.info.get("cmdline") or [])

        def text(path, limit=5000):
            try:
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - limit))
                    return stream.read().decode("utf-8", errors="replace")
            except OSError:
                return ""

        def url(path):
            return "/media/" + quote(directory.name) + "/" + quote(path.relative_to(directory).as_posix())

        tasks = []
        for record in manifest.get("tasks", []):
            folder = directory / record["task"]
            candidates = []
            for candidate in sorted(folder.glob("*/candidate_*")):
                if "interrupted" in candidate.parent.name:
                    continue
                result = _read_json(candidate / "result.json")
                request = _read_json(candidate / "request.json")
                instructions = _read_json(candidate.parent / "instructions.json")
                index = int(candidate.name.removeprefix("candidate_"))
                instruction = result.get("instruction") or (
                    instructions[index] if isinstance(instructions, list) and index < len(instructions) else {})
                if not instruction and request.get("prompt"):
                    instruction = {"prompt": request["prompt"]}
                videos = {path.stem: url(path) for path in candidate.glob("*.mp4") if result.get("status") == "complete"}
                frames = {}
                frame_dirs = sorted((candidate / "frames").glob("*"), reverse=True)
                for frame_dir in frame_dirs:
                    paths = list(frame_dir.glob("*.png"))
                    if paths:
                        frames = {path.stem: url(path) for path in paths}
                        break
                if not frames and not videos:
                    continue
                candidates.append({"key": str(candidate.relative_to(folder)), "id": result.get("id"),
                                   "complete": result.get("status") == "complete", "videos": videos,
                                   "frames": frames, "instruction": instruction,
                                   "steps": result.get("executed_steps", 0)})
            candidates.sort(key=lambda item: (bool(item["videos"]), not item["key"].startswith("initialization"), item["key"]))
            active = any(record["controller_uuid"] in args and "--session-id" in args for args in processes)
            simulator = any(str(ROOT / "sampling_multi_env.py") in args and any(str(folder) in arg for arg in args) for args in processes)
            selections = [{"file": path.name, "data": _read_json(path)} for path in sorted(folder.glob("selected_*.json"))]
            tasks.append({**record, "active": active, "simulator": simulator,
                          "progress": text(folder / "PROGRESS.md"), "console": text(folder / "console.log"),
                          "selections": selections, "candidates": candidates,
                          "completed": sum(item["complete"] for item in candidates)})
        sync = _read_json(Path.home() / ".blob-sync/instruction-results" / directory.name / "status.json")
        return {"session": directory.name, "tasks": tasks,
                "sync": {key: sync.get(key) for key in ("status", "uploaded_files", "checked_at", "errors")}}

    @app.get("/api/sampling")
    def sampling_runs():
        return {"runs": [{"id": path.name} for path in sorted(diagnostics.glob("instruction_*"), reverse=True)
                         if path.is_dir() and (path / "experiment.json").is_file()]}

    @app.get("/api/sampling/{run_id}")
    def sampling_run(run_id: str):
        directory = resolve_run(run_id)
        if not (directory / "experiment.json").is_file():
            raise HTTPException(404, "Sampling experiment not found")
        return _sampling_run(directory)

    @app.get("/api/runs")
    def runs():
        if not diagnostics.is_dir():
            return {"runs": []}
        candidates = [
            path for path in diagnostics.iterdir()
            if path.is_dir() and path.name.startswith("rollout_")
            and any((path / marker).exists() for marker in ("launch.log", "result.json", "frames", "chunks"))
        ]
        summaries = [_summarize(path) for path in candidates]
        summaries.sort(key=lambda item: item["updated_at"], reverse=True)
        return {"runs": summaries}

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        return _summarize(resolve_run(run_id))

    @app.get("/api/runs/{run_id}/chunks/{chunk_index}/viewport.mp4")
    def chunk_viewport_video(run_id: str, chunk_index: int):
        run_dir = resolve_run(run_id)
        result = _read_json(run_dir / "result.json")
        chunk = next((item for item in result.get("chunks", []) if item.get("index") == chunk_index), None)
        if chunk is None:
            raise HTTPException(404, "Chunk not found")
        try:
            path = _make_chunk_video(
                run_dir, chunk_index, int(chunk.get("start_step", 0)), int(chunk.get("executed_steps", 0))
            )
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
            raise HTTPException(404, str(error)) from error
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/runs/{run_id}/chunks/{chunk_index}/three-view.mp4")
    def chunk_three_view_video(run_id: str, chunk_index: int):
        run_dir = resolve_run(run_id)
        result = _read_json(run_dir / "result.json")
        chunk = next((item for item in result.get("chunks", []) if item.get("index") == chunk_index), None)
        if chunk is None:
            raise HTTPException(404, "Chunk not found")
        try:
            path = _make_three_view_video(
                run_dir, chunk_index, int(chunk.get("start_step", 0)), int(chunk.get("executed_steps", 0))
            )
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
            raise HTTPException(404, str(error)) from error
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/media/{run_id}/{asset_path:path}")
    def media(run_id: str, asset_path: str):
        run_dir = resolve_run(run_id)
        path = (run_dir / asset_path).resolve()
        if run_dir not in path.parents or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".mp4"}:
            raise HTTPException(404, "Asset not found")
        if not path.is_file():
            raise HTTPException(404, "Asset not found")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the live RoboLab rollout dashboard")
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "diagnostics")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.diagnostics), host=args.host, port=args.port)


if __name__ == "__main__":
    main()