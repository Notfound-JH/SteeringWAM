from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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


def _chunk_viewport_frames(run_dir: Path, start_step: int, executed_steps: int) -> list[Path]:
    camera = "egocentric_mirrored_camera.png"
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