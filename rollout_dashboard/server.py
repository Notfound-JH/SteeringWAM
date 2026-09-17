from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent / "web"
CAMERAS = (
    "egocentric_mirrored_camera",
    "wrist_cam",
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _image_url(run_name: str, relative_path: str) -> str:
    return f"/media/{run_name}/{relative_path}"


def _frame_sets(run_dir: Path, run_name: str) -> list[dict]:
    frames_dir = run_dir / "frames"
    sets = []
    if frames_dir.is_dir():
        for step_dir in sorted(frames_dir.glob("step_*")):
            try:
                step = int(step_dir.name.removeprefix("step_"))
            except ValueError:
                continue
            images = {
                camera: _image_url(run_name, f"frames/{step_dir.name}/{camera}.png")
                for camera in CAMERAS
                if (step_dir / f"{camera}.png").is_file()
            }
            if images:
                sets.append({"step": step, "label": f"Step {step}", "images": images})

    initial = {
        camera: _image_url(run_name, f"{camera}.png")
        for camera in CAMERAS
        if (run_dir / f"{camera}.png").is_file()
    }
    if initial:
        sets.insert(0, {"step": 0, "label": "Initial", "images": initial})

    final = {
        camera: _image_url(run_name, f"final/{camera}.png")
        for camera in CAMERAS
        if (run_dir / "final" / f"{camera}.png").is_file()
    }
    if final and (not sets or final != sets[-1]["images"]):
        final_step = max((item["step"] for item in sets), default=0)
        sets.append({"step": final_step, "label": "Final", "images": final})
    return sets


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
        output.append({
            "index": index,
            "start_step": item.get("start_step"),
            "executed_steps": item.get("executed_steps", 0),
            "inference_seconds": item.get("inference_seconds"),
            "prefix_matches": item.get("executed_prefix_matches"),
            "prompt": item.get("prompt", result.get("instruction")),
            "input": _image_url(run_name, f"chunks/{chunk_dir.name}/input.png")
            if (chunk_dir / "input.png").is_file() else None,
            "actions_ready": (chunk_dir / "actions.npy").is_file(),
        })
    return output


def _summarize(run_dir: Path) -> dict:
    result = _read_json(run_dir / "result.json")
    frames = _frame_sets(run_dir, run_dir.name)
    chunks = _chunks(run_dir, run_dir.name, result)
    stat = run_dir.stat()
    status = result.get("status") or ("running" if (run_dir / "launch.log").exists() else "preparing")
    steps = result.get("steps_executed")
    if steps is None:
        steps = max((frame["step"] for frame in frames), default=0)
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
        "frames": frames,
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

    @app.get("/api/runs/{run_id}/log")
    def log(run_id: str):
        path = resolve_run(run_id) / "launch.log"
        if not path.is_file():
            return {"lines": []}
        try:
            lines = path.read_text(errors="replace").splitlines()[-80:]
        except OSError:
            lines = []
        return {"lines": lines}

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