"""Small, disposable media previews; never rewrite recorded artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Lock

from PIL import Image


_LOCK = Lock()
_PROFILE = "640-crf28-g30-jpeg80-v1"
_FFMPEG = Path(__file__).resolve().parents[1] / "packages/ffmpeg/bin/ffmpeg"


def media_preview(directory: Path, source: Path) -> Path:
    directory, source = directory.resolve(), source.resolve()
    if not source.is_relative_to(directory) or not source.is_file():
        raise ValueError("Preview source is missing or outside the episode")
    video = source.suffix.lower() == ".mp4"
    if not video and source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Unsupported preview media")
    stat = source.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    key = hashlib.sha256(
        f"{source.relative_to(directory)}:{signature}:{_PROFILE}".encode()
    ).hexdigest()
    cache = (directory / ".dashboard_cache").resolve()
    if not cache.is_relative_to(directory):
        raise ValueError("Preview cache is outside the episode")
    output = cache / (key + (".mp4" if video else ".jpg"))
    with _LOCK:
        if output.is_symlink():
            raise ValueError("Preview cache must not contain symlinks")
        if output.is_file():
            return output
        cache.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, suffix=output.suffix, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            if video:
                executable = str(_FFMPEG) if _FFMPEG.is_file() else shutil.which("ffmpeg")
                if not executable:
                    raise FileNotFoundError("ffmpeg is required for video previews")
                subprocess.run([
                    executable, "-nostdin", "-y", "-v", "error", "-i", str(source),
                    "-map", "0:v:0", "-vf", "scale=w='min(640,iw)':h=-2",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "28", "-g", "30",
                    "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                    "-movflags", "+faststart", "-an", "-threads", "2", str(temporary),
                ], check=True)
            else:
                with Image.open(source) as image:
                    preview = image.convert("RGB")
                    preview.thumbnail((640, 480))
                    preview.save(temporary, format="JPEG", quality=80)
            current = source.stat()
            if (current.st_size, current.st_mtime_ns) != signature:
                raise RuntimeError("Preview source changed during generation; retry the request")
            if not temporary.stat().st_size:
                raise RuntimeError("Preview encoder produced an empty file")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output
