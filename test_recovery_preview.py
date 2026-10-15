from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import struct
import subprocess

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from rollout_dashboard.preview import _FFMPEG, media_preview
from rollout_dashboard.server import create_app
from test_recovery_dashboard import episode


def test_image_preview_cache_and_source_invalidation(episode):
    run, _ = episode
    source = run / "recovery/frame.png"
    Image.new("RGB", (1280, 720), "blue").save(source)
    original = source.read_bytes()
    preview = media_preview(run, source)
    with Image.open(preview) as image:
        assert image.size == (640, 360)
        assert image.format == "JPEG"
    assert media_preview(run, source) == preview
    assert source.read_bytes() == original
    Image.new("RGB", (1280, 720), "red").save(source)
    assert media_preview(run, source) != preview


def test_concurrent_cache_misses_encode_once(episode, monkeypatch):
    run, _ = episode
    source = run / "recovery/frame.png"
    Image.new("RGB", (32, 32), "blue").save(source)
    original_save = Image.Image.save
    calls = []

    def save(image, *args, **kwargs):
        calls.append(args[0])
        return original_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save)
    with ThreadPoolExecutor(max_workers=3) as executor:
        paths = list(executor.map(lambda _: media_preview(run, source), range(3)))
    assert len(set(paths)) == 1
    assert len(calls) == 1


def test_invalid_image_has_no_partial_cached_output(episode):
    run, _ = episode
    with pytest.raises(OSError):
        media_preview(run, run / "recovery/frame.png")
    assert not list((run / ".dashboard_cache").iterdir())


def test_preview_rejects_outside_sources_and_cache(episode, tmp_path):
    run, _ = episode
    outside = tmp_path / "outside.png"
    Image.new("RGB", (32, 32)).save(outside)
    with pytest.raises(ValueError, match="outside"):
        media_preview(run, outside)
    cache = tmp_path / "outside-cache"
    cache.mkdir()
    (run / ".dashboard_cache").symlink_to(cache)
    with pytest.raises(ValueError, match="cache is outside"):
        media_preview(run, run / "recovery/frame.png")


def test_preview_route_ranges_private_cache_and_explicit_errors(episode):
    run, _ = episode
    source = run / "recovery/frame.png"
    Image.new("RGB", (1280, 720), "blue").save(source)
    with TestClient(create_app(run.parent)) as client:
        url = f"/media/{run.name}/recovery/frame.png"
        original = client.get(url)
        preview = client.get(url + "?preview=true")
        assert original.headers["content-type"] == "image/png"
        assert original.headers["cache-control"] == "no-store"
        assert preview.headers["content-type"] == "image/jpeg"
        assert preview.headers["cache-control"] == "private, max-age=3600"
        part = client.get(url + "?preview=true", headers={"Range": "bytes=0-9"})
        assert part.status_code == 206
        assert part.content == preview.content[:10]
        source.write_bytes(b"invalid")
        assert client.get(url + "?preview=true").status_code == 500
        (run / "recovery/manifest.json").unlink()
        assert client.get(url + "?preview=true").status_code == 404
        assert client.get(url).status_code == 200


def test_new_recording_and_preview_have_front_index_and_same_frame_timing(tmp_path):
    from recovery_worker import _encode_video

    frames = []
    for index, color in enumerate(("blue", "red", "green", "white")):
        frame = tmp_path / f"{index}.png"
        Image.new("RGB", (64, 48), color).save(frame)
        frames.append(frame)
    source = tmp_path / "original.mp4"
    _encode_video(frames, source, fps=4)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    preview = media_preview(tmp_path, source)
    for path in (source, preview):
        boxes = []
        with path.open("rb") as handle:
            while header := handle.read(8):
                size, kind = struct.unpack(">I4s", header)
                boxes.append(kind)
                handle.seek(size - 8, 1)
        assert boxes.index(b"moov") < boxes.index(b"mdat")
        result = subprocess.run([
            str(_FFMPEG.with_name("ffprobe")), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate,duration", "-of", "json", str(path),
        ], check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)["streams"][0]
        assert stream["nb_frames"] == "4"
        assert stream["r_frame_rate"] == "4/1"
        assert float(stream["duration"]) == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
