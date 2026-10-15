"""Export a finalized episode as one private, self-contained HTML review."""

from __future__ import annotations

import argparse
import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import quote, unquote

from PIL import Image

from .recovery import load_episode


def _embedded_media(directory: Path, url: str, *, image: bool = False) -> str:
    prefix = f"/media/{quote(directory.name, safe='')}/"
    if not isinstance(url, str) or not url.startswith(prefix):
        raise ValueError("Expected media from the selected episode")
    path = (directory / unquote(url[len(prefix):])).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Media is missing or outside the selected episode")
    if image:
        with Image.open(path) as source:
            preview = source.convert("RGB")
            preview.thumbnail((640, 480))
            buffer = BytesIO()
            preview.save(buffer, format="JPEG", quality=80)
            raw = buffer.getvalue()
        mime = "image/jpeg"
    else:
        if path.suffix.lower() != ".mp4":
            raise ValueError("Expected an MP4 episode video")
        raw = path.read_bytes()
        mime = "video/mp4"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _details(title: str, value: object) -> str:
    text = "Not recorded" if value is None else json.dumps(value, indent=2, ensure_ascii=True)
    return f"<details open><summary>{escape(title)}</summary><pre>{escape(text)}</pre></details>"


def render_episode(directory: Path) -> str:
    directory = directory.resolve()
    data = load_episode(directory)
    video = _embedded_media(directory, data["video"])
    metadata = {key: data[key] for key in (
        "id", "task", "steps", "experiment_mode", "protocol_version", "success_criterion",
        "native_success", "acceptance_success", "agent_control_success", "mock_execution",
        "agent_input_mode", "agent_model_version",
        "budget_exhaustion_policy", "budget_fallback", "max_episode_steps",
    )}
    sections = []
    for attempt in data["attempts"]:
        mode = escape(attempt["proposal"].get("correction_mode", "Not recorded"))
        start, end = attempt["start"], attempt["end"]
        images = []
        for label, step in (("Before", start), ("After", end)):
            frames = data["observations"][step]["frames"]
            for camera in ("wrist_cam", "over_shoulder_left_camera"):
                if camera not in frames:
                    images.append(f"<p>{label}: {camera} image not recorded.</p>")
                    continue
                url = _embedded_media(directory, frames[camera], image=True)
                caption = f"{label} / step {step} / {camera}"
                images.append(
                    f'<figure><img src="{url}" alt="{caption}" loading="lazy">'
                    f"<figcaption>{caption}</figcaption></figure>"
                )
        title = f"Attempt {attempt['number']} / {mode}"
        sections.append(
            f'<section class="card"><h2>{title}</h2>'
            f"<p>Steps {start} to {end}; {attempt['executed_steps']} actual control steps. "
            f'Video interval: {start * data["dt"]:.2f} to {end * data["dt"]:.2f} seconds.</p>'
            f'<button data-seconds="{start * data["dt"]}">Jump to this attempt</button>'
            + _details("Agent proposal / bounded code / condition", attempt["proposal"])
            + _details("WAM evaluation window", attempt["evaluation"])
            + _details("Local assessment (not full task success)", attempt["assessment"])
            + '<div class="images">' + "".join(images) + "</div></section>"
        )
    if not sections:
        sections.append('<section class="card"><h2>No recorded correction attempts</h2></section>')
    phases = "".join(
        f'<li>{escape(phase["source"])}: steps {phase["start"]} to {phase["end"]}</li>'
        for phase in data["phases"]
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="'
        "default-src 'none'; img-src data:; media-src data:; "
        "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
        "connect-src 'none'; base-uri 'none'; form-action 'none'\">"
        "<title>Recovery Lab - Offline Review</title><style>"
        "body{margin:0;background:#edf2f1;color:#203a39;font:16px system-ui,sans-serif}"
        "main{max-width:1100px;margin:auto;padding:24px}h1,h2{overflow-wrap:anywhere}"
        ".card{background:white;padding:24px;border-radius:12px;margin:20px 0}"
        ".notice{border-left:5px solid #b56a00;padding:14px;background:#fff3dc}"
        "video{width:100%;max-height:600px;background:#182424}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f5;padding:12px}"
        "details{margin:16px 0}summary{cursor:pointer;font-weight:600}"
        ".images{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%}figcaption{font-size:13px;margin:6px 0}"
        "button{padding:10px 14px;cursor:pointer}a{color:#146458}"
        "</style></head><body><main><h1>Recovery Lab / Offline Review</h1>"
        f'<p>{escape(data["id"])}</p><p class="notice">'
        "PRIVATE OFFLINE SNAPSHOT. This file contains experiment data and video; share it carefully. "
        "No server, login, tunnel or network request is required. It is not a live dashboard. "
        "Still images are compressed previews; the original MP4 is embedded unchanged. "
        "Predicted futures, all per-step images and raw trajectories remain in the original archive. "
        "Local recovery labels, native success and verified full-chain acceptance are distinct.</p>"
        f'<section class="card"><h2>{escape(data["goal"])}</h2>'
        + _details("Recorded experiment and outcomes", metadata)
        + _details("Independent verification (unknown if not recorded)", data["verification"])
        + _details("Failure evidence", data["failure"])
        + '</section><section class="card" id="playback"><h2>Actual execution video</h2>'
        f'<video id="video" controls playsinline preload="metadata" src="{video}"></video>'
        '<p id="media-error" role="alert" hidden>'
        "Video could not load. Download the complete HTML and open it in a current Chrome or Edge "
        "browser, rather than the editor preview.</p>"
        f"<h3>Execution phases</h3><ul>{phases}</ul></section>"
        + "".join(sections)
        + """<script>
const video = document.getElementById("video");
video.addEventListener("error", () => {
  document.getElementById("media-error").hidden = false;
});
for (const button of document.querySelectorAll("[data-seconds]")) {
  button.addEventListener("click", () => {
    video.currentTime = Number(button.dataset.seconds);
    document.getElementById("playback").scrollIntoView({behavior: "smooth"});
  });
}
</script></main></body></html>"""
    )


def export_episode(directory: Path, output: Path) -> None:
    document = render_episode(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Finalized episode directory, above recovery/")
    parser.add_argument("--output", required=True, type=Path, help="New HTML file; never overwrite")
    args = parser.parse_args()
    export_episode(args.episode, args.output)
    print(f"Offline review: {args.output.resolve()} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
