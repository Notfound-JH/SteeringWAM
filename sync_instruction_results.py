"""Incremental SAS backup sidecar; never modifies the sampling session."""

import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from urllib.parse import urlsplit

from azure.storage.blob import ContainerClient, ContentSettings


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def signature(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def safe_path(session, path):
    path = Path(path).resolve(strict=True)
    path.relative_to(session)
    if not path.is_file():
        raise ValueError("Artifact must be a regular file inside this session")
    return path


def candidate_artifacts(session, marker):
    raw = marker.read_bytes()
    result = json.loads(raw)
    if result.get("status") != "complete":
        return None
    artifacts = []
    if marker.parent.name.startswith("candidate_"):
        for path in sorted(marker.parent.rglob("*")):
            if path.is_file() and path != marker and path.suffix != ".tmp":
                path = safe_path(session, path)
                artifacts.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
        artifacts.append((marker, hashlib.sha256(raw).hexdigest()))
        return artifacts
    for reference in result["artifacts"]:
        path = safe_path(session, reference["path"])
        path.relative_to(marker.parent)
        artifacts.append((path, reference["sha256"]))
    if not artifacts:
        raise ValueError("Completed candidate has no artifacts")
    artifacts.append((marker, hashlib.sha256(raw).hexdigest()))
    return artifacts


def metadata_paths(session):
    for path in sorted(session.rglob("*")):
        relative = path.relative_to(session)
        if any(part.startswith(".") for part in relative.parts) or path.suffix == ".tmp":
            continue
        if "candidates" in relative.parts or any(part.startswith("candidate_") for part in relative.parts) or not path.is_file():
            continue
        if relative.parts[0] == "services" and "runtime" in relative.parts:
            continue
        if path.name != "session.json":
            yield safe_path(session, path)
    yield from sorted(session.rglob("session.json"))


class Backup:
    def __init__(self, session, remote, state_dir, container):
        self.session = session.resolve()
        self.remote = remote.strip("/")
        if not self.remote or any(part in ("", ".", "..") for part in self.remote.split("/")):
            raise ValueError("Invalid destination prefix")
        self.state_dir = state_dir
        self.container = container
        self.cache_path = state_dir / "uploaded.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.uploads = 0
        self.completed = []
        self.failed = []

    def upload(self, path, expected=None):
        path = safe_path(self.session, path)
        key = path.relative_to(self.session).as_posix()
        before = signature(path)
        cached = self.cache.get(key)
        if cached and cached["signature"] == before and (expected is None or cached["sha256"] == expected):
            return
        checksum = hashlib.sha256()
        with tempfile.TemporaryFile(dir=self.state_dir) as snapshot:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    checksum.update(chunk)
                    snapshot.write(chunk)
            if signature(path) != before:
                raise RuntimeError("Source changed during snapshot; retry next pass")
            sha256 = checksum.hexdigest()
            if expected is not None and sha256 != expected:
                raise ValueError("Candidate artifact does not match its recorded SHA-256")
            size = snapshot.tell()
            snapshot.seek(0)
            blob = self.container.get_blob_client(f"{self.remote}/{key}")
            blob.upload_blob(snapshot, length=size, overwrite=True, max_concurrency=1,
                             validate_content=True, metadata={"sha256": sha256},
                             content_settings=ContentSettings(content_type="application/octet-stream"), timeout=120)
            properties = blob.get_blob_properties(timeout=30)
            if properties.size != size or properties.metadata.get("sha256") != sha256:
                raise RuntimeError("Remote length or checksum metadata differs")
        self.cache[key] = {"signature": before, "sha256": sha256, "bytes": size}
        self.uploads += 1

    def run_once(self):
        errors = []
        self.completed = []
        self.failed = []
        markers = list(self.session.rglob("candidates/*/result.json"))
        markers.extend(self.session.rglob("candidate_*/result.json"))
        for marker in sorted(markers):
            try:
                artifacts = candidate_artifacts(self.session, marker)
                if artifacts is None:
                    continue
                for path, expected in artifacts:
                    self.upload(path, expected)
                self.completed.append(marker.parent.name)
                atomic_json(self.cache_path, self.cache)
            except Exception as error:
                errors.append(self.error(marker, error))
        for failure in sorted(self.session.rglob("candidates/*/failure.json")):
            try:
                for path in sorted(failure.parent.rglob("*")):
                    if path.is_file() and path != failure and path.suffix != ".tmp":
                        self.upload(path)
                self.upload(failure)
                self.failed.append(failure.parent.name)
            except Exception as error:
                errors.append(self.error(failure, error))
        for path in metadata_paths(self.session):
            try:
                self.upload(path)
            except Exception as error:
                errors.append(self.error(path, error))
        atomic_json(self.cache_path, self.cache)
        status = {"pid": os.getpid(), "session": str(self.session), "remote_prefix": self.remote,
                  "checked_at": time.time(), "uploaded_files": len(self.cache), "uploads_this_process": self.uploads,
                  "completed_candidates": self.completed, "failed_candidates_backed_up": self.failed, "errors": errors,
                  "status": "synced" if not errors else "retrying"}
        atomic_json(self.state_dir / "status.json", status)
        return status

    def error(self, path, error):
        return {"path": str(path.relative_to(self.session)), "type": type(error).__name__,
                "http_status": getattr(error, "status_code", None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--sas-file", type=Path, default=Path.home() / ".blob-sync/temp_sas_url.txt")
    parser.add_argument("--remote-prefix", default="jinghe/steeringwam/instruction_sampling")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be at least one second")
    session = args.session.resolve(strict=True)
    if not (session / "experiment.json").is_file():
        parser.error("Expected a sampling experiment directory")
    os.umask(0o077)
    logging.getLogger("azure").setLevel(logging.CRITICAL)
    state_dir = args.state_dir or Path.home() / ".blob-sync/instruction-results" / session.name
    state_dir = state_dir.resolve()
    if state_dir.is_relative_to(session):
        parser.error("Keep uploader state outside the sampling session")
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    remote = f"{args.remote_prefix.strip('/')}/{session.name}"
    binding = {"session": str(session), "remote_prefix": remote}
    with (state_dir / "upload.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = state_dir / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != binding:
            raise ValueError("Uploader cache belongs to another session/destination")
        atomic_json(config_path, binding)
        stopped = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stopped.set())
        previous = None
        while not stopped.is_set():
            try:
                sas = args.sas_file.read_text().strip()
                parsed = urlsplit(sas)
                if parsed.scheme != "https" or not parsed.netloc or not parsed.query:
                    raise ValueError("An HTTPS container SAS URL is required")
                with ContainerClient.from_container_url(sas, logging_enable=False, retry_total=3,
                                                       connection_timeout=15, read_timeout=120,
                                                       max_single_put_size=4 * 1024 * 1024,
                                                       max_block_size=4 * 1024 * 1024) as container:
                    backup = Backup(session, remote, state_dir, container)
                    status = backup.run_once()
                summary = {key: status[key] for key in ("status", "uploaded_files", "completed_candidates", "failed_candidates_backed_up", "errors")}
                if summary != previous:
                    print(json.dumps(summary), flush=True)
                    previous = summary
                if args.once:
                    return 1 if status["errors"] else 0
            except Exception as error:
                summary = {"status": "retrying", "type": type(error).__name__,
                           "http_status": getattr(error, "status_code", None), "checked_at": time.time()}
                atomic_json(state_dir / "status.json", summary)
                print(json.dumps(summary), flush=True)
                if args.once:
                    return 1
            stopped.wait(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())