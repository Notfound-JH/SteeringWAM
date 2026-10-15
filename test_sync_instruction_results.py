import hashlib
import json
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest

from sync_instruction_results import Backup, candidate_artifacts
import sync_instruction_results


class Container:
    def __init__(self):
        self.files = {}
        self.order = []
        self.fail = False

    def get_blob_client(self, name):
        container = self

        class Blob:
            def upload_blob(self, stream, **kwargs):
                if container.fail:
                    raise RuntimeError("TEST failure")
                container.files[name] = (stream.read(), kwargs["metadata"])
                container.order.append(name)

            def get_blob_properties(self, **kwargs):
                raw, metadata = container.files[name]
                return SimpleNamespace(size=len(raw), metadata=metadata)

        return Blob()


def test_multi_env_candidate_marker_last(tmp_path):
    session = tmp_path / "session"
    state = tmp_path / "state"
    state.mkdir()
    candidate = session / "fruits/round_000/candidate_00"
    candidate.mkdir(parents=True)
    (candidate / "actions.npy").write_bytes(b"TEST ACTIONS")
    (candidate / "result.json").write_text('{"status":"complete"}')
    pending = candidate.parent / "candidate_01"
    pending.mkdir()
    (pending / "partial.npy").write_bytes(b"INCOMPLETE")
    container = Container()
    result = Backup(session, "prefix", state, container).run_once()
    assert result["status"] == "synced"
    assert container.order == ["prefix/fruits/round_000/candidate_00/actions.npy",
                               "prefix/fruits/round_000/candidate_00/result.json"]


def test_complete_candidate_last_incremental_retry(tmp_path):
    session = tmp_path / "session"
    state = tmp_path / "state"
    state.mkdir()
    candidate = session / "fruits/candidates/first"
    candidate.mkdir(parents=True)
    actions = candidate / "actions.npy"
    actions.write_bytes(b"TEST ACTIONS")
    marker = candidate / "result.json"
    marker.write_text(json.dumps({"status": "complete", "artifacts": [
        {"path": str(actions), "sha256": hashlib.sha256(actions.read_bytes()).hexdigest()}]}))
    incomplete = session / "fruits/candidates/pending"
    incomplete.mkdir()
    (incomplete / "partial.npy").write_bytes(b"NOT COMPLETE")
    metadata = session / "fruits/session.json"
    metadata.write_text('{"status":"pending_review"}')
    container = Container()
    backup = Backup(session, "prefix", state, container)
    container.fail = True
    assert backup.run_once()["status"] == "retrying"
    assert not backup.cache and not container.files
    container.fail = False
    assert backup.run_once()["status"] == "synced"
    assert container.order == ["prefix/fruits/candidates/first/actions.npy",
                               "prefix/fruits/candidates/first/result.json", "prefix/fruits/session.json"]
    assert backup.run_once()["status"] == "synced" and len(container.order) == 3
    metadata.write_text('{"status":"ready","review":"TEST ONLY"}')
    backup.run_once()
    assert len(container.order) == 4
    failure = incomplete / "failure.json"
    failure.write_text('{"operation_exit_code":1}')
    failed = backup.run_once()
    assert failed["status"] == "synced"
    assert failed["failed_candidates_backed_up"] == ["pending"]
    assert failed["completed_candidates"] == ["first"]
    assert container.order[-2:] == ["prefix/fruits/candidates/pending/partial.npy",
                                   "prefix/fruits/candidates/pending/failure.json"]
    actions.write_bytes(b"CHANGED")
    assert backup.run_once()["status"] == "retrying"


def test_no_artifact_escape(tmp_path):
    session = tmp_path / "session"
    candidate = session / "tools/candidates/first"
    candidate.mkdir(parents=True)
    secret = tmp_path / "private.txt"
    secret.write_text("NOT AN ARTIFACT")
    marker = candidate / "result.json"
    marker.write_text(json.dumps({"status": "complete", "artifacts": [{"path": str(secret), "sha256": "x"}]}))
    with pytest.raises(ValueError):
        candidate_artifacts(session, marker)


def test_nested_retry_result_uploaded_after_its_artifacts(tmp_path):
    session = tmp_path / "session"
    state = tmp_path / "state"
    state.mkdir()
    child = session / "retries/fruits_chunk_000_extra1/fruits"
    candidate = child / "candidates/retried"
    candidate.mkdir(parents=True)
    actions = candidate / "actions.npy"
    actions.write_bytes(b"TEST RETRY")
    (candidate / "result.json").write_text(json.dumps({"status": "complete", "artifacts": [
        {"path": str(actions), "sha256": hashlib.sha256(actions.read_bytes()).hexdigest()}]}))
    (child / "session.json").write_text('{"status":"pending_review"}')
    container = Container()
    result = Backup(session, "prefix", state, container).run_once()
    assert result["status"] == "synced" and result["completed_candidates"] == ["retried"]
    prefix = "prefix/retries/fruits_chunk_000_extra1/fruits/"
    assert container.order == [prefix + "candidates/retried/actions.npy",
                               prefix + "candidates/retried/result.json", prefix + "session.json"]


def test_cli_bounds_large_array_upload_requests(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    (session / "experiment.json").write_text("{}")
    sas = tmp_path / "test_sas.txt"
    sas.write_text("https://example.blob.core.windows.net/test?sig=TEST")
    container = Container()

    def connect(url, **kwargs):
        assert url == sas.read_text()
        assert kwargs["max_single_put_size"] == 4 * 1024 * 1024
        assert kwargs["max_block_size"] == 4 * 1024 * 1024
        return nullcontext(container)

    monkeypatch.setattr(sync_instruction_results.ContainerClient, "from_container_url", connect)
    monkeypatch.setattr(sync_instruction_results.os, "umask", lambda _: None)
    monkeypatch.setattr(sync_instruction_results.signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "argv", [
        "sync_instruction_results.py", "--session", str(session),
        "--sas-file", str(sas), "--state-dir", str(tmp_path / "state"), "--once",
    ])
    assert sync_instruction_results.main() == 0
    assert len(container.files) == 1