import subprocess

import pytest

import sync_recovery_code


@pytest.fixture
def repository(tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    remote = tmp_path / "remote.git"
    worktree.mkdir()
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(
        ["git", "init", "-b", "feature/wam-agent-recovery", str(worktree)],
        check=True, capture_output=True,
    )
    monkeypatch.setattr(sync_recovery_code, "ROOT", worktree)
    git = sync_recovery_code.git
    git("config", "user.name", "Recovery Sync Test")
    git("config", "user.email", "test@example.invalid")
    (worktree / "tracked.txt").write_text("committed\n")
    git("add", "tracked.txt")
    git("commit", "-m", "Initial checkpoint")
    git("remote", "add", "origin", str(remote))
    return worktree


def test_push_verifies_commit_without_staging_dirty_files(repository):
    (repository / "tracked.txt").write_text("uncommitted\n")
    (repository / "untracked.txt").write_text("must stay local\n")
    before = sync_recovery_code.git("status", "--porcelain")
    result = sync_recovery_code.sync_once("feature/wam-agent-recovery")
    assert result["status"] == "synced"
    assert result["remote_commit"] == sync_recovery_code.git("rev-parse", "HEAD")
    assert sync_recovery_code.git("status", "--porcelain") == before


def test_failed_push_is_not_reported_as_synced(repository):
    sync_recovery_code.git("remote", "set-url", "origin", str(repository / "missing-remote"))
    with pytest.raises(RuntimeError, match="push failed"):
        sync_recovery_code.sync_once("feature/wam-agent-recovery")
