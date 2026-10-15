"""Download selected final evidence and compare SHA256 without exposing SAS."""

import hashlib
import json
import logging
from pathlib import Path

from azure.core.exceptions import AzureError
from azure.storage.blob import ContainerClient


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
logging.getLogger("azure").setLevel(logging.CRITICAL)
results = json.loads((STUDY / "study_results.json").read_text())
groups = {STUDY: [
    "experiment.json", "study_results.json", "trial_results.csv", "chunk_metrics.csv",
    "comparison_report.md", "model_server.log", "initial_matching_first_batch.json",
    "initial_matching_second_batch.json", "drive_subgoal.py", "review_checkpoint.py",
    "audit_study.py", "verify_archive.py",
]}
for trial in results["episodes"]:
    run = ROOT / "diagnostics" / trial["run"]
    groups[run] = [
        "experiment.json", "study_episode.json", "subgoal_intents.jsonl",
        "subgoal_responses.jsonl", "recovery/manifest.json",
        "recovery/verification.json", "recovery/events.jsonl",
        "recovery/trajectory.hdf5", "recovery/episode_wrist.mp4",
        "recovery/segments/000_wam/predicted_actions.npy",
    ]
sas = (Path.home() / ".blob-sync/temp_sas_url.txt").read_text().strip()
checks = []
location = "container"
try:
    with ContainerClient.from_container_url(
        sas, logging_enable=False, retry_total=3, connection_timeout=15, read_timeout=120,
    ) as container:
        for directory, relatives in groups.items():
            for relative in relatives:
                location = f"{directory.name}/{relative}"
                path = directory / relative
                local = hashlib.sha256(path.read_bytes()).hexdigest()
                downloaded = hashlib.sha256()
                length = 0
                blob = f"jinghe/steeringwam/recovery/{directory.name}/{relative}"
                for block in container.download_blob(blob).chunks():
                    downloaded.update(block)
                    length += len(block)
                if length != path.stat().st_size or local != downloaded.hexdigest():
                    raise ValueError(f"Archive mismatch: {directory.name}/{relative}")
                checks.append({
                    "directory": directory.name, "path": relative, "bytes": length,
                    "local_sha256": local, "downloaded_sha256": downloaded.hexdigest(),
                    "match": True,
                })
            print(json.dumps({"verified_directory": directory.name, "verified_files": len(checks)}), flush=True)
except AzureError as error:
    raise RuntimeError(
        f"Azure readback failed for {location}: {type(error).__name__}, HTTP {getattr(error, 'status_code', None)}"
    ) from None
report = {
    "independent_readback_verified": True, "verified_files": len(checks),
    "scope": "Selected final critical artifacts, not a download of every camera image.",
    "checks": checks,
}
(STUDY / "azure_integrity_check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"verified_files": len(checks), "all_sha256_match": True}))
