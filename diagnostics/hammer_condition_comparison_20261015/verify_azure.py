"""Independently download and hash selected archived evidence without logging SAS."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from azure.core.exceptions import AzureError
from azure.storage.blob import ContainerClient


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=["A", "B", "C", "study"])
args = parser.parse_args()
logging.getLogger("azure").setLevel(logging.CRITICAL)
if args.variant == "study":
    directory = STUDY
    files = [
        "experiment.json", "study_design.json", "initial_matching_audit.json",
        "study_verification.json", "model_server.log", "motion_summary.json",
        "error_annotations.json", "error_distribution.csv", "comparison_report.md",
        "extract_motion.py", "audit_finished_study.py", "build_error_annotations.py",
        "verify_azure.py",
    ]
else:
    directory = ROOT / "diagnostics" / f"recovery_hammer_condition_{args.variant}_seed3407_20261015"
    files = [
        "experiment.json", "study_episode.json", "recovery/manifest.json",
        "recovery/verification.json", "recovery/events.jsonl",
        "recovery/trajectory.hdf5", "recovery/episode_wrist.mp4",
        "recovery/observations/00000_initial/scene.npz",
        "recovery/observations/00600_wam_008/scene.npz",
        "recovery/observations/00600_wam_008/wrist_cam.png",
        "recovery/segments/000_wam/predicted_actions.npy",
        "recovery/segments/037_wam/predicted_actions.npy",
        "recovery/segments/037_wam/executed_actions.npy",
    ]
prefix = f"jinghe/steeringwam/recovery/{directory.name}"
checks = []
sas = (Path.home() / ".blob-sync/temp_sas_url.txt").read_text().strip()
try:
    with ContainerClient.from_container_url(
        sas, logging_enable=False, retry_total=3, connection_timeout=15, read_timeout=120,
    ) as container:
        for relative in files:
            local = directory / relative
            expected = hashlib.sha256()
            with local.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    expected.update(block)
            actual = hashlib.sha256()
            downloaded = 0
            for block in container.download_blob(f"{prefix}/{relative}").chunks():
                actual.update(block)
                downloaded += len(block)
            if actual.digest() != expected.digest() or downloaded != local.stat().st_size:
                raise ValueError(f"Archive mismatch: {directory.name}/{relative}")
            checks.append({
                "path": relative, "bytes": downloaded,
                "local_sha256": expected.hexdigest(),
                "downloaded_sha256": actual.hexdigest(), "match": True,
            })
except AzureError as error:
    raise RuntimeError(
        f"Azure readback failed: {type(error).__name__}, HTTP {getattr(error, 'status_code', None)}"
    ) from None
report = {
    "remote_prefix": prefix, "independent_download_verified": True,
    "verified_files": len(checks), "checks": checks,
    "scope": "Selected critical evidence, not an independent download of every archived image.",
}
(directory / "azure_integrity_check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({
    "directory": directory.name, "verified_files": len(checks),
    "verified_bytes": sum(c["bytes"] for c in checks), "all_hashes_match": True,
}))
