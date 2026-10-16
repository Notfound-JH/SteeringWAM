"""Independently download final critical evidence and compare full SHA256."""

import hashlib
import json
import logging
from pathlib import Path

from azure.core.exceptions import AzureError
from azure.storage.blob import ContainerClient

RUN = Path(__file__).resolve().parent
logging.getLogger("azure").setLevel(logging.CRITICAL)
relatives = [
    "experiment.json", "study_episode.json", "audit_results.json", "task_progress.csv",
    "experiment_report.md", "audit_episode.py", "continue_wam.py", "verify_archive.py",
    "judge_05_raw.json", "judge_05_schema_rejection.json",
    "user_stop.json", "user_stop_endpoint_response.json", "user_stop_finalize_response.json",
    "services/model/server.log", "recovery/manifest.json", "recovery/verification.json",
    "recovery/events.jsonl", "recovery/trajectory.hdf5", "recovery/episode_wrist.mp4",
]
for provenance_path in sorted(RUN.glob("judge_*_provenance.json")):
    provenance = json.loads(provenance_path.read_text())
    relatives.extend([
        provenance_path.name, provenance["input"], provenance["raw_output"],
        provenance_path.name.replace("_provenance", "_submitted"),
        provenance_path.name.replace("_provenance", "_response"),
    ])
predictions = sorted((RUN / "recovery/segments").glob("*/predicted_actions.npy"))
for path in (predictions[0], predictions[-1]):
    for filename in ("request.json", "result.json", "predicted_actions.npy", "predicted_future.npy"):
        relatives.append(str(path.with_name(filename).relative_to(RUN)))
sas = (Path.home() / ".blob-sync/temp_sas_url.txt").read_text().strip()
checks, location = [], "container"
try:
    with ContainerClient.from_container_url(
        sas, logging_enable=False, retry_total=3, connection_timeout=15, read_timeout=120,
    ) as container:
        for relative in dict.fromkeys(relatives):
            location = relative
            path = RUN / relative
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            downloaded = hashlib.sha256()
            length = 0
            blob = f"jinghe/steeringwam/recovery/{RUN.name}/{relative}"
            for block in container.download_blob(blob).chunks():
                downloaded.update(block)
                length += len(block)
            if length != path.stat().st_size or expected != downloaded.hexdigest():
                raise ValueError(f"Archive mismatch: {relative}")
            checks.append({
                "path": relative, "bytes": length,
                "local_sha256": expected, "downloaded_sha256": downloaded.hexdigest(),
                "match": True,
            })
            print(json.dumps({"verified_file": relative, "verified_count": len(checks)}), flush=True)
except AzureError as error:
    raise RuntimeError(
        f"Azure readback failed for {location}: {type(error).__name__}, HTTP {getattr(error, 'status_code', None)}"
    ) from None
report = {
    "independent_readback_verified": True, "verified_files": len(checks),
    "scope": "Selected final critical artifacts, not a download of every per-step camera frame.",
    "checks": checks,
}
(RUN / "azure_integrity_check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"verified_files": len(checks), "all_sha256_match": True}))
