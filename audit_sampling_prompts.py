"""Read-only prompt/seed audit; no inference, state mutation or visual verdicts."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def audit(session):
    log_entries = defaultdict(list)
    for log in sorted(session.rglob("services/model_*/worker.log")):
        gpu = int(log.parent.name.removeprefix("model_"))
        for number, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
            match = re.search(r"\[(\d\d-\d\d \d\d:\d\d:\d\d).*\[robolab-policy-server\] prompt=.* seed=(\d+)\s*$", line)
            if match:
                log_entries[gpu].append((match[1], int(match[2]), str(log.relative_to(session)), number))
    groups = defaultdict(list)
    for path in sorted(session.rglob("candidates/*/result.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            continue
        request = json.loads((path.parent / "request.json").read_text())
        candidate = request["candidate"]
        gpu = candidate["slot"]["gpu"]
        start, end = result["started_at"], result["finished_at"]
        year = datetime.fromtimestamp(start, timezone.utc).year
        matches = []
        for stamp, seed, log, line in log_entries[gpu]:
            timestamp = datetime.strptime(f"{year}-{stamp}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            if start - 1 <= timestamp <= end + 1:
                matches.append({"seed": seed, "log": log, "line": line})
        scope = path.parent.parent.parent.relative_to(session).as_posix()
        group = (scope, candidate["chunk"], candidate["attempt"] // 4)
        groups[group].append({"candidate_id": candidate["id"], "attempt": candidate["attempt"], "gpu": gpu,
            "instruction": candidate["instruction"], "sent_prompt": result["prompt"],
            "prompt_sha256": hashlib.sha256(result["prompt"].encode()).hexdigest(),
            "configured_slot_seed": candidate["slot"]["seed"],
            "actual_seed": matches[0]["seed"] if len(matches) == 1 else None,
            "seed_evidence": matches, "result": str(path.relative_to(session))})
    batches = []
    for (scope, chunk, batch), entries in sorted(groups.items()):
        entries.sort(key=lambda item: item["attempt"])
        batches.append({"scope": scope, "chunk": chunk, "batch": batch + 1,
                        "candidate_count": len(entries),
                        "unique_instruction_texts": len({item["instruction"]["instruction"] for item in entries}),
                        "unique_sent_prompts": len({item["sent_prompt"] for item in entries}),
                        "all_seeds_log_verified": all(item["actual_seed"] is not None for item in entries),
                        "candidates": entries})
    rounds = defaultdict(list)
    for batch in batches:
        rounds[batch["scope"], batch["chunk"]].extend(batch["candidates"])
    return {"session": str(session), "method": "Exact recorded request/prompt comparison; actual seeds matched by GPU and request time interval to service log lines. Ambiguous or missing matches are null, never inferred.",
            "seed_semantics": "Slot seed initializes server RNG. deterministic_seed=False generates a new per-request integer. Scene seed is separate. Restarted retry services may use an offset seed; actual_seed comes only from logs.",
            "rounds": [{"scope": scope, "chunk": chunk, "candidate_count": len(entries),
                        "unique_instruction_texts": len({item["instruction"]["instruction"] for item in entries}),
                        "unique_sent_prompts": len({item["sent_prompt"] for item in entries}),
                        "verified_seed_count": sum(item["actual_seed"] is not None for item in entries)}
                       for (scope, chunk), entries in sorted(rounds.items())], "batches": batches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.session.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"rounds": report["rounds"], "report": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()