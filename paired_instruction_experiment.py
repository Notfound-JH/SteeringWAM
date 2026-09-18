"""Freeze equal-budget, matched-seed instruction arms without launching inference."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arm(path, task, checkpoint, instructions):
    plan = json.loads(Path(path).read_text())
    if plan.get("format") != "paired-instruction-v1" or plan.get("task") != task:
        raise ValueError("Experiment format or task mismatch")
    if checkpoint is None or sha256(checkpoint) != plan["checkpoint"]["sha256"]:
        raise ValueError("Experiment checkpoint mismatch")
    if sha256(instructions) != plan["instructions"]["sha256"]:
        raise ValueError("Experiment instructions mismatch")
    seeds = plan["seeds"]
    if len(seeds) != 12 or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError("Expected twelve integer seeds in [0, 2**31)")
    if len(set(seeds)) != 12:
        raise ValueError("Paired trial seeds must be distinct")
    if plan.get("condition") not in ("A", "B") or plan.get("execute_steps") != 16:
        raise ValueError("Invalid condition or execution budget")
    if not isinstance(plan.get("prompt"), str) or not plan["prompt"].strip():
        raise ValueError("An exact full prompt is required")
    return plan


def prepare(args):
    from sampling_multi_env import checkpoint_history, load_instructions

    history = checkpoint_history(args.checkpoint, args.task)
    if not history:
        raise ValueError("A selected checkpoint is required")
    seeds = json.loads(args.seeds.read_text())
    plans = []
    for condition, instructions, prompt in (
        ("A", args.instructions_a, args.prompt_a), ("B", args.instructions_b, args.prompt_b)
    ):
        values = load_instructions(instructions)
        if any(value != values[0] for value in values):
            raise ValueError("Each arm must repeat one fixed instruction twelve times")
        plan = {"format": "paired-instruction-v1", "task": args.task, "condition": condition,
                "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
                "instructions": {"path": str(instructions.resolve()), "sha256": sha256(instructions)},
                "prompt": prompt.read_text().strip(), "seeds": seeds, "execute_steps": 16,
                "best_of_groups": [list(range(start, start + 4)) for start in (0, 4, 8)]}
        plans.append((plan, values[0]))
    for field in ("subgoal", "completion_condition", "avoid"):
        if plans[0][1][field] != plans[1][1][field]:
            raise ValueError(f"Arms must share {field}")
    if plans[0][0]["prompt"] == plans[1][0]["prompt"]:
        raise ValueError("A and B prompts must differ")
    args.output.mkdir(parents=True, exist_ok=False)
    for plan, _ in plans:
        path = args.output / f"arm_{plan['condition']}.json"
        path.write_text(json.dumps(plan, indent=2) + "\n")
        load_arm(path, args.task, args.checkpoint, Path(plan["instructions"]["path"]))
    print(f"Frozen paired arms: {args.output}. No inference launched.")


def export_blind(args):
    from instruction_sampling_worker import verified, make_sheet

    arms = [json.loads((directory / "experiment_arm.json").read_text()) for directory in args.arms]
    if {arm["condition"] for arm in arms} != {"A", "B"}:
        raise ValueError("Exactly one A and one B arm required")
    for field in ("task", "checkpoint", "seeds", "execute_steps"):
        if arms[0][field] != arms[1][field]:
            raise ValueError(f"Unmatched arms: {field}")
    entries = []
    for directory, arm in zip(args.arms, arms, strict=True):
        result = json.loads((directory / "result.json").read_text())
        if result["status"] != "complete" or len(result["candidates"]) != 12:
            raise ValueError("Both full pools must complete before blind export")
        pair_ids = set()
        for candidate in result["candidates"]:
            receipt = candidate["experiment"]
            pair_id = receipt["pair_id"]
            if type(pair_id) is not int or pair_id not in range(12) or pair_id in pair_ids:
                raise ValueError("Invalid or duplicate pair id")
            pair_ids.add(pair_id)
            if receipt["condition"] != arm["condition"] or receipt["actual_seed"] != arm["seeds"][pair_id]:
                raise ValueError("Candidate seed receipt mismatch")
            media = []
            for frame in candidate["frames"]:
                source = Path(verified(frame)).resolve()
                source.relative_to(directory.resolve())
                media.append((source, f"frames/{frame['step']:03d}/{source.name}"))
            for video in candidate["videos"]:
                source = Path(verified(video)).resolve()
                source.relative_to(directory.resolve())
                media.append((source, source.name))
            if not media:
                raise ValueError("Candidate has no verified media")
            entries.append({"condition": arm["condition"], "pair_id": pair_id,
                            "frames": candidate["frames"],
                            "candidate_id": candidate["id"], "media": media})
    if args.mapping.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("Keep unblinding mapping outside the review directory")
    if args.mapping.exists() or args.output.exists():
        raise ValueError("Blind output and mapping must be new")
    random.SystemRandom().shuffle(entries)
    args.output.mkdir(parents=True)
    mapping = {}
    for index, entry in enumerate(entries):
        label = f"sample_{index:03d}"
        for source, relative in entry.pop("media"):
            destination = args.output / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        frames = entry.pop("frames")
        last_step = max(frame["step"] for frame in frames)
        times = {0, last_step // 4, last_step // 2, 3 * last_step // 4, last_step}
        sheet_frames = [{**frame, "path": str(args.output / label / "frames" / f"{frame['step']:03d}" / Path(frame["path"]).name)}
                        for frame in frames if frame["step"] in times and
                        (frame.get("camera") == "wrist_cam" or frame.get("debug_camera"))]
        if sheet_frames:
            make_sheet(sheet_frames, args.output / label / "review.png", label)
        mapping[label] = entry
    (args.output / "review_protocol.txt").write_text(args.protocol.read_text())
    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    with args.mapping.open("x") as stream:
        json.dump(mapping, stream, indent=2)
    args.mapping.chmod(0o600)
    print(f"Exported {len(entries)} anonymous candidates. Keep mapping hidden until reviews are frozen.")


def summarize(args):
    reviews = json.loads(args.reviews.read_text())
    mapping = json.loads(args.mapping.read_text())
    if set(reviews) != set(mapping) or len(mapping) != 24:
        raise ValueError("Require exactly all 24 anonymous reviews before unblinding")
    pairs = {}
    for label, review in reviews.items():
        score = review.get("score")
        unsafe = review.get("unsafe")
        if score is not None and (type(score) is not int or score not in (-1, 0, 1, 2)):
            raise ValueError("Score must be -1, 0, 1, 2 or null")
        if unsafe is not None and type(unsafe) is not bool:
            raise ValueError("Unsafe must be boolean or null")
        if not isinstance(review.get("evidence"), str) or not review["evidence"].strip():
            raise ValueError("Every review requires visual evidence")
        entry = mapping[label]
        pair = pairs.setdefault(entry["pair_id"], {})
        if entry["condition"] in pair:
            raise ValueError("Duplicate review condition in pair")
        pair[entry["condition"]] = {"label": label, **review}
    if set(pairs) != set(range(12)) or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Require twelve complete A/B pairs")
    counts = {"improved": 0, "tied": 0, "worse": 0, "uncertain": 0}
    for pair in pairs.values():
        if pair["A"]["score"] is None or pair["B"]["score"] is None:
            counts["uncertain"] += 1
        else:
            delta = pair["B"]["score"] - pair["A"]["score"]
            counts["improved" if delta > 0 else "worse" if delta < 0 else "tied"] += 1
    groups = []
    for start in (0, 4, 8):
        group = {"pair_ids": list(range(start, start + 4))}
        for condition in ("A", "B"):
            entries = [pairs[pair_id][condition] for pair_id in group["pair_ids"]]
            group[condition] = None if any(item["score"] is None or item["unsafe"] is None for item in entries) else max(
                entries, key=lambda item: (not item["unsafe"], item["score"]))
        groups.append(group)
    report = {"reviews_sha256": sha256(args.reviews), "mapping_sha256": sha256(args.mapping),
              "paired_counts": counts, "pairs": pairs, "best_of_four": groups,
              "unsafe_counts": {condition: sum(pair[condition]["unsafe"] is True for pair in pairs.values())
                                for condition in ("A", "B")},
              "scope": "One historical checkpoint; descriptive evidence, not full-task success or independent-task replication."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.set_defaults(operation=prepare)
    preparation.add_argument("--task", required=True)
    for name in ("checkpoint", "instructions-a", "instructions-b", "prompt-a", "prompt-b", "seeds", "output"):
        preparation.add_argument("--" + name, type=Path, required=True)
    blind = commands.add_parser("blind")
    blind.set_defaults(operation=export_blind)
    blind.add_argument("--arms", nargs=2, type=Path, required=True)
    for name in ("output", "mapping", "protocol"):
        blind.add_argument("--" + name, type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.set_defaults(operation=summarize)
    for name in ("reviews", "mapping", "output"):
        summary.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.operation(args)


if __name__ == "__main__":
    main()