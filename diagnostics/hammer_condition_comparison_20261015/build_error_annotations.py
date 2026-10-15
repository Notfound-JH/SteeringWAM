"""Create post-hoc annotations for the visually reviewed three-run study."""

import csv
import json
from pathlib import Path


study = Path(__file__).resolve().parent
summaries = json.loads((study / "motion_summary.json").read_text())
reviewed_starts = {"A": 65, "B": 65, "C": 49}
annotations = []
for summary in summaries:
    variant = summary["variant"]
    with (study / f"{variant}_chunk_metrics.csv").open() as handle:
        chunks = list(csv.DictReader(handle))
    full = [c for c in chunks if int(c["actual_steps"]) == 16]
    first = next(
        i for i in range(len(full) - 1)
        if all(
            c["near_static_motion_only"] == "True" and int(c["close_commands"]) == 0
            for c in full[i:i + 2]
        )
    )
    start = int(full[first]["first_step"])
    assert start == reviewed_starts[variant]
    assert summary["actual_steps_analyzed"] == 600
    assert summary["native_success"] is False and summary["close_commands"] == 0
    assert all(c["near_static_motion_only"] == "True" for c in chunks[first:])
    deferred = [c for c in full if c["closure_only_in_unexecuted_half"] == "True"]
    offsets = [int(c["predicted_first_close_offset"]) for c in deferred]
    annotations.append({
        "variant": variant,
        "condition": summary["condition"],
        "primary_error": "approach_or_contact_stall",
        "subtype": "no_transition_to_grasp",
        "independent_error_episodes": 1,
        "first_candidate_window": [start, int(full[first]["last_step"])],
        "confirmation_step": int(full[first + 1]["last_step"]),
        "exact_physical_error_onset_step": None,
        "estimated_error_coverage_start": start,
        "error_coverage_end": 600,
        "estimated_error_steps": 601 - start,
        "estimated_error_step_fraction": (601 - start) / 600,
        "resolved": False,
        "native_success": False,
        "actual_close_commands": 0,
        "full_chunks_with_closure_only_in_unexecuted_half": len(deferred),
        "full_chunks_denominator": len(full),
        "deferred_first_close_offset_range": [min(offsets), max(offsets)] if offsets else None,
        "full_chunks_with_first_close_at_offset_18_or_19": sum(x in (18, 19) for x in offsets),
        "max_hammer_root_displacement_mm": 1000 * summary["hammer_max_displacement_from_initial_m"],
        "max_hammer_root_height_gain_mm": 1000 * summary["hammer_max_root_height_gain_m"],
        "target_selection_assessment": "Approached the hammer cluster; no other object was acquired. Target identity at contact is not proven solely by approach direction.",
        "unreached_stages": ["retained_grasp", "extraction", "transport", "placement"],
        "visual_evidence": [f"{variant}_onset_multiview.jpg", f"{variant}_boundary_contact_sheet.jpg"],
        "measured_evidence": [f"{variant}_motion.csv", f"{variant}_chunk_metrics.csv"],
    })
total_error_steps = sum(a["estimated_error_steps"] for a in annotations)
report = {
    "study_id": "hammer_condition_comparison_20261015",
    "annotation_mode": "offline_parent_assistant_visual_review_and_measured_state",
    "independent_recovery_judge_used": False,
    "annotations_affected_execution": False,
    "threshold_status": "post_hoc_descriptive_not_preregistered_detector",
    "criterion": {
        "consecutive_full_chunks": 2,
        "steps_per_chunk": 16,
        "ee_position_ptp_norm_max_m": 0.02,
        "hammer_root_position_ptp_norm_max_m": 0.01,
        "actual_close_commands": 0,
        "visual_requirement": "Gripper remains open near the hammer cluster without acquisition.",
        "coverage_convention": "Backdate window-level coverage to the first candidate window after confirmation; not an exact physical error onset.",
        "resolution": "16-step windows; onset multi-view images additionally inspected every 8 steps.",
    },
    "total_actual_steps": 1800,
    "total_independent_error_episodes": 3,
    "total_estimated_error_steps": total_error_steps,
    "estimated_error_step_fraction": total_error_steps / 1800,
    "native_successes": 0,
    "episodes": annotations,
    "late_stage_error_rates": None,
    "late_stage_error_rates_reason": "No retained-grasp, extraction, transport, or placement opportunities were observed.",
    "causal_caveat": "Error manifestation time is not evidence of the best causal condition-intervention time. No same-state counterfactuals or execute-32 control were run.",
}
(study / "error_annotations.json").write_text(json.dumps(report, indent=2) + "\n")
columns = [
    "variant", "primary_error", "independent_error_episodes", "confirmation_step",
    "estimated_error_coverage_start", "error_coverage_end", "estimated_error_steps",
    "estimated_error_step_fraction", "actual_close_commands",
    "full_chunks_with_closure_only_in_unexecuted_half", "full_chunks_denominator",
    "max_hammer_root_displacement_mm", "native_success",
]
with (study / "error_distribution.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(annotations)
print(json.dumps({
    "independent_errors": len(annotations), "estimated_error_steps": total_error_steps,
    "total_steps": 1800, "error_step_fraction": total_error_steps / 1800,
    "first_candidate_starts": {a["variant"]: a["estimated_error_coverage_start"] for a in annotations},
}, indent=2))
