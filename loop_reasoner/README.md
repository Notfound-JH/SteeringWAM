# Recurrent Planning for Cosmos3 Edge Policy

This directory contains the recurrent-planning implementation, launch scripts,
and offline experiment reports. The workspace keeps runtime dependencies under
the ignored `packages/` directory, so the Cosmos changes are distributed as
`patches/cosmos3-edge-planning.patch`, not as an embedded Git repository.

## Restore the Code

The patch is based on NVIDIA/cosmos-framework commit
`efba2076c80648663a89a7fdbc92eb262fae39ac`.
From the SteeringWAM repository root, for a fresh checkout:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
git -C packages/cosmos3 checkout --detach efba2076c80648663a89a7fdbc92eb262fae39ac
git -C packages/cosmos3 apply ../../loop_reasoner/patches/cosmos3-edge-planning.patch
```

Do not repeat these commands against the existing modified development checkout.
The patch includes the planner, paired-FM training, sampling hooks, configs,
diagnostic tools, and regression test source. It excludes unrelated RoboLab
server edits, virtual environments, pretrained weights, and downloaded datasets.

## Experiments

- [Implementation and results](cosmos3_edge_policy_implementation.md)
- [Research design](cosmos3_recurrent_latent_planning_idea.md)
- [Paired attribution results](validation/edge_attribution_200.json)

The launch scripts use physical GPUs 0-3 only. Install the upstream training
environment and prepare the model, VAE, and DROID assets before running them;
these assets are not included in Git. The implementation document records the
local asset layout and pinned dataset revision. Report paths describe the
original machine and are not portable asset downloads.

The completed comparison uses B0-CONT and B4 trained independently for 200
updates, and the existing 200-update B5 checkpoint. B0-CONT uses the corrected
EMA warm-start run (`_ema_fix`); the earlier invalid run is excluded. Re-running
the attribution script on the original machine resumes completed runs and
re-evaluates them; it is not an independent training replicate.

Results are synthetic paired-FM evaluations on nine validation episodes, not
simulator success rates or evidence of a feedback advantage. No new tests or
training were run as part of preparing this GitHub upload.