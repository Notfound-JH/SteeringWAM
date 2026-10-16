"""Read-only privileged geometry measurement; never commands the simulator."""

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--context", type=Path, default=root / "judge_02_input_response.json")
parser.add_argument("--output", type=Path, default=root / "handle_geometry_step0112.json")
args = parser.parse_args()
context = json.loads(args.context.read_text())
stage = Usd.Stage.Open(str(root.parents[1] / "RoboLab/assets/scenes/tools_picking.usda"))
cache = UsdGeom.XformCache()
measurements = {}
with np.load(context["state"]["latest_observation"]["scene"]["path"]) as snapshot:
    for name in ("blue_hammer", "husky_hammer", "red_hammer", "wood_hammer"):
        prim = stage.GetPrimAtPath(f"/world/{name}")
        authored = cache.GetLocalToWorldTransform(prim)
        row_rotation = np.asarray(authored)[:3, :3]
        row_rotation = row_rotation / np.linalg.norm(row_rotation, axis=1, keepdims=True)
        translation = np.asarray(authored.ExtractTranslation())
        pose = snapshot[f"rigid_object/{name}/root_pose"][0]
        current_rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        meshes = []
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get())
                world = np.c_[points, np.ones(len(points))] @ np.asarray(cache.GetLocalToWorldTransform(child))
                local = (world[:, :3] - translation) @ row_rotation.T
                meshes.append(local @ current_rotation.T + pose[:3])
        vertices = np.concatenate(meshes)
        handle = vertices[(vertices[:, 0] > 0.47) & (vertices[:, 0] < 0.51)]
        if not len(handle):
            raise ValueError(f"No geometry in exposed handle slice for {name}")
        measurements[name] = {
            "root_pose": pose.tolist(), "slice_x_m": [0.47, 0.51],
            "handle_median_xyz": np.median(handle, axis=0).tolist(),
            "handle_min_xyz": handle.min(axis=0).tolist(),
            "handle_max_xyz": handle.max(axis=0).tolist(),
        }
result = {
    "episode_step": context["state"]["step"],
    "method": "Authored USD vertices transformed by measured current rigid-body poses; root is not the grasp point.",
    "privileged_state_assisted": True,
    "objects": measurements,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
