"""Load BVH files into [T, 15, 3] world-space joint position tensors.

We use ``bvhio`` to traverse the hierarchy and pick the joints listed in the
SIGK Projekt 5 PDF (15-joint stickman). CMU MoCap labels differ slightly by
subject, so we resolve joint names from a fallback list.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .skeleton import Joint, N_JOINTS

# Multiple candidate names — first match wins. Covers both CMU "biovision"
# rigs and the common bvhio normalised hierarchy.
_CANDIDATES: Dict[Joint, List[str]] = {
    Joint.HEAD: ["Head", "head"],
    Joint.NECK: ["Neck1", "Neck", "neck"],
    Joint.PELVIS: ["Hips", "hip", "Hip"],
    Joint.RIGHT_SHOULDER: ["RightArm", "rclavicle", "RightShoulder"],
    Joint.RIGHT_ELBOW: ["RightForeArm", "rhumerus", "RightForearm", "RightElbow"],
    Joint.RIGHT_WRIST: ["RightHand", "rwrist", "RightWrist"],
    Joint.LEFT_SHOULDER: ["LeftArm", "lclavicle", "LeftShoulder"],
    Joint.LEFT_ELBOW: ["LeftForeArm", "lhumerus", "LeftForearm", "LeftElbow"],
    Joint.LEFT_WRIST: ["LeftHand", "lwrist", "LeftWrist"],
    Joint.RIGHT_HIP: ["RightUpLeg", "rfemur", "RightHip"],
    Joint.RIGHT_KNEE: ["RightLeg", "rtibia", "RightKnee"],
    Joint.RIGHT_ANKLE: ["RightFoot", "rfoot", "RightAnkle"],
    Joint.LEFT_HIP: ["LeftUpLeg", "lfemur", "LeftHip"],
    Joint.LEFT_KNEE: ["LeftLeg", "ltibia", "LeftKnee"],
    Joint.LEFT_ANKLE: ["LeftFoot", "lfoot", "LeftAnkle"],
}


def _resolve_names(joint_index: Dict[str, object]) -> Dict[Joint, str]:
    resolved: Dict[Joint, str] = {}
    available = set(joint_index.keys())
    for j, cands in _CANDIDATES.items():
        for name in cands:
            if name in available:
                resolved[j] = name
                break
        if j not in resolved:
            raise KeyError(
                f"Could not resolve joint {j.name}. Tried {cands}. "
                f"Available joints: {sorted(available)[:30]}…"
            )
    return resolved


def load_bvh_as_tensor(bvh_path: str, frame_stride: int = 1) -> np.ndarray:
    """Return [T, 15, 3] world-space positions; PDF axis convention (X, Y_floor, Z_up).

    The CMU BVH stores Y as the up axis. The SIGK PDF visualises with Z as
    the up axis. We swap (X, Y, Z) → (X, Z, Y) to match that convention.
    """
    import bvhio  # local import: optional dependency at runtime

    root = bvhio.readAsHierarchy(bvh_path)
    layout = list(root.layout())
    joint_index = {j.Name: j for j, _, _ in layout}
    resolved = _resolve_names(joint_index)

    T = len(root.Keyframes)
    indices = range(0, T, frame_stride)
    out = np.zeros((len(list(indices)), N_JOINTS, 3), dtype=np.float32)

    for out_idx, frame_idx in enumerate(range(0, T, frame_stride)):
        root.loadPose(frame_idx)
        for j_enum, name in resolved.items():
            pos = joint_index[name].PositionWorld
            out[out_idx, int(j_enum)] = [pos.x, pos.z, pos.y]
    return out


def list_bvh_files(folder: str) -> List[str]:
    """Sorted list of .bvh files in a folder (non-recursive)."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".bvh")
    )


def load_cached(npy_path: str) -> Optional[np.ndarray]:
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None


def save_cached(npy_path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, arr)
