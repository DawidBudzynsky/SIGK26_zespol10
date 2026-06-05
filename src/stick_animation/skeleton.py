"""15-joint stickman skeleton as defined in SIGK Projekt 5 PDF."""
from enum import IntEnum
from typing import List, Tuple

import numpy as np


class Joint(IntEnum):
    HEAD = 0
    NECK = 1
    PELVIS = 2
    RIGHT_SHOULDER = 3
    RIGHT_ELBOW = 4
    RIGHT_WRIST = 5
    LEFT_SHOULDER = 6
    LEFT_ELBOW = 7
    LEFT_WRIST = 8
    RIGHT_HIP = 9
    RIGHT_KNEE = 10
    RIGHT_ANKLE = 11
    LEFT_HIP = 12
    LEFT_KNEE = 13
    LEFT_ANKLE = 14


N_JOINTS = 15
ROOT = Joint.PELVIS

# Parent index for each joint. Root (PELVIS) is its own parent (-1 sentinel by convention).
JOINT_PARENTS: List[int] = [-1] * N_JOINTS
JOINT_PARENTS[Joint.PELVIS] = -1
JOINT_PARENTS[Joint.NECK] = Joint.PELVIS
JOINT_PARENTS[Joint.HEAD] = Joint.NECK
JOINT_PARENTS[Joint.RIGHT_SHOULDER] = Joint.NECK
JOINT_PARENTS[Joint.RIGHT_ELBOW] = Joint.RIGHT_SHOULDER
JOINT_PARENTS[Joint.RIGHT_WRIST] = Joint.RIGHT_ELBOW
JOINT_PARENTS[Joint.LEFT_SHOULDER] = Joint.NECK
JOINT_PARENTS[Joint.LEFT_ELBOW] = Joint.LEFT_SHOULDER
JOINT_PARENTS[Joint.LEFT_WRIST] = Joint.LEFT_ELBOW
JOINT_PARENTS[Joint.RIGHT_HIP] = Joint.PELVIS
JOINT_PARENTS[Joint.RIGHT_KNEE] = Joint.RIGHT_HIP
JOINT_PARENTS[Joint.RIGHT_ANKLE] = Joint.RIGHT_KNEE
JOINT_PARENTS[Joint.LEFT_HIP] = Joint.PELVIS
JOINT_PARENTS[Joint.LEFT_KNEE] = Joint.LEFT_HIP
JOINT_PARENTS[Joint.LEFT_ANKLE] = Joint.LEFT_KNEE

# Edges (child, parent) — also used for visual stick connections.
BONES: List[Tuple[int, int]] = [
    (j, JOINT_PARENTS[j]) for j in range(N_JOINTS) if JOINT_PARENTS[j] >= 0
]

# Convenience: same edges in PDF visualization style (parent, child).
JOINT_CONNECTIONS: List[Tuple[Joint, Joint]] = [
    (Joint.PELVIS, Joint.NECK),
    (Joint.NECK, Joint.HEAD),
    (Joint.NECK, Joint.RIGHT_SHOULDER),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.NECK, Joint.LEFT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.PELVIS, Joint.RIGHT_HIP),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    (Joint.PELVIS, Joint.LEFT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
]

# Mirror pairs: (right_joint, left_joint). Used for left-right augmentation.
MIRROR_PAIRS: List[Tuple[int, int]] = [
    (Joint.RIGHT_SHOULDER, Joint.LEFT_SHOULDER),
    (Joint.RIGHT_ELBOW, Joint.LEFT_ELBOW),
    (Joint.RIGHT_WRIST, Joint.LEFT_WRIST),
    (Joint.RIGHT_HIP, Joint.LEFT_HIP),
    (Joint.RIGHT_KNEE, Joint.LEFT_KNEE),
    (Joint.RIGHT_ANKLE, Joint.LEFT_ANKLE),
]

# Ankle joints (used for foot-contact / foot-skating loss).
ANKLES = (int(Joint.RIGHT_ANKLE), int(Joint.LEFT_ANKLE))


def adjacency_matrix(symmetric: bool = True, self_loop: bool = True) -> np.ndarray:
    """[N_JOINTS, N_JOINTS] adjacency for the kinematic tree."""
    a = np.zeros((N_JOINTS, N_JOINTS), dtype=np.float32)
    for c, p in BONES:
        a[c, p] = 1.0
        if symmetric:
            a[p, c] = 1.0
    if self_loop:
        np.fill_diagonal(a, 1.0)
    return a


def graph_distance() -> np.ndarray:
    """Shortest-path hop distance between joints on the kinematic tree."""
    inf = 1_000
    d = np.full((N_JOINTS, N_JOINTS), inf, dtype=np.int32)
    for i in range(N_JOINTS):
        d[i, i] = 0
    for c, p in BONES:
        d[c, p] = 1
        d[p, c] = 1
    # Floyd-Warshall (tiny graph).
    for k in range(N_JOINTS):
        for i in range(N_JOINTS):
            for j in range(N_JOINTS):
                if d[i, k] + d[k, j] < d[i, j]:
                    d[i, j] = d[i, k] + d[k, j]
    return d
