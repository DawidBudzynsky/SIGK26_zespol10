"""3D stickman visualisation — adapted from the snippet in the project PDF."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .skeleton import JOINT_CONNECTIONS


def animate_skeleton_3d(
    tensor_data: np.ndarray,
    output_filename: Optional[str] = None,
    fps: int = 24,
    show: bool = False,
    title: str = "3D Stickman Motion",
    follow_root: bool = False,
) -> animation.FuncAnimation:
    """tensor_data: [T, 15, 3]."""
    T = tensor_data.shape[0]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    span = max(1.5, float(np.percentile(np.abs(tensor_data), 99)))
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    points = ax.scatter([], [], [], c="red", s=40, zorder=3)
    lines = [ax.plot([], [], [], c="blue", lw=2, zorder=2)[0] for _ in JOINT_CONNECTIONS]

    def init():
        points._offsets3d = ([], [], [])
        for ln in lines:
            ln.set_data(np.array([]), np.array([]))
            ln.set_3d_properties(np.array([]))
        return [points] + lines

    def update(frame_idx: int):
        frame = tensor_data[frame_idx]
        xs, ys, zs = frame[:, 0], frame[:, 1], frame[:, 2]
        points._offsets3d = (xs, ys, zs)
        for ln, (a, b) in zip(lines, JOINT_CONNECTIONS):
            ln.set_data(np.array([frame[int(a), 0], frame[int(b), 0]]),
                        np.array([frame[int(a), 1], frame[int(b), 1]]))
            ln.set_3d_properties(np.array([frame[int(a), 2], frame[int(b), 2]]))
        if follow_root:
            cx, cy, cz = frame[2]
            ax.set_xlim(cx - span, cx + span)
            ax.set_ylim(cy - span, cy + span)
            ax.set_zlim(cz - span, cz + span)
        else:
            ax.set_xlim(-span, span)
            ax.set_ylim(-span, span)
            ax.set_zlim(-span, span)
        return [points] + lines

    anim = animation.FuncAnimation(
        fig, update, frames=T, init_func=init, blit=False, interval=1000 / fps
    )
    if output_filename:
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)) or ".", exist_ok=True)
        anim.save(output_filename, writer="pillow", fps=fps)
    if show:
        plt.show()
    plt.close(fig)
    return anim


def grid_static(
    motions: np.ndarray,
    output_filename: str,
    frame_idx: int = -1,
    cols: int = 4,
    title: Optional[str] = None,
):
    """Static grid of one frame per sample, useful for paper-style figures."""
    N = motions.shape[0]
    rows = (N + cols - 1) // cols
    fig = plt.figure(figsize=(3 * cols, 3 * rows))
    if title:
        fig.suptitle(title)
    for i in range(N):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        frame = motions[i, frame_idx]
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c="red", s=20)
        for a, b in JOINT_CONNECTIONS:
            ax.plot([frame[int(a), 0], frame[int(b), 0]],
                    [frame[int(a), 1], frame[int(b), 1]],
                    [frame[int(a), 2], frame[int(b), 2]], c="blue", lw=1.5)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)) or ".", exist_ok=True)
    plt.savefig(output_filename, dpi=120)
    plt.close(fig)
