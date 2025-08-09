from __future__ import annotations

"""Simple benchmarking script for the synthetic PnP pipeline.

Measures runtime and accuracy for a few configurations.
"""

import time
import statistics as stats
from typing import List

import numpy as np

from vio.camera import PinholeCamera
from vio.io import generate_cube_scene, generate_camera_poses
from vio.pnp_demo import run_pnp_sequence
from vio.evaluate import trajectory_ATE, trajectory_RPE


def run_once(n_points: int, noise_px: float) -> tuple[float, float, float]:
    cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240, dist=np.zeros(5))
    pts3d = generate_cube_scene(n_points=n_points, cube_size=0.5, seed=1)
    gt_Ts = generate_camera_poses(n_poses=10, radius=3.0, height=0.2)
    t0 = time.time()
    est_Ts = run_pnp_sequence(pts3d, cam, gt_Ts, noise_px=noise_px, seed=42)
    dt_ms = (time.time() - t0) * 1000.0
    ate = trajectory_ATE(gt_Ts, est_Ts)
    rpe = trajectory_RPE(gt_Ts, est_Ts)
    return dt_ms, ate, rpe


def main():
    configs = [
        (100, 0.5),
        (200, 0.8),
        (500, 1.0),
    ]
    for n_points, noise in configs:
        times: List[float] = []
        ates: List[float] = []
        rpes: List[float] = []
        for _ in range(5):
            dt, ate, rpe = run_once(n_points, noise)
            times.append(dt)
            ates.append(ate)
            rpes.append(rpe)
        print(
            "n_points={:4d} noise={:.2f} | time(ms) mean={:.1f} sd={:.1f} | "
            "ATE mean={:.4f} | RPE mean={:.4f}".format(
                n_points,
                noise,
                stats.mean(times),
                stats.stdev(times),
                stats.mean(ates),
                stats.mean(rpes),
            )
        )


if __name__ == "__main__":
    main()
