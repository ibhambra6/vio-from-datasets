from __future__ import annotations

"""EuRoC dataset scaffold.

This is a minimal, non-downloading placeholder that validates the expected
folder structure and exposes basic metadata and GT pose listing if present.
"""

from pathlib import Path
from typing import List
import numpy as np

from .base import TrajectorySample


class EuRoCSequence:
    def __init__(self, root: str | Path, sequence_name: str):
        self.root = Path(root)
        self.sequence_name = sequence_name
        self._seq_root = self.root / sequence_name
        if not self._seq_root.exists():
            raise FileNotFoundError(f"Sequence path not found: {self._seq_root}")

    def camera_intrinsics(self) -> dict:
        # Placeholder: usually from calibration files
        return {
            "fx": 458.654,
            "fy": 457.296,
            "cx": 367.215,
            "cy": 248.375,
            "dist": np.zeros(5),
        }

    def list_pose_samples(self) -> List[TrajectorySample]:
        # Placeholder: would parse groundtruth.csv if present
        gt_file = self._seq_root / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        samples: List[TrajectorySample] = []
        if gt_file.exists():
            # Very small, lenient CSV parse (timestamp, px, py, pz, qx, qy, qz, qw)
            with gt_file.open("r") as f:
                for line in f:
                    if not line or line.startswith("#") or line.startswith("time"):
                        continue
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue
                    ts = int(parts[0])
                    px, py, pz = (float(parts[1]), float(parts[2]), float(parts[3]))
                    qx, qy, qz, qw = (
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    )
                    # Convert quaternion to rotation matrix
                    R = _quat_to_R(qx, qy, qz, qw)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = np.array([px, py, pz])
                    samples.append(TrajectorySample(timestamp_ns=ts, T_world_camera=T))
        return samples


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # Normalized quaternion to rotation matrix
    import numpy as np

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R
