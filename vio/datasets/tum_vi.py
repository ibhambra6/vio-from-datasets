from __future__ import annotations

"""TUM-VI dataset scaffold.

Validates expected folder structure and exposes minimal metadata.
"""

from pathlib import Path
from typing import List
import numpy as np

from .base import TrajectorySample


class TUMVISequence:
    def __init__(self, root: str | Path, sequence_name: str):
        self.root = Path(root)
        self.sequence_name = sequence_name
        self._seq_root = self.root / sequence_name
        if not self._seq_root.exists():
            raise FileNotFoundError(f"Sequence path not found: {self._seq_root}")

    def camera_intrinsics(self) -> dict:
        # Placeholder intrinsics typical for fisheye (set to pinhole defaults here)
        return {
            "fx": 190.0,
            "fy": 190.0,
            "cx": 224.0,
            "cy": 224.0,
            "dist": np.zeros(5),
        }

    def list_pose_samples(self) -> List[TrajectorySample]:
        # Placeholder: TUM-VI provides imu0/data.csv and groundtruth.csv in some conversions
        samples: List[TrajectorySample] = []
        gt_file = self._seq_root / "groundtruth.txt"
        if gt_file.exists():
            with gt_file.open("r") as f:
                for line in f:
                    if not line or line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue
                    ts = int(float(parts[0]) * 1e9)
                    px, py, pz = (float(parts[1]), float(parts[2]), float(parts[3]))
                    qx, qy, qz, qw = (
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    )
                    R = _quat_to_R(qx, qy, qz, qw)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = np.array([px, py, pz])
                    samples.append(TrajectorySample(timestamp_ns=ts, T_world_camera=T))
        return samples


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
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
