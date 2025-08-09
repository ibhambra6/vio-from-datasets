from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol
import numpy as np


@dataclass
class TrajectorySample:
    """A single timestamped pose sample from a dataset.

    Attributes:
        timestamp_ns: Nanoseconds timestamp.
        T_world_camera: 4x4 SE(3) homogeneous matrix.
    """

    timestamp_ns: int
    T_world_camera: np.ndarray


class DatasetSequence(Protocol):
    """Protocol for dataset sequences with pose ground truth.

    Real implementations should also expose camera intrinsics and image access.
    """

    root: Path
    sequence_name: str

    def list_pose_samples(self) -> List[TrajectorySample]: ...

    def camera_intrinsics(self) -> dict: ...
