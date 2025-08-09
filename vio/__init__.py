"""Core VIO/VO utilities for synthetic PnP-based pose estimation.

Exposes commonly used classes and functions and defines the package version.
"""

__all__ = [
    "PinholeCamera",
    "trajectory_ATE",
    "trajectory_RPE",
    "plot_traj",
    "generate_cube_scene",
    "generate_camera_poses",
    "run_pnp_sequence",
]

__version__ = "0.2.0"

from .camera import PinholeCamera  # noqa: E402
from .evaluate import trajectory_ATE, trajectory_RPE, plot_traj  # noqa: E402
from .io import generate_cube_scene, generate_camera_poses  # noqa: E402
from .pnp_demo import run_pnp_sequence  # noqa: E402
