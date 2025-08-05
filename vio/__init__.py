"""
VIO Package
-----------

A minimal, laptop-runnable library for Visual-Inertial Odometry experiments.
"""

from .camera import PinholeCamera
from .evaluate import trajectory_ATE, trajectory_RPE, plot_traj
from .geometry import (
    se3_to_T,
    T_to_se3,
    T_inv,
    compose,
    rvec_tvec_to_T,
)
from .io import generate_cube_scene, generate_camera_poses
from .pnp_demo import run_pnp_sequence

