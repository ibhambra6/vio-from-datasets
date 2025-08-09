import numpy as np

from vio.geometry import se3_to_T, T_inv
from vio.camera import PinholeCamera
from vio.io import generate_cube_scene, generate_camera_poses
from vio.pnp_demo import run_pnp_sequence
from vio.evaluate import trajectory_ATE, trajectory_RPE


def test_se3_inverse_roundtrip():
    R = np.eye(3)
    t = np.array([1.0, -2.0, 3.0])
    T = se3_to_T(R, t)
    T_i = T_inv(T)
    identity = T @ T_i
    assert np.allclose(identity, np.eye(4), atol=1e-9)


def test_pnp_pipeline_runs():
    cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240, dist=np.zeros(5))
    pts3d = generate_cube_scene(n_points=100, cube_size=0.5, seed=2)
    gt_Ts = generate_camera_poses(n_poses=5, radius=2.0, height=0.1)
    est_Ts = run_pnp_sequence(pts3d, cam, gt_Ts, noise_px=0.5, seed=3)
    assert len(est_Ts) == len(gt_Ts)
    ate = trajectory_ATE(gt_Ts, est_Ts)
    rpe = trajectory_RPE(gt_Ts, est_Ts, delta=1)
    assert np.isfinite(ate)
    assert np.isfinite(rpe)
