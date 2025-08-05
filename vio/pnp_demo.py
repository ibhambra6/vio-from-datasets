"""
This module contains the core logic for running a Perspective-n-Point (PnP)
based pose estimation pipeline on a sequence of camera views.
"""

import numpy as np
import cv2
from typing import List
from .camera import PinholeCamera
from .geometry import rvec_tvec_to_T, T_inv

def project_points(pts3d: np.ndarray, T_camera_world: np.ndarray, cam: PinholeCamera) -> np.ndarray:
    """
    Projects 3D world points into 2D image coordinates.

    Args:
        pts3d (np.ndarray): An (N, 3) array of 3D points in the world frame.
        T_camera_world (np.ndarray): A 4x4 SE(3) matrix that transforms points
                                     from the world frame to the camera frame.
        cam (PinholeCamera): The camera model containing intrinsic parameters.

    Returns:
        np.ndarray: An (N, 2) array of 2D image coordinates.
    """
    # Get rotation and translation from the transformation matrix
    R_camera_world = T_camera_world[:3, :3]
    t_camera_world = T_camera_world[:3, 3]

    # Project the points using OpenCV's projectPoints function.
    # It handles the full projection logic including distortion.
    uv, _ = cv2.projectPoints(pts3d, R_camera_world, t_camera_world, cam.K, cam.dist)
    
    # The output of projectPoints is (N, 1, 2), so we reshape it to (N, 2)
    return uv.reshape(-1, 2)

def run_pnp_sequence(
    pts3d: np.ndarray, 
    cam: PinholeCamera, 
    gt_poses_T_world_camera: List[np.ndarray], 
    noise_px: float = 0.5, 
    seed: int = 0
) -> List[np.ndarray]:
    """
    Runs a PnP-based pose estimation sequence for a synthetic dataset.

    For each ground truth pose, this function simulates the image capture process by
    projecting 3D points into the camera's view, adding noise, and then using
    PnP with RANSAC to estimate the camera's pose.

    Args:
        pts3d (np.ndarray): The (N, 3) array of 3D world points.
        cam (PinholeCamera): The camera model.
        gt_poses_T_world_camera (List[np.ndarray]): A list of ground truth 4x4 camera poses
                                                    (T_world_camera), which transform points
                                                    from camera to world coordinates.
        noise_px (float, optional): The standard deviation of Gaussian noise to add
                                    to the projected 2D points (in pixels). Defaults to 0.5.
        seed (int, optional): A seed for the random number generator. Defaults to 0.

    Returns:
        List[np.ndarray]: A list of the estimated 4x4 camera poses (T_world_camera).
    """
    rng = np.random.default_rng(seed)
    estimated_poses_T_world_camera = []

    for T_world_camera_gt in gt_poses_T_world_camera:
        # We need the transform from world to camera for projection
        T_camera_world_gt = T_inv(T_world_camera_gt)
        
        # Project 3D points to 2D image plane
        uv = project_points(pts3d, T_camera_world_gt, cam)
        
        # Add synthetic noise to the projections
        uv_noisy = uv + rng.normal(0, noise_px, size=uv.shape)
        
        # Use PnP with RANSAC to estimate the pose from the noisy 2D-3D correspondences.
        # This estimates the camera's pose relative to the world frame, giving us
        # a rotation (rvec) and translation (tvec) that transform world points
        # into the camera frame.
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, uv_noisy, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            raise RuntimeError("PnP failed to find a solution.")
            
        # The result (rvec, tvec) gives T_camera_world. We convert it to T_world_camera.
        T_camera_world_est = rvec_tvec_to_T(rvec, tvec)
        T_world_camera_est = T_inv(T_camera_world_est)
        
        estimated_poses_T_world_camera.append(T_world_camera_est)
        
    return estimated_poses_T_world_camera
