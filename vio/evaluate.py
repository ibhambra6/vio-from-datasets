"""
This module provides tools for evaluating trajectory estimates against ground
truth data, including metrics like ATE and RPE, and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def trajectory_ATE(gt_Ts: List[np.ndarray], est_Ts: List[np.ndarray]) -> float:
    """
    Calculates the Absolute Trajectory Error (ATE) between a ground truth and
    an estimated trajectory.

    ATE is computed as the Root Mean Square Error (RMSE) of the translation
    components between the two trajectories after they have been aligned at the
    first pose.

    Args:
        gt_Ts (List[np.ndarray]): A list of ground truth 4x4 transformation matrices.
        est_Ts (List[np.ndarray]): A list of estimated 4x4 transformation matrices.

    Returns:
        float: The calculated ATE value. Returns np.nan if trajectories are empty.
    """
    if len(gt_Ts) != len(est_Ts):
        raise ValueError("Ground truth and estimated trajectories must have the same length.")
    if not gt_Ts:
        return np.nan

    # Align the first pose of the estimated trajectory to the ground truth.
    # This transform aligns the entire estimated trajectory to the ground truth frame.
    align_transform = gt_Ts[0] @ np.linalg.inv(est_Ts[0])
    
    errors = []
    for gt_T, est_T in zip(gt_Ts, est_Ts):
        est_T_aligned = align_transform @ est_T
        translation_error = gt_T[:3, 3] - est_T_aligned[:3, 3]
        errors.append(np.linalg.norm(translation_error))
    
    return float(np.sqrt(np.mean(np.square(errors))))

def trajectory_RPE(gt_Ts: List[np.ndarray], est_Ts: List[np.ndarray], delta: int = 1) -> float:
    """
    Calculates the Relative Pose Error (RPE) between a ground truth and an
    estimated trajectory.

    RPE measures the drift over a fixed time interval (delta) by comparing
    the relative motion between pairs of poses. This implementation computes
    the translational RPE.

    Args:
        gt_Ts (List[np.ndarray]): A list of ground truth 4x4 transformation matrices.
        est_Ts (List[np.ndarray]): A list of estimated 4x4 transformation matrices.
        delta (int, optional): The step size between poses to compute relative
                               motion. Defaults to 1.

    Returns:
        float: The calculated RPE value. Returns np.nan if the trajectory is
               too short for the given delta.
    """
    n = len(gt_Ts)
    if n <= delta:
        return np.nan

    errors = []
    for i in range(n - delta):
        # Ground truth relative pose
        gt_rel_T = np.linalg.inv(gt_Ts[i]) @ gt_Ts[i + delta]
        # Estimated relative pose
        est_rel_T = np.linalg.inv(est_Ts[i]) @ est_Ts[i + delta]
        
        # The error is the difference in the translational part of the relative poses
        translation_error = gt_rel_T[:3, 3] - est_rel_T[:3, 3]
        errors.append(np.linalg.norm(translation_error))
        
    return float(np.sqrt(np.mean(np.square(errors))))

def plot_traj(gt_Ts: List[np.ndarray], est_Ts: List[np.ndarray]) -> None:
    """
    Plots a 3D comparison of a ground truth and an estimated trajectory.

    Args:
        gt_Ts (List[np.ndarray]): List of ground truth 4x4 transformation matrices.
        est_Ts (List[np.ndarray]): List of estimated 4x4 transformation matrices.
    """
    gt_translations = np.array([T[:3, 3] for T in gt_Ts])
    est_translations = np.array([T[:3, 3] for T in est_Ts])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_translations[:, 0], gt_translations[:, 1], gt_translations[:, 2], 'g-', label='Ground Truth')
    ax.plot(est_translations[:, 0], est_translations[:, 1], est_translations[:, 2], 'r-', label='Estimated')
    
    # Plot start and end points
    ax.scatter(gt_translations[0, 0], gt_translations[0, 1], gt_translations[0, 2], c='lime', marker='o', s=100, label='Start GT')
    ax.scatter(est_translations[0, 0], est_translations[0, 1], est_translations[0, 2], c='red', marker='o', s=100, label='Start EST')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True)
    plt.show()
