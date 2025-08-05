"""
This script runs a synthetic data demonstration for a PnP-based VIO pipeline.

It performs the following steps:
1.  Defines a virtual camera and scene parameters.
2.  Generates a synthetic 3D point cloud (a cube scene).
3.  Generates a ground truth camera trajectory (circular path).
4.  Runs the PnP sequence to estimate the camera poses from the synthetic data.
5.  Evaluates the estimated trajectory against the ground truth using ATE and RPE.
6.  Visualizes the ground truth and estimated trajectories in a 3D plot.
"""

import numpy as np
from vio.camera import PinholeCamera
from vio.io import generate_cube_scene, generate_camera_poses
from vio.pnp_demo import run_pnp_sequence
from vio.evaluate import plot_traj, trajectory_ATE, trajectory_RPE

def main():
    """
    Main function to run the synthetic VIO demo.
    """
    # --- 1. Define Scene and Camera Configuration ---
    
    # Define the virtual pinhole camera model
    cam = PinholeCamera(
        fx=500, fy=500,  # Focal lengths
        cx=320, cy=240,  # Principal point
        dist=np.zeros(5) # No lens distortion
    )
    
    # Configuration for the synthetic data generation
    scene_config = {
        'n_points': 200,      # Number of 3D points in the scene
        'cube_size': 0.5,     # Half-side length of the cube
        'seed': 1
    }
    
    trajectory_config = {
        'n_poses': 10,        # Number of camera poses in the trajectory
        'radius': 3.0,        # Radius of the circular camera path
        'height': 0.2
    }
    
    pnp_config = {
        'noise_px': 0.8,      # Std dev of noise added to 2D projections
        'seed': 42
    }

    # --- 2. Generate Synthetic Data ---
    print("Generating synthetic data...")
    pts3d = generate_cube_scene(**scene_config)
    gt_Ts = generate_camera_poses(**trajectory_config)
    
    # --- 3. Run PnP Pose Estimation ---
    print("Running PnP sequence...")
    est_Ts = run_pnp_sequence(
        pts3d,
        cam,
        gt_Ts,
        noise_px=pnp_config['noise_px'],
        seed=pnp_config['seed']
    )
    
    # --- 4. Evaluate Trajectory ---
    print("Evaluating trajectory...")
    ate = trajectory_ATE(gt_Ts, est_Ts)
    rpe = trajectory_RPE(gt_Ts, est_Ts, delta=1)
    
    print(f"\nEvaluation Results:")
    print(f"  Absolute Trajectory Error (ATE, RMS): {ate:.4f} m")
    print(f"  Relative Pose Error (RPE, RMS, delta=1): {rpe:.4f} m")
    
    # --- 5. Visualize Trajectory ---
    print("Plotting trajectories...")
    plot_traj(gt_Ts, est_Ts)

if __name__ == "__main__":
    main()
