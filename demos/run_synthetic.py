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

import argparse
import time
import numpy as np
from vio.logging_utils import get_logger
from vio.camera import PinholeCamera
from vio.io import generate_cube_scene, generate_camera_poses
from vio.pnp_demo import run_pnp_sequence
from vio.evaluate import plot_traj, trajectory_ATE, trajectory_RPE


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic PnP VIO demo")
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--cube-size", type=float, default=0.5)
    parser.add_argument("--scene-seed", type=int, default=1)
    parser.add_argument("--n-poses", type=int, default=10)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--height", type=float, default=0.2)
    parser.add_argument("--noise-px", type=float, default=0.8)
    parser.add_argument("--pnp-seed", type=int, default=42)
    parser.add_argument("--save-plot", type=str, default="")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    """
    Main function to run the synthetic VIO demo.
    """
    logger = get_logger("demo")
    args = build_arg_parser().parse_args(argv)

    # --- 1. Define Scene and Camera Configuration ---

    # Define the virtual pinhole camera model
    cam = PinholeCamera(
        fx=500,
        fy=500,  # Focal lengths
        cx=320,
        cy=240,  # Principal point
        dist=np.zeros(5),  # No lens distortion
    )

    # Configuration for the synthetic data generation
    scene_config = {
        "n_points": args.n_points,
        "cube_size": args.cube_size,
        "seed": args.scene_seed,
    }

    trajectory_config = {
        "n_poses": args.n_poses,
        "radius": args.radius,
        "height": args.height,
    }

    pnp_config = {
        "noise_px": args.noise_px,
        "seed": args.pnp_seed,
    }

    # --- 2. Generate Synthetic Data ---
    logger.info("Generating synthetic data...")
    pts3d = generate_cube_scene(**scene_config)
    gt_Ts = generate_camera_poses(**trajectory_config)

    # --- 3. Run PnP Pose Estimation ---
    logger.info("Running PnP sequence...")
    t0 = time.time()
    est_Ts, inliers = run_pnp_sequence(
        pts3d,
        cam,
        gt_Ts,
        noise_px=pnp_config["noise_px"],
        seed=pnp_config["seed"],
        return_inlier_counts=True,
    )
    logger.info(
        f"PnP done in {(time.time() - t0)*1000:.1f} ms; avg inliers: {np.mean(inliers):.1f}"
    )

    # --- 4. Evaluate Trajectory ---
    logger.info("Evaluating trajectory...")
    ate = trajectory_ATE(gt_Ts, est_Ts)
    rpe = trajectory_RPE(gt_Ts, est_Ts, delta=1)

    logger.info("Evaluation Results:")
    logger.info(f"  Absolute Trajectory Error (ATE, RMS): {ate:.4f} m")
    logger.info(f"  Relative Pose Error (RPE, RMS, delta=1): {rpe:.4f} m")

    # --- 5. Visualize Trajectory ---
    logger.info("Plotting trajectories...")
    save_path = args.save_plot or None
    plot_traj(gt_Ts, est_Ts, show=not args.no_show, save_path=save_path)


if __name__ == "__main__":
    main()
