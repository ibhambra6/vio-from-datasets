"""
This module handles data input/output operations, including the generation
of synthetic datasets for testing VIO/VO algorithms.
"""

import numpy as np
from typing import List
from .geometry import se3_to_T


def generate_cube_scene(n_points: int = 100, cube_size: float = 1.0, seed: int = 0) -> np.ndarray:
    """Generate a 3D point cloud by sampling points on the faces of a cube.

    The cube is centered at the origin. This provides a simple, structured
    scene for testing 3D algorithms.

    Args:
        n_points: The total number of points to generate.
        cube_size: The half-side length of the cube.
        seed: Seed for the RNG for reproducibility.

    Returns:
        An (n_points, 3) numpy array of 3D points.
    """
    rng = np.random.default_rng(seed)
    pts = []

    # Define the six faces of the cube.
    # For each point, we randomly select a face and then a random (u,v) on that face.
    for _ in range(n_points):
        face_index = rng.integers(0, 6)
        # Generate a point on a [-cube_size, cube_size] square
        u, v = (rng.random(2) - 0.5) * 2 * cube_size

        point = {
            0: [cube_size, u, v],  # +X face
            1: [-cube_size, u, v],  # -X face
            2: [u, cube_size, v],  # +Y face
            3: [u, -cube_size, v],  # -Y face
            4: [u, v, cube_size],  # +Z face
            5: [u, v, -cube_size],  # -Z face
        }[face_index]
        pts.append(point)

    return np.array(pts, dtype=np.float64)


def generate_camera_poses(
    n_poses: int = 10, radius: float = 3.0, height: float = 0.3
) -> List[np.ndarray]:
    """Generate a sequence of camera poses on a circular path.

    The camera moves around the origin in the XY plane at a constant height,
    always looking towards the origin. Poses are `T_world_camera` (camera to world).

    Args:
        n_poses: Number of camera poses to generate.
        radius: Radius of the circular path.
        height: Height of the camera path along the Z-axis.

    Returns:
        List of 4x4 SE(3) transformation matrices (T_world_camera).
    """
    poses_T_world_camera = []
    eps = 1e-9
    for k in range(n_poses):
        # Calculate angle for the current pose on the circle
        theta = 2 * np.pi * k / n_poses

        # Camera position (t_world_camera)
        cam_pos = np.array([radius * np.cos(theta), radius * np.sin(theta), height])

        # Determine camera orientation (R_world_camera) by creating a look-at matrix.
        # The camera's z-axis should point from the camera towards the origin.
        cam_pos_norm = max(np.linalg.norm(cam_pos), eps)
        z_axis = -cam_pos / cam_pos_norm
        # The world 'up' vector
        world_up = np.array([0, 0, 1.0])
        # The camera's x-axis is perpendicular to the world 'up' and its z-axis.
        x_axis = np.cross(world_up, z_axis)
        x_norm = max(np.linalg.norm(x_axis), eps)
        x_axis = x_axis / x_norm
        # The camera's y-axis is perpendicular to its z and x axes.
        y_axis = np.cross(z_axis, x_axis)

        # Rotation matrix columns are the camera's axes in world coordinates.
        R_world_camera = np.stack([x_axis, y_axis, z_axis], axis=1)

        # Create the SE(3) transformation matrix
        T_world_camera = se3_to_T(R_world_camera, cam_pos)
        poses_T_world_camera.append(T_world_camera)

    return poses_T_world_camera
