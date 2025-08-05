"""
This module handles data input/output operations, including the generation
of synthetic datasets for testing VIO/VO algorithms.
"""

import numpy as np
from typing import List
from .geometry import se3_to_T

def generate_cube_scene(n_points: int = 100, cube_size: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    Generates a 3D point cloud by sampling points on the faces of a cube.

    The cube is centered at the origin. This provides a simple, structured
    scene for testing 3D algorithms.

    Args:
        n_points (int, optional): The total number of points to generate. Defaults to 100.
        cube_size (float, optional): The half-side length of the cube. Defaults to 1.0.
        seed (int, optional): A seed for the random number generator for reproducibility.
                              Defaults to 0.

    Returns:
        np.ndarray: An (n_points, 3) numpy array of 3D points.
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
            1: [-cube_size, u, v], # -X face
            2: [u, cube_size, v],  # +Y face
            3: [u, -cube_size, v], # -Y face
            4: [u, v, cube_size],  # +Z face
            5: [u, v, -cube_size]  # -Z face
        }[face_index]
        pts.append(point)
        
    return np.array(pts, dtype=np.float64)

def generate_camera_poses(n_poses: int = 10, radius: float = 3.0, height: float = 0.3) -> List[np.ndarray]:
    """
    Generates a sequence of camera poses in a circular path.

    The camera moves in a circle in the XY plane at a specified height,
    always looking towards the origin. This simulates a common object-centric
    camera motion. The poses are defined as T_world_camera, transforming points
    from the camera frame to the world frame.

    Args:
        n_poses (int, optional): The number of camera poses to generate. Defaults to 10.
        radius (float, optional): The radius of the circular path. Defaults to 3.0.
        height (float, optional): The height of the camera path along the Z-axis.
                                 Defaults to 0.3.

    Returns:
        List[np.ndarray]: A list of 4x4 SE(3) transformation matrices (T_world_camera).
    """
    poses_T_world_camera = []
    for k in range(n_poses):
        # Calculate angle for the current pose on the circle
        theta = 2 * np.pi * k / n_poses
        
        # Camera position (t_world_camera)
        cam_pos = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            height
        ])
        
        # Determine camera orientation (R_world_camera) by creating a look-at matrix.
        # The camera's z-axis should point from the camera towards the origin.
        z_axis = -cam_pos / np.linalg.norm(cam_pos)
        # The world 'up' vector
        world_up = np.array([0, 0, 1.0])
        # The camera's x-axis is perpendicular to the world 'up' and its z-axis.
        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        # The camera's y-axis is perpendicular to its z and x axes.
        y_axis = np.cross(z_axis, x_axis)
        
        # Rotation matrix columns are the camera's axes in world coordinates.
        R_world_camera = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        # Create the SE(3) transformation matrix
        T_world_camera = se3_to_T(R_world_camera, cam_pos)
        poses_T_world_camera.append(T_world_camera)
        
    return poses_T_world_camera
