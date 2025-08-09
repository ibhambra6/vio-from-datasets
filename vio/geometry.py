"""Geometric transformations for 3D vision (SE(3) utilities).

Includes helpers to build, invert, compose, and convert between rotation/translation
representations frequently used in VO/VIO pipelines.
"""

import numpy as np
import cv2
from typing import Tuple


def se3_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Constructs a 4x4 SE(3) transformation matrix from a rotation matrix and
    a translation vector.

    Args:
        R (np.ndarray): A 3x3 rotation matrix.
        t (np.ndarray): A 3x1 translation vector.

    Returns:
        np.ndarray: The corresponding 4x4 homogeneous transformation matrix.
    """
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if t.shape != (3,) and t.shape != (3, 1):
        raise ValueError("Translation vector must be 3x1.")

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def T_to_se3(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a 4x4 SE(3) transformation matrix into a rotation matrix and
    a translation vector.

    Args:
        T (np.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the 3x3 rotation
                                       matrix and the 3x1 translation vector.
    """
    if T.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4.")

    return T[:3, :3], T[:3, 3]


def T_inv(T: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a 4x4 SE(3) transformation matrix.

    This is more efficient than a generic matrix inversion.

    Args:
        T (np.ndarray): The 4x4 homogeneous transformation matrix to invert.

    Returns:
        np.ndarray: The inverted 4x4 transformation matrix.
    """
    R, t = T_to_se3(T)
    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def compose(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    """
    Composes two SE(3) transformations.

    This is equivalent to matrix multiplication, but with explicit naming
    to clarify the transformation order. For example, composing T_a_b (transform from b to a)
    and T_b_c (transform from c to b) yields T_a_c.

    Args:
        T_a_b (np.ndarray): The first transformation matrix.
        T_b_c (np.ndarray): The second transformation matrix.

    Returns:
        np.ndarray: The composed transformation matrix (T_a_b @ T_b_c).
    """
    return T_a_b @ T_b_c


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Converts a rotation vector (rvec) and translation vector (tvec) from
    OpenCV format to a 4x4 SE(3) transformation matrix.

    Args:
        rvec (np.ndarray): A 3x1 rotation vector (Rodrigues representation).
        tvec (np.ndarray): A 3x1 translation vector.

    Returns:
        np.ndarray: The corresponding 4x4 homogeneous transformation matrix.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(rvec)
    return se3_to_T(R, tvec)


def R_to_rvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a Rodrigues vector (3,).

    Args:
        R: Rotation matrix.

    Returns:
        Rodrigues rotation vector with shape (3,).
    """
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)
