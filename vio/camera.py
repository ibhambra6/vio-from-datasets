"""A simple pinhole camera model with basic validation and utilities.

Exposes an intrinsic matrix `K` property and an `undistort` helper with optional output size.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


@dataclass
class PinholeCamera:
    """
    Represents a pinhole camera model with intrinsic parameters and distortion.

    This class encapsulates the camera matrix (K) and distortion coefficients,
    providing methods for operations like image undistortion.

    Attributes:
        fx (float): Focal length in the x-direction (in pixels).
        fy (float): Focal length in the y-direction (in pixels).
        cx (float): Principal point x-coordinate (in pixels).
        cy (float): Principal point y-coordinate (in pixels).
        dist (np.ndarray): Distortion coefficients in OpenCV's format
                           [k1, k2, p1, p2, k3].
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray

    def __post_init__(self) -> None:
        # Ensure numeric types
        for name in ("fx", "fy", "cx", "cy"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite float; got {value}")

        # Normalize distortion to shape (N,) float64; allow empty (no distortion)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        if self.dist.size not in (0, 4, 5, 8):
            raise ValueError("dist must have 0, 4, 5, or 8 coefficients (OpenCV format)")

    @property
    def K(self) -> np.ndarray:
        """
        Constructs and returns the 3x3 camera intrinsic matrix (K).

        The camera matrix relates 3D camera coordinates to 2D image plane
        coordinates.

        Returns:
            np.ndarray: A 3x3 numpy array representing the camera matrix,
                        with dtype=np.float64.
        """
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]], dtype=np.float64
        )

    def undistort(
        self, img: np.ndarray, alpha: float = 0.0, new_size: Optional[tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Removes lens distortion from an input image.

        This method applies the camera's distortion coefficients to correct
        for radial and tangential lens distortion, returning a rectified image.

        Args:
            img (np.ndarray): The distorted input image.

        Args:
            alpha: Free scaling parameter between 0 (crop black regions) and 1 (retain all pixels).
            new_size: Optional output size (width, height). Defaults to input size.

        Returns:
            np.ndarray: The undistorted (rectified) image.
        """
        if img is None or img.size == 0:
            raise ValueError("img must be a non-empty array")
        h, w = img.shape[:2]
        out_size = (w, h) if new_size is None else new_size
        # Get the optimal new camera matrix for undistortion
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), alpha, out_size)
        # Undistort the image
        return cv2.undistort(img, self.K, self.dist, None, new_K)
