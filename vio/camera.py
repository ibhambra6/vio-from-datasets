"""A simple pinhole camera model."""

from dataclasses import dataclass
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
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1.0]
        ], dtype=np.float64)

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """
        Removes lens distortion from an input image.

        This method applies the camera's distortion coefficients to correct
        for radial and tangential lens distortion, returning a rectified image.

        Args:
            img (np.ndarray): The distorted input image.

        Returns:
            np.ndarray: The undistorted (rectified) image.
        """
        h, w = img.shape[:2]
        # Get the optimal new camera matrix for undistortion
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 0)
        # Undistort the image
        return cv2.undistort(img, self.K, self.dist, None, new_K)
