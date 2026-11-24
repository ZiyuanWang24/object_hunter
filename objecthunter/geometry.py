from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
try:
    import cv2
except ImportError:  # Optional fallback keeps homography math unit-testable.
    cv2 = None
import numpy as np


@dataclass
class HomographyCalibrator:
    image_points: np.ndarray
    world_points: np.ndarray
    image_size: tuple[int, int]
    H: np.ndarray
    H_inv: np.ndarray
    rms_error: float

    @classmethod
    def from_points(cls, image_points, world_points, image_size, ransac_threshold=3.0):
        image = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        world = np.asarray(world_points, dtype=np.float32).reshape(-1, 2)
        if len(image) < 4 or len(world) != len(image):
            raise ValueError("At least four corresponding image/world points are required")
        if cv2 is not None:
            H, mask = cv2.findHomography(image, world, cv2.RANSAC, ransac_threshold)
        else:
            H = cls._dlt(image, world)
        if H is None:
            raise ValueError("Could not estimate a homography; check point geometry")
        H_inv = np.linalg.inv(H)
        pred = cls._transform(image, H)
        err = np.linalg.norm(pred - world, axis=1)
        return cls(image, world, tuple(map(int, image_size)), H, H_inv, float(np.sqrt(np.mean(err**2))))

    @staticmethod
    def _transform(points, matrix):
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.c_[p, np.ones(len(p))] @ matrix.astype(np.float64).T
        return (homogeneous[:, :2] / homogeneous[:, 2:3]).reshape(-1, 2)

    @staticmethod
    def _dlt(src, dst):
        rows = []
        for (x, y), (u, v) in zip(src, dst):
            rows.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u], [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
        _, _, vh = np.linalg.svd(np.asarray(rows, float))
        H = vh[-1].reshape(3, 3); return H / H[2, 2]

    def pixel_to_world(self, pixels):
        result = self._transform(pixels, self.H)
        return result[0] if np.asarray(pixels).ndim == 1 else result

    def world_to_pixel(self, world):
        result = self._transform(world, self.H_inv)
        return result[0] if np.asarray(world).ndim == 1 else result

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"image_points": self.image_points.tolist(), "world_points": self.world_points.tolist(), "image_size": list(self.image_size), "H": self.H.tolist(), "rms_error": self.rms_error}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        H = np.asarray(d["H"], dtype=float)
        image = np.asarray(d["image_points"], dtype=float)
        world = np.asarray(d["world_points"], dtype=float)
        return cls(image, world, tuple(d["image_size"]), H, np.linalg.inv(H), float(d.get("rms_error", 0)))


def mask_contact_point(mask: np.ndarray, lower_fraction: float = 0.2) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cutoff = ys.max() - max(1, int((ys.max() - ys.min()) * lower_fraction))
    chosen = xs[ys >= cutoff]
    return float(np.median(chosen)), float(ys.max())


def undistort_pixels(pixels, camera_matrix, distortion):
    """Optional pinhole-lens correction before homography projection."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for lens-distortion correction")
    p = np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2)
    corrected = cv2.undistortPoints(p, np.asarray(camera_matrix, dtype=np.float64), np.asarray(distortion, dtype=np.float64), P=np.asarray(camera_matrix, dtype=np.float64))
    corrected = corrected.reshape(-1, 2)
    return corrected[0] if np.asarray(pixels).ndim == 1 else corrected


def undistort_frame(frame, camera_matrix, distortion):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for lens-distortion correction")
    return cv2.undistort(frame, np.asarray(camera_matrix, dtype=np.float64), np.asarray(distortion, dtype=np.float64))
