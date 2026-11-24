from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class Lifecycle(str, Enum):
    TENTATIVE = "TENTATIVE"
    TRACKED = "TRACKED"
    OCCLUDED = "OCCLUDED"
    DORMANT = "DORMANT"


@dataclass
class Detection:
    timestamp: float
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None
    contact_pixel: Optional[tuple[float, float]] = None
    world_position: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)


@dataclass
class Observation:
    timestamp: float
    position: np.ndarray
    confidence: float
    embedding: Optional[np.ndarray]
    detection: Detection


@dataclass
class TrackRecord:
    object_id: int
    class_name: str
    created_at: float
    last_observed_at: float
    last_reliable_position: Optional[np.ndarray]
    state: np.ndarray
    covariance: np.ndarray
    gallery: list[np.ndarray] = field(default_factory=list)
    observation_count: int = 0
    missed_observations: int = 0
    detection_confidence: float = 0.0
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)
    lifecycle: Lifecycle = Lifecycle.TENTATIVE
    reidentification_count: int = 0
    last_detection: Optional[Detection] = None

    @property
    def position(self) -> np.ndarray:
        return self.state[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:]

    @property
    def reliable(self) -> bool:
        return self.lifecycle in (Lifecycle.TRACKED, Lifecycle.OCCLUDED)
