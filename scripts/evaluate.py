"""Reproducible validation for ObjectHunter's deterministic core.

This benchmark deliberately excludes YOLO/DINO inference and webcam capture.
It measures geometry, filtering, association, lifecycle memory, and core
throughput on deterministic synthetic observations. Real-camera metrics remain
unavailable until annotated replay data is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from objecthunter.geometry import HomographyCalibrator
from objecthunter.kalman import ConstantVelocityKalman
from objecthunter.tracker import MultiObjectTracker
from objecthunter.types import Detection


def homography_validation() -> dict[str, float]:
    image_corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=float)

    def project(pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=float)
        return np.column_stack(
            [0.002 * pixels[:, 0] + 0.0003 * pixels[:, 1] + 0.1,
             -0.0002 * pixels[:, 0] + 0.003 * pixels[:, 1] - 0.2]
        )

    calibrator = HomographyCalibrator.from_points(
        image_corners, project(image_corners), image_size=(640, 480)
    )
    validation_pixels = np.array([[80, 60], [320, 240], [550, 390]], dtype=float)
    expected_world = project(validation_pixels)
    predicted_world = calibrator.pixel_to_world(validation_pixels)
    roundtrip_pixels = calibrator.world_to_pixel(predicted_world)
    return {
        "validation_points": float(len(validation_pixels)),
        "world_rmse_m": float(np.sqrt(np.mean(np.sum((predicted_world - expected_world) ** 2, axis=1)))),
        "pixel_roundtrip_max_error_px": float(np.max(np.linalg.norm(roundtrip_pixels - validation_pixels, axis=1))),
        "calibration_fit_rms_m": float(calibrator.rms_error),
    }


def kalman_validation() -> dict[str, float]:
    rng = np.random.default_rng(2026)
    dt = 0.05
    steps = 240
    timestamps = np.arange(steps) * dt
    truth = np.column_stack([0.25 + 0.18 * timestamps, 0.40 + 0.06 * timestamps])
    measurements = truth + rng.normal(0.0, 0.025, size=truth.shape)

    filt = ConstantVelocityKalman(
        measurements[0], process_noise=0.03, measurement_noise=0.025**2
    )
    estimates = [filt.x[:2].copy()]
    for measurement in measurements[1:]:
        filt.predict(dt)
        filt.update(measurement)
        estimates.append(filt.x[:2].copy())
    estimates = np.asarray(estimates)
    stable_error = estimates[20:] - truth[20:]

    occlusion_filter = ConstantVelocityKalman(
        measurements[0], process_noise=0.03, measurement_noise=0.025**2
    )
    for measurement in measurements[1:61]:
        occlusion_filter.predict(dt)
        occlusion_filter.update(measurement)
    covariance_before = float(np.trace(occlusion_filter.P[:2, :2]))
    predictions = []
    for index in range(61, 71):
        occlusion_filter.predict(dt)
        predictions.append(occlusion_filter.x[:2].copy())
    covariance_after = float(np.trace(occlusion_filter.P[:2, :2]))
    prediction_error = np.asarray(predictions) - truth[61:71]

    return {
        "position_rmse_m_after_warmup": float(np.sqrt(np.mean(stable_error**2))),
        "occlusion_frames": 10.0,
        "occlusion_prediction_rmse_m": float(np.sqrt(np.mean(prediction_error**2))),
        "position_covariance_trace_growth_ratio": covariance_after / covariance_before,
    }


def detection(timestamp: float, class_name: str, position: tuple[float, float], embedding: tuple[float, float]) -> Detection:
    x, y = position
    return Detection(
        timestamp=timestamp,
        class_name=class_name,
        confidence=0.95,
        bbox=(x - 0.05, y - 0.05, x + 0.05, y + 0.05),
        world_position=np.asarray(position, dtype=float),
        embedding=np.asarray(embedding, dtype=float),
    )


def association_and_memory_validation() -> dict[str, float | bool]:
    tracker = MultiObjectTracker(
        confirm_hits=1, dormant_timeout_s=0.5, reid_threshold=0.7
    )
    identity_ids: dict[tuple[float, float], int] = {}
    consistent_frames = 0
    total_frames = 60
    for frame in range(total_frames):
        timestamp = frame * 0.1
        cup = detection(timestamp, "cup", (0.20 + frame * 0.002, 0.30), (1.0, 0.0))
        bottle = detection(timestamp, "bottle", (0.70 - frame * 0.002, 0.60), (0.0, 1.0))
        observations = [cup, bottle] if frame % 2 == 0 else [bottle, cup]
        objects = tracker.update(observations, timestamp)
        current_ids = {
            tuple(np.asarray(obj.last_detection.embedding, dtype=float)): obj.object_id
            for obj in objects
            if obj.last_detection is not None and obj.last_detection.embedding is not None
        }
        if frame == 0:
            identity_ids = current_ids
        elif all(current_ids.get(key) == value for key, value in identity_ids.items()):
            consistent_frames += 1

    reid_tracker = MultiObjectTracker(
        confirm_hits=1, dormant_timeout_s=0.5, reid_threshold=0.7
    )
    original = reid_tracker.update([detection(0.0, "cup", (0.2, 0.2), (1.0, 0.0))], 0.0)[0]
    original_id = original.object_id
    reid_tracker.update([], 1.0)
    recovered = reid_tracker.update(
        [detection(2.0, "cup", (4.0, 3.0), (1.0, 0.0))], 2.0
    )[0]

    start = time.perf_counter()
    throughput_tracker = MultiObjectTracker(confirm_hits=1)
    for frame in range(300):
        timestamp = frame * 0.05
        throughput_tracker.update(
            [
                detection(timestamp, "cup", (0.2 + frame * 0.001, 0.3), (1.0, 0.0)),
                detection(timestamp, "bottle", (0.7 - frame * 0.001, 0.6), (0.0, 1.0)),
            ],
            timestamp,
        )
    elapsed = time.perf_counter() - start

    return {
        "association_frames": float(total_frames),
        "identity_consistency_rate": float(consistent_frames / (total_frames - 1)),
        "reidentification_success": bool(recovered.object_id == original_id and recovered.reidentification_count == 1),
        "core_update_frames": 300.0,
        "core_update_rate_hz": float(300 / elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()

    result = {
        "benchmark": "objecthunter_deterministic_core_validation",
        "seed": 2026,
        "neural_inference_measured": False,
        "homography": homography_validation(),
        "kalman": kalman_validation(),
        "association_memory": association_and_memory_validation(),
        "unavailable_without_annotated_replay": [
            "IDF1",
            "ID switches",
            "camera-to-world error on a real scene",
            "YOLO/DINO latency and peak GPU memory",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
