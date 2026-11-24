"""Run ObjectHunter's tracker and dashboard against the virtual camera stream."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from objecthunter.tracker import MultiObjectTracker
from objecthunter.types import Detection
from objecthunter.visualization import draw_dashboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="assets/simulation_scene.mp4")
    parser.add_argument("--ground-truth", default="assets/simulation_ground_truth.jsonl")
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--record", default="outputs/simulation_dashboard.mp4")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    target_class = args.target_class or input("Simulation item to track [cup]: ").strip() or "cup"
    records = [json.loads(line) for line in Path(args.ground_truth).read_text(encoding="utf-8").splitlines()]
    tracker = MultiObjectTracker(confirm_hits=3, dormant_timeout_s=1.0)
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open virtual camera stream: {args.source}")

    writer = None
    paused = False
    previous_time = time.perf_counter()
    fps = 0.0
    frame_index = 0
    try:
        while frame_index < len(records):
            ok, frame = capture.read()
            if not ok:
                break
            record = records[frame_index]
            if not paused:
                detections = []
                for item in record["objects"]:
                    if item["class_name"].lower() != target_class.lower():
                        continue
                    detections.append(Detection(
                        timestamp=record["timestamp"], class_name=item["class_name"],
                        confidence=item["confidence"], bbox=tuple(item["bbox"]),
                        contact_pixel=tuple(item["contact_pixel"]),
                        world_position=np.asarray(item["world_position"], dtype=float),
                        embedding=np.asarray(item["embedding"], dtype=float),
                    ))
                start = time.perf_counter()
                objects = tracker.update(detections, record["timestamp"])
                latency_ms = (time.perf_counter() - start) * 1000.0
                now = time.perf_counter()
                instant_fps = 1.0 / max(now - previous_time, 1e-6)
                fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
                previous_time = now
            else:
                detections, objects, latency_ms = [], list(tracker.memory.objects.values()), 0.0

            target = next((o for o in objects if o.class_name.lower() == target_class.lower()), None)
            metrics = {"fps": fps, "latency_ms": latency_ms, "gpu_memory_mb": 0.0}
            if target is not None:
                position = target.last_reliable_position if target.lifecycle.value == "DORMANT" and target.last_reliable_position is not None else target.position
                metrics.update({"target_position": (float(position[0]), float(position[1])), "target_unit": "m"})
            view = draw_dashboard(frame, detections, objects, metrics, (0.0, 1.2, 0.0, 0.8), 600)
            cv2.putText(view, f"VIRTUAL CAMERA | target: {target_class}", (10, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 240), 1, cv2.LINE_AA)
            if writer is None:
                Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (view.shape[1], view.shape[0]))
            writer.write(view)
            if not args.headless:
                cv2.imshow("ObjectHunter Virtual Camera", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("r"):
                    tracker.memory.reset(); tracker.filters.clear(); tracker.last_timestamp = None
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Tracked {target_class} over {frame_index} virtual-camera frames")
    target_records = [o for o in tracker.memory.objects.values() if o.class_name.lower() == target_class.lower()]
    if target_records:
        target = target_records[0]
        print(f"Persistent ID: {target.object_id}; re-identifications: {target.reidentification_count}; final lifecycle: {target.lifecycle.value}")
    print(f"Wrote dashboard recording: {args.record}")


if __name__ == "__main__":
    main()
