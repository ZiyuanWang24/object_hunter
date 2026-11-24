from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import yaml

from objecthunter.camera import LatestFrameCamera
from objecthunter.detector import YOLOSegDetector
from objecthunter.embedder import DINOv2Embedder
from objecthunter.geometry import HomographyCalibrator
from objecthunter.pipeline import PerceptionPipeline
from objecthunter.tracker import MultiObjectTracker
from objecthunter.types import Lifecycle


class ObjectHunterUI:
    def __init__(self, args):
        self.args = args
        self.cfg = yaml.safe_load(Path(args.config).read_text())
        self.root = tk.Tk()
        self.root.title("ObjectHunter — Single-Target Spatial Memory")
        self.root.geometry("1280x820")
        self.root.configure(bg="#111827")
        self.stop_event = threading.Event()
        self.worker = None
        self.latest = queue.Queue(maxsize=1)
        self.reset_requested = False
        self.target_class = tk.StringVar(value=args.target_class or "cup")
        self.status = tk.StringVar(value="Enter an item and press Start")
        self.location = tk.StringVar(value="—")
        self.lifecycle = tk.StringVar(value="—")
        self.metrics = tk.StringVar(value="FPS —   Latency —   GPU —")
        self.video_label = None
        self.map_label = None
        self.video_image = None
        self.map_image = None
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._refresh)

    def _build_layout(self):
        header = tk.Frame(self.root, bg="#111827", padx=18, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="ObjectHunter", fg="#f9fafb", bg="#111827", font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="uncertainty-aware single-object spatial memory", fg="#9ca3af", bg="#111827", font=("Segoe UI", 11)).pack(side="left", padx=16, pady=6)

        controls = tk.Frame(self.root, bg="#1f2937", padx=14, pady=10)
        controls.pack(fill="x", padx=14)
        tk.Label(controls, text="Item to find:", fg="#e5e7eb", bg="#1f2937", font=("Segoe UI", 11, "bold")).pack(side="left")
        entry = tk.Entry(controls, textvariable=self.target_class, width=18, font=("Segoe UI", 12))
        entry.pack(side="left", padx=8)
        entry.bind("<Return>", lambda _event: self.start())
        tk.Button(controls, text="Start tracking", command=self.start, bg="#16a34a", fg="white", relief="flat", padx=14, pady=6).pack(side="left", padx=4)
        tk.Button(controls, text="Stop", command=self.stop, bg="#dc2626", fg="white", relief="flat", padx=14, pady=6).pack(side="left", padx=4)
        tk.Button(controls, text="Reset target", command=self.reset, bg="#374151", fg="white", relief="flat", padx=14, pady=6).pack(side="left", padx=4)
        tk.Label(controls, textvariable=self.status, fg="#fbbf24", bg="#1f2937", font=("Segoe UI", 10)).pack(side="right")

        body = tk.Frame(self.root, bg="#111827", padx=14, pady=14)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#030712", padx=6, pady=6)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="LIVE PERCEPTION", fg="#d1d5db", bg="#030712", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.video_label = tk.Label(left, bg="#030712")
        self.video_label.pack(fill="both", expand=True)

        right = tk.Frame(body, bg="#1f2937", width=360, padx=14, pady=14)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)
        tk.Label(right, text="TARGET LOCATION", fg="#f9fafb", bg="#1f2937", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(right, textvariable=self.location, fg="#4ade80", bg="#1f2937", font=("Consolas", 24, "bold"), pady=10).pack(anchor="w")
        tk.Label(right, textvariable=self.lifecycle, fg="#fbbf24", bg="#1f2937", font=("Segoe UI", 12)).pack(anchor="w")
        tk.Label(right, textvariable=self.metrics, fg="#d1d5db", bg="#1f2937", font=("Consolas", 10), pady=12).pack(anchor="w")
        tk.Label(right, text="TOP-DOWN MAP", fg="#f9fafb", bg="#1f2937", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(12, 4))
        self.map_label = tk.Label(right, bg="#111827")
        self.map_label.pack(fill="both", expand=True)
        tk.Label(right, text="Green = observed   Yellow = predicted   Gray = last known", fg="#9ca3af", bg="#1f2937", font=("Segoe UI", 9), wraplength=320, justify="left", pady=10).pack(anchor="w")

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        target = self.target_class.get().strip().lower()
        if not target:
            self.status.set("Enter an item class first")
            return
        self.stop_event.clear()
        self.status.set(f"Loading models for: {target}")
        self.worker = threading.Thread(target=self._worker, args=(target,), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status.set("Stopping…")

    def reset(self):
        self.reset_requested = True
        self.status.set("Target reset; waiting for a new match")

    def _worker(self, target_class):
        cap = None
        try:
            device = self.args.device
            if str(device).startswith("cuda"):
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
                    self._put_status("CUDA unavailable; using CPU")
            detector = YOLOSegDetector(self.cfg["detector"]["model"], self.cfg["detector"]["confidence"], self.cfg["detector"]["iou"], self.cfg["detector"]["imgsz"], device, self.cfg["detector"]["classes"])
            embedder = DINOv2Embedder(self.cfg["embedder"]["model"], self.cfg["embedder"]["device"])
            geometry = None
            calibration = self.cfg["geometry"]["calibration_path"]
            if os.path.exists(calibration): geometry = HomographyCalibrator.load(calibration)
            tracker = MultiObjectTracker(**self.cfg["tracker"])
            pipe = PerceptionPipeline(detector, tracker, geometry, embedder, self.cfg["embedder"]["refresh_interval"])
            source = self.args.source if self.args.source else self.args.camera
            cap = LatestFrameCamera(source, self.cfg["camera"]["width"], self.cfg["camera"]["height"]).start()
            target_id = None
            self._put_status(f"Looking for: {target_class}")
            while not self.stop_event.is_set() and cap.running:
                item = cap.read()
                if item is None:
                    time.sleep(.01); continue
                timestamp, frame = item
                if self.reset_requested:
                    tracker.memory.reset(); tracker.filters.clear(); target_id = None; self.reset_requested = False

                def filter_items(items):
                    target = tracker.memory.objects.get(target_id) if target_id is not None else None
                    wanted = target.class_name if target is not None else target_class
                    return [d for d in items if d.class_name.lower() == wanted.lower()][:1]

                detections, objects = pipe.process(frame, timestamp, filter_items)
                if target_id is None:
                    created = [o for o in objects if o.last_observed_at == timestamp and o.last_detection is not None]
                    if created: target_id = created[-1].object_id
                target = tracker.memory.objects.get(target_id) if target_id is not None else None
                target_objects = [target] if target is not None else []
                display = self._annotate(frame, detections, target)
                map_image = self._map_image(target_objects, geometry, frame.shape)
                info = self._target_info(target, geometry, pipe.metrics)
                self._put((display, map_image, info))
        except Exception as exc:
            self._put_status(f"Error: {exc}")
        finally:
            if cap is not None: cap.stop()

    def _annotate(self, frame, detections, target):
        view = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            if d.mask is not None and d.mask.shape[:2] == view.shape[:2]:
                overlay = view.copy(); overlay[d.mask > 0] = (40, 190, 70); view = cv2.addWeighted(view, .72, overlay, .28, 0)
            cv2.rectangle(view, (x1, y1), (x2, y2), (70, 230, 90), 3)
            cv2.putText(view, f"TARGET: {d.class_name} {d.confidence:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            if d.contact_pixel: cv2.circle(view, tuple(map(int, d.contact_pixel)), 7, (0, 255, 255), -1)
        return view

    def _map_image(self, objects, geometry, shape):
        size = 420; image = np.zeros((size, size, 3), np.uint8); image[:] = (17, 24, 39)
        if geometry is not None: bounds = (0, self.cfg["geometry"]["world_width"], 0, self.cfg["geometry"]["world_height"]); unit = "m"
        else: bounds = (0, shape[1], 0, shape[0]); unit = "px"
        xmin, xmax, ymin, ymax = bounds
        def pix(p): return int((p[0]-xmin) / max(1e-9, xmax-xmin) * (size-20) + 10), int(size - ((p[1]-ymin) / max(1e-9, ymax-ymin) * (size-20) + 10))
        for g in range(1, 5):
            cv2.line(image, (10, g*size//5), (size-10, g*size//5), (45, 55, 70), 1); cv2.line(image, (g*size//5, 10), (g*size//5, size-10), (45, 55, 70), 1)
        for obj in objects:
            p = obj.last_reliable_position if obj.lifecycle == Lifecycle.DORMANT and obj.last_reliable_position is not None else obj.position
            color = (140, 140, 140) if obj.lifecycle == Lifecycle.DORMANT else ((0, 220, 255) if obj.lifecycle == Lifecycle.OCCLUDED else (60, 220, 60))
            if len(obj.trajectory) > 1:
                pts = np.array([pix(t[1:]) for t in obj.trajectory[-60:]], np.int32); cv2.polylines(image, [pts], False, color, 2)
            q = pix(p); cv2.circle(image, q, 10, color, -1); cv2.putText(image, f"#{obj.object_id}", (q[0]+12, q[1]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        cv2.putText(image, unit, (size-35, size-12), cv2.FONT_HERSHEY_SIMPLEX, .5, (180, 190, 200), 1)
        return image

    def _target_info(self, target, geometry, metrics):
        if target is None: return {"location": "Waiting for target…", "lifecycle": "NOT FOUND", "metrics": f"FPS {metrics.get('fps', 0):.1f}   Latency {metrics.get('latency_ms', 0):.1f} ms", "target": None}
        pos = target.last_reliable_position if target.lifecycle == Lifecycle.DORMANT and target.last_reliable_position is not None else target.position
        unit = "m" if geometry is not None else "px"; state = "LAST KNOWN" if target.lifecycle == Lifecycle.DORMANT else target.lifecycle.value
        return {"location": f"({pos[0]:.3f}, {pos[1]:.3f}) {unit}", "lifecycle": f"#{target.object_id} {target.class_name} — {state}", "metrics": f"FPS {metrics.get('fps', 0):.1f}   Latency {metrics.get('latency_ms', 0):.1f} ms   GPU {metrics.get('gpu_memory_mb', 0):.0f} MB", "target": target}

    def _put(self, value):
        try: self.latest.get_nowait()
        except queue.Empty: pass
        try: self.latest.put_nowait(value)
        except queue.Full: pass

    def _put_status(self, text): self.root.after(0, lambda: self.status.set(text))

    def _refresh(self):
        try:
            frame, map_image, info = self.latest.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame_rgb = cv2.resize(frame_rgb, (820, 615), interpolation=cv2.INTER_AREA)
            self.video_image = ImageTk.PhotoImage(Image.fromarray(frame_rgb)); self.video_label.configure(image=self.video_image)
            map_rgb = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB); self.map_image = ImageTk.PhotoImage(Image.fromarray(map_rgb)); self.map_label.configure(image=self.map_image)
            self.location.set(info["location"]); self.lifecycle.set(info["lifecycle"]); self.metrics.set(info["metrics"]); self.status.set("Tracking active")
        except queue.Empty: pass
        self.root.after(50, self._refresh)

    def close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="ObjectHunter desktop single-target UI")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-class", help="Initial item class, e.g. cup")
    args = parser.parse_args()
    ObjectHunterUI(args).root.mainloop()


if __name__ == "__main__": main()
