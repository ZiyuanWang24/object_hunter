"""Interactive MuJoCo virtual-camera lab for ObjectHunter."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objecthunter.tracker import MultiObjectTracker
from objecthunter.types import Detection, Lifecycle
from objecthunter.visualization import COLORS
from generate_simulation import COLORS_RGB, EMBEDDINGS, FPS, SCENE_XML, WORLD_OFFSET, color_bbox, mocap_id, state


class MuJoCoLab:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ObjectHunter - MuJoCo Virtual Camera Lab")
        self.root.geometry("1880x900")
        self.root.configure(bg="#111827")
        self.target_var = tk.StringVar(value="cup")
        self.status_var = tk.StringVar(value="Press Start simulation")
        self.location_var = tk.StringVar(value="Waiting for target")
        self.lifecycle_var = tk.StringVar(value="NOT FOUND")
        self.metrics_var = tk.StringVar(value="FPS -   Latency -   Re-ID 0")
        self.latest: queue.Queue = queue.Queue(maxsize=1)
        self.commands: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.video_image = None
        self.scene_image = None
        self.map_image = None
        self.video_label = None
        self.scene_label = None
        self.map_label = None
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._refresh)
        self.root.after(250, self.start)

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#111827", padx=18, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="ObjectHunter", fg="#f9fafb", bg="#111827", font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="MuJoCo virtual-camera lab", fg="#60a5fa", bg="#111827", font=("Segoe UI", 12)).pack(side="left", padx=18)

        controls = tk.Frame(self.root, bg="#1f2937", padx=12, pady=10)
        controls.pack(fill="x", padx=14)
        tk.Label(controls, text="Target:", fg="#e5e7eb", bg="#1f2937", font=("Segoe UI", 11, "bold")).pack(side="left")
        entry = tk.Entry(controls, textvariable=self.target_var, width=12, font=("Segoe UI", 11))
        entry.pack(side="left", padx=7)
        entry.bind("<Return>", lambda _event: self.start())
        for label, command, color in [
            ("Start", self.start, "#16a34a"), ("Stop", self.stop, "#dc2626"),
            ("Reset", self.reset, "#374151"), ("Move left", lambda: self.commands.put("left"), "#2563eb"),
            ("Move right", lambda: self.commands.put("right"), "#2563eb"),
            ("Occlude", lambda: self.commands.put("hide"), "#b45309"),
            ("Reveal", lambda: self.commands.put("show"), "#0891b2"),
            ("Orbit 3-D ↺", lambda: self.commands.put("orbit_left"), "#7c3aed"),
            ("Orbit 3-D ↻", lambda: self.commands.put("orbit_right"), "#7c3aed"),
            ("Zoom +", lambda: self.commands.put("zoom_in"), "#9333ea"),
            ("Zoom -", lambda: self.commands.put("zoom_out"), "#9333ea"),
        ]:
            tk.Button(controls, text=label, command=command, bg=color, fg="white", relief="flat", padx=9, pady=5).pack(side="left", padx=3)
        tk.Label(controls, textvariable=self.status_var, fg="#fbbf24", bg="#1f2937", font=("Segoe UI", 10)).pack(side="right")

        body = tk.Frame(self.root, bg="#111827", padx=14, pady=14)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#030712", padx=6, pady=6)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="LIVE MUJOCO RGB CAMERA", fg="#d1d5db", bg="#030712", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.video_label = tk.Label(left, bg="#030712")
        self.video_label.pack(fill="both", expand=True)

        scene = tk.Frame(body, bg="#030712", padx=6, pady=6)
        scene.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(scene, text="3-D MUJOCO SCENE VIEW", fg="#d1d5db", bg="#030712", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.scene_label = tk.Label(scene, bg="#030712")
        self.scene_label.pack(fill="both", expand=True)

        right = tk.Frame(body, bg="#1f2937", width=390, padx=14, pady=14)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)
        tk.Label(right, text="TARGET LOCATION", fg="#f9fafb", bg="#1f2937", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(right, textvariable=self.location_var, fg="#4ade80", bg="#1f2937", font=("Consolas", 23, "bold"), pady=8).pack(anchor="w")
        tk.Label(right, textvariable=self.lifecycle_var, fg="#fbbf24", bg="#1f2937", font=("Segoe UI", 12)).pack(anchor="w")
        tk.Label(right, textvariable=self.metrics_var, fg="#d1d5db", bg="#1f2937", font=("Consolas", 10), pady=10).pack(anchor="w")
        tk.Label(right, text="TOP-DOWN METRIC MAP", fg="#f9fafb", bg="#1f2937", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 4))
        self.map_label = tk.Label(right, bg="#111827")
        self.map_label.pack(fill="both", expand=True)
        tk.Label(right, text="Green observed | Yellow predicted | Gray last known\nTry Occlude, wait one second, then Reveal.", fg="#9ca3af", bg="#1f2937", font=("Segoe UI", 9), justify="left", pady=8).pack(anchor="w")

    def start(self) -> None:
        target = self.target_var.get().strip().lower()
        if target not in COLORS_RGB:
            self.status_var.set("Use cup, bottle, book, or keys")
            return
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.status_var.set(f"Starting MuJoCo target: {target}")
        self.worker = threading.Thread(target=self._worker, args=(target,), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopped")

    def reset(self) -> None:
        self.commands.put("reset")
        self.status_var.set("Tracker reset")

    def _worker(self, target: str) -> None:
        try:
            model = mujoco.MjModel.from_xml_string(SCENE_XML)
            data = mujoco.MjData(model)
            # MuJoCo's renderer constructor uses (height, width), unlike
            # OpenCV's common (width, height) convention.
            renderer = mujoco.Renderer(model, 360, 640)
            scene_renderer = mujoco.Renderer(model, 360, 640)
            scene_camera = mujoco.MjvCamera()
            scene_camera.azimuth = 135.0
            scene_camera.elevation = -25.0
            scene_camera.distance = 1.9
            scene_camera.lookat[:] = [0.0, 0.0, 0.75]
            target_mocap = mocap_id(model, target)
            tracker = MultiObjectTracker(confirm_hits=3, dormant_timeout_s=1.0)
            target_id = None
            frame_index = 0
            manual_offset = np.zeros(2, dtype=float)
            hidden = False
            last_wall = time.perf_counter()
            fps = 0.0
            self._put_status(f"Tracking {target} in MuJoCo")
            while not self.stop_event.is_set():
                try:
                    while True:
                        command = self.commands.get_nowait()
                        if command == "left": manual_offset[0] = max(-0.35, manual_offset[0] - 0.05)
                        elif command == "right": manual_offset[0] = min(0.35, manual_offset[0] + 0.05)
                        elif command == "hide": hidden = True
                        elif command == "show": hidden = False
                        elif command == "orbit_left": scene_camera.azimuth -= 15.0
                        elif command == "orbit_right": scene_camera.azimuth += 15.0
                        elif command == "zoom_in": scene_camera.distance = max(1.0, scene_camera.distance - 0.15)
                        elif command == "zoom_out": scene_camera.distance = min(3.5, scene_camera.distance + 0.15)
                        elif command == "reset":
                            tracker.memory.reset(); tracker.filters.clear(); tracker.last_timestamp = None; target_id = None; manual_offset[:] = 0; hidden = False
                except queue.Empty:
                    pass

                xy, auto_visible, z = state(target, frame_index)
                xy = xy + manual_offset
                visible = auto_visible and not hidden
                data.mocap_pos[target_mocap] = [float(xy[0]), float(xy[1]), z if visible else 0.70]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="rgb_camera")
                rgb = renderer.render()
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                detections = []
                if visible:
                    bbox_contact = color_bbox(rgb, COLORS_RGB[target])
                    if bbox_contact is not None:
                        bbox, contact = bbox_contact
                        detections.append(Detection(
                            timestamp=frame_index / FPS, class_name=target, confidence=0.99,
                            bbox=tuple(bbox), contact_pixel=tuple(contact),
                            world_position=xy + WORLD_OFFSET, embedding=np.asarray(EMBEDDINGS[target], dtype=float),
                        ))
                start = time.perf_counter()
                objects = tracker.update(detections, frame_index / FPS)
                latency_ms = (time.perf_counter() - start) * 1000
                if target_id is None:
                    candidates = [o for o in objects if o.last_detection is not None and o.last_observed_at == frame_index / FPS]
                    if candidates:
                        target_id = candidates[-1].object_id
                target_record = tracker.memory.objects.get(target_id) if target_id is not None else None
                wall_now = time.perf_counter()
                instant = 1.0 / max(wall_now - last_wall, 1e-6)
                fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
                last_wall = wall_now
                cv2.putText(frame, f"MUJOCO CAMERA | target: {target}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (225, 235, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, "simulator ground truth -> ObjectHunter tracker", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 195, 215), 1, cv2.LINE_AA)
                scene_renderer.update_scene(data, camera=scene_camera)
                scene_frame = cv2.cvtColor(scene_renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(scene_frame, "OBLIQUE 3-D WORLD VIEW", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (225, 235, 245), 2, cv2.LINE_AA)
                cv2.putText(scene_frame, "desk height and object geometry", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 195, 215), 1, cv2.LINE_AA)
                self._put((frame, scene_frame, target_record, tracker, detections, fps, latency_ms))
                frame_index += 1
                time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - wall_now)))
        except Exception as exc:
            self._put_status(f"MuJoCo worker error: {type(exc).__name__}: {exc}")

    def _map_image(self, target) -> np.ndarray:
        size = 390
        image = np.zeros((size, size, 3), np.uint8); image[:] = (17, 24, 39)
        for g in range(1, 5):
            p = int(g * size / 5)
            cv2.line(image, (20, p), (size - 20, p), (45, 55, 70), 1)
            cv2.line(image, (p, 20), (p, size - 20), (45, 55, 70), 1)
        def pix(p): return int(20 + p[0] / 1.44 * (size - 40)), int(size - (20 + p[1] / 1.0 * (size - 40)))
        if target is not None:
            p = target.last_reliable_position if target.lifecycle == Lifecycle.DORMANT and target.last_reliable_position is not None else target.position
            color = (140, 140, 140) if target.lifecycle == Lifecycle.DORMANT else ((0, 220, 255) if target.lifecycle == Lifecycle.OCCLUDED else (60, 220, 60))
            if len(target.trajectory) > 1:
                pts = np.array([pix(t[1:]) for t in target.trajectory[-60:]], np.int32)
                cv2.polylines(image, [pts], False, color, 2)
            q = pix(p); cv2.circle(image, q, 9, color, -1); cv2.putText(image, f"#{target.object_id}", (q[0] + 12, q[1]), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            if target.lifecycle != Lifecycle.DORMANT:
                end = pix(p + target.velocity * .2); cv2.arrowedLine(image, q, end, color, 2, tipLength=.2)
            cov = target.covariance[:2, :2]
            radius = max(8, int(np.sqrt(max(0.0, float(np.trace(cov)))) * 180))
            cv2.ellipse(image, q, (radius, max(8, radius // 2)), 0, 0, 360, color, 1)
        cv2.putText(image, "meters", (size - 65, size - 10), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 190, 200), 1)
        return image

    def _put(self, value) -> None:
        try: self.latest.get_nowait()
        except queue.Empty: pass
        try: self.latest.put_nowait(value)
        except queue.Full: pass

    def _put_status(self, value: str) -> None:
        self.root.after(0, lambda: self.status_var.set(value))

    def _refresh(self) -> None:
        try:
            frame, scene_frame, target, tracker, detections, fps, latency = self.latest.get_nowait()
            view = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = map(int, d.bbox)
                cv2.rectangle(view, (x1, y1), (x2, y2), (70, 230, 90), 3)
                cv2.putText(view, f"TARGET {d.class_name} {d.confidence:.2f}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            frame_rgb = cv2.cvtColor(cv2.resize(view, (900, 506)), cv2.COLOR_BGR2RGB)
            self.video_image = ImageTk.PhotoImage(Image.fromarray(frame_rgb)); self.video_label.configure(image=self.video_image)
            scene_rgb = cv2.cvtColor(cv2.resize(scene_frame, (900, 506)), cv2.COLOR_BGR2RGB)
            self.scene_image = ImageTk.PhotoImage(Image.fromarray(scene_rgb)); self.scene_label.configure(image=self.scene_image)
            map_rgb = cv2.cvtColor(self._map_image(target), cv2.COLOR_BGR2RGB)
            self.map_image = ImageTk.PhotoImage(Image.fromarray(map_rgb)); self.map_label.configure(image=self.map_image)
            if target is None:
                self.location_var.set("Waiting for target")
                self.lifecycle_var.set("NOT FOUND")
            else:
                p = target.last_reliable_position if target.lifecycle == Lifecycle.DORMANT and target.last_reliable_position is not None else target.position
                label = "LAST KNOWN" if target.lifecycle == Lifecycle.DORMANT else target.lifecycle.value
                self.location_var.set(f"({p[0]:.3f}, {p[1]:.3f}) m")
                self.lifecycle_var.set(f"#{target.object_id} {target.class_name} - {label}")
            self.metrics_var.set(f"FPS {fps:.1f}   Latency {latency:.2f} ms   Re-ID {tracker.memory.total_reidentifications}")
            self.status_var.set("Simulation running")
        except queue.Empty:
            pass
        self.root.after(40, self._refresh)

    def close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    MuJoCoLab().root.mainloop()
