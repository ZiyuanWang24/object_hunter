"""Interactive living-room scene with a movable MuJoCo camera and tracker."""

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
from living_room import ROOM_EMBEDDINGS, ROOM_TARGETS, ROOM_XML


class LivingRoomLab:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ObjectHunter - Living Room 3-D Lab")
        self.root.geometry("1460x850")
        self.root.configure(bg="#111827")
        self.target_var = tk.StringVar(value="mug")
        self.status_var = tk.StringVar(value="Enter mug, bottle, book, or keys")
        self.location_var = tk.StringVar(value="Waiting for target")
        self.lifecycle_var = tk.StringVar(value="NOT FOUND")
        self.metrics_var = tk.StringVar(value="FPS -   Latency -   Re-ID 0")
        self.latest: queue.Queue = queue.Queue(maxsize=1)
        self.commands: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.image = None
        self.image_label = None
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._refresh)
        self.root.after(250, self.start)

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#111827", padx=18, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="ObjectHunter", fg="#f9fafb", bg="#111827", font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="Living-room movable-camera lab", fg="#60a5fa", bg="#111827", font=("Segoe UI", 12)).pack(side="left", padx=18)

        controls = tk.Frame(self.root, bg="#1f2937", padx=12, pady=9)
        controls.pack(fill="x", padx=14)
        tk.Label(controls, text="Target:", fg="#e5e7eb", bg="#1f2937", font=("Segoe UI", 11, "bold")).pack(side="left")
        entry = tk.Entry(controls, textvariable=self.target_var, width=10, font=("Segoe UI", 11))
        entry.pack(side="left", padx=7)
        entry.bind("<Return>", lambda _event: self.start())
        buttons = [
            ("Start", self.start, "#16a34a"), ("Stop", self.stop, "#dc2626"), ("Reset", self.reset, "#374151"),
            ("Occlude", lambda: self.commands.put("hide"), "#b45309"), ("Reveal", lambda: self.commands.put("show"), "#0891b2"),
            ("Orbit ↺", lambda: self.commands.put("orbit_left"), "#7c3aed"), ("Orbit ↻", lambda: self.commands.put("orbit_right"), "#7c3aed"),
            ("Zoom +", lambda: self.commands.put("zoom_in"), "#9333ea"), ("Zoom -", lambda: self.commands.put("zoom_out"), "#9333ea"),
            ("Pan ←", lambda: self.commands.put("pan_left"), "#2563eb"), ("Pan →", lambda: self.commands.put("pan_right"), "#2563eb"),
            ("Pan ↑", lambda: self.commands.put("pan_forward"), "#2563eb"), ("Pan ↓", lambda: self.commands.put("pan_back"), "#2563eb"),
        ]
        for label, command, color in buttons:
            tk.Button(controls, text=label, command=command, bg=color, fg="white", relief="flat", padx=7, pady=4).pack(side="left", padx=2)
        tk.Label(controls, textvariable=self.status_var, fg="#fbbf24", bg="#1f2937", font=("Segoe UI", 10)).pack(side="right")

        body = tk.Frame(self.root, bg="#111827", padx=14, pady=14)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#030712", padx=6, pady=6)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="MOVABLE 3-D MUJOCO ROOM CAMERA", fg="#d1d5db", bg="#030712", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.image_label = tk.Label(left, bg="#030712")
        self.image_label.pack(fill="both", expand=True)

        right = tk.Frame(body, bg="#1f2937", width=360, padx=14, pady=14)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)
        tk.Label(right, text="TARGET LOCATION", fg="#f9fafb", bg="#1f2937", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(right, textvariable=self.location_var, fg="#4ade80", bg="#1f2937", font=("Consolas", 22, "bold"), pady=8).pack(anchor="w")
        tk.Label(right, textvariable=self.lifecycle_var, fg="#fbbf24", bg="#1f2937", font=("Segoe UI", 12)).pack(anchor="w")
        tk.Label(right, textvariable=self.metrics_var, fg="#d1d5db", bg="#1f2937", font=("Consolas", 10), pady=10).pack(anchor="w")
        tk.Label(right, text="ROOM OBJECTS", fg="#f9fafb", bg="#1f2937", font=("Segoe UI", 12, "bold"), pady=8).pack(anchor="w")
        tk.Label(right, text="Sofa\nCoffee table\nTV + console\nBookshelf + books\nFloor lamp\nPlant\nMug / bottle / book / keys", fg="#d1d5db", bg="#1f2937", justify="left", font=("Segoe UI", 11), pady=6).pack(anchor="w")
        tk.Label(right, text="Green = observed\nYellow = predicted during occlusion\nCamera controls change the 3-D view", fg="#9ca3af", bg="#1f2937", justify="left", font=("Segoe UI", 9), pady=16).pack(anchor="w")

    def start(self) -> None:
        target = self.target_var.get().strip().lower()
        if target not in ROOM_TARGETS:
            self.status_var.set("Use mug, bottle, book, or keys")
            return
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.status_var.set(f"Starting room target: {target}")
        self.worker = threading.Thread(target=self._worker, args=(target,), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopped")

    def reset(self) -> None:
        self.commands.put("reset")
        self.status_var.set("Tracker and camera reset")

    def _worker(self, target: str) -> None:
        try:
            model = mujoco.MjModel.from_xml_string(ROOM_XML)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, 540, 960)
            camera = mujoco.MjvCamera()
            camera.azimuth = 90.0
            camera.elevation = -20.0
            camera.distance = 5.5
            camera.lookat[:] = [0.0, 1.5, 1.0]
            mocap_id = int(model.body(target).mocapid[0])
            tracker = MultiObjectTracker(confirm_hits=3, dormant_timeout_s=2.0)
            target_id = None
            frame_index = 0
            hidden = False
            last_wall = time.perf_counter()
            fps = 0.0
            self._put_status(f"Room camera tracking {target}")
            while not self.stop_event.is_set():
                try:
                    while True:
                        command = self.commands.get_nowait()
                        if command == "orbit_left": camera.azimuth -= 12.0
                        elif command == "orbit_right": camera.azimuth += 12.0
                        elif command == "zoom_in": camera.distance = max(2.5, camera.distance - 0.25)
                        elif command == "zoom_out": camera.distance = min(10.0, camera.distance + 0.25)
                        elif command == "pan_left": camera.lookat[0] -= 0.25
                        elif command == "pan_right": camera.lookat[0] += 0.25
                        elif command == "pan_forward": camera.lookat[1] += 0.25
                        elif command == "pan_back": camera.lookat[1] -= 0.25
                        elif command == "hide": hidden = True
                        elif command == "show": hidden = False
                        elif command == "reset":
                            tracker.memory.reset(); tracker.filters.clear(); tracker.last_timestamp = None; target_id = None; hidden = False
                            camera.azimuth = 90.0; camera.elevation = -20.0; camera.distance = 5.5; camera.lookat[:] = [0.0, 1.5, 1.0]
                except queue.Empty:
                    pass

                xy = ROOM_TARGETS[target]
                data.mocap_pos[mocap_id] = [float(xy[0]), float(xy[1]), 0.10 if hidden else 0.755]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                detections = [] if hidden else [Detection(
                    timestamp=frame_index / 30.0, class_name=target, confidence=0.99,
                    bbox=(0.0, 0.0, 1.0, 1.0), contact_pixel=(0.0, 0.0),
                    world_position=xy.copy(), embedding=ROOM_EMBEDDINGS[target].copy(),
                )]
                start = time.perf_counter()
                objects = tracker.update(detections, frame_index / 30.0)
                latency_ms = (time.perf_counter() - start) * 1000.0
                if target_id is None:
                    recent = [o for o in objects if o.last_detection is not None]
                    if recent:
                        target_id = recent[-1].object_id
                record = tracker.memory.objects.get(target_id) if target_id is not None else None
                now = time.perf_counter()
                instant = 1.0 / max(now - last_wall, 1e-6)
                fps = instant if not fps else 0.9 * fps + 0.1 * instant
                last_wall = now
                cv2.putText(frame, "LIVING ROOM | MOVABLE CAMERA", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 242, 250), 2, cv2.LINE_AA)
                cv2.putText(frame, "MuJoCo room geometry -> ObjectHunter tracker", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (175, 200, 220), 1, cv2.LINE_AA)
                if record is not None:
                    pos = record.last_reliable_position if record.lifecycle == Lifecycle.DORMANT and record.last_reliable_position is not None else record.position
                    cv2.putText(frame, f"TARGET {target}  #{record.object_id}  ({pos[0]:.2f}, {pos[1]:.2f}) m", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 245, 130), 2, cv2.LINE_AA)
                self._put((frame, record, tracker, fps, latency_ms))
                frame_index += 1
                time.sleep(max(0.0, 1.0 / 30.0 - (time.perf_counter() - now)))
        except Exception as exc:
            self._put_status(f"Room worker error: {type(exc).__name__}: {exc}")

    def _put(self, value) -> None:
        try: self.latest.get_nowait()
        except queue.Empty: pass
        try: self.latest.put_nowait(value)
        except queue.Full: pass

    def _put_status(self, value: str) -> None:
        self.root.after(0, lambda: self.status_var.set(value))

    def _refresh(self) -> None:
        try:
            frame, record, tracker, fps, latency = self.latest.get_nowait()
            view = cv2.resize(frame, (1000, 562))
            image = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            self.image = ImageTk.PhotoImage(Image.fromarray(image))
            self.image_label.configure(image=self.image)
            if record is None:
                self.location_var.set("Waiting for target")
                self.lifecycle_var.set("NOT FOUND")
            else:
                pos = record.last_reliable_position if record.lifecycle == Lifecycle.DORMANT and record.last_reliable_position is not None else record.position
                label = "LAST KNOWN" if record.lifecycle == Lifecycle.DORMANT else record.lifecycle.value
                self.location_var.set(f"({pos[0]:.3f}, {pos[1]:.3f}) m")
                self.lifecycle_var.set(f"#{record.object_id} {record.class_name} - {label}")
            self.metrics_var.set(f"FPS {fps:.1f}   Latency {latency:.2f} ms   Re-ID {tracker.memory.total_reidentifications}")
            self.status_var.set("Room simulation running")
        except queue.Empty:
            pass
        self.root.after(40, self._refresh)

    def close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    LivingRoomLab().root.mainloop()
