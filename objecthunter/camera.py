from __future__ import annotations

import threading
import time
import cv2


class LatestFrameCamera:
    def __init__(self, source=0, width=1280, height=720):
        self.source, self.width, self.height = source, width, height; self.cap = None; self.latest = None; self.lock = threading.Lock(); self.running = False; self.thread = None
    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened(): raise RuntimeError(f"Could not open camera/video source: {self.source}")
        if isinstance(self.source, int): self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    def start(self):
        if self.cap is None: self.open()
        self.running = True; self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start(); return self
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok: self.running = False; break
            with self.lock: self.latest = (time.perf_counter(), frame)
    def read(self):
        with self.lock: return self.latest
    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1)
        if self.cap: self.cap.release()
