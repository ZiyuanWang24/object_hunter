from __future__ import annotations

import time
import cv2
import numpy as np
from .types import Detection


class PerceptionPipeline:
    def __init__(self, detector, tracker, geometry=None, embedder=None, embedding_refresh=8):
        self.detector, self.tracker, self.geometry, self.embedder = detector, tracker, geometry, embedder
        self.embedding_refresh, self.frame_index = embedding_refresh, 0
        self.metrics = {"frames":0, "inference_ms":0.0, "latency_ms":0.0, "fps":0.0, "gpu_memory_mb":0.0}
        self._last_frame_time = None

    def _crops(self, frame, detections):
        crops=[]
        for d in detections:
            x1,y1,x2,y2 = map(int, d.bbox); x1=max(0,x1); y1=max(0,y1); x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
            crop=frame[y1:y2,x1:x2]
            if crop.size:
                if d.mask is not None and d.mask.shape[:2] == frame.shape[:2]:
                    crop = crop.copy(); crop[d.mask[y1:y2, x1:x2] == 0] = 0
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            else: crops.append(frame[:1,:1])
        return crops

    def process(self, frame, timestamp=None, detection_filter=None):
        t0=time.perf_counter(); timestamp=time.time() if timestamp is None else timestamp
        d0=time.perf_counter(); detections=self.detector.detect(frame, timestamp); d1=time.perf_counter()
        if detection_filter is not None:
            detections=list(detection_filter(detections))
        for d in detections:
            if d.contact_pixel is None: d.contact_pixel=(float((d.bbox[0]+d.bbox[2])/2), float(d.bbox[3]))
            if self.geometry is not None: d.world_position=np.asarray(self.geometry.pixel_to_world(d.contact_pixel), float)
        if self.embedder is not None and (self.frame_index % self.embedding_refresh == 0 or any(d.embedding is None for d in detections)):
            embeddings=self.embedder.embed(self._crops(frame, detections))
            for d,e in zip(detections, embeddings): d.embedding=e
        objects=self.tracker.update(detections, timestamp); self.frame_index += 1
        now=time.perf_counter(); self.metrics["frames"] += 1; self.metrics["inference_ms"] = (d1-d0)*1000; self.metrics["latency_ms"]=(now-t0)*1000
        try:
            import torch
            if torch.cuda.is_available(): self.metrics["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1e6
        except ImportError: pass
        if self._last_frame_time: self.metrics["fps"]=.9*self.metrics["fps"]+.1/(timestamp-self._last_frame_time) if timestamp>self._last_frame_time else self.metrics["fps"]
        self._last_frame_time=timestamp
        return detections, objects
