from __future__ import annotations

import json
import numpy as np
from dataclasses import asdict
from pathlib import Path
from .types import TrackRecord, Lifecycle


class ObjectMemory:
    def __init__(self, confirm_hits=3, dormant_timeout_s=3.0, dormant_ttl_s=3600, max_gallery=20):
        self.confirm_hits, self.dormant_timeout_s, self.dormant_ttl_s, self.max_gallery = confirm_hits, dormant_timeout_s, dormant_ttl_s, max_gallery
        self.objects: dict[int, TrackRecord] = {}
        self.next_id = 1
        self.events: list[str] = []
        self.total_reidentifications = 0

    def create(self, class_name, timestamp, state, covariance, detection):
        obj = TrackRecord(self.next_id, class_name, timestamp, timestamp, state[:2].copy(), state.copy(), covariance.copy(), last_detection=detection)
        self.objects[obj.object_id] = obj; self.next_id += 1; self.events.append(f"created object {obj.object_id}")
        return obj

    def observe(self, obj, detection, timestamp, state, covariance, reliable=True, reidentified=False):
        obj.state, obj.covariance = state.copy(), covariance.copy(); obj.last_observed_at = timestamp; obj.last_detection = detection
        obj.detection_confidence = detection.confidence; obj.observation_count += 1; obj.missed_observations = 0
        obj.trajectory.append((timestamp, float(obj.position[0]), float(obj.position[1])))
        if detection.world_position is not None and reliable: obj.last_reliable_position = detection.world_position.copy()
        if detection.embedding is not None and reliable:
            obj.gallery.append(detection.embedding.copy()); obj.gallery = obj.gallery[-self.max_gallery:]
        if reidentified:
            obj.reidentification_count += 1; self.total_reidentifications += 1; self.events.append(f"re-identified object {obj.object_id}")
        if obj.observation_count >= self.confirm_hits and obj.lifecycle == Lifecycle.TENTATIVE: obj.lifecycle = Lifecycle.TRACKED
        elif obj.lifecycle in (Lifecycle.OCCLUDED, Lifecycle.DORMANT): obj.lifecycle = Lifecycle.TRACKED

    def miss(self, obj, timestamp, dt):
        obj.missed_observations += 1
        if obj.lifecycle == Lifecycle.TENTATIVE: return
        if timestamp - obj.last_observed_at >= self.dormant_timeout_s:
            if obj.lifecycle != Lifecycle.DORMANT: self.events.append(f"dormant object {obj.object_id}")
            obj.lifecycle = Lifecycle.DORMANT
        else: obj.lifecycle = Lifecycle.OCCLUDED

    def prune(self, timestamp):
        self.objects = {i:o for i,o in self.objects.items() if timestamp-o.last_observed_at <= self.dormant_ttl_s}

    def reset(self): self.objects.clear(); self.next_id = 1; self.events.clear(); self.total_reidentifications = 0

    def export_json(self, path):
        def conv(v):
            if isinstance(v, np.ndarray): return v.tolist()
            if isinstance(v, Lifecycle): return v.value
            if isinstance(v, TrackRecord): return {k: conv(x) for k,x in asdict(v).items()}
            if isinstance(v, list): return [conv(x) for x in v]
            if isinstance(v, dict): return {k:conv(x) for k,x in v.items()}
            return v
        Path(path).write_text(json.dumps({"next_id": self.next_id, "objects": [conv(o) for o in self.objects.values()]}, indent=2))
