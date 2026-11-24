from __future__ import annotations

import numpy as np
from .association import associate, gallery_similarity
from .kalman import ConstantVelocityKalman
from .memory import ObjectMemory
from .types import Detection, Lifecycle


class _TrackView:
    def __init__(self, record, kalman): self.record, self.kalman = record, kalman
    @property
    def class_name(self): return self.record.class_name
    @property
    def gallery(self): return self.record.gallery
    @property
    def last_detection(self): return self.record.last_detection


class MultiObjectTracker:
    def __init__(self, memory=None, **kwargs):
        self.cfg = {"confirm_hits":3, "dormant_timeout_s":3.0, "dormant_ttl_s":3600.0, "max_gallery":20, "mahalanobis_gate":9.21, "appearance_weight":.55, "motion_weight":.35, "iou_weight":.10, "appearance_threshold":.55, "reid_threshold":.62, "ambiguity_margin":.08, "process_noise":1.0, "measurement_noise":.02}
        self.cfg.update(kwargs)
        self.memory = memory or ObjectMemory(**{k:self.cfg[k] for k in ["confirm_hits","dormant_timeout_s","dormant_ttl_s","max_gallery"]})
        self.filters: dict[int, ConstantVelocityKalman] = {}
        self.last_timestamp: float | None = None

    def _position(self, d): return np.asarray(d.world_position if d.world_position is not None else d.center, float)

    def update(self, detections: list[Detection], timestamp: float):
        dt = 0 if self.last_timestamp is None else max(0, timestamp-self.last_timestamp)
        active = [o for o in self.memory.objects.values() if o.lifecycle in (Lifecycle.TENTATIVE, Lifecycle.TRACKED, Lifecycle.OCCLUDED)]
        views = []
        for o in active:
            f = self.filters.setdefault(o.object_id, ConstantVelocityKalman(o.state[:2], o.covariance, self.cfg["process_noise"], self.cfg["measurement_noise"]))
            f.x, f.P = o.state.copy(), o.covariance.copy(); f.predict(dt); o.state, o.covariance = f.x.copy(), f.P.copy()
            views.append(_TrackView(o, f))
        matches, unmatched_tracks, unmatched_dets, _ = associate(detections, views, appearance_weight=self.cfg["appearance_weight"], motion_weight=self.cfg["motion_weight"], iou_weight=self.cfg["iou_weight"], gate=self.cfg["mahalanobis_gate"], appearance_threshold=self.cfg["appearance_threshold"], ambiguity_margin=self.cfg["ambiguity_margin"])
        matched_ids = set()
        for ti, di in matches:
            view, det = views[ti], detections[di]; view.kalman.update(self._position(det)); view.record.state, view.record.covariance = view.kalman.x.copy(), view.kalman.P.copy()
            self.memory.observe(view.record, det, timestamp, view.kalman.x, view.kalman.P); matched_ids.add(view.record.object_id)
        for ti in unmatched_tracks:
            o = views[ti].record; self.memory.miss(o, timestamp, dt)
        # Dormant re-identification intentionally uses gallery/class only; old position is not a gate.
        dormant = [o for o in self.memory.objects.values() if o.lifecycle == Lifecycle.DORMANT and o.object_id not in matched_ids]
        consumed = set()
        for di in list(unmatched_dets):
            det = detections[di]; candidates = [(gallery_similarity(det.embedding, o.gallery), o) for o in dormant if o.class_name == det.class_name and det.embedding is not None and o.gallery]
            candidates = sorted([(s,o) for s,o in candidates if s is not None], reverse=True, key=lambda x:x[0])
            if candidates and candidates[0][0] >= self.cfg["reid_threshold"] and (len(candidates)==1 or candidates[0][0]-candidates[1][0] >= self.cfg["ambiguity_margin"]):
                o = candidates[0][1]; f = ConstantVelocityKalman(self._position(det), process_noise=self.cfg["process_noise"], measurement_noise=self.cfg["measurement_noise"]); self.filters[o.object_id] = f
                self.memory.observe(o, det, timestamp, f.x, f.P, reidentified=True); matched_ids.add(o.object_id); consumed.add(di)
        for di in unmatched_dets:
            if di in consumed: continue
            det = detections[di]; p = self._position(det); f = ConstantVelocityKalman(p, process_noise=self.cfg["process_noise"], measurement_noise=self.cfg["measurement_noise"]); o = self.memory.create(det.class_name, timestamp, f.x, f.P, det); self.filters[o.object_id] = f
            self.memory.observe(o, det, timestamp, f.x, f.P)
        self.memory.prune(timestamp); self.last_timestamp = timestamp
        return list(self.memory.objects.values())
