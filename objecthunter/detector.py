from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import cv2
from .types import Detection
from .geometry import mask_contact_point


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]: ...


class YOLOSegDetector(Detector):
    def __init__(self, model_path="yolo11n-seg.pt", confidence=.35, iou=.45, imgsz=640, device="auto", classes=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("YOLO backend requires ultralytics. Install the perception extras.") from e
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not load YOLO weights '{model_path}'. Place the pretrained file locally "
                "or allow access to the Ultralytics asset host, then retry."
            ) from e
        self.confidence, self.iou, self.imgsz = confidence, iou, imgsz
        self.device = None if device == "auto" else device
        self.classes = classes

    def detect(self, frame, timestamp):
        result = self.model.predict(frame, conf=self.confidence, iou=self.iou, imgsz=self.imgsz, device=self.device, classes=self.classes, verbose=False)[0]
        names = result.names; detections = []
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []
        masks = result.masks.data.cpu().numpy() if result.masks is not None else [None] * len(boxes)
        for box, conf, c, mask in zip(boxes, confs, cls, masks):
            if mask is not None:
                mask = cv2.resize((mask > .5).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            detections.append(Detection(timestamp, names[int(c)], float(conf), tuple(map(float, box)), mask, mask_contact_point(mask) if mask is not None else (float((box[0]+box[2])/2), float(box[3]))))
        return detections


class MockDetector(Detector):
    def __init__(self, detections_by_frame): self.detections_by_frame = detections_by_frame; self.i = 0
    def detect(self, frame, timestamp):
        out = self.detections_by_frame[min(self.i, len(self.detections_by_frame)-1)] if self.detections_by_frame else []
        self.i += 1
        return [Detection(timestamp=d.timestamp if d.timestamp is not None else timestamp, class_name=d.class_name, confidence=d.confidence, bbox=d.bbox, mask=d.mask, contact_pixel=d.contact_pixel, world_position=None if d.world_position is None else d.world_position.copy(), embedding=None if d.embedding is None else d.embedding.copy()) for d in out]
