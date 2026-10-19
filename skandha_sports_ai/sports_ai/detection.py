"""Detection + tracking wrapper around Ultralytics YOLO (pretrained, COCO classes)."""
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    box: Tuple[float, float, float, float]   # x1, y1, x2, y2
    conf: float
    cls_id: int
    track_id: Optional[int] = None           # None when the tracker did not assign an ID


def resolve_device(pref="auto"):
    import torch
    if pref != "auto":
        return pref
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_results(result, person_cls=0, ball_cls=32):
    """Convert one Ultralytics Results object to (persons, balls). Safe for empty frames / missing IDs."""
    persons: List[Detection] = []
    balls: List[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return persons, balls
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    ids = None if boxes.id is None else boxes.id.cpu().numpy().astype(int)
    for i in range(len(xyxy)):
        det = Detection(tuple(float(v) for v in xyxy[i]), float(conf[i]), int(cls[i]),
                        int(ids[i]) if ids is not None else None)
        if det.cls_id == person_cls:
            persons.append(det)
        elif det.cls_id == ball_cls:
            balls.append(det)
    return persons, balls


class Detector:
    """Holds the YOLO model. Create a NEW Detector for every run so tracker state starts clean."""

    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = YOLO(cfg.model_name)          # auto-downloads pretrained weights

    def detect(self, frame):
        """Single-frame detection WITHOUT tracking (used for the Phase 2 prototype)."""
        r = self.model.predict(frame, conf=self.cfg.conf, classes=self.cfg.classes, imgsz=self.cfg.imgsz,
                               device=self.device, verbose=False)[0]
        return parse_results(r, self.cfg.person_cls, self.cfg.ball_cls)

    def track(self, frame):
        """Detection + tracking. persist=True keeps tracker state between consecutive frames."""
        r = self.model.track(frame, persist=True, tracker=self.cfg.tracker, conf=self.cfg.conf,
                             classes=self.cfg.classes, imgsz=self.cfg.imgsz, device=self.device,
                             verbose=False)[0]
        return parse_results(r, self.cfg.person_cls, self.cfg.ball_cls)
