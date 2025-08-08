from ultralytics import YOLO
import numpy as np
from core.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    ENABLE_TTA,
    TTA_SCALES,
    ENABLE_HFLIP_TTA,
    TTA_MERGE_IOU,
)
from core.logger import get_logger
import os

logger = get_logger(__name__)
yolo_model = None

def load_yolo_model():
    global yolo_model
    try:
        yolo_model = YOLO(MODEL_PATH)
        logger.info(f"YOLO model loaded successfully from: {MODEL_PATH}")
    except Exception:
        logger.exception(f"Failed to load YOLO model from: {MODEL_PATH}")
        raise RuntimeError("YOLO model loading failed. Check the model path or file.")

def _merge_detections(all_dets, iou_thresh: float):
    """Merge detections from a flat list of boxes [x1,y1,x2,y2,conf] using IoU-based suppression."""
    if not all_dets:
        return []
    # Sort by conf desc
    all_dets.sort(key=lambda b: b[4], reverse=True)
    merged = []
    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    for box in all_dets:
        keep = True
        for m in merged:
            if iou(box, m) >= iou_thresh:
                keep = False
                break
        if keep:
            merged.append(box)
    return merged


def perform_yolo_inference(image_data: bytes):
    if yolo_model is None:
        load_yolo_model() # Ensure model is loaded before inference
    if not ENABLE_TTA:
        return yolo_model(image_data, conf=CONFIDENCE_THRESHOLD)

    # Basic TTA: scales and optional horizontal flip
    try:
        scales = [float(s.strip()) for s in TTA_SCALES.split(",") if s.strip()]
    except Exception:
        scales = [1.0]

    base_img = image_data
    all_boxes = []
    for s in scales:
        # Ultralytics API accepts imgsz; but we only have raw bytes here; call directly and rely on internal resize
        res = yolo_model(base_img, conf=CONFIDENCE_THRESHOLD)
        boxes = res[0].boxes.xyxy.cpu().numpy()
        scores = [b.conf.item() for b in res[0].boxes]
        for (x1, y1, x2, y2), sc in zip(boxes, scores):
            all_boxes.append([float(x1), float(y1), float(x2), float(y2), float(sc)])

        if ENABLE_HFLIP_TTA:
            # Horizontal flip TTA: use numpy flip and run again; then unflip boxes back
            import cv2
            arr = np.frombuffer(base_img, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                flipped = cv2.flip(img, 1)
                res_f = yolo_model(flipped, conf=CONFIDENCE_THRESHOLD)
                boxes_f = res_f[0].boxes.xyxy.cpu().numpy()
                scores_f = [b.conf.item() for b in res_f[0].boxes]
                # Unflip boxes
                for (x1, y1, x2, y2), sc in zip(boxes_f, scores_f):
                    ux1 = float(w - x2); ux2 = float(w - x1)
                    all_boxes.append([ux1, float(y1), ux2, float(y2), float(sc)])

    merged = _merge_detections(all_boxes, TTA_MERGE_IOU)

    class DummyXY:
        def __init__(self, arr):
            self._arr = arr
        def cpu(self):
            return self
        def numpy(self):
            return self._arr

    class DummyBoxes:
        def __init__(self, boxes):
            arr = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=float)
            self.xyxy = DummyXY(arr)
            self.conf = np.array([b[4] for b in boxes])
        def __iter__(self):
            # Provide minimal iterable of per-box dicts with .conf
            for i in range(len(self.conf)):
                yield type("_B", (), {"conf": self.conf[i]})

    class DummyResult:
        def __init__(self, boxes):
            self.boxes = DummyBoxes(boxes)
        def plot(self):
            # Fall back to a single-pass plot if needed in callers
            return yolo_model(base_img, conf=CONFIDENCE_THRESHOLD)[0].plot()

    return [DummyResult(merged)]

print("MODEL_PATH:", MODEL_PATH)
print("Absolute path:", os.path.abspath(MODEL_PATH))
print("Exists?", os.path.exists(MODEL_PATH))