from ultralytics import YOLO
from core.config import MODEL_PATH, CONFIDENCE_THRESHOLD
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

def perform_yolo_inference(image_data: bytes):
    if yolo_model is None:
        load_yolo_model() # Ensure model is loaded before inference
    return yolo_model(image_data, conf=CONFIDENCE_THRESHOLD)

print("MODEL_PATH:", MODEL_PATH)
print("Absolute path:", os.path.abspath(MODEL_PATH))
print("Exists?", os.path.exists(MODEL_PATH))