import os
from dotenv import load_dotenv

load_dotenv()

# Application Constants
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "../runs/detect/librarea_yolov11_run/weights/best.pt"
)
MAX_SMALL_DIM = 1200
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", 0.4))
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)