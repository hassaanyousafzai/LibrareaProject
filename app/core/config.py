import os
from dotenv import load_dotenv

load_dotenv()

# Application Constants
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "../runs/detect/best.pt"
)
MAX_SMALL_DIM = 1200
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", 0.25))
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

# Spine Detection Configuration
SPINE_MIN_ASPECT_RATIO = float(os.getenv("SPINE_MIN_ASPECT_RATIO", 2.5))  # height/width ratio
SPINE_MAX_WIDTH_RATIO = float(os.getenv("SPINE_MAX_WIDTH_RATIO", 0.95))   # width/image_width ratio
SPINE_MIN_HEIGHT_RATIO = float(os.getenv("SPINE_MIN_HEIGHT_RATIO", 0.1))  # height/image_height ratio
SPINE_MIN_WIDTH_PX = int(os.getenv("SPINE_MIN_WIDTH_PX", 30))  # minimum width in pixels for readability

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)