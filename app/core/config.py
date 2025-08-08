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
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", 0.50))
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

# Spine Detection Configuration
# Vertical spine parameters
SPINE_MIN_ASPECT_RATIO = float(os.getenv("SPINE_MIN_ASPECT_RATIO", 2.5))  # height/width ratio for vertical spines
SPINE_MAX_WIDTH_RATIO = float(os.getenv("SPINE_MAX_WIDTH_RATIO", 0.95))   # width/image_width ratio
SPINE_MIN_HEIGHT_RATIO = float(os.getenv("SPINE_MIN_HEIGHT_RATIO", 0.1))  # height/image_height ratio

# Horizontal spine parameters
SPINE_MIN_HORIZONTAL_ASPECT_RATIO = float(os.getenv("SPINE_MIN_HORIZONTAL_ASPECT_RATIO", 2.0))  # width/height ratio for horizontal spines
SPINE_MAX_HEIGHT_RATIO = float(os.getenv("SPINE_MAX_HEIGHT_RATIO", 0.3))   # height/image_height ratio for horizontal spines
SPINE_MIN_WIDTH_RATIO = float(os.getenv("SPINE_MIN_WIDTH_RATIO", 0.1))    # width/image_width ratio for horizontal spines

# Common parameters
SPINE_MIN_WIDTH_PX = int(os.getenv("SPINE_MIN_WIDTH_PX", 30))  # minimum width in pixels for readability

# Shelf grouping parameters
SHELF_BOUNDARY_PADDING = float(os.getenv("SHELF_BOUNDARY_PADDING", 0.05))  # 5% padding for shelf boundaries
SHELF_MIN_VERTICAL_BOOKS = int(os.getenv("SHELF_MIN_VERTICAL_BOOKS", 3))  # minimum vertical books to establish a shelf
SHELF_CENTER_TOLERANCE = float(os.getenv("SHELF_CENTER_TOLERANCE", 0.15))  # 15% tolerance for book center point placement
HORIZONTAL_BOOK_TOLERANCE_MULTIPLIER = float(os.getenv("HORIZONTAL_BOOK_TOLERANCE_MULTIPLIER", 1.2))  # slightly increased tolerance for horizontal books

# Deprecated hardcoded ranges removed in favor of clustering

# Horizontal book detection
HORIZONTAL_ASPECT_RATIO = 1.5  # width/height ratio to identify horizontal books
HORIZONTAL_OVERLAP_THRESHOLD = 0.3  # minimum overlap required with shelf

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Layout analysis configuration
ENABLE_LLM_SHELF_COUNT = os.getenv("ENABLE_LLM_SHELF_COUNT", "true").lower() == "true"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
MAX_SHELVES = int(os.getenv("MAX_SHELVES", 6))
SHELF_CLUSTER_PADDING_RATIO = float(os.getenv("SHELF_CLUSTER_PADDING_RATIO", 0.08))
SHELF_SILHOUETTE_MIN = float(os.getenv("SHELF_SILHOUETTE_MIN", 0.2))