from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api import upload, organize
from services.yolo import load_yolo_model
from core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(upload.router)
app.include_router(organize.router)

# Load YOLO model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup event triggered.")
    load_yolo_model()
    logger.info("Application startup complete.")