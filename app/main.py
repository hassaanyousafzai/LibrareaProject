import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api import upload, organize, task_manager
from services.yolo import load_yolo_model
from core.logger import get_logger
from fastapi.exceptions import RequestValidationError

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup event triggered.")
    load_yolo_model()
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Try to extract total_shelves from the request body if possible
    try:
        body = await request.json()
        image_id = body.get("image_id")
        from core.cache import processed_images_cache
        cached_data = processed_images_cache.get(image_id)
        total_shelves = len(cached_data) if cached_data else None
    except Exception:
        total_shelves = None
    if total_shelves == 1:
        range_msg = "1"
    elif total_shelves:
        range_msg = f"between 1 and {total_shelves}"
    else:
        range_msg = "a positive integer (e.g., 1, 2, 3)"
    return JSONResponse(
        status_code=400,
        content={
            "detail": f"Invalid shelf format. Shelf number must be {range_msg}."
        },
    )

@app.middleware("http")
async def check_duplicate_json_keys_middleware(request: Request, call_next):
    if "application/json" in request.headers.get("content-type", ""):
        body = await request.body()

        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive

        if body:
            try:
                pairs = json.JSONDecoder(object_pairs_hook=lambda x: x).decode(body.decode())
                
                seen_keys = {}
                for key, value in pairs:
                    if key in seen_keys:
                        # If key is already seen, check if the value is different
                        if seen_keys[key] != value:
                            return JSONResponse(
                                status_code=400,
                                content={"detail": f"Duplicate key '{key}' found with conflicting values."}
                            )
                        # If values are the same, it's still a duplicate key scenario
                        return JSONResponse(
                            status_code=400,
                            content={"detail": f"Duplicate key found in JSON body: {key}"}
                        )
                    seen_keys[key] = value

            except json.JSONDecodeError:
                # Let FastAPI's built-in validation handle malformed JSON
                pass

    response = await call_next(request)
    return response

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
app.include_router(task_manager.router)

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
