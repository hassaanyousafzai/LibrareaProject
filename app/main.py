import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
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
    """
    Handles Pydantic validation errors to return simple, user-friendly messages.
    """
    try:
        error = exc.errors()[0]
    except IndexError:
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation error with unknown structure."},
        )

    error_type = error.get("type")
    
    if error_type == 'json_invalid':
        parser_message = error.get('msg')
        detailed_message = f"Invalid JSON syntax: {parser_message}. Please correct the formatting and try again."
        return JSONResponse(status_code=400, content={"detail": detailed_message})

    # Proceed with field-specific errors
    field = error.get("loc", ["body", "unknown"])[-1]

    # Define valid values for suggestion messages
    valid_sort_by = ["author", "title", "genre", "height"]
    valid_sort_order = ["asc", "desc"]

    if field == 'sort_by':
        detailed_message = f"The 'sort_by' field cannot be empty. It must be one of: {', '.join(valid_sort_by)}."
    elif field == 'sort_order':
        detailed_message = f"The 'sort_order' field cannot be empty. It must be 'asc' or 'desc'."
    elif field == 'image_id':
        detailed_message = "The 'image_id' field is required and cannot be empty."
    elif field == 'shelf_number':
        detailed_message = "The 'shelf_number' must be a valid integer and cannot be empty."
    else:
        # Fallback for any other validation error
        detailed_message = f"There was an error with the '{field}' field: {error.get('msg')}"

    return JSONResponse(
        status_code=422,  # Unprocessable Entity
        content={"detail": detailed_message},
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
                        if seen_keys[key] != value:
                            return JSONResponse(
                                status_code=400,
                                content={"detail": f"Duplicate key '{key}' found with conflicting values."}
                            )
                        return JSONResponse(
                            status_code=400,
                            content={"detail": f"Duplicate key found in JSON body: {key}"}
                        )
                    seen_keys[key] = value

            except json.JSONDecodeError:
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
