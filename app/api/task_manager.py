from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from core.tasks_store import upload_tasks
from core.cache import processed_images_cache
from core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/upload-image/{image_id}/status")
async def get_image_upload_status(image_id: str):
    task = upload_tasks.get(image_id)
    
    if not task:
        if image_id in processed_images_cache:
            return {"status": "completed", "result": processed_images_cache.get(image_id)}
        
        return JSONResponse(
            status_code=404,
            content={
                "status": "not_found",
                "detail": "Image data not found for the provided ID. Please upload the image or check the ID for typos."
            }
        )

    status = task.get("status")

    if status == "completed":
        result = processed_images_cache.get(image_id)
        return {"status": "completed", "result": result}
    
    if status == "processing":
        return {"status": "processing", "detail": "The image is currently being processed."}
        
    if status == "failed":
        return {"status": "failed", "detail": task.get("error", "An unknown error occurred.")}

    return {"status": status, "detail": None}

@router.post("/upload-image/{image_id}/cancel")
async def cancel_image_upload(image_id: str):
    task = upload_tasks.get(image_id)
    if not task:
        raise HTTPException(status_code=404, detail="Upload task not found.")

    if task["status"] != "processing":
        return JSONResponse(
            status_code=400,
            content={"message": f"Cannot cancel a task with status '{task['status']}'."}
        )

    task["cancel"] = True
    logger.info(f"Cancellation requested for upload task {image_id}.")
    return {"message": "Cancellation request received. The task will be terminated shortly."}