from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from core.tasks_store import upload_tasks
from core.cache import processed_images_cache
from core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/status/{image_id}", tags=["Status"])
async def get_task_status(image_id: str):
    """
    Provides real-time status updates for the image processing task.
    """
    task = upload_tasks.get(image_id)

    # 1. Check live tasks first
    if task:
        return JSONResponse(content=task)

    # 2. If not in live tasks, check the cache for completed tasks
    if image_id in processed_images_cache:
        cached_result = processed_images_cache[image_id]
        
        # Format a final "completed" status response from cache
        if isinstance(cached_result, dict) and "message" in cached_result:
            # Case: Completed but no spines found
            final_status = {
                "status": "completed",
                "message": cached_result["message"],
                "result": cached_result
            }
        else:
            # Case: Completed successfully with shelves
            final_status = {
                "status": "completed",
                "message": "Processing complete. Found shelves and books.",
                "result": cached_result
            }
        return JSONResponse(content=final_status)

    # 3. If not found anywhere, the task is unknown
    raise HTTPException(status_code=404, detail=f"Task with image_id '{image_id}' not found.")
