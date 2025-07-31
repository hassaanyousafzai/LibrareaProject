from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from core.tasks_store import upload_tasks
from core.cache import processed_images_cache
from services.cleanup import cleanup_task_resources
from core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/upload-image/{image_id}/cancel")
async def cancel_image_upload(image_id: str):
    task = upload_tasks.get(image_id)
    if not task:
        raise HTTPException(status_code=404, detail="Upload task not found.")

    # Allow cancellation only if the task is 'queued' or 'processing'
    if task["status"] not in ["queued", "processing"]:
        return JSONResponse(
            status_code=400,
            content={"message": f"Cannot cancel a task with status '{task['status']}'."}
        )

    task["cancel"] = True
    task["status"] = "cancelling"
    task["message"] = "Cancellation request received. The task will be terminated shortly."
    logger.info(f"Cancellation requested for upload task {image_id}.")
    
    return JSONResponse(
        status_code=202,
        content={"message": task["message"]}
    )

@router.get("/upload-image/{image_id}/status")
async def get_image_upload_status(image_id: str):
    # This endpoint is deprecated and will be removed in a future version.
    # Please use the new GET /status/{image_id} endpoint for real-time updates.
    logger.warning(f"DeprecationWarning: '/upload-image/{image_id}/status' is deprecated. Use '/status/{image_id}' instead.")
    
    task = upload_tasks.get(image_id)
    
    # If the task is not in our live tracking dictionary...
    if not task:
        # It might be because it's already completed and its result is in the cache.
        if image_id in processed_images_cache:
            logger.info(f"Task {image_id} not in live tasks, but found in cache. Reporting as completed.")
            result = processed_images_cache.get(image_id)
            
            # Check if result indicates no spines were detected
            if isinstance(result, dict) and "message" in result:
                return {
                    "status": "completed", 
                    "result": result,
                    "spine_detection_summary": {
                        "spines_found": False,
                        "total_detections": result.get("total_detections", 0),
                        "spine_count": len(result.get("spine_detections", [])),
                        "rejected_count": len(result.get("rejected_detections", []))
                    }
                }
            else:
                # Normal case with shelf data
                spine_count = sum(len(shelf_data[list(shelf_data.keys())[0]]) 
                                 for shelf_data in (result or []))
                return {
                    "status": "completed", 
                    "result": result,
                    "spine_detection_summary": {
                        "spines_found": True,
                        "spine_count": spine_count,
                        "shelf_count": len(result or [])
                    }
                }
        
        # If it's not in the cache either, then it's an unknown task.
        logger.warning(f"Status requested for unknown task ID: {image_id}")
        raise HTTPException(status_code=404, detail="Upload task not found.")

    status = task.get("status")

    if status == "completed":
        logger.info(f"Task {image_id} is completed. Returning result and cleaning up.")
        result = processed_images_cache.get(image_id)
        
        # Check if result indicates no spines were detected
        if isinstance(result, dict) and "message" in result:
            response = {
                "status": "completed", 
                "result": result,
                "spine_detection_summary": {
                    "spines_found": False,
                    "total_detections": result.get("total_detections", 0),
                    "spine_count": len(result.get("spine_detections", [])),
                    "rejected_count": len(result.get("rejected_detections", []))
                }
            }
        else:
            # Normal case with shelf data
            spine_count = sum(len(shelf_data[list(shelf_data.keys())[0]]) 
                             for shelf_data in (result or []))
            response = {
                "status": "completed", 
                "result": result,
                "spine_detection_summary": {
                    "spines_found": True,
                    "spine_count": spine_count,
                    "shelf_count": len(result or [])
                }
            }
        
        # Once the result is retrieved, we can clean up the task entry.
        cleanup_task_resources(image_id)
        return response
    
    if status == "processing":
        return {"status": "processing", "detail": "The image is currently being processed."}
        
    if status == "failed":
        return {"status": "failed", "detail": task.get("error", "An unknown error occurred.")}

    # For any other status (like 'cancelled' or unexpected values),
    # return the status without a specific detail message.
    return {"status": status, "detail": None}
