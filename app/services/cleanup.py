import os
import shutil
from core.logger import get_logger
from core.config import UPLOAD_DIR
from core.cache import processed_images_cache
from core.tasks_store import upload_tasks

logger = get_logger(__name__)

def cleanup_task_resources(image_id: str):
    """Removes files and directories associated with a given task."""
    logger.info(f"Cleaning up resources for task {image_id}.")
    
    # Remove from cache if it exists
    processed_images_cache.pop(image_id, None)

    # Remove the annotated image file
    annotated_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    if os.path.exists(annotated_path):
        os.remove(annotated_path)
        logger.debug(f"Removed annotated image: {annotated_path}")

    # Remove the directory of cropped spines
    spine_dir = os.path.join(UPLOAD_DIR, "cropped_spines", image_id)
    if os.path.isdir(spine_dir):
        shutil.rmtree(spine_dir)
        logger.debug(f"Removed spine directory: {spine_dir}")
    
    # Finally, remove the task from the tracking dictionary
    upload_tasks.pop(image_id, None)
    logger.info(f"Finished cleanup for task {image_id}.")
