import os
import cv2
import uuid
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from core.config import UPLOAD_DIR
from core.logger import get_logger
from core.cache import processed_images_cache
from core.tasks_store import upload_tasks
from services.yolo import perform_yolo_inference
from services.gemini_ocr import get_book_metadata_from_spine
from services.google_books import enrich_book_metadata
from services.image_processing import (
    preprocess_multi_scale, 
    group_books_into_shelves, 
    filter_spine_detections,
    merge_overlapping_boxes,
    is_image_blurred
)
from services.cleanup import cleanup_task_resources

router = APIRouter()
logger = get_logger(__name__)

def is_blank_spine(metadata: dict) -> bool:
    """
    Determines if a spine should be considered blank and filtered out.
    
    Args:
        metadata: The combined metadata dictionary from OCR and enrichment
        
    Returns:
        True if the spine should be filtered out, False otherwise
    """
    title = metadata.get("Book Name", "").strip()
    author = metadata.get("Author", "").strip()
    
    # Consider blank if both title and author are empty
    if not title and not author:
        return True
        
    # Consider blank if title is very short (likely OCR noise)
    if title and len(title) < 3:
        return True
        
    # Consider blank if author is very short (likely OCR noise)
    if author and len(author) < 2:
        return True
    
    return False

def run_image_processing_task(image_id: str, contents: bytes, original_filename: str):
    """The main function to be run in the background for processing an uploaded image."""
    
    upload_tasks[image_id] = {
        "status": "processing",
        "message": f"Starting processing for '{original_filename}'...",
        "filename": original_filename,
        "cancel": False
    }

    try:
        task_state = upload_tasks.get(image_id)
        if not task_state:
            logger.warning(f"Task {image_id} started but no state found.")
            return

        logger.info(f"Starting image processing for task {image_id} ('{original_filename}').")

        upload_tasks[image_id]["message"] = "Preprocessing image..."
        img_full_norm, img_small_norm, scale = preprocess_multi_scale(contents)
        original_image_height, original_image_width = img_full_norm.shape[:2]
        
        if task_state.get("cancel"):
            logger.info(f"Cancellation detected for task {image_id} after preprocessing.")
            upload_tasks[image_id].update({"status": "cancelled", "message": "Task was cancelled by user."})
            cleanup_task_resources(image_id)
            return
        
        upload_tasks[image_id]["message"] = "Detecting books with YOLO model..."
        img_small_uint8 = (img_small_norm * 255).astype(np.uint8)
        results = perform_yolo_inference(img_small_uint8)
        
        if task_state.get("cancel"):
            logger.info(f"Cancellation detected for task {image_id} after YOLO inference.")
            upload_tasks[image_id].update({"status": "cancelled", "message": "Task was cancelled by user."})
            cleanup_task_resources(image_id)
            return

        boxes_small = results[0].boxes.xyxy.cpu().numpy()
        boxes_full = (boxes_small / scale).tolist()
        scores = [b.conf.item() for b in results[0].boxes]

        merged_boxes, merged_scores = merge_overlapping_boxes(boxes_full, scores)
        logger.info(f"Initial detections: {len(boxes_full)}, After merging: {len(merged_boxes)}")

        preliminary_detections = []
        for idx, (box, score) in enumerate(zip(merged_boxes, merged_scores), start=1):
            book_id = f"{image_id}_book_{idx}"
            preliminary_detections.append({
                "book_id": book_id,
                "bbox": box,
                "confidence": f"{round(float(score) * 100)}%",
            })

        upload_tasks[image_id]["message"] = "Filtering detections to identify book spines..."
        spine_detections, rejected_detections = filter_spine_detections(
            preliminary_detections, original_image_width, original_image_height
        )
        
        logger.info(f"YOLO detected {len(preliminary_detections)} objects, "
                   f"{len(spine_detections)} identified as spines, "
                   f"{len(rejected_detections)} rejected as non-spines")
        
        if not spine_detections:
            logger.info(f"No book spines detected in image {image_id}")
            no_spines_message = "No book spines were detected in the image."
            upload_tasks[image_id].update({"status": "completed", "message": no_spines_message})
            processed_images_cache[image_id] = {
                "message": no_spines_message,
                "total_detections": len(preliminary_detections),
                "rejected_detections": rejected_detections,
                "spine_detections": []
            }
            return

        annotated = results[0].plot()
        annotated_full = cv2.resize(annotated, (original_image_width, original_image_height), interpolation=cv2.INTER_LINEAR)
        _, buf = cv2.imencode('.jpg', annotated_full)
        
        annotated_filename = f"{image_id}.jpg"
        annotated_path = os.path.join(UPLOAD_DIR, annotated_filename)
        with open(annotated_path, 'wb') as f:
            f.write(buf.tobytes())
        logger.info(f"Annotated image saved to {annotated_path}")

        img_full_uint8 = (img_full_norm * 255).astype(np.uint8)
        spine_dir = os.path.join(UPLOAD_DIR, "cropped_spines", image_id)
        os.makedirs(spine_dir, exist_ok=True)

        total_spines = len(spine_detections)
        detections = []
        filtered_blank_spines = 0
        
        for idx, spine_detection in enumerate(spine_detections, start=1):
            if task_state.get("cancel"):
                logger.info(f"Cancellation detected for task {image_id} during book processing.")
                upload_tasks[image_id].update({"status": "cancelled", "message": "Task was cancelled by user."})
                cleanup_task_resources(image_id)
                return
            
            upload_tasks[image_id]["message"] = f"Extracting text from spine {idx} of {total_spines}..."

            box = spine_detection["bbox"]
            x1, y1, x2, y2 = map(int, box)
            crop = img_full_uint8[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning(f"Skipping empty crop for box {idx}: {box}")
                continue

            crop_name = f"{image_id}_spine_{idx}.jpg"
            crop_path = os.path.join(spine_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            with open(crop_path, 'rb') as img_f:
                crop_bytes = img_f.read()
            
            ocr_meta = get_book_metadata_from_spine(crop_bytes, crop_name)
            title = ocr_meta.get("Book Name") or ""
            author = ocr_meta.get("Author") or ""
            series_name = ocr_meta.get("Series_Name") or ""
            
            upload_tasks[image_id]["message"] = f"Enriching metadata for spine {idx} of {total_spines}..."
            enrichment = enrich_book_metadata(title, author, series_name, crop_name)

            if (not author or author.strip() == "") and enrichment.get("author_from_api"):
                ocr_meta["Author"] = enrichment["author_from_api"]
                logger.info(f"[UPLOAD] Updated Author for {crop_name}: OCR='{author}' → API='{enrichment['author_from_api']}'")

            enrichment_clean = {k: v for k, v in enrichment.items() if k != "author_from_api"}
            combined_meta = {**ocr_meta, **enrichment_clean}
            
            # Check if this spine should be filtered out as blank
            if is_blank_spine(combined_meta):
                filtered_blank_spines += 1
                logger.info(f"[UPLOAD] Filtering out blank spine {crop_name}: title='{combined_meta.get('Book Name', '')}', author='{combined_meta.get('Author', '')}'")
                continue
            
            book_id = spine_detection["book_id"]
            detections.append({
                "book_id": book_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": spine_detection["confidence"],
                "crop_path": crop_path,
                "metadata": combined_meta
            })

        upload_tasks[image_id]["message"] = "Grouping books into shelves..."
        shelf_mapped_books = group_books_into_shelves(detections, original_image_height)
        processed_images_cache[image_id] = shelf_mapped_books
        
        final_message = f"Processing complete. Found {len(detections)} books on {len(shelf_mapped_books)} shelves."
        if filtered_blank_spines > 0:
            final_message += f" Filtered out {filtered_blank_spines} blank spines."
        
        upload_tasks[image_id].update({"status": "completed", "message": final_message})
        logger.info(f"Image processing for task {image_id} completed successfully. Filtered {filtered_blank_spines} blank spines.")

    except ValueError as ve:
        error_message = f"Processing failed: {ve}"
        logger.error(error_message)
        upload_tasks[image_id].update({"status": "failed", "message": error_message, "error": str(ve)})
    except Exception as e:
        logger.exception(f"An error occurred during image processing for task {image_id}: {e}")
        error_message = "An unexpected error occurred during processing."
        upload_tasks[image_id].update({"status": "failed", "message": error_message, "error": str(e)})

@router.post("/upload-image/")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()
    
    # Perform quick, synchronous checks first
    image_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=415, message="Unsupported file format")

    is_blurred, fft_score = is_image_blurred(img)
    if is_blurred:
        raise HTTPException(
            status_code=400, 
            message=f"The image is too blurry to process (FFT score: {fft_score:.2f})"
        )

    # If checks pass, proceed to background processing
    image_id = str(uuid.uuid4())
    
    upload_tasks[image_id] = {
        "status": "queued",
        "message": "Upload received, task is queued to start.",
        "filename": file.filename,
        "cancel": False
    }
    
    background_tasks.add_task(run_image_processing_task, image_id, contents, file.filename)
    
    return JSONResponse(
        status_code=202,
        content={"message": "Upload accepted and is being processed.", "image_id": image_id}
    )