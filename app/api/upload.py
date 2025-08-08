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
from services.layout_analysis import estimate_shelf_count_with_llm
from services.cleanup import cleanup_task_resources

router = APIRouter()
logger = get_logger(__name__)

def is_blank_spine(metadata: dict) -> bool:
    """
    Determines if a spine should be considered blank and filtered out.
    """
    title = metadata.get("Book Name", "").strip()
    author = metadata.get("Author", "").strip()
    
    # Filter out books where both Author and Book Name are "Unrecognized"
    # if title == "Unrecognized" and author == "Unrecognized":
    #     return True
    
    if not title and not author:
        return True
        
    if title and len(title) < 3:
        return True
        
    if author and len(author) < 2:
        return True
    
    return False

def determine_ocr_action(ocr_meta: dict) -> tuple[str, dict]:
    """
    Determines what action to take based on OCR results.
    
    Returns:
        tuple: (action, processed_metadata)
        action can be: "unrecognized", "use_google_books"
    """
    title = ocr_meta.get("Book Name", "").strip()
    author = ocr_meta.get("Author", "").strip()
    series_name = ocr_meta.get("Series_Name", "").strip()
    
    # Check if OCR failed completely or neither title nor author found
    if not title and not author:
        return "unrecognized", {
            "Author": "Unrecognized",
            "Book Name": "Unrecognized",
            "Series_Name": series_name
        }
    
    # If Book Name exists, use Google Books API (regardless of Author status)
    # Google Books will find missing Author if possible
    if title:
        return "use_google_books", {
            "Author": author if author else "",
            "Book Name": title,
            "Series_Name": series_name
        }
    
    # Only Author found, no Book Name - don't fill Book Name with anything
    return "author_only", {
        "Author": author,
        "Book Name": "",
        "Series_Name": series_name
    }

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
            return

        upload_tasks[image_id]["message"] = "Preprocessing image..."
        img_full_norm, img_small_norm, scale = preprocess_multi_scale(contents)
        original_image_height, original_image_width = img_full_norm.shape[:2]
        
        if task_state.get("cancel"):
            upload_tasks[image_id].update({"status": "cancelled", "message": "Task was cancelled by user."})
            cleanup_task_resources(image_id)
            return
        
        upload_tasks[image_id]["message"] = "Estimating shelf count..."
        # Estimate shelf count with LLM prior (optional)
        img_full_uint8 = (img_full_norm * 255).astype(np.uint8)
        _, buf_full = cv2.imencode('.jpg', img_full_uint8)
        expected_k, llm_conf = estimate_shelf_count_with_llm(buf_full.tobytes())

        upload_tasks[image_id]["message"] = "Detecting books with YOLO model..."
        img_small_uint8 = (img_small_norm * 255).astype(np.uint8)
        results = perform_yolo_inference(img_small_uint8)
        
        if task_state.get("cancel"):
            upload_tasks[image_id].update({"status": "cancelled", "message": "Task was cancelled by user."})
            cleanup_task_resources(image_id)
            return

        boxes_small = results[0].boxes.xyxy.cpu().numpy()
        boxes_full = (boxes_small / scale).tolist()
        scores = [b.conf.item() for b in results[0].boxes]

        merged_boxes, merged_scores = merge_overlapping_boxes(boxes_full, scores)

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
        
        if not spine_detections:
            # Provide more helpful feedback based on what was detected
            if preliminary_detections:
                no_spines_message = f"Found {len(preliminary_detections)} object(s) in the image, but none were identified as book spines. Please ensure the image shows clear book spines and try again."
            else:
                no_spines_message = "No objects were detected in the image. Please ensure the image is clear and contains visible book spines."
            
            upload_tasks[image_id].update({"status": "completed", "message": no_spines_message})
            processed_images_cache[image_id] = {
                "message": no_spines_message,
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

        spine_dir = os.path.join(UPLOAD_DIR, "cropped_spines", image_id)
        os.makedirs(spine_dir, exist_ok=True)

        total_spines = len(spine_detections)
        detections = []
        filtered_blank_spines = 0
        
        for idx, spine_detection in enumerate(spine_detections, start=1):
            if task_state.get("cancel"):
                upload_tasks[image_id].update({"status": "cancelled", "message": "Task was cancelled by user."})
                cleanup_task_resources(image_id)
                return
            
            upload_tasks[image_id]["message"] = f"Extracting text from spine {idx} of {total_spines}..."

            box = spine_detection["bbox"]
            x1, y1, x2, y2 = map(int, box)
            crop = img_full_uint8[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"{image_id}_spine_{idx}.jpg"
            crop_path = os.path.join(spine_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            with open(crop_path, 'rb') as img_f:
                crop_bytes = img_f.read()
            
            ocr_meta = get_book_metadata_from_spine(crop_bytes, crop_name)
            
            # Determine action based on OCR results
            action, processed_meta = determine_ocr_action(ocr_meta)
            
            if action == "unrecognized":
                # OCR failed completely or found neither title nor author
                upload_tasks[image_id]["message"] = f"OCR could not recognize spine {idx} of {total_spines}..."
                combined_meta = processed_meta
                logger.info(f"[DEBUG] Spine {crop_name} marked as unrecognized - no Google Books API call")
                
            elif action == "author_only":
                # OCR found only author, no book name - don't use Google Books API
                upload_tasks[image_id]["message"] = f"OCR found author only for spine {idx} of {total_spines}..."
                combined_meta = processed_meta
                logger.info(f"[DEBUG] Spine {crop_name} has author only - no Google Books API call")
                
            else:  # action == "use_google_books"
                # OCR found book name (with or without author) - use Google Books API
                title = processed_meta["Book Name"]
                author = processed_meta["Author"]
                series_name = processed_meta["Series_Name"]
                
                upload_tasks[image_id]["message"] = f"Enriching metadata for spine {idx} of {total_spines}..."
                enrichment = enrich_book_metadata(title, author, series_name, crop_name)
                
                # Use author from Google Books API if OCR didn't find one
                final_author = author if author else enrichment.get("author_from_api", "")
                
                # Combine OCR results with Google Books enrichment
                enrichment_clean = {k: v for k, v in enrichment.items() if k != "author_from_api"}
                combined_meta = {
                    "Author": final_author,
                    "Book Name": title,
                    "Series_Name": series_name,
                    **enrichment_clean
                }
                logger.info(f"[DEBUG] Spine {crop_name} enriched with Google Books API - Author: '{final_author}'")
            
            if is_blank_spine(combined_meta):
                filtered_blank_spines += 1
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
        shelf_mapped_books = group_books_into_shelves(detections, original_image_height, expected_k=expected_k)
        processed_images_cache[image_id] = shelf_mapped_books
        
        final_message = f"Processing complete. Found {len(detections)} books on {len(shelf_mapped_books)} shelves."
        if filtered_blank_spines > 0:
            final_message += f" Filtered out {filtered_blank_spines} blank spines."
        
        upload_tasks[image_id].update({"status": "completed", "message": final_message})

    except ValueError as ve:
        error_message = f"Processing failed: {ve}"
        upload_tasks[image_id].update({"status": "failed", "message": error_message, "error": str(ve)})
    except Exception as e:
        error_message = "An unexpected error occurred during processing."
        upload_tasks[image_id].update({"status": "failed", "message": error_message, "error": str(e)})

@router.post("/upload-image/")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()
    # Add file size check (10MB = 10 * 1024 * 1024 bytes)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail="The uploaded image exceeds the 10MB size limit. Please upload a smaller image."
        )
    
    image_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file format. Please upload an image in one of these formats: JPEG/JPG, PNG, BMP, TIFF, or WebP. Maximum file size is 10MB."
        )

    is_blurred, fft_score = is_image_blurred(img)
    if is_blurred:
        raise HTTPException(
            status_code=400, 
            detail=f"The image is too blurry to process. Please try uploading a clearer image."
        )

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
