import os
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from core.config import UPLOAD_DIR
from core.logger import get_logger
from core.cache import processed_images_cache
from services.yolo import perform_yolo_inference
from services.gemini_ocr import get_book_metadata_from_spine
from services.google_books import enrich_book_metadata
from services.image_processing import preprocess_multi_scale, group_books_into_shelves

router = APIRouter()
logger = get_logger(__name__)

@router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=415, detail="Unsupported media type")

    contents = await file.read()
    try:
        img_full_norm, img_small_norm, scale = preprocess_multi_scale(contents)
        original_image_height = img_full_norm.shape[0]
        original_image_width = img_full_norm.shape[1]

        img_small_uint8 = (img_small_norm * 255).astype(np.uint8)
        try:
            results = perform_yolo_inference(img_small_uint8)
        except Exception:
            logger.exception("YOLO inference failed")
            raise HTTPException(status_code=500, detail="YOLO inference failed")

        boxes_small = results[0].boxes.xyxy.cpu().numpy()
        boxes_full = (boxes_small / scale).tolist()
        scores = [b.conf.item() for b in results[0].boxes]

        annotated = results[0].plot()
        annotated_full = cv2.resize(
            annotated,
            (original_image_width, original_image_height),
            interpolation=cv2.INTER_LINEAR
        )
        _, buf = cv2.imencode('.jpg', annotated_full)
        annotated_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(annotated_path, 'wb') as f:
            f.write(buf.tobytes())
        logger.info(f"Annotated image saved to {annotated_path}")

        img_full_uint8 = (img_full_norm * 255).astype(np.uint8)
        base = os.path.splitext(file.filename)[0]
        spine_dir = os.path.join(UPLOAD_DIR, "cropped_spines", base)
        os.makedirs(spine_dir, exist_ok=True)

        detections = []
        for idx, (box, score) in enumerate(zip(boxes_full, scores), start=1):
            x1, y1, x2, y2 = map(int, box)
            crop = img_full_uint8[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning(f"Skipping empty crop for box {idx}: {box}")
                continue

            crop_name = f"{base}_spine{idx}.jpg"
            crop_path = os.path.join(spine_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            with open(crop_path, 'rb') as img_f:
                crop_bytes = img_f.read()
            
            ocr_meta = get_book_metadata_from_spine(crop_bytes, crop_name)
            
            title = ocr_meta.get("Book Name") or ""
            author = ocr_meta.get("Author") or ""
            enrichment = enrich_book_metadata(title, author, crop_name)

            combined_meta = {**ocr_meta, **enrichment}

            book_id = f"{base}_book_{idx}"
            detections.append({
                "book_id": book_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(score), 2),
                "crop_path": crop_path,
                "metadata": combined_meta
            })

        shelf_mapped_books = group_books_into_shelves(detections, original_image_height)

        image_key = base
        processed_images_cache[image_key] = shelf_mapped_books
        logger.info(f"Processed image '{image_key}' stored in cache.")

        return JSONResponse({
            "image_id": image_key,
            "status": "success",
            "shelves": shelf_mapped_books
        })

    except ValueError as ve:
        logger.error(f"Preprocess error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error during processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")