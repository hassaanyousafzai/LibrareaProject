import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from core.config import (MAX_SMALL_DIM, SPINE_MIN_ASPECT_RATIO, SPINE_MAX_WIDTH_RATIO, 
                          SPINE_MIN_HEIGHT_RATIO, SPINE_MIN_WIDTH_PX)
from core.logger import get_logger

logger = get_logger(__name__)

def merge_overlapping_boxes(boxes: List[List[float]], scores: List[float], iou_threshold: float = 0.5) -> Tuple[List[List[float]], List[float]]:
    """
    Merges overlapping bounding boxes using Intersection over Union (IoU).
    """
    if not boxes:
        return [], []

    sorted_indices = np.argsort(scores)[::-1]
    
    merged_boxes = []
    merged_scores = []
    
    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        current_box = boxes[current_idx]
        current_score = scores[current_idx]
        
        merged_boxes.append(current_box)
        merged_scores.append(current_score)
        
        remaining_indices = []
        
        for i in range(1, len(sorted_indices)):
            next_idx = sorted_indices[i]
            next_box = boxes[next_idx]
            
            x1 = max(current_box[0], next_box[0])
            y1 = max(current_box[1], next_box[1])
            x2 = min(current_box[2], next_box[2])
            y2 = min(current_box[3], next_box[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area_next = (next_box[2] - next_box[0]) * (next_box[3] - next_box[1])
            union = area_current + area_next - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou < iou_threshold:
                remaining_indices.append(next_idx)
        
        sorted_indices = [idx for idx in sorted_indices if idx in remaining_indices]
        sorted_indices = sorted(remaining_indices, key=lambda idx: scores[idx], reverse=True)

    return merged_boxes, merged_scores

def is_image_blurred(image: np.ndarray, threshold: float = 20.0, size: int = 60) -> Tuple[bool, float]:
    """
    Checks if an image is blurred using Fast Fourier Transform (FFT).
    """
    if image is None:
        return True, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    cX, cY = int(w / 2.0), int(h / 2.0)
    
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    
    fft_shift[cY - size:cY + size, cX - size:cX + size] = 0
    
    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)
    
    magnitude = 20 * np.log(np.abs(recon))
    mean_magnitude = np.mean(magnitude)
    
    is_blurred_flag = mean_magnitude < threshold
    
    return is_blurred_flag, mean_magnitude

def is_likely_book_spine(bbox: List[float], image_width: int, image_height: int, 
                        min_aspect_ratio: float = None, max_width_ratio: float = None, 
                        min_height_ratio: float = None, is_single_book_context: bool = False) -> Tuple[bool, str]:
    """
    Determines if a detected bounding box is likely a book spine.
    For single book images, uses more relaxed criteria.
    """
    if min_aspect_ratio is None:
        # More relaxed aspect ratio for single books
        min_aspect_ratio = 1.5 if is_single_book_context else SPINE_MIN_ASPECT_RATIO
    if max_width_ratio is None:
        max_width_ratio = SPINE_MAX_WIDTH_RATIO
    if min_height_ratio is None:
        # More relaxed height requirement for single books
        min_height_ratio = 0.05 if is_single_book_context else SPINE_MIN_HEIGHT_RATIO
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    if width <= 0 or height <= 0:
        return False, "Invalid width or height"

    # Check minimum width first - if spine is too narrow, text won't be readable
    if width < SPINE_MIN_WIDTH_PX:
        return False, f"Too narrow ({width}px < {SPINE_MIN_WIDTH_PX}px minimum)"

    aspect_ratio = height / width
    width_ratio = width / image_width
    height_ratio = height / image_height

    if aspect_ratio < min_aspect_ratio:
        return False, f"Low aspect ratio ({aspect_ratio:.2f} < {min_aspect_ratio})"

    if width_ratio > max_width_ratio:
        return False, f"Too wide ({width_ratio:.2f} > {max_width_ratio})"

    if height_ratio < min_height_ratio:
        return False, f"Too short ({height_ratio:.2f} < {min_height_ratio})"
    
    context_note = " (single book context)" if is_single_book_context else ""
    return True, f"Valid spine (AR: {aspect_ratio:.2f}, W: {width_ratio:.2f}, H: {height_ratio:.2f}){context_note}"

def analyze_text_orientation(crop_image: np.ndarray) -> Tuple[str, float]:
    """
    Analyzes the predominant text orientation in a cropped image.
    """
    try:
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_image.copy()
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None or len(lines) == 0:
            return 'unclear', 0.0
        
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            angles.append(angle_deg)
        
        angles = np.array(angles)
        
        vertical_mask = (angles < 15) | (angles > 165)
        horizontal_mask = (angles > 75) & (angles < 105)
        
        vertical_count = np.sum(vertical_mask)
        horizontal_count = np.sum(horizontal_mask)
        total_lines = len(angles)
        
        if total_lines == 0:
            return 'unclear', 0.0
        
        vertical_ratio = vertical_count / total_lines
        horizontal_ratio = horizontal_count / total_lines
        
        if vertical_ratio > 0.4:
            return 'vertical', vertical_ratio
        elif horizontal_ratio > 0.4:
            return 'horizontal', horizontal_ratio
        else:
            return 'rotated', max(vertical_ratio, horizontal_ratio)
            
    except Exception as e:
        logger.warning(f"Text orientation analysis failed: {e}")
        return 'unclear', 0.0

def filter_spine_detections(detections: List[Dict], image_width: int, image_height: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Filters detections to separate likely spines from non-spine objects.
    Uses relaxed criteria if only 1-2 detections exist (single book scenario).
    """
    spine_detections = []
    rejected_detections = []
    
    # Determine if this is likely a single book scenario
    is_single_book_context = len(detections) <= 2
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        if len(bbox) != 4:
            rejected_detections.append({
                **detection,
                'rejection_reason': 'Invalid bounding box'
            })
            continue
        
        is_spine, reason = is_likely_book_spine(bbox, image_width, image_height, 
                                               is_single_book_context=is_single_book_context)
        
        if is_spine:
            spine_detections.append(detection)
        else:
            rejected_detections.append({
                **detection,
                'rejection_reason': reason
            })
    
    # If we still have no spine detections in a single book context, 
    # try with the most confident detection using very relaxed criteria
    if not spine_detections and is_single_book_context and detections:
        logger.info("No spines detected in single book context, trying with relaxed criteria...")
        
        # Get the most confident detection
        best_detection = max(detections, key=lambda d: float(d.get('confidence', '0%').rstrip('%')))
        bbox = best_detection.get('bbox', [])
        
        if len(bbox) == 4:
            # Very relaxed criteria for single books
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / width if width > 0 else 0
            width_ratio = width / image_width
            height_ratio = height / image_height
            
            # Accept if it looks remotely book-like
            if (aspect_ratio >= 1.0 and  # At least as tall as wide
                width_ratio <= 0.98 and  # Not the entire image width
                height_ratio >= 0.03):   # At least 3% of image height
                
                spine_detections.append(best_detection)
                # Remove from rejected if it was there
                rejected_detections = [r for r in rejected_detections 
                                     if r.get('book_id') != best_detection.get('book_id')]
                logger.info(f"Accepted detection with relaxed criteria: AR={aspect_ratio:.2f}, W={width_ratio:.2f}, H={height_ratio:.2f}")
    
    return spine_detections, rejected_detections

def preprocess_multi_scale(image_bytes: bytes):
    """
    Decodes and resizes an image for model processing.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img_full = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_full is None:
        raise ValueError("Could not decode image in background task.")

    h, w = img_full.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_SMALL_DIM:
        scale = MAX_SMALL_DIM / float(max(h, w))
        img_small = cv2.resize(img_full, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        img_small = img_full.copy()
        
    img_full_norm = img_full.astype(np.float32) / 255.0
    img_small_norm = img_small.astype(np.float32) / 255.0
    return img_full_norm, img_small_norm, scale

def group_books_into_shelves(books_data: list, original_image_height: int, vertical_proximity_threshold_ratio: float = 0.08) -> List[Dict[str, Any]]:
    """
    Groups books into shelves based on their vertical proximity.
    """
    if not books_data:
        return []

    vertical_threshold = original_image_height * vertical_proximity_threshold_ratio

    sorted_books_by_y = sorted(books_data, key=lambda b: (b['bbox'][1] + b['bbox'][3]) / 2)

    shelves = []
    if not sorted_books_by_y:
        return []

    current_shelf_books = [sorted_books_by_y[0]]

    for i in range(1, len(sorted_books_by_y)):
        prev_book = sorted_books_by_y[i-1]
        current_book = sorted_books_by_y[i]

        prev_center_y = (prev_book['bbox'][1] + prev_book['bbox'][3]) / 2
        current_center_y = (current_book['bbox'][1] + current_book['bbox'][3]) / 2

        if abs(current_center_y - prev_center_y) < vertical_threshold:
            current_shelf_books.append(current_book)
        else:
            shelves.append(current_shelf_books)
            current_shelf_books = [current_book]

    if current_shelf_books:
        shelves.append(current_shelf_books)

    final_shelf_map = []
    for shelf_idx, shelf_books in enumerate(shelves):
        sorted_books_on_current_shelf = sorted(shelf_books, key=lambda b: b['bbox'][0])

        shelf_output = {
            f"Shelf {shelf_idx + 1}": []
        }

        for position_in_shelf, book in enumerate(sorted_books_on_current_shelf):
            book_output = {
                "book_id": book['book_id'],
                "position": position_in_shelf + 1,
                "bbox": book['bbox'],
                "confidence": book['confidence'],
                "metadata": book['metadata']
            }
            shelf_output[f"Shelf {shelf_idx + 1}"].append(book_output)

        final_shelf_map.append(shelf_output)

    return final_shelf_map
