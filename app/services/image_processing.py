import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from core.config import (MAX_SMALL_DIM, SPINE_MIN_ASPECT_RATIO, SPINE_MAX_WIDTH_RATIO,
                          SPINE_MIN_HEIGHT_RATIO, SPINE_MIN_WIDTH_PX, SPINE_MIN_HORIZONTAL_ASPECT_RATIO,
                          SPINE_MAX_HEIGHT_RATIO, SPINE_MIN_WIDTH_RATIO, SHELF_BOUNDARY_PADDING,
                          SHELF_MIN_VERTICAL_BOOKS, SHELF_CENTER_TOLERANCE, HORIZONTAL_BOOK_TOLERANCE_MULTIPLIER,
                          HORIZONTAL_ASPECT_RATIO, HORIZONTAL_OVERLAP_THRESHOLD)
from core.logger import get_logger
from services.layout_analysis import cluster_shelves_from_boxes

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

    width_ratio = width / image_width
    height_ratio = height / image_height

    # Determine if the book is vertical or horizontal
    is_vertical = height > width
    
    if is_vertical:
        # Vertical book spine checks
        aspect_ratio = height / width
        
        if aspect_ratio < min_aspect_ratio:
            return False, f"Low vertical aspect ratio ({aspect_ratio:.2f} < {min_aspect_ratio})"
        
        if width_ratio > max_width_ratio:
            return False, f"Too wide for vertical spine ({width_ratio:.2f} > {max_width_ratio})"
        
        if height_ratio < min_height_ratio:
            return False, f"Too short for vertical spine ({height_ratio:.2f} < {min_height_ratio})"
            
    else:
        # Horizontal book spine checks
        aspect_ratio = width / height
        
        if aspect_ratio < SPINE_MIN_HORIZONTAL_ASPECT_RATIO:
            return False, f"Low horizontal aspect ratio ({aspect_ratio:.2f} < {SPINE_MIN_HORIZONTAL_ASPECT_RATIO})"
        
        if height_ratio > SPINE_MAX_HEIGHT_RATIO:
            return False, f"Too tall for horizontal spine ({height_ratio:.2f} > {SPINE_MAX_HEIGHT_RATIO})"
        
        if width_ratio < SPINE_MIN_WIDTH_RATIO:
            return False, f"Too narrow for horizontal spine ({width_ratio:.2f} < {SPINE_MIN_WIDTH_RATIO})"
    
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

def determine_shelf_boundaries(books_data: list, original_image_height: int) -> List[Tuple[float, float]]:
    """
    Enhanced shelf boundary detection that considers both vertical and horizontal books.
    Returns list of shelf ranges [(top1, bottom1), (top2, bottom2), ...] sorted top to bottom.
    """
    # Separate vertical and horizontal books
    vertical_books = []
    horizontal_books = []
    
    for book in books_data:
        bbox = book['bbox']
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        book_info = {
                'book': book,
                'top': bbox[1],
                'bottom': bbox[3],
                'center': (bbox[1] + bbox[3]) / 2
        }
        
        if height > width:  # Vertical book
            vertical_books.append(book_info)
        else:  # Horizontal book
            horizontal_books.append(book_info)
    
    # If no vertical books, use horizontal books to establish boundaries
    if not vertical_books:
        if not horizontal_books:
            return []
        # Use horizontal books only - group them by proximity
        all_books = horizontal_books
    else:
        # Use vertical books as primary reference, but include horizontal books for validation
        all_books = vertical_books
    
    if not all_books:
        return []
    
    # Sort books by their center point (top to bottom)
    all_books.sort(key=lambda x: x['center'])
    
    # Initialize boundaries with the first book
    current_shelf_top = all_books[0]['top']
    current_shelf_bottom = all_books[0]['bottom']
    current_shelf_books = [all_books[0]]
    shelf_boundaries = []
    
    # Group books into shelves
    for book in all_books[1:]:
        # Check if this book belongs to current shelf
        vertical_gap = book['top'] - current_shelf_bottom
        shelf_height = current_shelf_bottom - current_shelf_top
        
        # Use smaller gap threshold for horizontal books as they're typically closer to shelf edges
        gap_threshold = shelf_height * SHELF_BOUNDARY_PADDING
        if not vertical_books:  # Only horizontal books - use tighter clustering
            gap_threshold = shelf_height * 0.3  # Reduced threshold for horizontal-only detection
        
        if vertical_gap > gap_threshold:
            # If we have enough books in the current shelf, add it
            min_books_threshold = SHELF_MIN_VERTICAL_BOOKS if vertical_books else 1  # Allow single horizontal book shelves
            if len(current_shelf_books) >= min_books_threshold:
                # Add padding to boundaries
                padding = shelf_height * SHELF_BOUNDARY_PADDING
                shelf_boundaries.append((
                    current_shelf_top - padding,
                    current_shelf_bottom + padding
                ))
            # Start new shelf
            current_shelf_top = book['top']
            current_shelf_bottom = book['bottom']
            current_shelf_books = [book]
        else:
            # Expand current shelf
            current_shelf_bottom = max(current_shelf_bottom, book['bottom'])
            current_shelf_books.append(book)
    
    # Add the last shelf if it has enough books
    min_books_threshold = SHELF_MIN_VERTICAL_BOOKS if vertical_books else 1
    if len(current_shelf_books) >= min_books_threshold:
        shelf_height = current_shelf_bottom - current_shelf_top
        padding = shelf_height * SHELF_BOUNDARY_PADDING
        shelf_boundaries.append((
            current_shelf_top - padding,
            current_shelf_bottom + padding
        ))
    
    # Second pass: Refine boundaries by including horizontal books if we used vertical books primarily
    if vertical_books and horizontal_books:
        refined_boundaries = []
        for top, bottom in shelf_boundaries:
            # Find horizontal books that fall within this shelf range
            shelf_horizontal_books = [
                hb for hb in horizontal_books 
                if top <= hb['center'] <= bottom
            ]
            
            # If we have horizontal books in this range, potentially adjust boundaries
            if shelf_horizontal_books:
                all_tops = [top] + [hb['top'] for hb in shelf_horizontal_books]
                all_bottoms = [bottom] + [hb['bottom'] for hb in shelf_horizontal_books]
                
                # Expand boundaries to include horizontal books, but don't shrink existing boundaries
                adjusted_top = min(all_tops)
                adjusted_bottom = max(all_bottoms)
                refined_boundaries.append((adjusted_top, adjusted_bottom))
            else:
                refined_boundaries.append((top, bottom))
        
        shelf_boundaries = refined_boundaries
    
    logger.info(f"[SHELF_DETECTION] Detected {len(shelf_boundaries)} shelves using {len(vertical_books)} vertical and {len(horizontal_books)} horizontal books")
    
    return shelf_boundaries

def assign_book_to_shelf(book: Dict, shelf_boundaries: List[Tuple[float, float]]) -> int:
    """
    Assigns a book to a shelf based on its position with enhanced logic for horizontal books.
    Returns shelf index (0-based) or -1 if no shelf found.
    """
    bbox = book['bbox']
    book_top = bbox[1]
    book_bottom = bbox[3]
    book_height = book_bottom - book_top
    book_width = bbox[2] - bbox[0]
    book_center = (book_top + book_bottom) / 2
    
    # Enhanced horizontal book detection
    aspect_ratio = book_width / book_height if book_height > 0 else float('inf')
    is_horizontal = aspect_ratio > HORIZONTAL_ASPECT_RATIO
    
    if is_horizontal:
        # For extremely wide books (like GALLANT), prioritize bottom edge
        if aspect_ratio > 4.0:  # Extra wide books
            reference_point = book_bottom - (book_height * 0.25)  # Use point closer to bottom
        else:
            reference_point = book_center
            
        # Find the shelf where this book has maximum overlap
        max_overlap = 0
        best_shelf = -1
        
        for idx, (shelf_top, shelf_bottom) in enumerate(shelf_boundaries):
            # Calculate overlap
            overlap_top = max(book_top, shelf_top)
            overlap_bottom = min(book_bottom, shelf_bottom)
            overlap = max(0, overlap_bottom - overlap_top)
            
            # For bottom shelf books, give extra weight to overlap
            if idx == len(shelf_boundaries) - 1 and book_bottom > shelf_bottom:
                overlap *= 1.2
                
            # Check if reference point is within shelf bounds (with tolerance)
            shelf_height = shelf_bottom - shelf_top
            tolerance = shelf_height * SHELF_CENTER_TOLERANCE
            if (shelf_top - tolerance) <= reference_point <= (shelf_bottom + tolerance):
                overlap *= 1.5  # Boost overlap score if reference point is in range
                
            if overlap > max_overlap:
                max_overlap = overlap
                best_shelf = idx
                
        if best_shelf >= 0:
            return best_shelf

        # If still no match, check overlap
        book_top = bbox[1]
        book_bottom = bbox[3]
        
        # Calculate overlap with each shelf
        overlaps = []
        for idx, (shelf_top, shelf_bottom) in enumerate(shelf_boundaries):
            overlap_top = max(book_top, shelf_top)
            overlap_bottom = min(book_bottom, shelf_bottom)
            overlap = max(0, overlap_bottom - overlap_top)
            overlap_ratio = overlap / book_height if book_height > 0 else 0
            
            if overlap_ratio >= HORIZONTAL_OVERLAP_THRESHOLD:
                overlaps.append((overlap_ratio, idx))
        
        if overlaps:
            # Return shelf with maximum overlap
            return max(overlaps, key=lambda x: x[0])[1]
    
    # For vertical books, use standard tolerance-based approach
    base_tolerance = SHELF_CENTER_TOLERANCE
    
    for idx, (shelf_top, shelf_bottom) in enumerate(shelf_boundaries):
        shelf_height = shelf_bottom - shelf_top
        tolerance = shelf_height * base_tolerance
        
        if (shelf_top - tolerance) <= book_center <= (shelf_bottom + tolerance):
            return idx
    
    # If book doesn't fit in any shelf, find the closest one
    if shelf_boundaries:
        distances = []
        for idx, (shelf_top, shelf_bottom) in enumerate(shelf_boundaries):
            shelf_center = (shelf_top + shelf_bottom) / 2
            distance = abs(book_center - shelf_center)
            
            # For horizontal books, also consider overlap with shelf boundaries
            if is_horizontal:
                book_top = bbox[1]
                book_bottom = bbox[3]
                
                # Check if any part of the horizontal book overlaps with shelf boundaries
                overlap_top = max(book_top, shelf_top)
                overlap_bottom = min(book_bottom, shelf_bottom)
                overlap = max(0, overlap_bottom - overlap_top)
                
                # If there's significant overlap, reduce the distance score
                overlap_ratio = overlap / book_height if book_height > 0 else 0
                if overlap_ratio > 0.3:  # 30% overlap threshold
                    distance = distance * (1 - overlap_ratio * 0.5)  # Reduce distance by up to 50%
            
            distances.append((distance, idx))
        
        closest_shelf = min(distances, key=lambda x: x[0])[1]
        logger.info(f"[BOOK_ASSIGNMENT] Book {book.get('book_id', 'unknown')} ({'horizontal' if is_horizontal else 'vertical'}) assigned to closest shelf {closest_shelf + 1}")
        return closest_shelf
    
    return -1

def group_books_into_shelves(books_data: list, original_image_height: int, expected_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Groups books into shelves based on their vertical proximity and alignment.
    Uses clustering approach to identify distinct shelves at different vertical levels.
    """
    if not books_data:
        return []

    # If only one book, create a single shelf
    if len(books_data) == 1:
        return [{
            "Shelf 1": [{
                "book_id": books_data[0]['book_id'],
                "position": 1,
                "bbox": books_data[0]['bbox'],
                "confidence": books_data[0]['confidence'],
                "metadata": books_data[0]['metadata']
            }]
        }]

    # Determine shelf boundaries using clustering (data-driven)
    shelf_boundaries = cluster_shelves_from_boxes(books_data, original_image_height, expected_k)
    
    if not shelf_boundaries:
        # If no clear shelf boundaries found, treat all books as one shelf
        return [{
            "Shelf 1": [
                {
                    "book_id": book['book_id'],
                    "position": idx + 1,
                    "bbox": book['bbox'],
                    "confidence": book['confidence'],
                    "metadata": book['metadata']
                }
                for idx, book in enumerate(sorted(books_data, key=lambda x: x['bbox'][0]))
            ]
        }]

    # Second pass: Assign all books to shelves
    shelves = [[] for _ in range(len(shelf_boundaries))]
    
    # Assign each book to a shelf
    for book in books_data:
        shelf_idx = assign_book_to_shelf(book, shelf_boundaries)
        if shelf_idx >= 0:
            shelves[shelf_idx].append(book)
    
    logger.info(f"[SHELF_GROUPING] Detected {len(shelves)} shelves with books: {[len(shelf) for shelf in shelves]}")
    
    # Create final shelf map with sorted books
    final_shelf_map = []
    for shelf_idx, shelf_books in enumerate(shelves):
        if not shelf_books:
            continue
            
        # Sort books horizontally (left to right)
        shelf_books.sort(key=lambda x: x['bbox'][0])
        
        shelf_output = {
            f"Shelf {shelf_idx + 1}": []
        }
        
        for position_in_shelf, book in enumerate(shelf_books):
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
