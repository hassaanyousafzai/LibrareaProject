import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from core.config import MAX_SMALL_DIM, SPINE_MIN_ASPECT_RATIO, SPINE_MAX_WIDTH_RATIO, SPINE_MIN_HEIGHT_RATIO
from core.logger import get_logger

logger = get_logger(__name__)

def is_image_blurred(image: np.ndarray,
                     fft_threshold: float = 1.0,
                     laplacian_threshold: float = 50.0,
                     tenengrad_threshold: float = 2000.0,
                     size: int = 60) -> bool:
    """
    Checks if an image is blurred using Fast Fourier Transform (FFT).
    A lower FFT magnitude score indicates a blurrier image.
    """
    if image is None:
        return False, 0.0

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- FFT-based Blur Detection ---
    h, w = gray.shape

    cX, cY = int(w / 2.0), int(h / 2.0)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    # Remove low-frequency region (which corresponds to overall illumination)
    fft_shift[cY - size:cY + size, cX - size:cX + size] = 0
    recon = np.fft.ifft2(np.fft.ifftshift(fft_shift))
    magnitude = np.abs(recon)
    # Avoid log(0) by adding a small epsilon
    magnitude_db = 20 * np.log10(magnitude + 1e-8)
    fft_score = float(np.mean(magnitude_db))

    # --- Laplacian Variance ---
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # --- Tenengrad (Sobel) ---
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))

    # Combine the three metrics.  Treat the image as blurry only if TWO or more metrics
    # indicate blurriness.  This makes the decision more robust across varying content.
    blur_votes = 0
    if fft_score < fft_threshold:
        blur_votes += 1
    if lap_var < laplacian_threshold:
        blur_votes += 1
    if tenengrad < tenengrad_threshold:
        blur_votes += 1

    is_blurred = blur_votes >= 2

    logger.info(
        f"Blur metrics — FFT mean: {fft_score:.2f} (th {fft_threshold}), "
        f"LapVar: {lap_var:.1f} (th {laplacian_threshold}), "
        f"Tenengrad: {tenengrad:.1f} (th {tenengrad_threshold}) => Blurred={is_blurred}")

    return is_blurred, fft_score

def is_likely_book_spine(bbox: List[float], image_width: int, image_height: int, 
                        min_aspect_ratio: float = None, max_width_ratio: float = None, 
                        min_height_ratio: float = None) -> Tuple[bool, str]:
    """
    Determines if a detected bounding box is likely a book spine based on geometric characteristics.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        image_width: Width of the original image
        image_height: Height of the original image
        min_aspect_ratio: Minimum height/width ratio for a valid spine
        max_width_ratio: Maximum width as fraction of image width for a valid spine
        min_height_ratio: Minimum height as fraction of image height for a valid spine
    
    Returns:
        Tuple of (is_spine: bool, reason: str)
    """
    # Use config defaults if not provided
    if min_aspect_ratio is None:
        min_aspect_ratio = SPINE_MIN_ASPECT_RATIO
    if max_width_ratio is None:
        max_width_ratio = SPINE_MAX_WIDTH_RATIO
    if min_height_ratio is None:
        min_height_ratio = SPINE_MIN_HEIGHT_RATIO
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Calculate aspect ratio (height/width)
    if width <= 0:
        return False, "Invalid width"
    
    aspect_ratio = height / width
    width_ratio = width / image_width
    height_ratio = height / image_height
    
    # Check aspect ratio - spines should be tall and narrow
    if aspect_ratio < min_aspect_ratio:
        return False, f"Low aspect ratio ({aspect_ratio:.2f} < {min_aspect_ratio})"
    
    # Check width relative to image - spines shouldn't be too wide
    if width_ratio > max_width_ratio:
        return False, f"Too wide ({width_ratio:.2f} > {max_width_ratio})"
    
    # Check height relative to image - spines should be reasonably tall
    if height_ratio < min_height_ratio:
        return False, f"Too short ({height_ratio:.2f} < {min_height_ratio})"
    
    return True, f"Valid spine (AR: {aspect_ratio:.2f}, W: {width_ratio:.2f}, H: {height_ratio:.2f})"

def analyze_text_orientation(crop_image: np.ndarray) -> Tuple[str, float]:
    """
    Analyzes the predominant text orientation in a cropped image.
    
    Args:
        crop_image: The cropped book region as numpy array
    
    Returns:
        Tuple of (orientation: str, confidence: float)
        orientation can be: 'vertical', 'horizontal', 'rotated', 'unclear'
    """
    try:
        # Convert to grayscale if needed
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_image.copy()
        
        # Apply edge detection to find text-like features
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use HoughLines to detect line orientations
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None or len(lines) == 0:
            return 'unclear', 0.0
        
        # Analyze line angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            angles.append(angle_deg)
        
        angles = np.array(angles)
        
        # Classify orientations
        # Vertical lines: ~0° or ~180°
        vertical_mask = (angles < 15) | (angles > 165)
        # Horizontal lines: ~90°
        horizontal_mask = (angles > 75) & (angles < 105)
        
        vertical_count = np.sum(vertical_mask)
        horizontal_count = np.sum(horizontal_mask)
        total_lines = len(angles)
        
        if total_lines == 0:
            return 'unclear', 0.0
        
        vertical_ratio = vertical_count / total_lines
        horizontal_ratio = horizontal_count / total_lines
        
        # Determine predominant orientation
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
    Filters detections to separate likely spines from covers/non-spine objects.
    
    Args:
        detections: List of detection dictionaries with bbox information
        image_width: Width of the original image
        image_height: Height of the original image
    
    Returns:
        Tuple of (spine_detections, rejected_detections)
    """
    spine_detections = []
    rejected_detections = []
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        if len(bbox) != 4:
            rejected_detections.append({
                **detection,
                'rejection_reason': 'Invalid bounding box'
            })
            continue
        
        is_spine, reason = is_likely_book_spine(bbox, image_width, image_height)
        
        if is_spine:
            spine_detections.append(detection)
            logger.info(f"Accepted detection {detection.get('book_id', 'unknown')}: {reason}")
        else:
            rejected_detections.append({
                **detection,
                'rejection_reason': reason
            })
            logger.info(f"Rejected detection {detection.get('book_id', 'unknown')}: {reason}")
    
    return spine_detections, rejected_detections

def preprocess_multi_scale(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img_full = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_full is None:
        raise ValueError("Could not decode image")

    # Check for blurriness with the new FFT-based method
    is_blurred, fft_score = is_image_blurred(img_full)
    if is_blurred:
        raise ValueError(f"The image is too blurry to process (FFT score: {fft_score:.2f})")

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
    Groups books into shelves based on their vertical proximity and then
    sorts them horizontally within each shelf.
    """
    if not books_data:
        return []

    vertical_threshold = original_image_height * vertical_proximity_threshold_ratio
    logger.info(f"Calculated vertical proximity threshold: {vertical_threshold:.2f} pixels (from image height {original_image_height})")

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
