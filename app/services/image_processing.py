import cv2
import numpy as np
from typing import List, Dict, Any
from core.config import MAX_SMALL_DIM
from core.logger import get_logger

logger = get_logger(__name__)

def preprocess_multi_scale(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img_full = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_full is None:
        raise ValueError("Could not decode image")
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