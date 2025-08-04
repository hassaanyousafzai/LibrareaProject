from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from core.models import OrganizeRequest
from core.logger import get_logger
from core.cache import processed_images_cache
from typing import Any

router = APIRouter()
logger = get_logger(__name__)

@router.post("/organize-shelf/")
async def organize_shelf(request: OrganizeRequest):
    """
    Provides reorganization suggestions for books on a shelf based on specified criteria.
    """
    image_id = request.image_id
    sort_by = request.sort_by
    shelf_number = request.shelf_number
    sort_order = request.sort_order

    if image_id not in processed_images_cache:
        raise HTTPException(status_code=404, detail="Image data not found. Please upload the image first.")

    cached_data = processed_images_cache[image_id]
    
    if isinstance(cached_data, dict) and "message" in cached_data:
        return JSONResponse({
            "image_id": image_id,
            "message": cached_data["message"],
            "total_detections": cached_data.get("total_detections", 0),
            "rejected_detections": cached_data.get("rejected_detections", []),
            "spine_detections": cached_data.get("spine_detections", []),
            "can_organize": False
        })
    
    current_shelf_layout = cached_data
    total_shelves = len(current_shelf_layout)

    if shelf_number is not None:
        if shelf_number <= 0 or shelf_number > total_shelves:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid shelf number. Please enter a number between 1 and {total_shelves}."
            )

    valid_sort_by = ["author", "title", "genre", "height"]
    valid_sort_order = ["asc", "desc"]

    if sort_by not in valid_sort_by:
        raise HTTPException(status_code=400, detail=f"Invalid sort_by criteria. Must be one of: {', '.join(valid_sort_by)}")
    if sort_order not in valid_sort_order:
        raise HTTPException(status_code=400, detail=f"Invalid sort_order. Must be 'asc' or 'desc'.")

    organized_layout = []
    all_reorder_instructions = []

    def get_sort_key(book: Any):
        if sort_by == "author":
            author = book['metadata'].get("Author")
            return author.strip().lower() if author else ""
        elif sort_by == "title":
            title = book['metadata'].get("Book Name")
            return title.strip().lower() if title else ""
        elif sort_by == "genre":
            genres = book['metadata'].get("genre", [])
            return genres[0].strip().lower() if genres else ""
        elif sort_by == "height":
            if book['bbox'] and len(book['bbox']) == 4:
                return book['bbox'][3] - book['bbox'][1]
            return 0
        return ""

    for shelf_idx, shelf_data_dict in enumerate(current_shelf_layout):
        shelf_name = list(shelf_data_dict.keys())[0]
        books_on_current_shelf = shelf_data_dict[shelf_name]

        if shelf_number is not None and shelf_idx + 1 != shelf_number:
            organized_layout.append(shelf_data_dict)
            if shelf_number is None:
                if not books_on_current_shelf:
                    all_reorder_instructions.append({
                        "action": "info",
                        "shelf": shelf_name,
                        "message": f"No books detected on Shelf {shelf_name}."
                    })
                else:
                    all_reorder_instructions.append({
                        "action": "info",
                        "shelf": shelf_name,
                        "message": f"Shelf {shelf_name} was not targeted for reordering and remains in its original layout."
                    })
            continue

        books_to_sort = list(books_on_current_shelf)

        try:
            books_to_sort.sort(key=get_sort_key, reverse=(sort_order == "desc"))
        except Exception as e:
            logger.error(f"Error sorting books on {shelf_name} by {sort_by}: {e}")
            books_to_sort = books_on_current_shelf

        shelf_reorder_moves = []
        
        original_positions = {book['book_id']: book['position'] for book in books_on_current_shelf}
        
        original_book_ids = [book['book_id'] for book in books_on_current_shelf]
        sorted_book_ids = [book['book_id'] for book in books_to_sort]
        
        shelf_order_changed = original_book_ids != sorted_book_ids
        
        if shelf_order_changed:
            for new_pos, book in enumerate(books_to_sort, start=1):
                book_id = book['book_id']
                original_pos = original_positions[book_id]
                
                shelf_reorder_moves.append({
                    "action": "move",
                    "shelf": shelf_name,
                    "book_id": book_id,
                    "book_name": book['metadata'].get("Book Name", "N/A"),
                    "author": book['metadata'].get("Author", "N/A"),
                    "original_position": original_pos,
                    "new_position": new_pos
                })

        shelf_reorder_moves.sort(key=lambda x: x['new_position'])

        if not shelf_reorder_moves and len(books_on_current_shelf) > 0:
             all_reorder_instructions.append({
                 "action": "info",
                 "shelf": shelf_name,
                 "message": f"Shelf {shelf_name} is already in the requested order or no rearrangement needed."
             })
        elif len(books_on_current_shelf) == 0:
             all_reorder_instructions.append({
                 "action": "info",
                 "shelf": shelf_name,
                 "message": f"No books detected on Shelf {shelf_name}."
             })
        else:
            all_reorder_instructions.extend(shelf_reorder_moves)

        sorted_books_with_new_positions = []
        for i, book in enumerate(books_to_sort):
            book_copy = book.copy()
            book_copy['position'] = i + 1
            sorted_books_with_new_positions.append(book_copy)

        organized_layout.append({shelf_name: sorted_books_with_new_positions})

    return JSONResponse({
        "image_id": image_id,
        "requested_sort_by": sort_by,
        "sort_order": sort_order,
        "shelf_number_processed": shelf_number,
        "current_layout": current_shelf_layout,
        "organized_layout": organized_layout,
        "reorder_instructions": all_reorder_instructions,
        "message": "Shelf organization complete."
    })
