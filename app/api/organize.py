from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
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

    current_shelf_layout = processed_images_cache[image_id]

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
        sorted_book_id_to_new_position = {book['book_id']: i + 1 for i, book in enumerate(books_to_sort)}

        for original_book in books_on_current_shelf:
            book_id = original_book['book_id']
            original_pos = original_book['position']
            new_pos = sorted_book_id_to_new_position.get(book_id)

            if new_pos is not None and original_pos != new_pos:
                shelf_reorder_moves.append({
                    "action": "move",
                    "shelf": shelf_name,
                    "book_id": book_id,
                    "book_name": original_book['metadata'].get("Book Name", "N/A"),
                    "author": original_book['metadata'].get("Author", "N/A"),
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