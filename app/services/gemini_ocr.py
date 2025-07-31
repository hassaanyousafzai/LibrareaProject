import json
from google import genai
from google.genai import types
from core.models import gemini_response_schema
from core.logger import get_logger

logger = get_logger(__name__)
client = genai.Client()

prompt = """
You are an expert at extracting book information from spine images. Your task is to identify the AUTHOR, BOOK TITLE, and SERIES NAME.

EXTRACTION RULES:
1. AUTHOR: Identify the author's name. Look for it anywhere on the spine, it may be:
    - At the top, middle, or bottom of the spine.
    - In small or large text.
    - Part of series information (e.g., "ERIN HUNTER" in "Warriors" series).
    - Sometimes abbreviated (first initial + last name).
    - Multiple authors separated by "and" or "&".

2. BOOK TITLE: Extract the main book title.
    - May include a subtitle if present after a colon.
    - Exclude the general series name (e.g., "Warriors") from the book title unless the series name is explicitly part of the unique title (e.g., "The Chronicles of Narnia: The Lion, the Witch, and the Wardrobe").
    - Include volume numbers if they are an integral part of the title (e.g., "Book One: The Fellowship of the Ring").

3. SERIES NAME: Identify the series name.
    - Look for distinct text identifying a series (e.g., "Warriors", "Harry Potter", "The Chronicles of Narnia").
    - Return the most prominent or official series name.
    - Do not include subtitles or volume numbers in the Series Name field.

IMPORTANT CONSIDERATIONS:
- Carefully scan the ENTIRE image for all relevant text.
- Author names are often in smaller text than titles.
- Extract exactly what you see; do not modify, interpret, or infer information not present in the image.
- If a piece of information (Author, Book Name, or Series Name) is truly not visible on the spine, use an empty string ("") for that field.

Return ONLY a JSON object with the following fields: "Author", "Book Name", and "Series Name".
"""

generate_content_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=gemini_response_schema,
    system_instruction=[
        types.Part.from_text(text=prompt),
    ],
)

def get_book_metadata_from_spine(image_bytes: bytes, crop_name: str):
    try:
        logger.info(f"[DEBUG] Starting OCR processing for {crop_name}")
        
        ocr_resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')],
            config=generate_content_config
        )
        
        logger.info(f"[DEBUG] Gemini raw response for {crop_name}: '{ocr_resp.text}'")
        logger.info(f"[DEBUG] Response type: {type(ocr_resp.text)}")
        logger.info(f"[DEBUG] Response length: {len(ocr_resp.text) if ocr_resp.text else 0}")
        
        if ocr_resp.text and ocr_resp.text.strip():
            try:
                ocr_meta = json.loads(ocr_resp.text)
                logger.info(f"[DEBUG] Parsed JSON for {crop_name}: {ocr_meta}")
                logger.info(f"[DEBUG] Author field: '{ocr_meta.get('Author')}' (type: {type(ocr_meta.get('Author'))})")
                logger.info(f"[DEBUG] Book Name field: '{ocr_meta.get('Book Name')}' (type: {type(ocr_meta.get('Book Name'))})")
                logger.info(f"[DEBUG] Series_Name field: '{ocr_meta.get('Series_Name')}' (type: {type(ocr_meta.get('Series_Name'))})")
                
                # Clean up the fields and ensure they're strings
                original_author = ocr_meta.get("Author")
                original_book_name = ocr_meta.get("Book Name")
                original_series_name = ocr_meta.get("Series_Name")
                
                # Handle Author field
                if original_author and isinstance(original_author, str) and original_author.strip():
                    ocr_meta["Author"] = original_author.replace('\n', ' ').strip()
                    logger.info(f"[DEBUG] Cleaned Author: '{original_author}' → '{ocr_meta['Author']}'")
                else:
                    ocr_meta["Author"] = ""
                    logger.warning(f"[DEBUG] Author field was null/empty/invalid for {crop_name}: {original_author}")
                
                # Handle Book Name field  
                if original_book_name and isinstance(original_book_name, str) and original_book_name.strip():
                    ocr_meta["Book Name"] = original_book_name.replace('\n', ' ').strip()
                    logger.info(f"[DEBUG] Cleaned Book Name: '{original_book_name}' → '{ocr_meta['Book Name']}'")
                else:
                    ocr_meta["Book Name"] = ""
                    logger.warning(f"[DEBUG] Book Name field was null/empty/invalid for {crop_name}: {original_book_name}")
                
                # Handle Series_Name field (optional)
                if original_series_name and isinstance(original_series_name, str) and original_series_name.strip():
                    ocr_meta["Series_Name"] = original_series_name.replace('\n', ' ').strip()
                    logger.info(f"[DEBUG] Cleaned Series_Name: '{original_series_name}' → '{ocr_meta['Series_Name']}'")
                else:
                    # Remove the Series_Name field if it's empty/null to keep it truly optional
                    if "Series_Name" in ocr_meta:
                        del ocr_meta["Series_Name"]
                    logger.info(f"[DEBUG] Series_Name field was null/empty/invalid for {crop_name}, removing from response: {original_series_name}")
                
                logger.info(f"[DEBUG] Final metadata for {crop_name}: {ocr_meta}")
                
            except json.JSONDecodeError as jde:
                logger.error(f"[DEBUG] JSON decode error for {crop_name}: {jde}")
                logger.error(f"[DEBUG] Problematic JSON string: '{ocr_resp.text}'")
                ocr_meta = {"Author": "", "Book Name": ""}
        else:
            logger.warning(f"[DEBUG] Gemini returned empty/whitespace response for {crop_name}")
            ocr_meta = {"Author": "", "Book Name": ""}
            
    except Exception as e:
        logger.exception(f"[DEBUG] Gemini OCR failed for {crop_name}: {e}")
        ocr_meta = {"Author": "", "Book Name": ""}
    
    logger.info(f"[DEBUG] Returning metadata for {crop_name}: {ocr_meta}")
    return ocr_meta