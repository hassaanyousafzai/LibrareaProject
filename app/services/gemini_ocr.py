import json
from google import genai
from google.genai import types
from core.models import gemini_response_schema
from core.logger import get_logger

logger = get_logger(__name__)
client = genai.Client()

prompt = """
You are an expert in extracting book metadata from spine images. Your task is to identify three key elements: "Author", "Book Name", and "Series Name".

Please follow these strict rules:

1. AUTHOR:
- Look for names anywhere on the spine (top, middle, or bottom).
- Names may be in small text or initials (e.g., "J.K. Rowling").
- May include multiple authors joined by "and" or "&".
- Do not infer — extract only visible names.

2. BOOK NAME:
- Extract ONLY the actual book title, not captions, series arc names, or promotional text.
- Look for the most prominent, standalone book title on the spine.
- EXCLUDE series arc names (e.g., if you see "The Prophecies Begin 2 Fire and Ice", the book name is "Fire and Ice", not the full text).
- EXCLUDE promotional text, captions, or descriptive phrases that aren't part of the core title.
- If the actual book title has legitimate subtitles connected by punctuation (e.g., "Harry Potter and the Sorcerer's Stone", "A Storm of Swords: Blood and Gold"), include those as they are part of the official title.
- When in doubt, choose the shorter, more specific title rather than including extra text.

3. SERIES NAME:
- Look for the PRIMARY brand name only (e.g., "Warriors", "Percy Jackson", "Goosebumps").
- DO NOT include subtitles, volume info, or descriptive text (use "Warriors" not "Warriors: The Prophecies Begin").
- DO NOT include phrases like "The Original Series", "Book One", "Volume 1", etc.
- Extract only the core series brand that would identify all books in that series.
- If not clearly mentioned, leave as an empty string.

ADDITIONAL INSTRUCTIONS:
- Read the entire image carefully.
- Do not guess or infer missing information.
- If any field is not present, return it as an empty string: "".
- Output a JSON object with these three keys: "Author", "Book Name", and "Series Name".

Return only the JSON.
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