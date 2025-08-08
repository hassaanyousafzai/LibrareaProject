import json
from google import genai
from google.genai import types
from core.models import gemini_response_schema
from core.logger import get_logger
from core.config import (
    GEMINI_MIN_INTERVAL,
    GEMINI_MAX_RETRIES,
    GEMINI_BACKOFF_INITIAL,
    GEMINI_BACKOFF_FACTOR,
)
import time

logger = get_logger(__name__)
client = genai.Client()
_last_call_ts: float = 0.0

prompt = """
You must transcribe ONLY text that is clearly visible on the book spine image. Never use outside knowledge, memory, or guessing.

Your output MUST be a JSON with exactly these keys (values are strings): "Author", "Book Name", and "Series Name".

Hard rules (no exceptions):
- If any field is unclear, cut off, too small, blurred, occluded, or not confidently readable, return "" for that field.
- Do NOT infer, expand, translate, or complete partial text (e.g., do not expand initials, do not add missing surnames).
- The returned values MUST be substrings that visibly appear on the spine. If you cannot see it, return "".
- Treat publishers, imprints, retailers, slogans, and logos as NOT the author or title. Examples to ignore as author/title: "Penguin", "Puffin", "Vintage", "Random House", "Scholastic", "HarperCollins", "Oxford", "Cambridge", "Penguin Classics", "Anchor", "Pan Macmillan".
- If only a series brand is visible, set "Series Name" to that brand and leave the other fields "".

Author:
- Visible person name(s) only; initials allowed; multiple authors joined by " and " or " & ".

Book Name:
- Extract ONLY the core title as printed on the spine; EXCLUDE subtitles, taglines, and series arcs.
- If the spine shows a separator like ":", " - ", " – ", or " — " (colon, dash, en dash, em dash), keep ONLY the text before the separator as the title.
- If multiple stacked lines are present, prefer the largest standalone title line; exclude smaller descriptive lines.
- When in doubt, choose the shorter main title; if still unsure, return "".

Series Name:
- Only the core series brand (e.g., "Warriors"); exclude volume labels (e.g., "Book One"), arcs, and descriptors.

Output format:
- Return ONLY JSON with keys "Author", "Book Name", "Series Name". No extra keys, notes, or prose.

Negative examples:
- If you see only "Penguin Classics": {"Author":"", "Book Name":"", "Series Name":"Penguin Classics"}
- If you see a partial author like "J. R. R." with no surname: {"Author":"", "Book Name":"", "Series Name":""}
- If multiple big words exist but none clearly looks like a title: {"Author":"", "Book Name":"", "Series Name":""}
"""

generate_content_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=gemini_response_schema,
    system_instruction=[types.Part.from_text(text=prompt)],
)

def get_book_metadata_from_spine(image_bytes: bytes, crop_name: str):
    try:
        logger.info(f"[DEBUG] Starting OCR processing for {crop_name}")
        # Simple rate limit: ensure minimum interval between calls
        global _last_call_ts
        elapsed = time.monotonic() - _last_call_ts
        if elapsed < GEMINI_MIN_INTERVAL:
            time.sleep(GEMINI_MIN_INTERVAL - elapsed)

        # Exponential backoff on rate limits / transient failures
        attempt = 0
        backoff = GEMINI_BACKOFF_INITIAL
        while True:
            try:
                ocr_resp = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')],
                    config=generate_content_config
                )
                _last_call_ts = time.monotonic()
                break
            except Exception as e:
                attempt += 1
                # Check if it's a rate-limit or transient error; conservatively backoff for any exception
                if attempt > GEMINI_MAX_RETRIES:
                    logger.exception(f"[DEBUG] Gemini OCR failed after {attempt-1} retries for {crop_name}: {e}")
                    raise
                logger.warning(f"[DEBUG] Gemini OCR error on attempt {attempt} for {crop_name}: {e}. Backing off {backoff:.2f}s")
                time.sleep(backoff)
                backoff *= GEMINI_BACKOFF_FACTOR
        
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