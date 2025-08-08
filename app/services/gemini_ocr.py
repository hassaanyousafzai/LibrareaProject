import json
from google import genai
from google.genai import types
from core.models import gemini_response_schema
from core.logger import get_logger

logger = get_logger(__name__)
client = genai.Client()

prompt = """
You are an expert in extracting book metadata strictly from spine images. Your task is to return a JSON with three keys only: "Author", "Book Name", and "Series Name".

CRITICAL RULES (Hallucination guard):
- NEVER use outside knowledge, prior memory, or guesses. ONLY transcribe text that is clearly visible in the crop.
- If text is blurred, cut off, too small, blocked, or ambiguous, return an empty string for that field.
- If you see only logos, artwork, publishers, imprints, or decorative words (e.g., "Penguin", "HarperCollins"), do not treat them as Author or Book Name.
- If only a series brand is visible, set "Series Name" and leave other fields empty.
- If you are not at least reasonably certain a word belongs to that field from the spine itself, return an empty string.

AUTHOR:
- Extract visible author name(s) only (initials allowed). Multiple authors allowed with "and" or "&".
- Do NOT infer missing parts or expand initials.

BOOK NAME:
- Extract ONLY the actual title on the spine (not captions, slogans, taglines, or series arcs).
- If the title includes a valid subtitle joined by punctuation, include it (e.g., "Title: Subtitle").
- If unsure whether a long phrase is the title, choose the shorter, core title. If still unsure, return empty.

SERIES NAME:
- Extract only the core series brand (e.g., "Warriors").
- Exclude volume labels, arcs, phrases like "Book One", "Volume 1", "The Original Series".
- If not clearly present, return empty.

OUTPUT FORMAT:
- Return ONLY JSON with keys: "Author", "Book Name", "Series Name" (no extra keys).
- All values must be strings.

EXAMPLES:
1) If the crop is mostly blank or unreadable:
{"Author": "", "Book Name": "", "Series Name": ""}

2) If only a series logo like "Goosebumps" is clearly visible:
{"Author": "", "Book Name": "", "Series Name": "Goosebumps"}

3) If author initials and a short clear title are visible:
{"Author": "J.K. Rowling", "Book Name": "Harry Potter", "Series Name": ""}
"""

generate_content_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=gemini_response_schema,
    system_instruction=[types.Part.from_text(text=prompt)],
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