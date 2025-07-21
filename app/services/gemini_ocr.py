import json
from google import genai
from google.genai import types
from core.models import gemini_response_schema
from core.logger import get_logger

logger = get_logger(__name__)
client = genai.Client()

prompt = """
- You are an expert at identifying book information from spine images.
- Your sole task is to extract the **Author** and **Book Name** (title) from the provided book spine image.
- **Strictly adhere to the JSON format provided in the schema.**
- Fill "Author" with the name of the author as it appears on the spine.
- Fill "Book Name" with the full title of the book as it appears on the spine, including any subtitles if clearly visible.
- If either the Author or Book Name is completely absent or unreadable on the spine, use an empty string ("") for that field.
- **Do not hallucinate or guess.** Only provide information that is directly visible and legible on the spine.
- Prioritize text that clearly looks like an author or title. Ignore other decorative or irrelevant text.
- Your response MUST contain ONLY the JSON object, and nothing else (no conversational text, no preambles, no postscripts).
- The JSON object must strictly conform to the provided schema.
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
        ocr_resp = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=[types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')],
            config=generate_content_config
        )
        logger.info(f"Gemini raw response text for {crop_name}: {ocr_resp.text}")
        if ocr_resp.text:
            ocr_meta = json.loads(ocr_resp.text)
            if ocr_meta.get("Author"):
                ocr_meta["Author"] = ocr_meta["Author"].replace('\n', ' ').strip()
            if ocr_meta.get("Book Name"):
                ocr_meta["Book Name"] = ocr_meta["Book Name"].replace('\n', ' ').strip()
        else:
            logger.warning(f"Gemini returned empty response for {crop_name}. Setting metadata to None.")
            ocr_meta = {"Author": None, "Book Name": None}
    except json.JSONDecodeError as jde:
        logger.error(f"JSONDecodeError for {crop_name}: {jde}. Raw response was: '{ocr_resp.text}'")
        ocr_meta = {"Author": None, "Book Name": None}
    except Exception as e:
        logger.exception(f"Gemini OCR failed for {crop_name}: {e}")
        ocr_meta = {"Author": None, "Book Name": None}
    return ocr_meta