from typing import Optional
from pydantic import BaseModel
from google.genai import types

class OrganizeRequest(BaseModel):
    image_id: str
    sort_by: str
    shelf_number: Optional[int] = None
    sort_order: str = "asc"

# Gemini response schema
gemini_response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "Author": types.Schema(type=types.Type.STRING),
        "Book Name": types.Schema(type=types.Type.STRING),
    },
)