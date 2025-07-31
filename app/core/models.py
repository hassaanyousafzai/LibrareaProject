from typing import Optional
from pydantic import BaseModel, field_validator, model_validator, ValidationInfo
from google.genai import types

class OrganizeRequest(BaseModel):
    image_id: str
    sort_by: str
    shelf_number: Optional[int] = None
    sort_order: str = "asc"

    @model_validator(mode='before')
    @classmethod
    def check_for_empty_body(cls, data):
        """
        Ensures the request body is not empty.
        """
        if not data:
            raise ValueError("Request body cannot be empty")
        return data

    @field_validator("image_id", "sort_by", "sort_order")
    @classmethod
    def not_empty(cls, v: str, info: ValidationInfo) -> str:
        """
        Validates that the given string is not empty and provides a clear error message.
        """
        if not v.strip():
            # Include the field name in the error message for clarity.
            raise ValueError(f"'{info.field_name}' is a required field and cannot be empty.")
        return v

# Gemini response schema
gemini_response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "Author": types.Schema(
            type=types.Type.STRING,
            description="The author's name as it appears on the book spine. Use empty string if not visible."
        ),
        "Book Name": types.Schema(
            type=types.Type.STRING, 
            description="The book title as it appears on the spine. Use empty string if not visible."
        ),
        "Series_Name": types.Schema(
            type=types.Type.STRING,
            description="The series name if visible on the spine. Only include if Book Name is empty or unclear. Optional field."
        ),
    },
    required=["Author", "Book Name"]
)
