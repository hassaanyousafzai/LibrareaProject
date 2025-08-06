from typing import Optional, Any
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

    @field_validator("shelf_number", mode='before')
    @classmethod
    def validate_shelf_number(cls, v: Any) -> Optional[int]:
        """
        Validates that shelf_number is a proper integer and provides human-readable error messages.
        """
        if v is None:
            return v
        
        # Reject ALL float inputs (even whole number floats like 1.0)
        if isinstance(v, float):
            raise ValueError("Please enter an integer number (whole number without decimals).")
        
        # Check if it's a string or other type that needs conversion
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            
            # Check for leading zeros (invalid JSON number format)
            if len(v) > 1 and v.startswith('0') and v.isdigit():
                raise ValueError("Please enter a valid number without leading zeros (e.g., use 1 instead of 00001).")
            
            try:
                return int(v)
            except ValueError:
                raise ValueError("Please enter a valid integer number.")
        
        # For integers, return as-is
        if isinstance(v, int):
            return v
            
        # For other types, try to convert to int
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError("Please enter a valid integer number.")

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
