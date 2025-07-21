from typing import Dict, List, Any

# Global Cache for processed image data
# Key: image_id (base filename), Value: List of shelf dictionaries
processed_images_cache: Dict[str, List[Dict[str, Any]]] = {}