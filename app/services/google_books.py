import requests
import re
from core.config import GOOGLE_BOOKS_API_KEY
from core.logger import get_logger

logger = get_logger(__name__)

def enrich_book_metadata(title: str, author: str, crop_name: str):
    enrichment = {"year_published": None, "genre": [], "isbn": []}
    if title and author and GOOGLE_BOOKS_API_KEY:
        try:
            gb_query = f'intitle:"{title}"+inauthor:"{author}"'
            gb_url = "https://www.googleapis.com/books/v1/volumes"
            gb_params = {
                "q": gb_query,
                "key": GOOGLE_BOOKS_API_KEY,
                "maxResults": 1,
                "fields": "items(volumeInfo(publishedDate,categories,industryIdentifiers))"
            }
            gb_resp = requests.get(gb_url, params=gb_params)
            gb_resp.raise_for_status()
            gb_data = gb_resp.json()
            if gb_data.get("items"):
                info = gb_data["items"][0]["volumeInfo"]

                published_date_raw = info.get("publishedDate")
                year_published = None
                if published_date_raw:
                    match = re.match(r"^(\d{4})", published_date_raw)
                    if match:
                        try:
                            year_published = int(match.group(1))
                        except ValueError:
                            logger.warning(f"Could not convert year '{match.group(1)}' to int for {crop_name}")

                enrichment = {
                    "year_published": year_published,
                    "genre": info.get("categories", []),
                    "isbn": [ident["identifier"] for ident in info.get("industryIdentifiers", []) if ident.get("type") in ("ISBN_10", "ISBN_13")]
                }
        except Exception:
            logger.exception("Google Books API failed during enrichment")
    return enrichment