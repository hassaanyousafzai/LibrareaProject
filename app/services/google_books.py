import requests
import re
import Levenshtein
from core.config import GOOGLE_BOOKS_API_KEY
from core.logger import get_logger

logger = get_logger(__name__)

def get_best_match(query_title: str, query_author: str, api_results: list) -> dict:
    """
    Finds the best match from Google Books API results based on title and author similarity.
    """
    best_match = None
    highest_score = 0.0

    for item in api_results:
        info = item.get("volumeInfo", {})
        api_title = info.get("title", "").lower()
        api_authors = [a.lower() for a in info.get("authors", [])]

        title_similarity = Levenshtein.ratio(query_title.lower(), api_title)

        author_similarity = 0.0
        if query_author and api_authors:
            author_scores = [Levenshtein.ratio(query_author.lower(), api_author) for api_author in api_authors]
            author_similarity = max(author_scores) if author_scores else 0.0
        elif not query_author:
            author_similarity = 1.0

        total_score = (title_similarity * 0.7) + (author_similarity * 0.3)

        if total_score > highest_score:
            highest_score = total_score
            best_match = info

    if highest_score > 0.8:
        return best_match
    
    return None

def enrich_book_metadata(title: str, author: str, series_name: str, crop_name: str):
    enrichment = {"year_published": None, "genre": [], "isbn": [], "author_from_api": None}
    
    if not title or not title.strip():
        return enrichment
    
    if not GOOGLE_BOOKS_API_KEY:
        return enrichment
    
    try:
        # First attempt: Precise search with both title and author
        query_parts = [f'intitle:"{title.strip()}"']
        if author and author.strip():
            query_parts.append(f'inauthor:"{author.strip()}"')
            
        gb_query = "+".join(query_parts)
        
        gb_url = "https://www.googleapis.com/books/v1/volumes"
        gb_params = {
            "q": gb_query,
            "key": GOOGLE_BOOKS_API_KEY,
            "maxResults": 5,
            "fields": "items(volumeInfo(title,authors,publishedDate,categories,industryIdentifiers))"
        }
            
        gb_resp = requests.get(gb_url, params=gb_params)
        gb_resp.raise_for_status()
        gb_data = gb_resp.json()

        best_match_info = None
        if gb_data.get("items"):
            best_match_info = get_best_match(title, author, gb_data["items"])
        
        # Fallback: If no confident match, search by title only
        if not best_match_info:
            title_only_query = f'intitle:"{title.strip()}"'
            gb_params["q"] = title_only_query
            
            gb_resp = requests.get(gb_url, params=gb_params)
            gb_resp.raise_for_status()
            gb_data = gb_resp.json()

            if gb_data.get("items"):
                best_match_info = get_best_match(title, author, gb_data["items"])

        if best_match_info:
            published_date_raw = best_match_info.get("publishedDate")
            year_published = None
            if published_date_raw:
                match = re.match(r"^(\d{4})", published_date_raw)
                if match:
                    try:
                        year_published = int(match.group(1))
                    except ValueError:
                        pass

            raw_genres = best_match_info.get("categories", [])
            processed_genres = []
            if raw_genres:
                for genre_item in raw_genres:
                    sub_genres = [g.strip() for g in genre_item.replace('/', ',').split(',')]
                    processed_genres.extend(sub_genres)
            
            unique_genres = sorted(list(set(g for g in processed_genres if g)))

            isbn_list = []
            identifiers = best_match_info.get("industryIdentifiers", [])
            if identifiers:
                for ident in identifiers:
                    if ident.get("type") in ("ISBN_10", "ISBN_13"):
                        isbn_list.append(ident["identifier"])
            
            api_authors = best_match_info.get("authors", [])
            author_from_api = " & ".join(api_authors) if api_authors else None

            enrichment.update({
                "year_published": year_published,
                "genre": unique_genres[:5],
                "isbn": isbn_list,
                "author_from_api": author_from_api
            })

    except Exception as e:
        logger.exception(f"[GOOGLE_BOOKS] API failed for {crop_name}: {e}")
    
    return enrichment
