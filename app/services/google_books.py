import requests
import re
from core.config import GOOGLE_BOOKS_API_KEY
from core.logger import get_logger

logger = get_logger(__name__)

def get_best_match(query_title: str, query_author: str, api_results: list) -> dict:
    """
    Finds the best match from Google Books API results based on simple title and author matching.
    """
    best_match = None
    highest_score = 0.0

    for item in api_results:
        info = item.get("volumeInfo", {})
        api_title = info.get("title", "").lower()
        api_authors = [a.lower() for a in info.get("authors", [])]

        # Simple title matching: check if query title is contained in API title or vice versa
        query_title_lower = query_title.lower().strip()
        title_similarity = 0.0
        
        # Check exact match first
        if query_title_lower == api_title:
            title_similarity = 1.0
        # Check if query title is contained in API title
        elif query_title_lower in api_title:
            title_similarity = 0.8
        # Check if API title is contained in query title
        elif api_title in query_title_lower:
            title_similarity = 0.8
        # Check core title (after colon) for series books
        else:
            api_core_title = api_title.split(':')[-1].strip()
            query_core_title = query_title_lower.split(':')[-1].strip()
            if query_core_title == api_core_title:
                title_similarity = 0.7
            elif query_core_title in api_core_title or api_core_title in query_core_title:
                title_similarity = 0.6

        # Simple author matching
        author_similarity = 0.0
        if query_author and api_authors:
            query_author_lower = query_author.lower().strip()
            for api_author in api_authors:
                # Check exact match
                if query_author_lower == api_author:
                    author_similarity = 1.0
                    break
                # Check if author names contain each other
                elif query_author_lower in api_author or api_author in query_author_lower:
                    author_similarity = max(author_similarity, 0.8)
        elif not query_author:
            # When no author is provided, don't penalize
            author_similarity = 0.0

        # Calculate total score with title being more important
        total_score = (title_similarity * 0.8) + (author_similarity * 0.2)

        if total_score > highest_score:
            highest_score = total_score
            best_match = info

    # Return best match if score is reasonable
    # Lower threshold since we're using simpler matching
    required_score = 0.6 if query_author else 0.5
    if highest_score >= required_score:
        return best_match
    
    return None

def get_best_match_with_series(query_title: str, query_author: str, query_series: str, api_results: list) -> dict:
    """
    Enhanced matching that considers series information to avoid wrong matches.
    """
    best_match = None
    highest_score = 0.0

    for item in api_results:
        info = item.get("volumeInfo", {})
        api_title = info.get("title", "").lower()
        api_authors = [a.lower() for a in info.get("authors", [])]
        
        # Check if series name appears in the title or description
        series_in_title = False
        if query_series:
            query_series_lower = query_series.lower().strip()
            series_in_title = query_series_lower in api_title
        
        # Title matching (same as before)
        query_title_lower = query_title.lower().strip()
        title_similarity = 0.0
        
        if query_title_lower == api_title:
            title_similarity = 1.0
        elif query_title_lower in api_title:
            title_similarity = 0.8
        elif api_title in query_title_lower:
            title_similarity = 0.8
        else:
            api_core_title = api_title.split(':')[-1].strip()
            query_core_title = query_title_lower.split(':')[-1].strip()
            if query_core_title == api_core_title:
                title_similarity = 0.7
            elif query_core_title in api_core_title or api_core_title in query_core_title:
                title_similarity = 0.6

        # Author matching (same as before)
        author_similarity = 0.0
        if query_author and api_authors:
            query_author_lower = query_author.lower().strip()
            for api_author in api_authors:
                if query_author_lower == api_author:
                    author_similarity = 1.0
                    break
                elif query_author_lower in api_author or api_author in query_author_lower:
                    author_similarity = max(author_similarity, 0.8)
        elif not query_author:
            author_similarity = 0.0

        # Series bonus: if series name is found in title, boost the score
        series_bonus = 0.3 if series_in_title else 0.0
        
        # Calculate total score - higher weight on title, series bonus important
        total_score = (title_similarity * 0.7) + (author_similarity * 0.1) + series_bonus

        if total_score > highest_score:
            highest_score = total_score
            best_match = info

    # Higher threshold when we have series info to ensure accuracy
    required_score = 0.7 if query_series else 0.5
    if highest_score >= required_score:
        return best_match
    
    return None

def enrich_book_metadata(title: str, author: str, series_name: str, crop_name: str):
    enrichment = {"year_published": None, "genre": [], "isbn": [], "author_from_api": ""}
    
    if not title or not title.strip():
        return enrichment
    
    if not GOOGLE_BOOKS_API_KEY:
        return enrichment
    
    try:
        found_author = author.strip() if author and author.strip() else None
        # If author is missing, try to find it using book name and series name
        if not found_author:
            # Try with series name first if present - this gives more targeted results
            if series_name and series_name.strip():
                # Search for series + title combination for better accuracy
                author_query = f'"{series_name.strip()}" "{title.strip()}"'
            else:
                author_query = f'intitle:"{title.strip()}"'
            gb_url = "https://www.googleapis.com/books/v1/volumes"
            gb_params = {
                "q": author_query,
                "key": GOOGLE_BOOKS_API_KEY,
                "maxResults": 10,  # Get more results to find the correct match
                "fields": "items(volumeInfo(title,authors))"
            }
            logger.info(f"[GOOGLE_BOOKS] Searching for author with query: '{author_query}' for {crop_name}")
            gb_resp = requests.get(gb_url, params=gb_params)
            gb_resp.raise_for_status()
            gb_data = gb_resp.json()
            logger.info(f"[GOOGLE_BOOKS] Found {len(gb_data.get('items', []))} results for author search")
            best_match_info = None
            if gb_data.get("items"):
                # When searching for author, be more strict about matching
                best_match_info = get_best_match_with_series(title, "", series_name, gb_data["items"])
            if best_match_info:
                api_authors = best_match_info.get("authors", [])
                found_author = " & ".join(api_authors) if api_authors else None
                logger.info(f"[GOOGLE_BOOKS] Found author from API: '{found_author}' for {crop_name}")
            else:
                logger.warning(f"[GOOGLE_BOOKS] No suitable author match found for {crop_name}")
        # Now, use found_author (if any) to enrich metadata
        query_parts = [f'intitle:"{title.strip()}"']
        if found_author:
            query_parts.append(f'inauthor:"{found_author}"')
        gb_query = "+".join(query_parts)
        logger.info(f"[GOOGLE_BOOKS] Main search query: '{gb_query}' for {crop_name}")
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
            # Use series-aware matching when we have series information
            if series_name and series_name.strip():
                best_match_info = get_best_match_with_series(title, found_author or author, series_name, gb_data["items"])
            else:
                best_match_info = get_best_match(title, found_author or author, gb_data["items"])
        # Fallback: If no confident match, search by title only
        if not best_match_info:
            title_only_query = f'intitle:"{title.strip()}"'
            gb_params["q"] = title_only_query
            gb_resp = requests.get(gb_url, params=gb_params)
            gb_resp.raise_for_status()
            gb_data = gb_resp.json()
            if gb_data.get("items"):
                # Use series-aware matching in fallback too
                if series_name and series_name.strip():
                    best_match_info = get_best_match_with_series(title, found_author or author, series_name, gb_data["items"])
                else:
                    best_match_info = get_best_match(title, found_author or author, gb_data["items"])
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
            author_from_api = " & ".join(api_authors) if api_authors else ""
            api_title = best_match_info.get("title", "")
            logger.info(f"[GOOGLE_BOOKS] Final match for {crop_name}: '{api_title}' by '{author_from_api}' (year: {year_published})")
            enrichment.update({
                "year_published": year_published,
                "genre": unique_genres[:5],
                "isbn": isbn_list,
                "author_from_api": author_from_api
            })
        else:
            logger.warning(f"[GOOGLE_BOOKS] No suitable match found for {crop_name}")
    except Exception as e:
        logger.exception(f"[GOOGLE_BOOKS] API failed for {crop_name}: {e}")
    return enrichment