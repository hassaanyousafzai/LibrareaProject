import requests
import re
from core.config import GOOGLE_BOOKS_API_KEY
from core.logger import get_logger

logger = get_logger(__name__)

def enrich_book_metadata(title: str, author: str, series_name: str, crop_name: str):
    # Initialize enrichment with default structure
    enrichment = {"year_published": None, "genre": [], "isbn": [], "author_from_api": None}
    
    # 1. Handle Missing 'Book Name' - Skip entirely if no title
    if not title or title.strip() == "":
        logger.info(f"[GOOGLE_BOOKS] Skipping {crop_name} - missing book title")
        return enrichment
    
    # Check if we have API key
    if not GOOGLE_BOOKS_API_KEY:
        logger.warning(f"[GOOGLE_BOOKS] No API key configured for {crop_name}")
        return enrichment
    
    try:
        # 2. FIRST: Check if Author is missing and find it
        if not author or author.strip() == "":
            logger.info(f"[GOOGLE_BOOKS] Author missing for {crop_name}, searching for author first")
            
            # Build enhanced search query to find the author
            if series_name and series_name.strip():
                gb_query = f'intitle:"{title.strip()}" "{series_name.strip()}"'
                logger.info(f"[GOOGLE_BOOKS] Author search query for {crop_name}: {gb_query}")
            else:
                gb_query = f'intitle:"{title.strip()}"'
                logger.info(f"[GOOGLE_BOOKS] Author search query (title-only) for {crop_name}: {gb_query}")
            
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
            
            logger.info(f"[GOOGLE_BOOKS] Author search response for {crop_name}: found {len(gb_data.get('items', []))} items")
            
            if gb_data.get("items"):
                info = gb_data["items"][0]["volumeInfo"]
                api_authors = info.get("authors", [])
                
                if api_authors:
                    # Found the author! Join multiple authors with " & " if there are multiple
                    found_author = " & ".join(api_authors)
                    enrichment["author_from_api"] = found_author
                    logger.info(f"[GOOGLE_BOOKS] ✅ Found missing author for {crop_name}: {found_author}")
                    
                    # NOW search again with the found author to get complete metadata
                    logger.info(f"[GOOGLE_BOOKS] Now searching for complete metadata with found author")
                    gb_query_complete = f'intitle:"{title.strip()}"+inauthor:"{found_author}"'
                    logger.info(f"[GOOGLE_BOOKS] Complete metadata query for {crop_name}: {gb_query_complete}")
                    
                    gb_params_complete = {
                        "q": gb_query_complete,
                        "key": GOOGLE_BOOKS_API_KEY,
                        "maxResults": 5,
                        "fields": "items(volumeInfo(title,authors,publishedDate,categories,industryIdentifiers))"
                    }
                    
                    gb_resp_complete = requests.get(gb_url, params=gb_params_complete)
                    gb_resp_complete.raise_for_status()
                    gb_data_complete = gb_resp_complete.json()
                    
                    if gb_data_complete.get("items"):
                        info = gb_data_complete["items"][0]["volumeInfo"]
                        logger.info(f"[GOOGLE_BOOKS] Using complete metadata result for {crop_name}: '{info.get('title', 'N/A')}'")
                    else:
                        logger.warning(f"[GOOGLE_BOOKS] No complete metadata found, using author-search result for {crop_name}")
                        # Keep using the original info from author search
                else:
                    logger.warning(f"[GOOGLE_BOOKS] ❌ No authors found in API response for {crop_name}")
                    # Still extract other metadata even if no author found
            else:
                logger.warning(f"[GOOGLE_BOOKS] ❌ No search results found when looking for author for {crop_name}")
                return enrichment  # Return early if no results at all
        else:
            # Author exists, use standard search with both title and author
            logger.info(f"[GOOGLE_BOOKS] Author exists, searching for complete metadata")
            gb_query = f'intitle:"{title.strip()}"+inauthor:"{author.strip()}"'
            logger.info(f"[GOOGLE_BOOKS] Standard query for {crop_name}: {gb_query}")
            
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
            
            logger.info(f"[GOOGLE_BOOKS] Standard search response for {crop_name}: found {len(gb_data.get('items', []))} items")
            
            if gb_data.get("items"):
                info = gb_data["items"][0]["volumeInfo"]
                logger.info(f"[GOOGLE_BOOKS] Using standard result for {crop_name}: '{info.get('title', 'N/A')}'")
            else:
                logger.warning(f"[GOOGLE_BOOKS] No results found for {crop_name}")
                return enrichment

        # Extract metadata from the final info object
        # Extract publication year
        published_date_raw = info.get("publishedDate")
        year_published = None
        if published_date_raw:
            match = re.match(r"^(\d{4})", published_date_raw)
            if match:
                try:
                    year_published = int(match.group(1))
                except ValueError:
                    logger.warning(f"[GOOGLE_BOOKS] Could not convert year '{match.group(1)}' to int for {crop_name}")

        # Genre Extraction Enhancement
        raw_genres = info.get("categories", [])
        processed_genres = []
        
        if raw_genres:
            # Flatten and clean genres (sometimes they come as nested or comma-separated)
            for genre_item in raw_genres:
                if isinstance(genre_item, str):
                    # Split on common separators and clean
                    sub_genres = [g.strip() for g in genre_item.replace('/', ',').split(',')]
                    processed_genres.extend(sub_genres)
            
            # Remove duplicates while preserving order
            unique_genres = []
            for genre in processed_genres:
                if genre and genre not in unique_genres:
                    unique_genres.append(genre)
            
            # Apply genre count logic: 3-5 genres
            if len(unique_genres) > 5:
                final_genres = unique_genres[:5]  # Take first 5
                logger.info(f"[GOOGLE_BOOKS] Trimmed genres for {crop_name}: {len(unique_genres)} → 5")
            elif len(unique_genres) >= 3:
                final_genres = unique_genres  # Keep all (3-5)
                logger.info(f"[GOOGLE_BOOKS] Using all genres for {crop_name}: {len(unique_genres)}")
            else:
                final_genres = unique_genres  # Keep all available (< 3)
                logger.info(f"[GOOGLE_BOOKS] Limited genres available for {crop_name}: {len(unique_genres)}")
        else:
            final_genres = []
            logger.info(f"[GOOGLE_BOOKS] No genres found for {crop_name}")

        # Extract ISBNs
        isbn_list = []
        identifiers = info.get("industryIdentifiers", [])
        for ident in identifiers:
            if ident.get("type") in ("ISBN_10", "ISBN_13"):
                isbn_list.append(ident["identifier"])

        # Update final enrichment
        enrichment.update({
            "year_published": year_published,
            "genre": final_genres,
            "isbn": isbn_list
        })
        
        logger.info(f"[GOOGLE_BOOKS] Final enrichment for {crop_name}: {enrichment}")
            
    except Exception as e:
        logger.exception(f"[GOOGLE_BOOKS] API failed during enrichment for {crop_name}: {e}")
    
    return enrichment