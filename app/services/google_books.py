import requests
import re
import unicodedata
from difflib import SequenceMatcher
from core.config import GOOGLE_BOOKS_API_KEY
import time
from typing import Optional, Dict, Any, List, Tuple
from core.logger import get_logger

logger = get_logger(__name__)

# Simple global rate limiter state
_LAST_REQUEST_TS: float = 0.0
# Minimum interval between requests (seconds). 0.2s ~= 5 requests/second
MIN_REQUEST_INTERVAL: float = 0.2

# Simple in-process cache to avoid duplicate calls within a run
_REQ_CACHE: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], dict] = {}

def _respect_rate_limit() -> None:
    global _LAST_REQUEST_TS
    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_TS
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _LAST_REQUEST_TS = time.monotonic()

def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    if not header_value:
        return None
    try:
        # Most services return seconds as an integer
        return float(int(header_value.strip()))
    except Exception:
        # If it's a date format, we ignore for simplicity here
        return None

def get_with_backoff(url: str, params: dict, *, max_retries: int = 5, backoff_initial: float = 0.5, backoff_factor: float = 2.0, timeout: float = 15.0) -> requests.Response:
    """
    Perform a GET request with a minimum delay between calls and exponential backoff on 429/5xx.
    """
    attempt = 0
    while True:
        _respect_rate_limit()
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            # Handle rate limit and transient server errors with backoff
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt >= max_retries:
                    logger.warning(f"[GOOGLE_BOOKS] Giving up after {attempt} retries (status {resp.status_code})")
                    resp.raise_for_status()
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                delay = retry_after if retry_after is not None else backoff_initial * (backoff_factor ** attempt)
                delay = max(delay, MIN_REQUEST_INTERVAL)
                attempt += 1
                logger.warning(f"[GOOGLE_BOOKS] Backoff retry {attempt}/{max_retries} after status {resp.status_code}; sleeping {delay:.2f}s")
                time.sleep(delay)
                continue

            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            # Network/timeout errors also get backoff
            if attempt >= max_retries:
                logger.exception(f"[GOOGLE_BOOKS] Request failed after {attempt} retries: {e}")
                raise
            delay = backoff_initial * (backoff_factor ** attempt)
            delay = max(delay, MIN_REQUEST_INTERVAL)
            attempt += 1
            logger.warning(f"[GOOGLE_BOOKS] Exception on request; retry {attempt}/{max_retries} in {delay:.2f}s: {e}")
            time.sleep(delay)


# ----------------------
# Normalization & Fuzzy
# ----------------------

_PUNCT_RE = re.compile(r"[^\w\s]")


def _strip_accents(text: str) -> str:
    if not text:
        return ""
    text_nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text_nfd if unicodedata.category(ch) != "Mn")


def _normalize(text: str) -> str:
    text = _strip_accents(text.lower())
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _normalize(text).split()


def _sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _jaccard_tokens(a: str, b: str) -> float:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


def _author_similarity(query_author: str, api_author: str) -> float:
    if not query_author or not api_author:
        return 0.0
    qa = _tokenize(query_author)
    aa = _tokenize(api_author)
    if not qa or not aa:
        return 0.0

    # Exact normalized match
    if _normalize(query_author) == _normalize(api_author):
        return 1.0

    score = 0.0

    # Token prefix / initials: e.g., "Yasmeen M" vs "Yasmeen Maxamuud"
    if len(qa) >= 2 and len(aa) >= 2:
        # First token must match closely
        if qa[0] == aa[0]:
            # Initial matches surname first letter
            if len(qa[1]) == 1 and aa[1].startswith(qa[1]):
                score = max(score, 0.9)
            # Prefix match for surnames
            if aa[1].startswith(qa[1]) or qa[1].startswith(aa[1]):
                score = max(score, 0.85)

    # Jaccard overlap across tokens
    score = max(score, _jaccard_tokens(query_author, api_author))
    # Sequence ratio fallback (handles diacritics, small edits)
    score = max(score, _sequence_ratio(query_author, api_author))

    return float(score)


def _title_similarity(query_title: str, api_title: str) -> float:
    if not query_title or not api_title:
        return 0.0

    q = _normalize(query_title)
    a = _normalize(api_title)

    if q == a:
        return 1.0
    if q in a or a in q:
        return 0.9

    # Core title after colon often holds the main title
    q_core = q.split(":")[-1].strip()
    a_core = a.split(":")[-1].strip()
    if q_core == a_core:
        return 0.85
    if q_core in a_core or a_core in q_core:
        return 0.8

    # Token and edit distance similarities
    jac = _jaccard_tokens(q, a)
    ratio = _sequence_ratio(q, a)
    return max(jac, ratio)


def _series_bonus(query_series: str, api_title: str) -> float:
    if not query_series:
        return 0.0
    qs = _normalize(query_series)
    at = _normalize(api_title)
    return 0.2 if qs and qs in at else 0.0


def _score_item(info: dict, q_title: str, q_author: str, q_series: str) -> float:
    api_title = info.get("title", "")
    api_authors = info.get("authors", []) or []
    t_sim = _title_similarity(q_title, api_title)
    a_sim = 0.0
    for api_author in api_authors:
        a_sim = max(a_sim, _author_similarity(q_author, api_author))
    bonus = _series_bonus(q_series, api_title)
    # Weight title more heavily; author still matters
    return (t_sim * 0.75) + (a_sim * 0.25) + bonus


def _parse_year(published_date: Optional[str]) -> Optional[int]:
    if not published_date:
        return None
    m = re.match(r"^(\d{4})", str(published_date).strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def get_best_match(query_title: str, query_author: str, api_results: list) -> dict:
    """Fuzzy best match by title+author without series bonus."""
    best_info = None
    best_score = 0.0
    for item in api_results:
        info = item.get("volumeInfo", {})
        score = _score_item(info, query_title, query_author, "")
        if score > best_score:
            best_score = score
            best_info = info
    threshold = 0.6 if query_author else 0.5
    return best_info if best_score >= threshold else None

def get_best_match_with_series(query_title: str, query_author: str, query_series: str, api_results: list) -> dict:
    """Fuzzy best match with series considered."""
    best_info = None
    best_score = 0.0
    for item in api_results:
        info = item.get("volumeInfo", {})
        score = _score_item(info, query_title, query_author, query_series)
        if score > best_score:
            best_score = score
            best_info = info
    threshold = 0.7 if query_series else 0.5
    return best_info if best_score >= threshold else None

def enrich_book_metadata(title: str, author: str, series_name: str, crop_name: str):
    """
    Enrich metadata using at most two Google Books calls:
      - If author is missing: (1) author discovery by title/series, (2) main metadata by title+author
      - If author present: (1) main metadata by title+author only
    Fuzzy matching is applied locally (no extra calls). Earliest publication year is selected from
    candidates in the main metadata response.
    """
    enrichment = {"year_published": None, "genre": [], "isbn": [], "author_from_api": ""}

    if not title or not title.strip():
        return enrichment
    if not GOOGLE_BOOKS_API_KEY:
        return enrichment

    try:
        found_author = author.strip() if author and author.strip() else None

        # Call #1 (optional): discover author only if missing
        if not found_author:
            gb_url = "https://www.googleapis.com/books/v1/volumes"
            # Use a broad token query (avoid strict intitle/quotes)
            if series_name and series_name.strip():
                author_query = f"{series_name.strip()} {title.strip()}"
            else:
                author_query = title.strip()
            gb_params = {
                "q": author_query,
                "key": GOOGLE_BOOKS_API_KEY,
                "maxResults": 10,
                "fields": "items(volumeInfo(title,authors))",
            }
            logger.info(f"[GOOGLE_BOOKS] Author discovery query: '{author_query}' for {crop_name}")
            cache_key = (gb_url, tuple(sorted(gb_params.items())))
            if cache_key in _REQ_CACHE:
                gb_data = _REQ_CACHE[cache_key]
            else:
                gb_resp = get_with_backoff(gb_url, gb_params)
                gb_resp.raise_for_status()
                gb_data = gb_resp.json()
                _REQ_CACHE[cache_key] = gb_data
            logger.info(f"[GOOGLE_BOOKS] Author discovery results: {len(gb_data.get('items', []))}")
            if gb_data.get("items"):
                info = get_best_match_with_series(title, "", series_name, gb_data["items"]) or \
                       get_best_match(title, "", gb_data["items"])  # fallback scoring without series
                if info:
                    api_authors = info.get("authors", []) or []
                    found_author = " & ".join(api_authors) if api_authors else None
                    logger.info(f"[GOOGLE_BOOKS] Discovered author: '{found_author}' for {crop_name}")

        # Call #2 (single): main metadata query (broad token query; rely on fuzzy matching)
        gb_url = "https://www.googleapis.com/books/v1/volumes"
        gb_query_parts = [title.strip()]
        if found_author:
            gb_query_parts.append(found_author)
        if series_name and series_name.strip():
            gb_query_parts.append(series_name.strip())
        gb_query = " ".join([p for p in gb_query_parts if p])
        gb_params = {
            "q": gb_query,
            "key": GOOGLE_BOOKS_API_KEY,
            "maxResults": 10,
            "fields": "items(volumeInfo(title,authors,publishedDate,categories,industryIdentifiers))",
        }
        logger.info(f"[GOOGLE_BOOKS] Metadata query: '{gb_query}' for {crop_name}")
        cache_key = (gb_url, tuple(sorted(gb_params.items())))
        if cache_key in _REQ_CACHE:
            gb_data = _REQ_CACHE[cache_key]
        else:
            gb_resp = get_with_backoff(gb_url, gb_params)
            gb_resp.raise_for_status()
            gb_data = gb_resp.json()
            _REQ_CACHE[cache_key] = gb_data

        items = gb_data.get("items", []) or []
        if not items:
            logger.warning(f"[GOOGLE_BOOKS] No metadata results for {crop_name}")
            return enrichment

        # Score all items; collect acceptable candidates
        scored: List[Tuple[float, dict]] = []
        for item in items:
            info = item.get("volumeInfo", {})
            score = _score_item(info, title, found_author or author or "", series_name or "")
            scored.append((score, info))
        # Accept candidates above threshold
        threshold = 0.6 if (found_author or author) else 0.5
        candidates = [info for sc, info in scored if sc >= threshold]
        if not candidates:
            # keep top-1 anyway for minimal enrichment if score low
            best_info = max(scored, key=lambda x: x[0])[1]
            candidates = [best_info]

        # Earliest year across candidates
        years = [y for y in (_parse_year(info.get("publishedDate")) for info in candidates) if y]
        year_published = min(years) if years else None

        # Choose the single best item for genres/ISBNs/author display
        best_info = max(scored, key=lambda x: x[0])[1]
        raw_genres = best_info.get("categories", []) or []
        processed_genres: List[str] = []
        for genre_item in raw_genres:
            sub_genres = [g.strip() for g in genre_item.replace('/', ',').split(',')]
            processed_genres.extend(sub_genres)
        unique_genres = sorted(list(set(g for g in processed_genres if g)))[:5]

        isbn_list: List[str] = []
        for ident in best_info.get("industryIdentifiers", []) or []:
            if ident.get("type") in ("ISBN_10", "ISBN_13"):
                val = ident.get("identifier")
                if val:
                    isbn_list.append(val)

        api_authors = best_info.get("authors", []) or []
        author_from_api = " & ".join(api_authors)
        api_title = best_info.get("title", "")

        logger.info(f"[GOOGLE_BOOKS] Selected match for {crop_name}: '{api_title}' by '{author_from_api}' (earliest year: {year_published})")
        enrichment.update({
            "year_published": year_published,
            "genre": unique_genres,
            "isbn": isbn_list,
            "author_from_api": author_from_api,
        })

    except Exception as e:
        logger.exception(f"[GOOGLE_BOOKS] API failed for {crop_name}: {e}")
    return enrichment
