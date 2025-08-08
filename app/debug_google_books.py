import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


def _ensure_import_path() -> None:
    """
    Ensure the repository's `app` directory is on sys.path so that imports like
    `from core.config import ...` work when running this script from repo root.
    """
    repo_root = Path(__file__).resolve().parents[1]
    app_dir = repo_root / "app"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))


_ensure_import_path()

# Now safe to import project modules that expect `app/` on PYTHONPATH
from core.config import GOOGLE_BOOKS_API_KEY  # noqa: E402
from services.google_books import (  # noqa: E402
    enrich_book_metadata,
    get_with_backoff,
    MIN_REQUEST_INTERVAL,
)


VOLUMES_URL = "https://www.googleapis.com/books/v1/volumes"


def _require_api_key() -> None:
    if not GOOGLE_BOOKS_API_KEY:
        print("ERROR: GOOGLE_BOOKS_API_KEY is not set. Create a .env with the key or export the env var.")
        sys.exit(1)


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_sequential_enrichment(examples: List[Tuple[str, str, str]], delay_between: float = 0.0) -> None:
    _print_header("Sequential enrich_book_metadata test")
    print(f"MIN_REQUEST_INTERVAL: {MIN_REQUEST_INTERVAL}s")
    for idx, (title, author, series) in enumerate(examples, start=1):
        crop_name = f"seq_{idx:02d}"
        t0 = time.monotonic()
        try:
            result = enrich_book_metadata(title, author, series, crop_name)
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
            status = "error"
        elapsed = time.monotonic() - t0
        print(f"[{idx:02d}] {status} {title!r} | {author!r} | {series!r} -> {result} (elapsed {elapsed:.2f}s)")
        if delay_between > 0:
            time.sleep(delay_between)


def run_concurrent_enrichment(examples: List[Tuple[str, str, str]], max_workers: int = 8) -> None:
    _print_header("Concurrent enrich_book_metadata test")
    print(f"MIN_REQUEST_INTERVAL: {MIN_REQUEST_INTERVAL}s | max_workers={max_workers}")

    def task(example: Tuple[str, str, str], idx: int) -> Dict[str, Any]:
        title, author, series = example
        crop_name = f"con_{idx:02d}"
        t0 = time.monotonic()
        try:
            result = enrich_book_metadata(title, author, series, crop_name)
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
            status = "error"
        elapsed = time.monotonic() - t0
        return {"idx": idx, "status": status, "elapsed": elapsed, "result": result, "title": title}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, ex, i + 1): i for i, ex in enumerate(examples)}
        for fut in as_completed(futures):
            r = fut.result()
            print(
                f"[{r['idx']:02d}] {r['status']} {r['title']!r} -> elapsed {r['elapsed']:.2f}s | result keys: {list(r['result'].keys())}"
            )


def run_raw_rate_limiter_test(num_calls: int = 20, max_workers: int = 10, query: str = "intitle:\"Harry Potter\"") -> None:
    _print_header("Raw get_with_backoff test against /volumes")
    print(f"MIN_REQUEST_INTERVAL: {MIN_REQUEST_INTERVAL}s | calls={num_calls} | max_workers={max_workers}")

    call_times: List[float] = []
    statuses: List[int] = []
    errors: List[str] = []

    def single_call(i: int) -> Dict[str, Any]:
        params = {
            "q": query,
            "key": GOOGLE_BOOKS_API_KEY,
            "maxResults": 1,
            "fields": "items(volumeInfo(title))",
        }
        t0 = time.monotonic()
        try:
            resp = get_with_backoff(VOLUMES_URL, params)
            status = resp.status_code
            payload = resp.json() if resp is not None else {}
            ok = True
            err = ""
        except Exception as exc:  # noqa: BLE001
            status = -1
            payload = {}
            ok = False
            err = str(exc)
        t1 = time.monotonic()
        return {"i": i, "ok": ok, "status": status, "elapsed": t1 - t0, "err": err, "payload": payload}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_call, i) for i in range(num_calls)]
        for fut in as_completed(futures):
            r = fut.result()
            call_times.append(r["elapsed"])  # duration, not start time
            statuses.append(r["status"])
            if not r["ok"]:
                errors.append(r["err"])
            code = r["status"]
            code_str = str(code) if code != -1 else "EXC"
            print(f"call {r['i']:02d}: status={code_str} elapsed={r['elapsed']:.2f}s")

    # Summary
    total = len(statuses)
    n_429 = sum(1 for s in statuses if s == 429)
    n_403 = sum(1 for s in statuses if s == 403)
    n_ok = sum(1 for s in statuses if 200 <= s < 300)
    print("-" * 80)
    if call_times:
        print(
            f"Completed {total} calls | ok={n_ok} | 429={n_429} | 403={n_403} | avg_duration={sum(call_times)/len(call_times):.2f}s"
        )
    if errors:
        print("Errors (first 3):")
        for e in errors[:3]:
            print(f"  - {e}")


def default_examples() -> List[Tuple[str, str, str]]:
    return [
        ("Harry Potter and the Chamber of Secrets", "J.K. Rowling", "Harry Potter"),
        ("The Fellowship of the Ring", "J. R. R. Tolkien", "The Lord of the Rings"),
        ("A Game of Thrones", "George R. R. Martin", "A Song of Ice and Fire"),
        ("The Hunger Games", "Suzanne Collins", ""),
        ("Dune", "Frank Herbert", ""),
        ("The Name of the Wind", "Patrick Rothfuss", "The Kingkiller Chronicle"),
        ("The Girl with the Dragon Tattoo", "Stieg Larsson", ""),
        ("The Lightning Thief", "Rick Riordan", "Percy Jackson & the Olympians"),
        ("The Hobbit", "", ""),  # missing author path
        ("The Two Towers", "", "The Lord of the Rings"),  # missing author but with series
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Google Books API rate limiting behavior")
    parser.add_argument("--mode", choices=["seq", "conc", "raw"], default="seq", help="Test mode")
    parser.add_argument("--workers", type=int, default=8, help="Max workers for concurrent tests")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between sequential calls (seconds)")
    parser.add_argument("--calls", type=int, default=20, help="Number of calls for raw mode")
    parser.add_argument("--query", type=str, default='intitle:"Harry Potter"', help="Query for raw mode")

    args = parser.parse_args()

    _require_api_key()

    examples = default_examples()

    if args.mode == "seq":
        run_sequential_enrichment(examples, delay_between=args.delay)
    elif args.mode == "conc":
        run_concurrent_enrichment(examples, max_workers=args.workers)
    elif args.mode == "raw":
        run_raw_rate_limiter_test(num_calls=args.calls, max_workers=args.workers, query=args.query)


if __name__ == "__main__":
    main()


