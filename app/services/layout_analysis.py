import json
from typing import List, Tuple, Dict, Optional

import numpy as np
from google import genai
from google.genai import types

from core.logger import get_logger
from core.config import (
    HORIZONTAL_ASPECT_RATIO,
    SHELF_BOUNDARY_PADDING,
    SHELF_CENTER_TOLERANCE,
    MAX_SHELVES,
    SHELF_CLUSTER_PADDING_RATIO,
    SHELF_SILHOUETTE_MIN,
    ENABLE_LLM_SHELF_COUNT,
    LLM_MODEL_NAME,
)


logger = get_logger(__name__)
client = genai.Client()


def _kmeans_1d(values: np.ndarray, k: int, n_init: int = 5, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 1D k-means implementation.
    Returns labels and centroids.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    values = values.reshape(-1, 1)
    best_inertia = np.inf
    best_labels = None
    best_centroids = None

    rng = np.random.default_rng()
    for _ in range(n_init):
        # k-means++ like init for 1D
        centroids = values[rng.choice(len(values), size=1, replace=False)]
        while len(centroids) < k:
            # distance to nearest centroid
            dists = np.min(((values - centroids.T) ** 2), axis=1)
            probs = dists / dists.sum() if dists.sum() > 0 else np.ones_like(dists) / len(dists)
            next_idx = rng.choice(len(values), p=probs)
            centroids = np.vstack([centroids, values[next_idx]])

        for _ in range(max_iter):
            # assign
            dists = np.square(values - centroids.T)
            labels = np.argmin(dists, axis=1)
            new_centroids = []
            for i in range(k):
                cluster_points = values[labels == i]
                if len(cluster_points) == 0:
                    # reinitialize empty centroid
                    new_centroids.append(values[rng.integers(0, len(values))])
                else:
                    new_centroids.append(np.mean(cluster_points, axis=0))
            new_centroids = np.vstack(new_centroids)
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        inertia = np.sum(np.min(np.square(values - centroids.T), axis=1))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids.reshape(-1)


def _silhouette_score_1d(values: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute a basic silhouette score for 1D data. Returns 0.0 when not defined (e.g., single cluster).
    """
    if len(np.unique(labels)) <= 1:
        return 0.0

    score_sum = 0.0
    for idx in range(len(values)):
        own_label = labels[idx]
        intra = values[labels == own_label]
        if len(intra) > 1:
            a = np.mean(np.abs(intra - values[idx]))  # avg intra-cluster distance
        else:
            a = 0.0

        b_vals = []
        for other_label in np.unique(labels):
            if other_label == own_label:
                continue
            inter = values[labels == other_label]
            if len(inter) > 0:
                b_vals.append(np.mean(np.abs(inter - values[idx])))
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b) if max(a, b) > 0 else 1.0
        score_sum += (b - a) / denom
    return float(score_sum / len(values))


def estimate_shelf_count_with_llm(image_bytes: bytes) -> Tuple[Optional[int], float]:
    """
    Use an LLM to estimate the number of shelves visible. Returns (k, confidence).
    If LLM is disabled or fails, returns (None, 0.0).
    """
    if not ENABLE_LLM_SHELF_COUNT:
        return None, 0.0

    try:
        system_text = (
            "You are an expert visual assistant. Count the number of distinct book shelves visible in the image. "
            "A shelf is a horizontal level holding books. Return a JSON object with keys 'count' (integer) and 'confidence' (0..1)."
        )
        resp = client.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=[types.Part.from_text(text=system_text)],
            ),
        )
        if not resp or not getattr(resp, "text", None):
            return None, 0.0
        payload = json.loads(resp.text)
        k = int(payload.get("count", 0))
        conf = float(payload.get("confidence", 0.0))
        if k <= 0:
            return None, max(0.0, min(conf, 1.0))
        logger.info(f"[LLM_SHELVES] Estimated shelves: {k} (confidence {conf:.2f})")
        return k, max(0.0, min(conf, 1.0))
    except Exception as e:
        logger.warning(f"LLM shelf count estimation failed: {e}")
        return None, 0.0


def cluster_shelves_from_boxes(
    books_data: List[Dict],
    image_height: int,
    expected_k: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """
    Cluster books into shelves using 1D k-means on vertical positions. Builds shelf bands (top,bottom).
    """
    if not books_data:
        return []

    # Compute reference y-value and bounds for each book
    refs = []
    tops = []
    bottoms = []
    heights = []

    for book in books_data:
        x1, y1, x2, y2 = book["bbox"]
        height = float(y2 - y1)
        width = float(x2 - x1)
        aspect = width / height if height > 0 else np.inf
        if aspect > HORIZONTAL_ASPECT_RATIO:
            # Horizontal: use weighted center towards bottom
            ref = y1 + 0.65 * height
        else:
            # Vertical: center
            ref = y1 + 0.5 * height
        refs.append(ref)
        tops.append(float(y1))
        bottoms.append(float(y2))
        heights.append(height)

    refs = np.array(refs, dtype=float)
    tops = np.array(tops, dtype=float)
    bottoms = np.array(bottoms, dtype=float)
    heights = np.array(heights, dtype=float)

    # Decide K
    best_labels = None
    best_centroids = None
    best_k = 1
    best_score = -1.0

    candidate_ks: List[int]
    if expected_k and 1 <= expected_k <= MAX_SHELVES:
        candidate_ks = [expected_k]
    else:
        candidate_ks = list(range(1, MAX_SHELVES + 1))

    # Try expected K first, if silhouette below threshold, sweep
    for k in candidate_ks:
        labels, centroids = _kmeans_1d(refs, k=max(1, k))
        score = _silhouette_score_1d(refs, labels) if k > 1 else 0.0
        if score > best_score:
            best_score = score
            best_labels = labels
            best_centroids = centroids
            best_k = k

    if expected_k and (best_k != expected_k or best_score < SHELF_SILHOUETTE_MIN):
        # Sweep all ks to find a better fit
        for k in range(1, MAX_SHELVES + 1):
            labels, centroids = _kmeans_1d(refs, k=k)
            score = _silhouette_score_1d(refs, labels) if k > 1 else 0.0
            if score > best_score + 1e-6:
                best_score = score
                best_labels = labels
                best_centroids = centroids
                best_k = k

    if best_labels is None or best_centroids is None:
        return []

    logger.info(f"[SHELF_CLUSTER] Chosen K={best_k}, silhouette={best_score:.3f}, expected={expected_k}")

    # Build bands per cluster using tops/bottoms with padding
    shelf_bands: List[Tuple[float, float]] = []
    for c in range(best_k):
        idx = (best_labels == c)
        if not np.any(idx):
            continue
        c_tops = tops[idx]
        c_bottoms = bottoms[idx]
        c_heights = heights[idx]
        base_top = float(np.min(c_tops))
        base_bottom = float(np.max(c_bottoms))
        pad = float(np.median(c_heights) * SHELF_CLUSTER_PADDING_RATIO)
        top = max(0.0, base_top - pad)
        bottom = min(float(image_height), base_bottom + pad)
        shelf_bands.append((top, bottom))

    # Sort bands and merge overlaps
    shelf_bands.sort(key=lambda b: (b[0] + b[1]) / 2.0)
    merged: List[Tuple[float, float]] = []
    for band in shelf_bands:
        if not merged:
            merged.append(band)
            continue
        prev_top, prev_bottom = merged[-1]
        cur_top, cur_bottom = band
        # If bands overlap significantly, merge
        overlap = min(prev_bottom, cur_bottom) - max(prev_top, cur_top)
        prev_h = prev_bottom - prev_top
        cur_h = cur_bottom - cur_top
        if overlap > 0 and (overlap > 0.3 * min(prev_h, cur_h)):
            new_top = min(prev_top, cur_top)
            new_bottom = max(prev_bottom, cur_bottom)
            merged[-1] = (new_top, new_bottom)
        else:
            merged.append(band)

    return merged


