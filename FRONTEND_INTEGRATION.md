# Librarea Frontend Integration Guide

This guide explains how to integrate a frontend application with the Librarea backend. It covers endpoints, request/response shapes, recommended flows, examples, and error handling.

## Base URL
- Local development: `http://127.0.0.1:8000`
- Behind a reverse proxy (optional): prefix routes accordingly (e.g., `/api`)

## High-level Flow
1. Upload a bookshelf image (starts async processing) → receive `image_id`.
2. Poll processing status using `image_id` until `status == completed`.
3. Fetch organized layout and move plan using the GET organize endpoint.
4. Render shelves, books, and reordering instructions.

## Endpoints

### 1) POST `/upload-image/`
- Description: Upload a bookshelf image; begins detection, OCR, and enrichment.
- Content-Type: `multipart/form-data`
- Body fields:
  - `file`: Image file. Max 10MB. Supported: JPEG/JPG, PNG, BMP, TIFF, WebP.
- Responses:
  - 202 Accepted
    ```json
    { "message": "Upload accepted and is being processed.", "image_id": "<uuid>" }
    ```
  - 400 Bad Request (image too blurry)
    ```json
    { "detail": "The image is too blurry to process. Please try uploading a clearer image." }
    ```
  - 413 Payload Too Large
    ```json
    { "detail": "The uploaded image exceeds the 10MB size limit. Please upload a smaller image." }
    ```
  - 415 Unsupported Media Type
    ```json
    { "detail": "Unsupported file format. Please upload an image in one of these formats: JPEG/JPG, PNG, BMP, TIFF, or WebP. Maximum file size is 10MB." }
    ```

#### Example (browser fetch)
```javascript
async function uploadImage(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/upload-image/', { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json(); // { image_id }
}
```

### 2) GET `/upload-image/{image_id}/status`
- Description: Check processing status and (when completed) fetch the raw grouped layout.
- Path params:
  - `image_id` (UUID): Task ID from the upload step.
- Responses:
  - 200 OK – Processing
    ```json
    { "status": "processing", "detail": "The image is currently being processed." }
    ```
  - 200 OK – Completed
    ```json
    { "status": "completed", "result": [ { "Shelf 1": [ /* books */ ] }, ... ] }
    ```
  - 200 OK – Failed
    ```json
    { "status": "failed", "detail": "<error message>" }
    ```
  - 404 Not Found
    ```json
    { "status": "not_found", "detail": "Image data not found for the provided ID. Please upload the image or check the ID for typos." }
    ```

#### Example (polling)
```javascript
async function pollStatus(imageId, intervalMs = 1200) {
  while (true) {
    const res = await fetch(`/upload-image/${imageId}/status`);
    const data = await res.json();
    if (data.status === 'completed' || data.status === 'failed') return data;
    await new Promise(r => setTimeout(r, intervalMs));
  }
}
```

### 3) POST `/upload-image/{image_id}/cancel`
- Description: Cancel a running processing task (only while `status == processing`).
- Path params: `image_id` (UUID)
- Responses:
  - 200 OK
    ```json
    { "message": "Cancellation request received. The task will be terminated shortly." }
    ```
  - 400 Bad Request
    ```json
    { "message": "Cannot cancel a task with status '<status>'." }
    ```
  - 404 Not Found
    ```json
    { "detail": "Upload task not found." }
    ```

### 4) GET `/organize-shelf/`
- Description: Generate an organized layout and move plan per shelf (idempotent/read-only).
- Query params:
  - `image_id` (required, UUID)
  - `sort_by` (optional, default `title`): one of `author`, `title`, `genre`, `height`
  - `sort_order` (optional, default `asc`): `asc` or `desc`
  - `shelf_number` (optional, default `1`): 1-based shelf index
- Responses:
  - 200 OK (matches `organize-api.json` shape)
    ```json
    {
      "image_id": "<uuid>",
      "requested_sort_by": "title",
      "sort_order": "asc",
      "shelf_number_processed": 1,
      "current_layout": [ { "Shelf 1": [ /* detected books */ ] } ],
      "organized_layout": [ { "Shelf 1": [ /* sorted books */ ] } ],
      "reorder_instructions": [
        {
          "action": "move",
          "shelf": "Shelf 1",
          "book_id": "<id>",
          "book_name": "...",
          "author": "...",
          "original_position": 4,
          "new_position": 1
        }
      ],
      "message": "Shelf organization complete."
    }
    ```
  - 202 Accepted (processing not finished)
    ```json
    { "status": "processing", "detail": "The image is currently being processed." }
    ```
  - 400 Bad Request – common cases
    - Invalid `sort_by` / `sort_order`
    - Invalid `shelf_number` (<= 0 or > number of shelves)
    - No shelves detected in the image
  - 404 Not Found – missing `image_id` cache

#### Example (browser fetch)
```javascript
async function getOrganizedLayout(imageId, { sortBy = 'title', sortOrder = 'asc', shelfNumber = 1 } = {}) {
  const params = new URLSearchParams({
    image_id: imageId,
    sort_by: sortBy,
    sort_order: sortOrder,
    shelf_number: String(shelfNumber),
  });
  const res = await fetch(`/organize-shelf/?${params.toString()}`);
  if (res.status === 202) return { status: 'processing' };
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

## Response Data Model
Each book object contains:
- `book_id` (string): unique per detection
- `position` (int): order within shelf (1-based)
- `bbox` ([x1, y1, x2, y2]) (ints): coordinates in original image
- `confidence` (string): detector confidence (e.g., "92%")
- `metadata` (object):
  - `Author` (string)
  - `Book Name` (string)
  - `Series_Name` (string; may be omitted if empty)
  - `year_published` (int|null)
  - `genre` (string[])
  - `isbn` (string[])

## UI Integration Tips
- Draw `bbox` overlays over the original image for a visual.
- Use `position` for left-to-right ordering within a shelf.
- Provide UI controls to change `sort_by`, `sort_order`, `shelf_number` and re-fetch.
- Show `reorder_instructions` as a checklist for physical reordering.

## Error Handling
- 202 processing: show progress indicator; keep polling status.
- 400 bad request: display the error message; validate query inputs on the client.
- 404 not found: prompt user to re-upload or verify `image_id`.
- Network/5xx: retry with exponential backoff.

## CORS
- Development allows `*`. In production, restrict allowed origins in the backend.

## Notes on Backend Behavior
- The backend uses YOLO for detection, clustering for shelf assignment, Gemini for OCR, and Google Books for enrichment.
- The organize endpoint is GET-only and safe to call repeatedly.
- The system selects the earliest publication year across matched editions and aggregates ISBNs across good matches.

## End-to-End Example
```javascript
async function runFlow(file) {
  const { image_id } = await uploadImage(file);
  const status = await pollStatus(image_id);
  if (status.status !== 'completed') throw new Error('Processing failed');
  const layout = await getOrganizedLayout(image_id, { sortBy: 'title', sortOrder: 'asc', shelfNumber: 1 });
  return layout; // render this in your UI
}
```
