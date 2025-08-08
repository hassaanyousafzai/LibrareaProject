
# Librarea: AI-Powered Bookshelf Organizer

Librarea is an intelligent bookshelf organization tool that uses computer vision and AI to help you manage your physical book collection. Simply upload a picture of your bookshelf, and Librairea will automatically detect, identify, and organize your books, providing you with a digital inventory and smart reorganization suggestions.

## Features

- **Automatic Book Detection:** Utilizes a fine-tuned YOLO11 model to accurately detect the location of each book on your shelves from a single image.
- **Intelligent Spine Detection:** Advanced filtering system distinguishes between book spines and front/back covers, ensuring only valid spines are processed.
- **Rich Metadata Extraction:** Employs Google's Gemini 2.0 Flash for precise OCR to extract titles and authors from the book spines.
- **Data Enrichment:** Leverages the Google Books API to enrich the extracted data with additional information like publication year, genre, and ISBN.
- **Smart Organization:** Groups detected books into shelves and allows you to sort them by various criteria (author, title, genre, height) to generate a reorganized shelf layout.
- **Reordering Instructions:** Provides clear, step-by-step instructions to physically reorganize your books according to your desired order.
- **RESTful API:** A robust backend built with FastAPI provides a clean and easy-to-use interface for all the functionalities.

## How It Works

1.  **Image Upload:** The user uploads an image of a bookshelf through the `/upload-image/` endpoint.
2.  **Image Preprocessing:** The image is preprocessed and resized for efficient and accurate object detection.
3.  **Book Detection:** The YOLO11 model processes the image and identifies the bounding boxes for each book.
4.  **Spine Filtering:** Advanced geometric analysis filters detections to distinguish book spines from front/back covers based on aspect ratio, width, and height characteristics.
5.  **Spine Cropping:** The identified spine regions are cropped to isolate the spines for processing.
6.  **OCR and Metadata Extraction:** Each spine image is sent to the Gemini 2.0 Flash model, which performs OCR to read the book's title and author.
7.  **Metadata Enrichment:** The extracted title and author are used to query the Google Books API, fetching additional details like genre, publication year, and ISBN.
8.  **Shelf Grouping:** The processed books are intelligently grouped into shelves based on their vertical alignment in the original image.
9.  **Organization:** The user can then call the `/organize-shelf/` endpoint (GET) with their desired sorting criteria to receive an organized layout and move plan.
10. **Reordering Plan:** The system generates a new, organized layout for the shelf and provides a list of moves to achieve the desired arrangement.

## Technologies Used

- **Backend:** Python, FastAPI
- **Computer Vision:** OpenCV, Ultralytics YOLO11
- **AI/ML:** Google Gemini 2.0 Flash
- **APIs:** Google Books API
- **Core Libraries:** Uvicorn, Pydantic, Requests

## Getting Started

### Prerequisites

- Python 3.8+
- An environment with the required packages installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hassaanyousafzai/LibrareaProject.git
    cd LibrareaProject
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    - You will need to create a `.env` file in the root directory and add your Google Books API key and configure other settings as defined in `app/core/config.py`.
    
    **Optional Configuration Variables:**
    ```env
    # Google Books API
    GOOGLE_BOOKS_API_KEY=your_api_key_here
    
    # YOLO Configuration
    YOLO_CONF=0.4
    YOLO_MODEL_PATH=../runs/detect/librarea_yolov11_run/weights/best.pt
    
    # Spine Detection Tuning (adjust these to fine-tune spine vs cover detection)
    SPINE_MIN_ASPECT_RATIO=2.0      # Minimum height/width ratio for valid spines
    SPINE_MAX_WIDTH_RATIO=0.15      # Maximum width as fraction of image width
    SPINE_MIN_HEIGHT_RATIO=0.1      # Minimum height as fraction of image height
    ```

### Running the Application

To start the FastAPI server, run the following command from the root of the project:

```bash
uvicorn app.main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `POST /upload-image/`: Upload an image of a bookshelf.
- `GET /status/{image_id}`: Get real-time status updates for an upload task.
- `GET /organize-shelf/`: Fetch a reorganization plan for a specific shelf.
  - Query params:
    - `image_id` (required): the upload task ID (UUID)
    - `sort_by` (optional, default `title`): one of `author`, `title`, `genre`, `height`
    - `sort_order` (optional, default `asc`): `asc` or `desc`
    - `shelf_number` (optional, default `1`): shelf index starting from 1
  - Returns the same response body as shown in `organize-api.json`.
- `POST /upload-image/{image_id}/cancel`: Cancel a running upload task.
- `GET /upload-image/{image_id}/status`: **(Deprecated)** Get the status of an upload task. 
