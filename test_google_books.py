import requests
import json

# Define the book title you want to search for
book_title = "SOMALIA: ECONOMY WITHOUT STATE"

# Construct the API URL. We'll use the 'q' parameter for a general search.
api_url = f"https://www.googleapis.com/books/v1/volumes?q={requests.utils.quote(book_title)}"

print(f"Searching for book metadata using URL: {api_url}\n")

try:
    # Make the GET request to the Google Books API
    response = requests.get(api_url)
    response.raise_for_status()

    # Parse the JSON response
    data = response.json()

    # Check if any books were found
    if data['totalItems'] > 0:
        print("Book found! Extracting metadata...\n")

        # The results are in an array called 'items'
        first_result = data['items'][0]
        volume_info = first_result['volumeInfo']

        # Extract the required information from the API response
        book_name = volume_info.get('title', 'N/A')
        authors = volume_info.get('authors', [])
        authors_str = ", ".join(authors) if authors else "N/A"
        
        # Extract the published year
        published_date = volume_info.get('publishedDate')
        year_published = published_date.split('-')[0] if published_date else None

        # Extract series name (not a standard field, often found in title)
        # We'll assume for this example it's not present, so we set it to empty string
        series_name = ""

        # Extract genres (categories)
        genres = volume_info.get('categories', [])

        # Extract ISBNs
        isbns = []
        industry_identifiers = volume_info.get('industryIdentifiers', [])
        for identifier in industry_identifiers:
            if 'identifier' in identifier:
                isbns.append(identifier['identifier'])
        
        # Create the metadata dictionary
        metadata = {
            "metadata": {
                "Author": authors_str,
                "Book Name": book_name,
                "Series_Name": series_name,
                "year_published": int(year_published) if year_published else None,
                "genre": genres,
                "isbn": isbns
            }
        }
        
        # Print the metadata in a nicely formatted JSON string
        print(json.dumps(metadata, indent=4))

    else:
        print("No results found for this book title.")

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    print("This could be a rate-limiting issue (Error 429).")
except Exception as err:
    print(f"An unexpected error occurred: {err}")
