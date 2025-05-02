"""
Book Downloader for Friendship Bot

This script downloads public domain books related to women's psychology,
communication patterns, and relationship dynamics from Project Gutenberg
and other sources.
"""

import os
import requests
import time
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup

# Create directories
os.makedirs('data/books/raw', exist_ok=True)

# List of public domain books related to women's psychology and communication
# These are Project Gutenberg IDs or direct URLs to text files
BOOK_SOURCES = [
    # Project Gutenberg books
    {
        "type": "gutenberg",
        "id": "66049",  # Woman in Modern Society by Earl Barnes
        "title": "Woman in Modern Society"
    },
    {
        "type": "gutenberg",
        "id": "67811",  # The Psychology of Female Violence by Anna Motz
        "title": "The Psychology of Female Violence"
    },
    {
        "type": "gutenberg",
        "id": "17573",  # Woman and the New Race by Margaret Sanger
        "title": "Woman and the New Race"
    },
    {
        "type": "gutenberg",
        "id": "9105",  # The Psychology of Beauty by Ethel D. Puffer
        "title": "The Psychology of Beauty"
    },
    {
        "type": "gutenberg",
        "id": "65083",  # Woman as Decoration by Emily Burbank
        "title": "Woman as Decoration"
    },
    # Add more books as needed
]

def download_gutenberg_book(book_id, output_dir, title=None):
    """
    Download a book from Project Gutenberg.
    
    Args:
        book_id: Project Gutenberg book ID
        output_dir: Directory to save the book
        title: Book title (for filename)
        
    Returns:
        Path to the downloaded file
    """
    # Construct the URL
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    # Construct the output filename
    if title:
        filename = f"{title.replace(' ', '_').lower()}.txt"
    else:
        filename = f"gutenberg_{book_id}.txt"
    
    output_path = os.path.join(output_dir, filename)
    
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"Book already exists: {output_path}")
        return output_path
    
    # Try to download the book
    try:
        response = requests.get(url)
        if response.status_code != 200:
            response = requests.get(alt_url)
            if response.status_code != 200:
                print(f"Failed to download book {book_id}: {response.status_code}")
                return None
        
        # Save the book
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Downloaded book: {output_path}")
        
        # Be nice to the server
        time.sleep(1)
        
        return output_path
    
    except Exception as e:
        print(f"Error downloading book {book_id}: {e}")
        return None

def download_direct_url(url, output_dir, title=None):
    """
    Download a book from a direct URL.
    
    Args:
        url: URL to the text file
        output_dir: Directory to save the book
        title: Book title (for filename)
        
    Returns:
        Path to the downloaded file
    """
    # Construct the output filename
    if title:
        filename = f"{title.replace(' ', '_').lower()}.txt"
    else:
        filename = url.split('/')[-1]
        if not filename.endswith('.txt'):
            filename += '.txt'
    
    output_path = os.path.join(output_dir, filename)
    
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"Book already exists: {output_path}")
        return output_path
    
    # Try to download the book
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to download book from {url}: {response.status_code}")
            return None
        
        # Save the book
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Downloaded book: {output_path}")
        
        # Be nice to the server
        time.sleep(1)
        
        return output_path
    
    except Exception as e:
        print(f"Error downloading book from {url}: {e}")
        return None

def search_gutenberg(keywords, max_results=10):
    """
    Search Project Gutenberg for books matching keywords.
    
    Args:
        keywords: List of keywords to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of book dictionaries with type, id, and title
    """
    search_query = '+'.join(keywords)
    search_url = f"https://www.gutenberg.org/ebooks/search/?query={search_query}&submit_search=Go%21"
    
    try:
        response = requests.get(search_url)
        if response.status_code != 200:
            print(f"Failed to search Gutenberg: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Find book entries
        for book_entry in soup.select('.booklink'):
            try:
                # Extract book ID from link
                link = book_entry.select_one('a.link')
                if not link:
                    continue
                    
                href = link.get('href', '')
                book_id = href.split('/')[-1]
                
                # Extract title
                title_elem = book_entry.select_one('span.title')
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                
                results.append({
                    "type": "gutenberg",
                    "id": book_id,
                    "title": title
                })
                
                if len(results) >= max_results:
                    break
            
            except Exception as e:
                print(f"Error parsing book entry: {e}")
                continue
        
        return results
    
    except Exception as e:
        print(f"Error searching Gutenberg: {e}")
        return []

def main():
    """Main function to download books."""
    parser = argparse.ArgumentParser(description='Download books for training the friendship bot')
    parser.add_argument('--output_dir', default='data/books/raw', help='Directory to save the books')
    parser.add_argument('--search', action='store_true', help='Search for additional books')
    parser.add_argument('--keywords', nargs='+', default=['women', 'psychology', 'communication'],
                        help='Keywords to search for books')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download predefined books
    downloaded_files = []
    
    for book in tqdm(BOOK_SOURCES, desc="Downloading predefined books"):
        if book["type"] == "gutenberg":
            file_path = download_gutenberg_book(book["id"], args.output_dir, book.get("title"))
        elif book["type"] == "url":
            file_path = download_direct_url(book["url"], args.output_dir, book.get("title"))
        else:
            print(f"Unknown book type: {book['type']}")
            file_path = None
        
        if file_path:
            downloaded_files.append(file_path)
    
    # Search for additional books if requested
    if args.search:
        print(f"Searching for books with keywords: {args.keywords}")
        search_results = search_gutenberg(args.keywords)
        
        if search_results:
            print(f"Found {len(search_results)} additional books")
            
            for book in tqdm(search_results, desc="Downloading search results"):
                if book["type"] == "gutenberg":
                    file_path = download_gutenberg_book(book["id"], args.output_dir, book.get("title"))
                    if file_path:
                        downloaded_files.append(file_path)
        else:
            print("No additional books found")
    
    print(f"Downloaded {len(downloaded_files)} books to {args.output_dir}")
    
    if downloaded_files:
        print("\nNext steps:")
        print("1. Run the book extractor to process the books:")
        print("   python data_collection/book_extractor.py")
        print("2. Integrate the book data with other training data:")
        print("   python data_collection/data_processor.py")
        print("3. Train the model:")
        print("   python train_model.py")

if __name__ == "__main__":
    main()
