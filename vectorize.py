import re
import pickle
import time
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

def extract_chapters_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html_content = item.get_content().decode('utf-8')
            paragraphs = extract_paragraphs(html_content)
            if paragraphs:  # Only add non-empty chapters
                chapters.append('\n\n'.join(paragraphs))
    return chapters

def extract_paragraphs(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    return [p.get_text().strip() for p in paragraphs if p.get_text().strip()]

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the EPUB file content
epub_path = 'moby-dick.epub'  # Update this to your EPUB file path
chapters = extract_chapters_from_epub(epub_path)

# Start timing
start_time = time.time()

print(f"Starting vectorization of {len(chapters)} chapters...")

# Vectorize chapters and paragraphs
vectorized_data = []
total_paragraphs = 0

for chapter_number, chapter in enumerate(chapters, start=1):
    chapter_start_time = time.time()
    
    chapter_vector = model.encode(chapter)
    paragraphs = chapter.split('\n\n')  # Split the chapter text into paragraphs
    total_paragraphs += len(paragraphs)
    
    vectorized_paragraphs = []
    for paragraph in paragraphs:
        paragraph_vector = model.encode(paragraph)
        vectorized_paragraphs.append({
            'paragraph_text': paragraph,
            'paragraph_vector': paragraph_vector
        })
    
    vectorized_data.append({
        'chapter_text': chapter,
        'chapter_vector': chapter_vector,
        'paragraphs': vectorized_paragraphs
    })
    
    chapter_end_time = time.time()
    chapter_duration = chapter_end_time - chapter_start_time
    total_duration = chapter_end_time - start_time
    
    print(f"Completed Chapter {chapter_number}/{len(chapters)} - "
          f"Paragraphs: {len(paragraphs)} - "
          f"Time: {chapter_duration:.2f}s - "
          f"Total Time: {total_duration:.2f}s")

# Save using pickle
with open('vectorized_data.pkl', 'wb') as f:
    pickle.dump(vectorized_data, f)

end_time = time.time()
total_duration = end_time - start_time

print(f"\nVectorization completed!")
print(f"Total chapters: {len(chapters)}")
print(f"Total paragraphs: {total_paragraphs}")
print(f"Total time: {total_duration:.2f} seconds")
print(f"Average time per chapter: {total_duration/len(chapters):.2f} seconds")
print(f"Average time per paragraph: {total_duration/total_paragraphs:.2f} seconds")
print("Vectorized data saved to 'vectorized_data.pkl'")