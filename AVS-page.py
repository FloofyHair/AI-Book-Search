from aerospike_vector_search import AdminClient, Client, types
from flask import Flask, request, render_template_string
from flask_cors import CORS  # Import CORS for cross-origin requests
import ebooklib
from ebooklib import epub
import bs4
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

global NAMESPACE, INDEX_CHAPTERS, INDEX_PARAGRAPHS, VECTOR_KEY, MODEL_DISTANCE_CALC, MODEL_DIM

NAMESPACE = "test"
BOOK_NAME = "moby-dick"
BOOK_TITLE = "Moby Dick"

PARAGRAPH_INDEX = BOOK_NAME + "-paragraphs"
CHAPTER_INDEX = BOOK_NAME + "-chapters"

VECTOR_KEY = "vector"
MODEL_DISTANCE_CALC = types.VectorDistanceMetric.COSINE
MODEL_DIM = 768

seeds = types.HostPort(
    host=HOST,
    port=PORT
)
client = Client(seeds=seeds)

# ------------- Embedding ------------- #

def create_embedding(query):
    embedding = requests.get(f"https://server.vector-rag.aerospike.com/rest/v1/embed/?q={query}")
    return embedding.json()

# ------------- Display Results ------------- #

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
book = epub.read_epub("./books/" + BOOK_NAME + ".epub")

@app.route('/', methods=['GET', 'POST'])
def display_book():
    # Extract text from the EPUB
    text_content = ""
    index = -1
    for item in book.get_items():
        index += 1
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html_content = item.get_content().decode('utf-8')
            soup = bs4.BeautifulSoup(html_content, 'html.parser')
                
            # Remove <img> tags
            for img in soup.find_all('img'):
                img.decompose()  # Remove the <img> tag

            # Remove <a> tags
            for a in soup.find_all('a'):
                a.unwrap()  # Remove the <a> tag but keep the text
            
            # Add anchor tags to the paragraphs
            paragraphs = soup.find_all('p')
            for paragraph_index, p in enumerate(paragraphs):
                chapter_id = index 
                p['id'] = f"{(chapter_id-7) * 10000 + paragraph_index}"

            text_content += str(soup)
            
    query = request.form.get('query', '')  # Get the query from the form
    results_html = ""

    if query:  # If a query is provided, perform the search
        results = client.vector_search(
            namespace=NAMESPACE,
            index_name=PARAGRAPH_INDEX,
            query=create_embedding(query),  # Create embedding for the query
            limit=20,
        )

        # Create HTML for results
        results_html = ''.join(f'''
            <section>
                <h3>
                    <a href="#{result.key.key}">{result.fields['chapter-title']}</a>
                    <span style="color: rgb({int(255 * result.distance)}, 255, {int(255 * result.distance)});"> ({result.distance:.2f}) </span>
                </h3>
                <p>{result.fields['text'][:300]}...</p>
            </section>
        ''' for result in results)

    return render_template_string('''<!doctype html>
    <html lang="en">
    <head>
        <link rel="icon" href="{{ url_for('static', filename='icon.png') }}" type="image/png">  <!-- Updated line -->
        <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='styles.css') }}">
        <title>AI Book Search</title>
    </head>
    <body>
    <div class="container">
        <div class="epub-content">
            {{ content|safe }}
        </div>
        <div class="search-engine">
            <form method="POST" action="/">
                <input type="text" name="query" placeholder="Enter your search query" required>
                <button type="submit">Search</button>
            </form>
            <div class="results">
                {{ results|safe }}
            </div>
        </div>
    </div>
    </body>
    </html>
    ''', content=text_content, results=results_html)

# ------------- Run the App ------------- #

if __name__ == '__main__':
    from waitress import serve  # Import Waitress
    try:
        serve(app, host='0.0.0.0', port=5000)  # Use Waitress to serve the app
    finally:
        client.close()  # Ensure the client is closed on exit