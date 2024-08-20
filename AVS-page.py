from aerospike_vector_search import AdminClient, Client, types
from flask import Flask, request, render_template_string
import ebooklib
from ebooklib import epub
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import bs4

from dotenv import load_dotenv
import os

load_dotenv()
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

global NAMESPACE, INDEX_CHAPTERS, INDEX_PARAGRAPHS, VECTOR_KEY, MODEL_DISTANCE_CALC, MODEL_DIM

NAMESPACE = "test"
BOOK_NAME = "moby-dick"

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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
model.eval()
def create_embedding(data, type: str = "document"):
    doc = f"search_{type}: {data}"
    encoded_input = tokenizer(doc, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
    
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].tolist()

# ------------- Query Search ------------- #

query = "The whale's skeleton is described"
query_vector = create_embedding(query, "query")

results = client.result = client.vector_search(
    namespace=NAMESPACE,
    index_name=CHAPTER_INDEX,
    query=query_vector,
    limit=20,
)

# ------------- Display Results ------------- #

app = Flask(__name__)
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
            query=create_embedding(query, "query"),  # Create embedding for the query
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

    return render_template_string('''
    <!doctype html>
    <html lang="en">
    <head>
        <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='styles.css') }}">
        <title>{{ book_name }}</title>
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
    ''', book_name=BOOK_NAME, content=text_content, results=results_html)

# ------------- Run the App ------------- #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

client.close()