from flask import Flask, request, render_template_string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from aerospike_vector_search import Client, types

HOST = "34.136.186.255"
PORT = 5000
seeds = types.HostPort(host=HOST, port=PORT)
client = Client(seeds=seeds)

app = Flask(__name__)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vectorized_data():
    # Load vectorized data from Aerospike
    vectorized_data = []
    # Assuming you have a method to fetch data from Aerospike
    # Replace 'your_namespace' and 'your_set' with actual values
    records = client.query('your_namespace', 'your_set').results()
    for record in records:
        vectorized_data.append(record)  # Adjust based on your record structure
    return vectorized_data

# Load the vectorized data
vectorized_data = load_vectorized_data()


def search(query, top_n_chapters=5):
    # Vectorize the query
    query_vector = model.encode(query)

    # Compare query to each chapter
    chapter_similarities = []
    for chapter_index, chapter in enumerate(vectorized_data, start=1):
        similarity = cosine_similarity([query_vector], [chapter['chapter_vector']])[0][0]
        chapter_similarities.append((chapter_index, similarity))

    # Get top N chapters
    top_n_chapters = sorted(chapter_similarities, key=lambda x: x[1], reverse=True)[:top_n_chapters]

    results = []
    for chapter_index, chapter_similarity in top_n_chapters:
        chapter = vectorized_data[chapter_index - 1]
        
        # Compare query to each paragraph in the chapter
        colored_text = color_text(chapter['chapter_text'], chapter['paragraphs'], query_vector)
        
        results.append({
            'chapter_number': chapter_index,
            'chapter_similarity': chapter_similarity,
            'colored_text': colored_text
        })

    return results

def color_text(text, paragraphs, query_vector):
    colored_paragraphs = []
    for paragraph in paragraphs:
        similarity = cosine_similarity([query_vector], [paragraph['paragraph_vector']])[0][0]
        color = get_color_for_similarity(similarity)
        colored_paragraphs.append((paragraph['paragraph_text'], color))
    
    # Sort paragraphs by length (longest first) to avoid issues with substrings
    colored_paragraphs.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Replace each paragraph with its colored version
    for paragraph, color in colored_paragraphs:
        escaped_paragraph = re.escape(paragraph)
        text = re.sub(
            escaped_paragraph,
            f'<div style="color: {color};">{paragraph}</div>',
            text,
            flags=re.DOTALL
        )
    
    return text

def get_color_for_similarity(similarity):
    # Convert similarity to a color between red (low) and green (high)
    r = int(255 * (1 - similarity))
    g = int(255)
    b = int(255 * (1 - similarity))
    return f'rgb({r},{g},{b})'

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        results = search(query)
    
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Book Search</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            body { 
                font-family: 'Roboto', sans-serif; 
                line-height: 1.6; 
                padding: 20px; 
                background-color: #1e1e1e; 
                color: #e0e0e0;
                margin: 0;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 0 20px;
            }
            h1 { 
                color: #bb86fc; 
                text-align: center;
            }
            form { 
                margin-bottom: 20px; 
                text-align: center;
            }
            input[type="text"] { 
                width: 300px; 
                padding: 10px; 
                background-color: #333; 
                border: 1px solid #555; 
                color: #e0e0e0; 
                font-size: 16px;
            }
            input[type="submit"] { 
                padding: 10px 20px; 
                background-color: #bb86fc; 
                color: #1e1e1e; 
                border: none; 
                cursor: pointer;
            }
            .chapter { 
                margin-bottom: 20px; 
                border: 1px solid #444; 
                padding: 15px; 
                background-color: #2d2d2d;
                width: 100%;
                box-sizing: border-box;
            }
            .chapter h2 { 
                margin-top: 0; 
                color: #03dac6;
            }
            .chapter-similarity {
                color: #03dac6;
                font-size: 20px;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Book Search</h1>
            <form method="post">
                <input type="text" name="query" placeholder="Enter your search query" required>
                <input type="submit" value="Search">
            </form>
            {% if results %}
                {% for result in results %}
                    <div class="chapter">
                        <h2>Chapter {{ result['chapter_number'] }}</h2>
                        <p>Chapter similarity: <span class="chapter-similarity">{{ "%.4f"|format(result['chapter_similarity']) }}</span></p>
                        <p>{{ result['colored_text']|safe }}</p>
                    </div>
                {% endfor %}
            {% endif %}
        </div>
    </body>
    </html>
    ''', results=results)

if __name__ == '__main__':
    app.run(debug=True)