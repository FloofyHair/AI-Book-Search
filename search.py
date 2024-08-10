from flask import Flask, request, render_template_string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

app = Flask(__name__)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the vectorized data
with open('vectorized_data.pkl', 'rb') as f:
    vectorized_data = pickle.load(f)

def search(query, top_n_scenes=5):
    # Vectorize the query
    query_vector = model.encode(query)

    # Compare query to each scene
    scene_similarities = []
    for scene_index, scene in enumerate(vectorized_data, start=1):
        similarity = cosine_similarity([query_vector], [scene['scene_vector']])[0][0]
        scene_similarities.append((scene_index, similarity))

    # Get top N scenes
    top_n_scenes = sorted(scene_similarities, key=lambda x: x[1], reverse=True)[:top_n_scenes]

    results = []
    for scene_index, scene_similarity in top_n_scenes:
        scene = vectorized_data[scene_index - 1]
        
        # Compare query to each sentence in the scene
        colored_text = color_text(scene['scene_text'], scene['sentences'], query_vector)
        
        results.append({
            'scene_number': scene_index,
            'scene_similarity': scene_similarity,
            'colored_text': colored_text
        })

    return results

def color_text(text, sentences, query_vector):
    colored_sentences = []
    for sentence, sentence_vector in sentences:
        similarity = cosine_similarity([query_vector], [sentence_vector])[0][0]
        color = get_color_for_similarity(similarity)
        colored_sentences.append((sentence, color))
    
    # Sort sentences by length (longest first) to avoid issues with substrings
    colored_sentences.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Replace each sentence with its colored version
    for sentence, color in colored_sentences:
        text = text.replace(sentence, f'<span style="color: {color};">{sentence}</span>')
    
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
            }
            h1 { color: #bb86fc; }
            form { margin-bottom: 20px; }
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
            .scene { 
                margin-bottom: 20px; 
                border: 1px solid #444; 
                padding: 15px; 
                background-color: #2d2d2d;
                margin-left: 25vw;
                margin-right: 25vw;
            }
            .scene h2 { 
                margin-top: 0; 
                color: #03dac6;
            }
            .scene-similarity {
                color: #03dac6;
                font-size: 20px;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <h1>Book Search</h1>
        <form method="post">
            <input type="text" name="query" placeholder="Enter your search query" required>
            <input type="submit" value="Search">
        </form>
        {% if results %}
            {% for result in results %}
                <div class="scene">
                    <h2>Scene {{ result['scene_number'] }}</h2>
                    <p>Scene similarity: <span class="scene-similarity">{{ "%.4f"|format(result['scene_similarity']) }}</span></p>
                    <p>{{ result['colored_text']|safe }}</p>
                </div>
            {% endfor %}
        {% endif %}
    </body>
    </html>
    ''', results=results)

if __name__ == '__main__':
    app.run(debug=True)