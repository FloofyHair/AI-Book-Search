import re
import pickle
from sentence_transformers import SentenceTransformer

def extract_scenes(text):
    scenes = []
    
    lines = text.split('\n')
    start = 0
    
    for line in lines:
        if line.isdigit():
            end = lines.index(line)
            
            scene = lines[start:end]
            scene = [line.strip().replace('"', '') for line in scene if line.strip()]
            scene = ' '.join(scene)
            scenes.append(scene)
            
            start = end
    
    scenes = scenes[1:]
    return scenes

def extract_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the file content
with open('the-catcher-in-the-rye.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Extract scenes
scenes = extract_scenes(content)

# Vectorize scenes and sentences
vectorized_data = []
for scene_number, scene in enumerate(scenes, start=1):
    scene_vector = model.encode(scene)
    sentences = extract_sentences(scene)
    
    vectorized_sentences = []
    for sentence in sentences:
        sentence_vector = model.encode(sentence)
        vectorized_sentences.append((sentence, sentence_vector))
    
    vectorized_data.append({
        'scene_text': scene,
        'scene_vector': scene_vector,
        'sentences': vectorized_sentences
    })

# Save using pickle
with open('vectorized_data.pkl', 'wb') as f:
    pickle.dump(vectorized_data, f)

# Now vectorized_data contains all the required information
# and is saved to disk using pickle