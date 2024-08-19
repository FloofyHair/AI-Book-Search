from aerospike_vector_search import AdminClient, Client, types
from flask import Flask, request, render_template_string
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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

IGNORE_CHAPTERS = [5, 6]

def print_results(results):
    print()
    
    yellow_color = "\033[93m"  # Yellow color
    reset_color = "\033[0m"  # Reset to default color

    print(f"Query: {yellow_color}\n{query}\n{reset_color}")
    
    red_color = "\033[91m"  # Red color
    reset_color = "\033[0m"  # Reset to default color

    print(f"{red_color}Keys{reset_color}", end="\t\t\t\t\t\t\t")
    print(f"{red_color}Distance{reset_color}", end="\t")
    print(f"{red_color}Chapter{reset_color}", end="\t\t")
    print(f"{red_color}Title{reset_color}", end="\t\t\t")
    print(f"{red_color}Text{reset_color}", end="\t\t\t\t\t\t")
    print(f"{red_color}Vector{reset_color}")
    
    for result in results:
        if result.fields["chapter-index"] in IGNORE_CHAPTERS:
            continue
        
        keys = result.key
        fields = result.fields
        
        vector = str([round(v, 2) for v in fields["vector"][:2]])[:-1]+", ...]"

        chapter_index = str(fields["chapter-index"]).ljust(5)
        chapter_title = str(fields["chapter-title"][:20]).ljust(20)
        text = (str(fields["text"][fields["text"].find(".\n")+2:fields["text"].find(".\n")+40]).replace("\n", " ") + "...").ljust(40)
        distance = round(result.distance, 2)

        # Determine color based on distance
        # More green for more similar (lower distance), more red for more different (higher distance)
        red = int(255 * distance)
        green = int(255 * (1 - distance))
        color = f"\033[38;2;{red};{green};0m"
        reset_color = "\033[0m"  # Reset to default color
        
        print(keys, end="   \t")
        print("--", end="")
        
        print(f'({color}{distance}{reset_color})', end="")  # Print distance in color
        
        print("->", end="\t")
        print(chapter_index, end="\t\t")
        print(chapter_title, end="\t")
        print(text, end="\t")
        print(vector)


results = client.result = client.vector_search(
    namespace=NAMESPACE,
    index_name=CHAPTER_INDEX,
    query=query_vector,
    limit=20,
)

print_results(results)

client.close()
