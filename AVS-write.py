from aerospike_vector_search import AdminClient, Client, types
import ebooklib
from ebooklib import epub
import bs4
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import time

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

BOOK_PATH = "./books/" + BOOK_NAME + ".epub"

start_time = time.time()

seeds = types.HostPort(
    host=HOST,
    port=PORT
)

client = Client(seeds=seeds)
admin_client = AdminClient(seeds=seeds)

# ----------------- Index Creation -----------------

def create_index(INDEX_NAME, SET_NAME):
    index_exists = False
    # Check if the index already exists. If not, create it
    for index in admin_client.index_list():
        if index["id"]["namespace"] == NAMESPACE and index["id"]["name"] == INDEX_NAME:
            index_exists = True
            print(f"{INDEX_NAME} already exists. Skipping creation")
            break

    if not index_exists:
        print(f"{INDEX_NAME} does not exist. Creating index")
        
        admin_client.index_create(
        namespace=NAMESPACE,
        name=INDEX_NAME,    
        sets=SET_NAME,
         
        vector_field=VECTOR_KEY,
        vector_distance_metric=MODEL_DISTANCE_CALC,
            dimensions=MODEL_DIM,
        )

create_index(CHAPTER_INDEX, "chapters")
create_index(PARAGRAPH_INDEX, "paragraphs")
print()

admin_client.close()


# ----------------- Vector Insertion -----------------

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

def extract_chapters_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    soups = []
    
    index = 0
    
    # Start the timer for chapter extraction
    start_time = time.time()
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html_content = item.get_content().decode('utf-8')
            soup = bs4.BeautifulSoup(html_content, 'html.parser')
            
            title = soup.find('h1')
            if title:
                title = title.get_text()
            else:
                title = ""
            
            chapter = {
                "chapter-index": index,
                "chapter-title": title,
                "text": soup.get_text(),
                "vector": None  # create_embedding(soup.get_text(), "chapter")
            }
            
            chapters.append(chapter)
            soups.append(soup)
            
            print(f"{index:3}", end=", ")
            if index % 10 == 9:
                print(f"\033[92m({time.time()-start_time:.2f}s)\033[0m")
                
            index += 1
    
    print()
    return chapters, soups

def extract_paragraphs_from_chapter(chapter, soup):
    yellow_color = "\033[93m"
    reset_color = "\033[0m"
    print(f"{yellow_color}CHAPTER {chapter['chapter-index']}:{reset_color}")
    
    chapter_index = chapter["chapter-index"]
    chapter_title = chapter["chapter-title"]
    chapter_html = soup
    
    paragraphs = []
        
    index = 0
    for p in chapter_html.find_all('p'):
        paragraph = {
            "chapter-index": chapter_index,
            "chapter-title": chapter_title,
            "paragraph-index": index,
            "text": p.get_text(),
            "vector": create_embedding(p.get_text(), "paragraph")
        }
        paragraphs.append(paragraph)
        
        print(f"{chapter_index*10000+index:5}", end=", ")
        if index % 10 == 9:
            print(f"\033[92m({time.time()-start_time:.2f}s)\033[0m")
        
        index += 1
    
    if index % 10 != 0:
        print(f"\033[92m({time.time()-start_time:.2f}s)\033[0m")
    
    return paragraphs


def insert_record(SET_NAME, KEY, RECORD_DATA):
    client.upsert(
        namespace=NAMESPACE,
        set_name=SET_NAME,
        key=KEY,
        record_data=RECORD_DATA,
    )

def delete_record(SET_NAME, KEY):
    client.delete(
        namespace=NAMESPACE,
        set_name=SET_NAME,
        key=KEY,
    )

vector = [] # create dummy 384 dimension vector
for i in range(MODEL_DIM):
    vector.append(0.0)

# INSERT CHAPTERS

print("Embedding chapters...")
chapters, soups = extract_chapters_from_epub(BOOK_PATH)
print("Chapters embedded")

print()

# print("Inserting chapters...")
# print("KEYS: ")
# i = 0
# for chapter in chapters:
#     insert_record("chapters", chapter["chapter-index"], chapter)
#     print(f"{chapter['chapter-index']:3}", end=", ")
#     if i % 10 == 9:
#         print()
#     i += 1
# print()
# print("Chapters inserted")

# INSERT PARAGRAPHS

print("Embedding and inserting paragraphs...")
for chapter in chapters:
    paragraphs = extract_paragraphs_from_chapter(chapter, soups[chapter["chapter-index"]])
    for paragraph in paragraphs:
        insert_record("paragraphs", paragraph["chapter-index"]*10000+paragraph["paragraph-index"], paragraph)
print("Paragraphs embedded and inserted")






# ----------------- Wait for Indexing -----------------

print()
print("Waiting for indexing to complete...")
#client.wait_for_index_completion(namespace=NAMESPACE, name=CHAPTER_INDEX)
print("CHAPTERS indexing complete")
#client.wait_for_index_completion(namespace=NAMESPACE, name=PARAGRAPH_INDEX)
#print("PARAGRAPHS indexing complete")
print()

# ----------------- Read Index -----------------

def check_if_indexed(INDEX_NAME, SET_NAME, KEY):
    return client.is_indexed(
        namespace=NAMESPACE,
        key=KEY,
        index_name=INDEX_NAME,
        set_name=SET_NAME
    )

# print("CHAPTERS:")
# for i in range(len(chapters)):
#     print(f"{chapters[i]['chapter-index']:3}", end=": ")
#     print(check_if_indexed(CHAPTER_INDEX, "chapters", chapters[i]["chapter-index"]), end=", ")
#     if i % 10 == 9:
#         print()

# ----------------- Search -----------------

def print_results(results):
    for result in results:
        keys = result.key
        fields = result.fields
        
        vector = [round(v, 2) for v in fields["vector"][:2]]
        chapter_index = str(fields["chapter-index"])
        chapter_title = str(fields["chapter-title"][:12])
        text = str(fields["text"][fields["text"].find(".\n")+2:fields["text"].find(".\n")+60])+"..."
        text = text.replace("\n", " ")
        
        print(keys, end="\t")
        print("->", end="\t")
        print(chapter_index, end="\t")
        print(chapter_title, end="\t")
        print(text, end="\t")
        print(vector)

print("\n-------------------------------------------------------------- Search --------------------------------------------------------------")

results = client.result = client.vector_search(
    namespace=NAMESPACE,
    index_name=CHAPTER_INDEX,
    query=vector,
    limit=144,
)
#print("CHAPTERS:")
#print_results(results)

print()

results = client.result = client.vector_search(
    namespace=NAMESPACE,
    index_name=PARAGRAPH_INDEX,
    query=vector,
    limit=144,
)
print("PARAGRAPHS:")
print_results(results)

client.close()