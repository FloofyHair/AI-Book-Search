from aerospike_vector_search import AdminClient, Client, types
import ebooklib
from ebooklib import epub
import bs4
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import time
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn
import warnings

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

console = Console()

from dotenv import load_dotenv
import os

load_dotenv()
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
NAMESPACE = "test"

VECTOR_KEY = "vector"
MODEL_DISTANCE_CALC = types.VectorDistanceMetric.COSINE
MODEL_DIM = 768

seeds = types.HostPort(
    host=HOST,
    port=PORT
)

client = Client(seeds=seeds)
admin_client = AdminClient(seeds=seeds)

def create_index(INDEX_NAME, SET_NAME):
    index_exists = False
    # Check if the index already exists. If not, create it
    for index in admin_client.index_list():
        if index["id"]["namespace"] == NAMESPACE and index["id"]["name"] == INDEX_NAME:
            index_exists = True
            console.print(f"\t{SET_NAME.upper()}:\tIndex '{INDEX_NAME}' already exists. Skipping creation")
            break

    if not index_exists:
        console.print(f"\t[green]{SET_NAME.upper()}:\tCreating index '{INDEX_NAME}'...[/green]")
        
        admin_client.index_create(
        namespace=NAMESPACE,
        name=INDEX_NAME,    
        sets=SET_NAME,
         
        vector_field=VECTOR_KEY,
        vector_distance_metric=MODEL_DISTANCE_CALC,
            dimensions=MODEL_DIM,
        )

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
    
    # Get total number of chapters
    total_chapters = sum(1 for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT)

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
                "vector": create_embedding(soup.get_text(), "chapter")
            }
            
            chapters.append(chapter)
            soups.append(soup)
            
            console.print(f"\tCHAPTERS:\tEmbedding [green]chapter {index + 1}[/green] [magenta]({index + 1}/{total_chapters})[/magenta]", end='\r', highlight=False)    
            index += 1
        
    console.print()
    return chapters, soups

def extract_paragraphs_from_chapters(chapters, soups):
    all_paragraphs = []
    total_paragraphs = sum(len(soup.find_all('p')) for soup in soups)
    total_index = 0
    
    for chapter in chapters:
        chapter_index = chapter["chapter-index"]
        chapter_title = chapter["chapter-title"]
        chapter_html = soups[chapter_index]
    
        chapter_paragraphs = []
        
        index = 0
        for p in chapter_html.find_all('p'):
            paragraph = {
                "chapter-index": chapter_index,
                "chapter-title": chapter_title,
                "paragraph-index": index,
                "text": p.get_text(),
                "vector": create_embedding(p.get_text(), "paragraph")
            }
            chapter_paragraphs.append(paragraph)
            
            total_index += 1
            console.print(f"\tPARAGRAPHS:\tEmbedding [green]paragraph {total_index}[/green] [magenta]({total_index}/{total_paragraphs})[/magenta]", end='\r', highlight=False)    

            
            index += 1
            
        all_paragraphs.extend(chapter_paragraphs)
    console.print()
    return all_paragraphs

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
    
def check_if_indexed(INDEX_NAME, SET_NAME, KEY):
    return client.is_indexed(
        namespace=NAMESPACE,
        key=KEY,
        index_name=INDEX_NAME,
        set_name=SET_NAME
    )

def upload_book(BOOK_NAME, BOOK_PATH):
    start_time = time.time()
    
    # Step 1: Create Indexes
    current_time = time.time() - start_time
    console.print(f"[green]Creating Indexes ({current_time:.2f}s)[/green]", highlight=False)
    CHAPTER_INDEX = BOOK_NAME + "-chapters"
    PARAGRAPH_INDEX = BOOK_NAME + "-paragraphs"
    
    create_index(CHAPTER_INDEX, "chapters")
    create_index(PARAGRAPH_INDEX, "paragraphs")
    
    # Step 2: Embedding
    current_time = time.time() - start_time
    console.print(f"[green]Embedding({current_time:.2f}s)[/green]", highlight=False)
    
    chapters, soups = extract_chapters_from_epub(BOOK_PATH)
    paragraphs = extract_paragraphs_from_chapters(chapters, soups)
    
    # Step 3: Insert Records
    current_time = time.time() - start_time
    console.print(f"[green]Inserting Records ({current_time:.2f}s)[/green]", highlight=False)
    for chapter in chapters:
        console.print(f"\tCHAPTERS:\tInserting [green]chapter {chapter['chapter-index']+1}[/green] [magenta]({chapter['chapter-index']+1}/{len(chapters)})[/magenta]", highlight=False, end="\r")
        insert_record("chapters", chapter["chapter-index"], chapter)
    console.print()
    total_index = 0
    for paragraph in paragraphs:
        console.print(f"\tPARAGRAPHS:\tInserting [green]paragraph {total_index+1}[/green] [magenta]({total_index+1}/{len(paragraphs)})[/magenta]", highlight=False, end="\r")
        insert_record("paragraphs", paragraph["chapter-index"]*10000+paragraph["paragraph-index"], paragraph)
        total_index += 1
    console.print()
    
    # Step 4: Wait for Indexing
    current_time = time.time() - start_time
    console.print(f"[green]Waiting for Indexing ({current_time:.2f}s)[/green]", highlight=False)
    
    console.print(f"\tCHAPTERS:\tWaiting for [green]'{CHAPTER_INDEX}'[/green]", highlight=False, end="\r")
    client.wait_for_index_completion(namespace=NAMESPACE, name=CHAPTER_INDEX)
    console.print(f"\tCHAPTERS:\t[green]'{CHAPTER_INDEX}'[/green] Indexed          ")
    console.print(f"\tPARAGRAPHS:\tWaiting for [green]'{PARAGRAPH_INDEX}'[/green]", highlight=False, end="\r")
    client.wait_for_index_completion(namespace=NAMESPACE, name=PARAGRAPH_INDEX)
    console.print(f"\tPARAGRAPHS:\t[green]'{PARAGRAPH_INDEX}'[/green] Indexed          ")
    
    # Step 5: Verify Records
    current_time = time.time() - start_time
    console.print(f"[green]Verifying Records ({current_time:.2f}s)[/green]", highlight=False)
    
    failed_chapters = []
    for chapter in chapters:
        console.print(f"\tCHAPTERS:\tChecking if [green]{chapter['chapter-index']+1}[/green] is indexed [magenta]({chapter['chapter-index']+1}/{len(chapters)})[/magenta]:\t", highlight=False, end="")
        indexed = check_if_indexed(CHAPTER_INDEX, "chapters", chapter["chapter-index"])
        if indexed:
            console.print(f"[green]True[/green]", highlight=False, end="\r")
        else:
            console.print(f"[red]False[/red]", highlight=False, end="\r")
            failed_chapters.append(chapter["chapter-index"])
    console.print()
    
    failed_paragraphs = []
    total_index = 1
    for paragraph in paragraphs:
        console.print(f"\tPARAGRAPHS:\tChecking if [green]{paragraph['chapter-index']*10000+paragraph['paragraph-index']+1}[/green] is indexed [magenta]({total_index}/{len(paragraphs)})[/magenta]:\t", highlight=False, end="")
        indexed = check_if_indexed(PARAGRAPH_INDEX, "paragraphs", paragraph["chapter-index"]*10000+paragraph["paragraph-index"])
        if indexed:
            console.print(f"[green]True[/green]", highlight=False, end="\r")
        else:
            console.print(f"[red]False[/red]", highlight=False, end="\r")
            failed_paragraphs.append(paragraph["chapter-index"]*10000+paragraph["paragraph-index"])
        total_index += 1
    console.print()
    
    if len(failed_chapters) > 0:
        console.print(f"[red]Failed to index {len(failed_chapters)} chapters[/red]")
    else:
        console.print(f"[green]All chapters indexed![/green]")
    if len(failed_paragraphs) > 0:
        console.print(f"[red]Failed to index {len(failed_paragraphs)} paragraphs[/red]")
    else:
        console.print(f"[green]All paragraphs indexed![/green]")



for index, book in enumerate(os.listdir("./books")):
    total_books = len(os.listdir("./books"))
    
    BOOK_PATH = f"./books/{book}"
    BOOK_NAME = book.replace(".epub", "")
    
    book_title = book.replace(".epub", "").replace("-", " ").title()
    console.print(f"\n[magenta]Uploading '{book_title}' ({index + 1}/{total_books})[/magenta]", highlight=False)

    
    try:
        upload_book(BOOK_NAME, BOOK_PATH)
    except Exception as e:
        console.print(f"[red]Error uploading '{book_title}': {e}[/red]", highlight=False)
    
admin_client.close()
client.close()