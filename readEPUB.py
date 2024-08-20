# Just a tool to see what the inside of the EPUB looks like

import os
import ebooklib
from ebooklib import epub


PATH = "./books"

for index, file in enumerate(os.listdir(PATH)):
    print(f"({index}) {file}")

selected_book = int(input("Select a book: "))

try:
    book = epub.read_epub(os.path.join(PATH, os.listdir(PATH)[selected_book]))
except Exception as e:
    print(f"Error: {e}")
    exit()

items = []
for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        items.append(item)
        print(f"({len(items)}) {item}")

selected_item = int(input("Select an item: "))

html_content = items[selected_item].get_content().decode('utf-8')
print(html_content)