# Vector_Doc_Chat

Vector Doc Chat is a command-line tool that allows you to search and interact with the contents of your local documents using semantic search powered by embeddings. It reads text from PDF, Word (DOCX), and TXT files, splits the content into meaningful chunks, converts them into embeddings using a pretrained model, and stores them in a local Chroma vector database. You can then ask questions and retrieve the most relevant answers based on semantic similarity.

---

## Features

- Reads documents from the `documents/` folder (PDF, DOCX, TXT supported)
- Splits content into paragraphs or logical chunks
- Converts chunks into embeddings using `all-MiniLM-L6-v2` model
- Stores and retrieves vectors using ChromaDB
- Supports querying with top-K semantic matches
- CLI-based, lightweight, and open source
  
---

## Project Structure

```
Vector-Doc-Chat/
├── main.py
├── documents/ # Place your source documents here
    ├── document1
    ├── document2
          -
          -
          -
    ├── documentn     
├── chunks/ # Stores preprocessed chunks in JSON format
├── vectorstore/ # ChromaDB vector index data
├── main.py # Main CLI interface
├── processor.py # Handles document reading and chunking
├── indexer.py # Handles embedding and semantic search
└── README.md # Project info
├── LICENSE
```

---

## Requirements

- Python 3.8+
- pip packages:
  - chromadb
  - sentence-transformers
  - pypdf2
  - python-docx
  - rich

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Vector-Doc-Chat.git
   cd Vector-Doc-Chat

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the Script:

   ```bash
   python main.py

---

## Usage

1. Place your documents inside the documents/ folder.

2. Run the script

3. Follow the prompts:
  - Choose a document to process
  - It will automatically chunk, embed, and store (if not already done)
  - You can then ask questions about the content
  - Optionally choose how many top results you want

4. To exit at any time, type exit, quit, or q.

---

## License

This project is open-source and available under the MIT License.
