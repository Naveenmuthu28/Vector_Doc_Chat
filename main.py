# import libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import json
import re
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
import chromadb
from rich import print, prompt

# project directories
DOCUMENTS_DIR = "documents"
CHUNKS_DIR = "chunks"
VECTORSTORE_DIR = "vectorstore"

# functions for document loading and splitting into chunks
class DocumentProcessor:
    def __init__(self):
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(CHUNKS_DIR, exist_ok=True)
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    # funtion to list documents
    def list_documents(self):
        return [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    
    # funtion to read pdf
    def read_pdf(self, filepath):
        reader = PdfReader(filepath)
        all_paragraphs = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Split inside page into paragraphs
                paragraphs = re.split(r'\n\s*\n', text)
                for p in paragraphs:
                    p = p.strip()
                    if p:
                        all_paragraphs.append(p)
        return "\n\n".join(all_paragraphs)
    
    # funtion to word document
    def read_docx(self, filepath):
        doc = DocxDocument(filepath)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    # funtion to text file
    def read_txt(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Normalize line breaks
        content = content.replace('\r\n', '\n')
        # Treat blank lines as paragraph breaks
        paragraphs = re.split(r'\n\s*\n', content)
        cleaned_paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return "\n\n".join(cleaned_paragraphs)

    # funtion to split into chunks
    def split_into_chunks(self, text, chunk_size=100):
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        for para in paragraphs:
            words = para.strip().split()
            if not words:
                continue
            # Split this paragraph into smaller fixed-size chunks
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if chunk:
                    chunks.append(chunk)
        return chunks

    # funtion to save chunks
    def save_chunks(self, filename, chunks):
        save_path = os.path.join(CHUNKS_DIR, f"{filename}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[green]Saved {len(chunks)} chunks to {save_path}[/green]")

    # funtion to load chunks
    def load_chunks(self, filename):
        path = os.path.join(CHUNKS_DIR, f"{filename}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            print(f"[cyan]Loaded existing chunks from {path}[/cyan]")
            return chunks
        return None

    # funtion to process chunks
    def process(self, document_name):
        base = os.path.splitext(document_name)[0]
        chunks = self.load_chunks(base)
        if chunks:
            return chunks

        print("[yellow]Splitting text into chunks...[/yellow]")
        path = os.path.join(DOCUMENTS_DIR, document_name)

        if document_name.lower().endswith(".pdf"):
            text = self.read_pdf(path)
        elif document_name.lower().endswith(".docx"):
            text = self.read_docx(path)
        elif document_name.lower().endswith(".txt"):
            text = self.read_txt(path)
        else:
            print(f"[red]Unsupported file type: {document_name}[/red]")
            return []

        chunks = self.split_into_chunks(text)
        self.save_chunks(base, chunks)
        return chunks

# functions for converting chunks into vectors, storing it and search it
class VectorIndexer:
    def __init__(self, collection_name="mydocs", model_name="all-MiniLM-L6-v2"):
        print("[cyan]Loading embedding model...[/cyan]")
        self.model = SentenceTransformer(model_name)
        print("[green]Model loaded![/green]")
        self.client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    # funtion to check chunks
    def check_if_indexed(self, base_name):
        results = self.collection.get()
        ids = results.get("ids", [])
        for id in ids:
            if id.startswith(f"{base_name}_"):
                return True
        return False

    # funtion to convert chunks into vectors
    def index_chunks(self, base_name, chunks):
        print(f"[yellow] Embedding and storing {len(chunks)} chunks...[/yellow]")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        ids = [f"{base_name}_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        print(f"[green]Stored {len(chunks)} embeddings in Chroma![/green]")

    # function to do query top results
    def semantic_search(self, query, top_k=3):
        print(f"[yellow]Embedding and searching top {top_k} results...[/yellow]")
        query_embedding = self.model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            print("[red]No relevant results found.[/red]")
            return

        print(f"\n[bold green]Top {top_k} Results:[/bold green]\n")
        for i, (doc, dist) in enumerate(zip(documents, distances), start=1):
            similarity_score = 1 - dist
            print(f"[cyan]{i}. [Score: {similarity_score:.2f}][/cyan]")
            print(f"{doc}\n")


# main function
def main():
    print("[bold blue]\nVECTOR DOC CHAT - Unified CLI[/bold blue]\n")

    # document selection
    processor = DocumentProcessor()
    docs = processor.list_documents()
    if not docs:
        print("[red]No documents found in /documents[/red]")
        return

    print("[bold]Available documents:[/bold]")
    print("0. Exit")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc}")

    choices = ["0"] + [str(i) for i in range(1, len(docs) + 1)]
    choice = prompt.Prompt.ask("\nWhich document number do you want to use?", choices=choices)

    if choice == "0":
        print("[red]Exiting. Bye![/red]")
        return

    selected_doc = docs[int(choice) - 1]
    base_name = os.path.splitext(selected_doc)[0]

    print(f"[green]You selected: {selected_doc}[/green]")

    # Chunking phase
    print("\n[bold][blue]Chunking phase[/blue][/bold]")
    chunks = processor.process(selected_doc)
    print(f"[green]{len(chunks)} chunks ready.[/green]")

    # Indexing phase
    print("\n[bold][blue]Indexing phase[/blue][/bold]")
    indexer = VectorIndexer()
    if indexer.check_if_indexed(base_name):
        print("[cyan]This document is already indexed in Chroma.[/cyan]")
    else:
        indexer.index_chunks(base_name, chunks)

    # Search phase
    print("\n[bold][blue]Search phase[/blue][/bold]")
    while True:
        question = prompt.Prompt.ask("\nAsk your question (or type 'exit' to quit)").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("[red]Exiting. Bye![/red]")
            break

        # Ask how many top results
        top_k_input = prompt.Prompt.ask("How many top results? [default=3]", default="3")
        try:
            top_k = int(top_k_input)
        except ValueError:
            top_k = 3

        # Call indexer's search, which will handle printing results
        indexer.semantic_search(question, top_k=top_k)


if __name__ == "__main__":
    main()

