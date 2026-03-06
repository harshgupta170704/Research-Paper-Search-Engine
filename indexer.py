"""
indexer.py
- Scan a folder of .txt and .pdf files
- Extract text (PyPDF2 for pdf)
- Make embeddings (sentence-transformers 'all-MiniLM-L6-v2', 384-dim)
- Create Endee index (dim=384) and upsert documents
"""

import os
import json
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from endee import Endee, Precision

DATA_DIR = "docs"       # put your pdf/txt files here
INDEX_NAME = "papers"   # name in Endee
EMBED_MODEL = "all-MiniLM-L6-v2"  # outputs 384-d vectors

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)

def read_documents(data_dir: str) -> List[Dict]:
    docs = []
    for p in Path(data_dir).iterdir():
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}:
            if p.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(str(p))
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
            title = p.stem
            docs.append({"id": str(p.name), "title": title, "text": text, "path": str(p)})
    return docs

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Yield text chunks (simple whitespace split). Good enough for indexing abstracts/sections."""
    tokens = text.split()
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        yield " ".join(chunk)
        i += chunk_size - overlap

def prepare_upsert_records(model, docs):
    records = []
    for doc in docs:
        # chunk a big document into multiple vectors (keeps retrieval precise)
        chunks = list(chunk_text(doc["text"], chunk_size=400, overlap=50))
        for idx, chunk in enumerate(chunks):
            vec = model.encode(chunk).tolist()
            rec = {
                "id": f"{doc['id']}__{idx}",
                "vector": vec,
                "meta": {"title": doc["title"], "source_path": doc["path"], "chunk_index": idx},
                "filter": {"filename": doc["id"]}  # optional
            }
            records.append(rec)
    return records

def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Reading documents...")
    docs = read_documents(DATA_DIR)
    if not docs:
        print(f"No documents found in `{DATA_DIR}`. Add PDF/TXT files and retry.")
        return

    print(f"Found {len(docs)} documents. Chunking and embedding...")
    records = prepare_upsert_records(model, docs)
    print(f"Prepared {len(records)} vectors to upsert.")

    print("Connecting to local Endee server...")
    client = Endee()  # defaults to localhost:8080; pass auth token if needed

    # Create index (if not exists)
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=384,
            space_type="cosine",
            precision=Precision.INT8
        )
        print(f"Created index `{INDEX_NAME}` (dim=384).")
    except Exception as e:
        print(f"Index create may have failed or already exists: {e}")

    index = client.get_index(name=INDEX_NAME)

    # Upsert in batches
    BATCH = 256
    for i in range(0, len(records), BATCH):
        batch = records[i:i+BATCH]
        index.upsert(batch)
        print(f"Upserted batch {i//BATCH + 1} ({len(batch)} vectors)")

    print("Indexing complete.")

if __name__ == "__main__":
    main()
