"""
app.py
- FastAPI app with a single /search endpoint
- Takes JSON: {"query": "your question", "top_k": 5}
- Embeds the query and queries Endee index
"""

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from endee import Endee

INDEX_NAME = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"

app = FastAPI()
model = SentenceTransformer(EMBED_MODEL)
client = Endee()
index = client.get_index(name=INDEX_NAME)

class QueryReq(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search(q: QueryReq):
    vec = model.encode(q.query).tolist()
    results = index.query(vector=vec, top_k=q.top_k)

    # results: list of dicts with id, similarity, meta (if stored)
    out = []
    for item in results:
        out.append({
            "id": item.get("id"),
            "similarity": item.get("similarity"),
            "title": item.get("meta", {}).get("title"),
            "source_path": item.get("meta", {}).get("source_path"),
            "chunk_index": item.get("meta", {}).get("chunk_index")
        })
    return {"query": q.query, "results": out}
