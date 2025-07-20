from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, uuid

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from fastembed import TextEmbedding

###############################################################################
# 1.  FastEmbed wrapper – makes FastEmbed look like a Chroma EmbeddingFunction
###############################################################################
class FastEmbedEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self,
                 model_name: str = "BAAI/bge-small-en-v1.5",
                 batch_size: int = 64):
        self._model = TextEmbedding(model_name=model_name, batch_size=batch_size)

    def __call__(self, docs: Documents) -> Embeddings:       # type: ignore[override]
        # FastEmbed returns a generator; wrap it in a list so Chroma can use it
        return list(self._model.embed(list(docs)))

    # Optional helpers --------------------------------------------------------
    def max_tokens(self) -> int:
        # FastEmbed’s small-bge models cap at 512 tokens, keep chunks < 512 chars
        return 512


###############################################################################
# 2.  FastAPI app + Chroma persistent client
###############################################################################
DB_PATH = "./db"                 # creates ./db on first run
os.makedirs(DB_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="rag_docs",
    embedding_function=FastEmbedEmbeddingFunction()
)

app = FastAPI(title="RAG Ingestion Service")


###############################################################################
# 3.  Helper: fixed-window chunker (500 chars, 50-char overlap)
###############################################################################
def chunk_text(txt: str,
               window: int = 500,
               overlap: int = 50) -> List[str]:
    if overlap >= window:
        raise ValueError("overlap must be smaller than window size")
    chunks = []
    start = 0
    while start < len(txt):
        end = start + window
        chunks.append(txt[start:end])
        start += window - overlap
    return chunks


###############################################################################
# 4.  API schema + endpoint
###############################################################################
class IngestRequest(BaseModel):
    document_id: str
    text: str
    source: Optional[str] = "manual"
    use_content_hash_ids: bool = False   # optional: stable IDs if text changes


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text payload is empty")

    chunks = chunk_text(req.text)
    ids, metas = [], []

    for idx, chunk in enumerate(chunks):
        chunk_id = (
            f"{req.document_id}_{idx}"
            if not req.use_content_hash_ids
            else f"{req.document_id}_{uuid.uuid5(uuid.NAMESPACE_URL, chunk)}"
        )
        ids.append(chunk_id)
        metas.append({
            "document_id": req.document_id,
            "chunk_index": idx,
            "source": req.source,
            "chars": len(chunk)
        })

    collection.upsert(ids=ids, documents=chunks, metadatas=metas)
    return {
        "ingested": len(ids),
        "collection": collection.name,
        "ids": ids
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)