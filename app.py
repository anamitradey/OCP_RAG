from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, uuid
import shutil
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from fastembed import TextEmbedding
import openai

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
DB_PATH = Path("./db")                 # creates ./db on first run
load_dotenv(override=False)  # respects existing env vars from Secrets

DB_PATH.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="rag_docs",
    embedding_function=FastEmbedEmbeddingFunction()
)

# --------------------------- OpenAI --------------------------- #
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it via env/Secret or .env file for local dev."
    )

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Or leave empty if the env variable is set
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

class SearchRequest(BaseModel):
    query: str
    top_k: int = 4
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

class ChatRequest(SearchRequest):
    model: str = DEFAULT_OPENAI_MODEL
    temperature: float = 0.2

# --------------------------- Routes --------------------------- #
@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_model": DEFAULT_OPENAI_MODEL,
    }

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
@app.delete("/collection/reset")
def reset_collection():
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    DB_PATH.mkdir(parents=True, exist_ok=True)

    # 3. Recreate client & collection
    chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = chroma_client.get_or_create_collection(
        name="rag_docs",
        embedding_function=FastEmbedEmbeddingFunction()
    )
    return {"status": "ok"}
@app.delete("/docs/{document_id}")
def delete_doc(document_id: str):
    deleted = collection.delete(where={"document_id": document_id})
    return {"deleted_ids": deleted}
@app.post("/search")
def search(req: SearchRequest):
    res = collection.query(
    query_texts=[req.query],
    n_results=req.top_k,
    include=["documents", "metadatas"],   # <- drop "ids"
    )
    results = [
        {"id": i, "text": d, "meta": m}
        for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ]
    return {"query": req.query, "top_k": req.top_k, "results": results}

@app.post("/chat")
def chat(req: ChatRequest):
    res = collection.query(
        query_texts=[req.query],
        n_results=1,
        include=["documents", "metadatas"],
    )
    context = "\n\n".join(res["documents"][0])
    ids = res["ids"][0]

    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "If the answer is not in the context, say you don't know." 
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.query}\nAnswer:"

    try:
        response = client.chat.completions.create(
            model=req.model,
            temperature=req.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return {
        "question": req.query,
        "answer": answer,
        "sources": ids,
        "model": req.model,
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)