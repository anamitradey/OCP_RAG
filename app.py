from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from typing import List, Optional

app = FastAPI(title="RAG Application", description="A RAG application with ChromaDB integration")

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create or get collection
collection_name = "documents"
try:
    collection = chroma_client.get_collection(name=collection_name)
except:
    collection = chroma_client.create_collection(name=collection_name)

# Initialize text splitter with recursive chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

class DocumentInput(BaseModel):
    text: str
    metadata: Optional[dict] = None
    document_id: Optional[str] = None

class DocumentResponse(BaseModel):
    document_id: str
    chunks_created: int
    message: str

@app.get('/')
def hello():
    return {"message": "RAG Application with ChromaDB", "endpoints": ["/ingest", "/search"]}

@app.post('/ingest', response_model=DocumentResponse)
async def ingest_document(document: DocumentInput):
    """
    Ingest text document into ChromaDB with recursive chunking.
    
    - chunk_size: 750 characters
    - chunk_overlap: 100 characters
    """
    try:
        # Generate document ID if not provided
        if not document.document_id:
            document.document_id = str(uuid.uuid4())
        
        # Split text into chunks
        chunks = text_splitter.split_text(document.text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks created")
        
        # Prepare metadata for each chunk
        chunk_metadata = []
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document.document_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Create metadata for this chunk
            chunk_meta = {
                "document_id": document.document_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            }
            
            # Add custom metadata if provided
            if document.metadata:
                chunk_meta.update(document.metadata)
            
            chunk_metadata.append(chunk_meta)
        
        # Add chunks to ChromaDB collection
        collection.add(
            documents=chunks,
            metadatas=chunk_metadata,
            ids=chunk_ids
        )
        
        return DocumentResponse(
            document_id=document.document_id,
            chunks_created=len(chunks),
            message=f"Successfully ingested document with {len(chunks)} chunks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

@app.get('/search')
async def search_documents(query: str, n_results: int = 5):
    """
    Search for documents in ChromaDB using semantic similarity.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            "query": query,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.get('/documents')
async def list_documents():
    """
    List all documents in the collection.
    """
    try:
        # Get all documents from collection
        results = collection.get()
        
        # Group by document_id
        documents = {}
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            document_id = metadata.get('document_id', doc_id)
            
            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "chunks": [],
                    "total_chunks": metadata.get('total_chunks', 1)
                }
            
            documents[document_id]["chunks"].append({
                "chunk_id": doc_id,
                "chunk_index": metadata.get('chunk_index', 0),
                "text": results['documents'][i][:100] + "..." if len(results['documents'][i]) > 100 else results['documents'][i]
            })
        
        return {
            "total_documents": len(documents),
            "documents": list(documents.values())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    uvicorn.run(app, host='0.0.0.0', port=port)