# RAG Application with ChromaDB

A Retrieval-Augmented Generation (RAG) application built with FastAPI and ChromaDB, featuring recursive text chunking for efficient document processing and retrieval.

## Features

- **Recursive Text Chunking**: Splits documents into 750-character chunks with 100-character overlap
- **ChromaDB Integration**: Vector database for semantic search and storage
- **RESTful API**: Easy-to-use endpoints for document ingestion and retrieval
- **Metadata Support**: Custom metadata for each document and chunk
- **Semantic Search**: Find relevant documents using natural language queries

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The application will start on `http://localhost:8080`

## API Endpoints

### 1. GET `/`
Returns basic information about the application.

**Response:**
```json
{
  "message": "RAG Application with ChromaDB",
  "endpoints": ["/ingest", "/search"]
}
```

### 2. POST `/ingest`
Ingest a text document into ChromaDB with recursive chunking.

**Request Body:**
```json
{
  "text": "Your document text here...",
  "metadata": {
    "source": "document_source",
    "category": "document_category",
    "author": "author_name"
  },
  "document_id": "optional_custom_id"
}
```

**Response:**
```json
{
  "document_id": "generated_or_custom_id",
  "chunks_created": 5,
  "message": "Successfully ingested document with 5 chunks"
}
```

### 3. GET `/search`
Search for documents using semantic similarity.

**Query Parameters:**
- `query` (required): Search query text
- `n_results` (optional): Number of results to return (default: 5)

**Example:**
```
GET /search?query=artificial intelligence&n_results=3
```

**Response:**
```json
{
  "query": "artificial intelligence",
  "results": {
    "ids": [["doc1_chunk_0", "doc2_chunk_1"]],
    "metadatas": [[{"document_id": "doc1", "chunk_index": 0}]],
    "documents": [["AI is a branch of computer science..."]],
    "distances": [[0.1, 0.3]]
  }
}
```

### 4. GET `/documents`
List all documents in the collection.

**Response:**
```json
{
  "total_documents": 2,
  "documents": [
    {
      "document_id": "doc1",
      "chunks": [
        {
          "chunk_id": "doc1_chunk_0",
          "chunk_index": 0,
          "text": "AI is a branch of computer science..."
        }
      ],
      "total_chunks": 3
    }
  ]
}
```

## Text Chunking Configuration

The application uses recursive text chunking with the following settings:

- **Chunk Size**: 750 characters
- **Chunk Overlap**: 100 characters
- **Separators**: `["\n\n", "\n", " ", ""]`

This configuration ensures:
- Meaningful text chunks that preserve context
- Overlap between chunks to maintain continuity
- Intelligent splitting at natural boundaries

## Testing the Application

Run the test script to see the application in action:

```bash
python test_rag.py
```

This will:
1. Ingest a sample document about AI and machine learning
2. Perform semantic searches
3. List all documents in the collection

## Example Usage

### Using curl

1. **Ingest a document:**
```bash
curl -X POST "http://localhost:8080/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text here...",
    "metadata": {
      "source": "web",
      "category": "technology"
    }
  }'
```

2. **Search for documents:**
```bash
curl "http://localhost:8080/search?query=machine learning&n_results=3"
```

3. **List all documents:**
```bash
curl "http://localhost:8080/documents"
```

### Using Python requests

```python
import requests

# Ingest document
response = requests.post("http://localhost:8080/ingest", json={
    "text": "Your document text...",
    "metadata": {"source": "test"}
})

# Search documents
response = requests.get("http://localhost:8080/search", params={
    "query": "your search query",
    "n_results": 5
})
```

## Data Storage

- ChromaDB data is stored in the `./chroma_db` directory
- Documents are automatically chunked and indexed for semantic search
- Each chunk maintains metadata about its source document and position

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│  Text Splitter  │───▶│   ChromaDB      │
│                 │    │  (Recursive)    │    │   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │   REST API      │    │   750 char      │    │  Semantic       │
   │   Endpoints     │    │   chunks        │    │  Search         │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Dependencies

- **FastAPI**: Web framework for building APIs
- **ChromaDB**: Vector database for embeddings and similarity search
- **LangChain**: Text splitting utilities
- **Sentence Transformers**: Embedding models for semantic search
- **Uvicorn**: ASGI server for running the application

## Notes

- The application automatically creates a ChromaDB collection named "documents"
- Each document is assigned a unique ID if not provided
- Chunks are named with the pattern: `{document_id}_chunk_{index}`
- The application uses DuckDB with Parquet for ChromaDB storage 