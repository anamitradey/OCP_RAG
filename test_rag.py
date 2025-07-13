#!/usr/bin/env python3
"""
Test script for the RAG application
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8080"

def test_ingest():
    """Test document ingestion"""
    print("Testing document ingestion...")
    
    # Sample text to ingest
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. 
    Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning, and problem solving.
    
    Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. 
    Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. 
    These neural networks are designed to mimic the human brain's structure and function, allowing them to process data in a way that's similar to how humans think.
    
    Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages. 
    It combines computational linguistics with statistical, machine learning, and deep learning models.
    """
    
    payload = {
        "text": sample_text,
        "metadata": {
            "source": "test_document",
            "category": "technology",
            "author": "test_user"
        }
    }
    
    response = requests.post(f"{BASE_URL}/ingest", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Successfully ingested document: {result['document_id']}")
        print(f"   Created {result['chunks_created']} chunks")
        print(f"   Message: {result['message']}")
        return result['document_id']
    else:
        print(f"❌ Failed to ingest document: {response.text}")
        return None

def test_search(query, n_results=3):
    """Test document search"""
    print(f"\nTesting search with query: '{query}'")
    
    response = requests.get(f"{BASE_URL}/search", params={"query": query, "n_results": n_results})
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Search successful")
        print(f"   Query: {result['query']}")
        print(f"   Found {len(result['results']['ids'][0])} results")
        
        for i, (doc_id, metadata, document) in enumerate(zip(
            result['results']['ids'][0],
            result['results']['metadatas'][0],
            result['results']['documents'][0]
        )):
            print(f"\n   Result {i+1}:")
            print(f"     Document ID: {doc_id}")
            print(f"     Chunk Index: {metadata.get('chunk_index', 'N/A')}")
            print(f"     Text: {document[:150]}...")
    else:
        print(f"❌ Search failed: {response.text}")

def test_list_documents():
    """Test listing all documents"""
    print("\nTesting document listing...")
    
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Found {result['total_documents']} documents")
        
        for doc in result['documents']:
            print(f"\n   Document: {doc['document_id']}")
            print(f"     Total chunks: {doc['total_chunks']}")
            print(f"     Chunks: {len(doc['chunks'])}")
    else:
        print(f"❌ Failed to list documents: {response.text}")

def main():
    """Run all tests"""
    print("🚀 Starting RAG Application Tests\n")
    
    # Test 1: Ingest a document
    doc_id = test_ingest()
    
    if doc_id:
        # Test 2: Search for documents
        test_search("artificial intelligence")
        test_search("machine learning")
        test_search("neural networks")
        
        # Test 3: List all documents
        test_list_documents()
    
    print("\n✨ Tests completed!")

if __name__ == "__main__":
    main() 