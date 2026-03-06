AI Semantic Document Search using Endee

This project implements a semantic document search system using vector embeddings and the Endee.io vector database.
Unlike traditional keyword-based search, this system retrieves documents based on semantic similarity, enabling more accurate and context-aware information retrieval.

The system converts documents into vector embeddings and stores them in a vector database. When a user submits a query, it is also converted into an embedding and matched with the most similar document vectors.

This approach is widely used in modern AI systems such as:

AI-powered search engines

recommendation systems

Retrieval Augmented Generation (RAG)

intelligent document assistants

Project Overview

The goal of this project is to demonstrate how vector databases and embeddings can be used to build a semantic search engine for documents such as research papers, notes, or articles.

The system:

Extracts text from documents (PDF/TXT)

Converts text into vector embeddings using an NLP model

Stores embeddings in a vector database

Retrieves the most relevant documents based on semantic similarity

System Architecture
Documents (PDF / TXT)
        │
        ▼
Text Extraction
        │
        ▼
Embedding Model (Sentence Transformers)
        │
        ▼
Vector Embeddings
        │
        ▼
Endee Vector Database
        │
        ▼
User Query → Embedding
        │
        ▼
Similarity Search
        │
        ▼
Most Relevant Documents
