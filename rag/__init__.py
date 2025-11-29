"""
RAG Pipeline using LlamaIndex with MongoDB Vector Store

This package provides a complete RAG (Retrieval-Augmented Generation) pipeline
that converts user context text to vector embeddings, stores them in MongoDB
with hierarchical parent-child chunking, and supports user/tenant filtering.

Features:
- Two chunking strategies: Hierarchical and Semantic
- User context ingestion (health data, summaries, reports)
- Document ingestion (research papers, PDFs, text files)
- MongoDB Atlas Vector Search with filtering
- Query both user context and documents together
"""

from rag.pipeline import RAGPipeline
from rag.retriever import RAGRetriever
from rag.update_service import UpdateService
from rag.chunking import HierarchicalChunker
from rag.semantic_chunking import SemanticChunker
from rag.document_loader import DocumentLoader

__all__ = [
    "RAGPipeline",
    "RAGRetriever", 
    "UpdateService",
    "HierarchicalChunker",
    "SemanticChunker",
    "DocumentLoader"
]

