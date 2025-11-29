"""
Configuration for RAG Pipeline
"""
import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    """Configuration for RAG pipeline"""
    
    # MongoDB Configuration
    mongodb_uri: str = Field(default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    mongodb_db_name: str = Field(default=os.getenv("MONGODB_DB_NAME", "swastya"))
    vector_collection_name: str = Field(default="user_context_vectors")
    vector_index_name: str = Field(default="vector_search_index")
    
    # OpenAI Embedding Configuration
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)  # text-embedding-3-small default
    
    # Hierarchical Chunking Configuration
    # Three-level hierarchy: large parent chunks -> medium chunks -> small detail chunks
    chunk_sizes: List[int] = Field(default=[2048, 512, 128])
    chunk_overlap: int = Field(default=20)
    
    # Semantic Chunking Configuration
    semantic_buffer_size: int = Field(default=1)  # Sentences per group
    semantic_breakpoint_threshold: int = Field(default=95)  # Percentile (0-100)
    
    # Retrieval Configuration
    top_k_results: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)
    
    # Update Service Configuration
    stale_threshold_hours: int = Field(default=24)  # Re-embed if data older than 24 hours
    enable_auto_update: bool = Field(default=True)
    
    # Evaluation Configuration (optional)
    llamaindex_evaluation_enabled: bool = Field(default=False)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from environment


# Global config instance
rag_config = RAGConfig()

