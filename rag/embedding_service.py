"""
OpenAI Embedding Service

Wrapper for OpenAI embeddings API using LlamaIndex's embedding interface.
Handles text-to-vector conversion with error handling and retry logic.
"""
from typing import List
from loguru import logger
from llama_index.embeddings.openai import OpenAIEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.config import rag_config


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI models
    
    Uses LlamaIndex's OpenAIEmbedding wrapper with retry logic
    for reliability.
    """
    
    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        """
        Initialize the embedding service
        
        Args:
            model_name: OpenAI embedding model (default from config)
            api_key: OpenAI API key (default from config/env)
        """
        self.model_name = model_name or rag_config.embedding_model
        self.api_key = api_key or rag_config.openai_api_key
        
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        # Initialize LlamaIndex OpenAI embedding
        self.embed_model = OpenAIEmbedding(
            model=self.model_name,
            api_key=self.api_key
        )
        
        logger.info(f"Initialized EmbeddingService with model: {self.model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        try:
            embedding = self.embed_model.get_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embed_model.get_text_embedding_batch(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def embed_text_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (async)
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        try:
            embedding = await self.embed_model.aget_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding (async): {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (async batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = await self.embed_model.aget_text_embedding_batch(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts (async)")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings (async): {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            Dimension of embedding vectors
        """
        return rag_config.embedding_dimensions
