"""
RAG Pipeline Orchestrator

Main pipeline that orchestrates text ingestion, chunking, embedding,
and storage in MongoDB vector store.
"""
from typing import List, Dict, Any, Optional, Union
from loguru import logger

from llama_index.core.schema import BaseNode

from rag.config import rag_config
from rag.chunking import HierarchicalChunker
from rag.semantic_chunking import SemanticChunker
from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore


class RAGPipeline:
    """
    Complete RAG pipeline for user context processing
    
    Orchestrates:
    1. Text chunking (hierarchical or semantic)
    2. Embedding generation
    3. Storage in MongoDB with user/tenant metadata
    
    Supports two chunking strategies:
    - Hierarchical: Fixed multi-level chunks (default)
    - Semantic: Adaptive chunks based on semantic similarity
    """
    
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        chunker: Union[HierarchicalChunker, SemanticChunker, None] = None
    ):
        """
        Initialize RAG pipeline with optional custom components
        
        Args:
            vector_store: Custom vector store instance
            embedding_service: Custom embedding service instance
            chunker: Custom chunker instance (HierarchicalChunker or SemanticChunker)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = chunker or HierarchicalChunker()
        
        # Log which chunking strategy is being used
        chunker_type = type(self.chunker).__name__
        logger.info(f"Initialized RAGPipeline with {chunker_type}")
    
    def ingest_user_context(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Ingest user context text into the RAG pipeline
        
        Complete workflow:
        1. Chunk text (using configured chunking strategy)
        2. Generate embeddings for chunks
        3. Store in MongoDB with metadata
        
        Args:
            text: User context text to ingest
            user_id: User identifier
            tenant_id: Tenant identifier
            additional_metadata: Additional metadata to attach
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Ingesting user context for user {user_id}, tenant {tenant_id}")
            
            # Step 1: Chunk text using configured strategy
            nodes = self.chunker.chunk_text(
                text=text,
                user_id=user_id,
                tenant_id=tenant_id,
                additional_metadata=additional_metadata
            )
            
            if not nodes:
                logger.warning("No nodes created from text")
                return {
                    "success": False,
                    "message": "No nodes created",
                    "nodes_processed": 0
                }
            
            # Step 2: Filter to leaf nodes for embedding
            # (For hierarchical: embed leaf nodes to avoid redundancy)
            # (For semantic: all nodes are leaf nodes, so this returns all)
            leaf_nodes = self.chunker.filter_leaf_nodes(nodes)
            
            if not leaf_nodes:
                logger.warning("No leaf nodes to embed")
                # Fall back to embedding all nodes if no clear hierarchy
                leaf_nodes = nodes
            
            # Step 3: Generate embeddings for leaf nodes
            texts_to_embed = [node.get_content() for node in leaf_nodes]
            embeddings = self.embedding_service.embed_texts(texts_to_embed)
            
            # Attach embeddings to nodes
            for node, embedding in zip(leaf_nodes, embeddings):
                node.embedding = embedding
            
            # Step 4: Store in MongoDB
            node_ids = self.vector_store.add_nodes(
                nodes=leaf_nodes,
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            logger.info(
                f"Successfully ingested {len(leaf_nodes)} nodes for user {user_id}"
            )
            
            return {
                "success": True,
                "message": "User context ingested successfully",
                "nodes_processed": len(leaf_nodes),
                "node_ids": node_ids,
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error ingesting user context: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "nodes_processed": 0
            }
    
    async def ingest_user_context_async(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Async version of ingest_user_context
        
        Args:
            text: User context text to ingest
            user_id: User identifier
            tenant_id: Tenant identifier
            additional_metadata: Additional metadata to attach
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Ingesting user context (async) for user {user_id}")
            
            # Step 1: Chunk text
            nodes = self.chunker.chunk_text(
                text=text,
                user_id=user_id,
                tenant_id=tenant_id,
                additional_metadata=additional_metadata
            )
            
            if not nodes:
                logger.warning("No nodes created from text")
                return {
                    "success": False,
                    "message": "No nodes created",
                    "nodes_processed": 0
                }
            
            # Step 2: Filter to leaf nodes
            leaf_nodes = self.chunker.filter_leaf_nodes(nodes)
            
            if not leaf_nodes:
                leaf_nodes = nodes
            
            # Step 3: Generate embeddings (async)
            texts_to_embed = [node.get_content() for node in leaf_nodes]
            embeddings = await self.embedding_service.embed_texts_async(texts_to_embed)
            
            # Attach embeddings
            for node, embedding in zip(leaf_nodes, embeddings):
                node.embedding = embedding
            
            # Step 4: Store in MongoDB
            node_ids = self.vector_store.add_nodes(
                nodes=leaf_nodes,
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            logger.info(
                f"Successfully ingested {len(leaf_nodes)} nodes (async) for user {user_id}"
            )
            
            return {
                "success": True,
                "message": "User context ingested successfully",
                "nodes_processed": len(leaf_nodes),
                "node_ids": node_ids,
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error ingesting user context (async): {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "nodes_processed": 0
            }
    
    def delete_user_data(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Delete all data for a specific user
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with deletion results
        """
        try:
            deleted_count = self.vector_store.delete_user_data(user_id, tenant_id)
            
            return {
                "success": True,
                "message": f"Deleted {deleted_count} documents",
                "deleted_count": deleted_count,
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error deleting user data: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "deleted_count": 0
            }
    
    def get_user_data_info(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Get information about stored data for a user
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with user data information
        """
        try:
            return self.vector_store.get_user_data_info(user_id, tenant_id)
        except Exception as e:
            logger.error(f"Error getting user data info: {str(e)}")
            return {
                "count": 0,
                "latest_timestamp": None,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "error": str(e)
            }
    
    def update_user_context(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Update user context by deleting old data and ingesting new
        
        Args:
            text: New user context text
            user_id: User identifier
            tenant_id: Tenant identifier
            additional_metadata: Additional metadata to attach
            
        Returns:
            Dictionary with update results
        """
        try:
            logger.info(f"Updating user context for user {user_id}, tenant {tenant_id}")
            
            # Delete existing data
            delete_result = self.delete_user_data(user_id, tenant_id)
            
            # Ingest new data
            ingest_result = self.ingest_user_context(
                text=text,
                user_id=user_id,
                tenant_id=tenant_id,
                additional_metadata=additional_metadata
            )
            
            return {
                "success": ingest_result.get("success", False),
                "message": "User context updated successfully",
                "deleted_count": delete_result.get("deleted_count", 0),
                "nodes_processed": ingest_result.get("nodes_processed", 0),
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error updating user context: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "deleted_count": 0,
                "nodes_processed": 0
            }

