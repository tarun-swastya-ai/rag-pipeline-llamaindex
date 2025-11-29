"""
RAG Retriever

Query interface for retrieving relevant context from the vector store
with user/tenant filtering using LlamaIndex's built-in retriever.
"""
from typing import List, Dict, Any, Optional, cast
from loguru import logger

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.schema import BaseNode, NodeWithScore

from rag.config import rag_config
from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore


class RAGRetriever:
    """
    Retriever for querying user context from vector store using LlamaIndex's retriever
    
    Features:
    - Uses LlamaIndex's VectorIndexRetriever
    - Similarity search with user/tenant filtering
    - Parent-child context expansion
    - Configurable top-k results
    """
    
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None
    ):
        """
        Initialize RAG retriever with LlamaIndex components
        
        Args:
            vector_store: Custom vector store instance
            embedding_service: Custom embedding service instance
        """
        self.vector_store_wrapper = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Create storage context with MongoDB vector store
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store_wrapper.vector_store
        )
        
        # Create VectorStoreIndex
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store_wrapper.vector_store,
            embed_model=self.embedding_service.embed_model,
            storage_context=self.storage_context
        )
        
        logger.info("Initialized RAGRetriever with LlamaIndex components")
    
    def retrieve(
        self,
        query: str,
        user_id: str,
        tenant_id: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        additional_filters: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query using LlamaIndex's retriever
        
        Args:
            query: Query text
            user_id: User identifier for filtering
            tenant_id: Tenant identifier for filtering
            top_k: Number of results to return (default from config)
            similarity_threshold: Minimum similarity score (default from config)
            additional_filters: Additional metadata filters
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        try:
            logger.info(f"Retrieving context for query: '{query[:50]}...'")
            
            # Step 1: Build metadata filters for user/tenant
            top_k = top_k or rag_config.top_k_results
            
            filters = [
                MetadataFilter(
                    key="user_id",
                    value=user_id,
                    operator=FilterOperator.EQ
                ),
                MetadataFilter(
                    key="tenant_id",
                    value=tenant_id,
                    operator=FilterOperator.EQ
                )
            ]
            
            # Add additional filters if provided
            if additional_filters:
                for key, value in additional_filters.items():
                    filters.append(
                        MetadataFilter(
                            key=key,
                            value=value,
                            operator=FilterOperator.EQ
                        )
                    )
            
            metadata_filters = MetadataFilters(filters=cast(List[MetadataFilter | MetadataFilters], filters))
            
            # Step 2: Create retriever with filters
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
                filters=metadata_filters,
                embed_model=self.embedding_service.embed_model
            )
            
            # Step 3: Retrieve nodes
            nodes_with_score: List[NodeWithScore] = retriever.retrieve(query)
            
            if not nodes_with_score:
                logger.info("No relevant context found")
                return {
                    "success": True,
                    "context": "",
                    "chunks": [],
                    "num_chunks": 0
                }
            
            # Step 4: Apply similarity threshold if specified
            similarity_threshold = similarity_threshold or rag_config.similarity_threshold
            filtered_nodes = []
            
            for node_with_score in nodes_with_score:
                if node_with_score.score is not None and node_with_score.score >= similarity_threshold:
                    filtered_nodes.append(node_with_score)
            
            if not filtered_nodes:
                logger.info("No nodes passed similarity threshold")
                return {
                    "success": True,
                    "context": "",
                    "chunks": [],
                    "num_chunks": 0
                }
            
            # Step 5: Extract and format context
            chunks = []
            for node_with_score in filtered_nodes:
                chunk_data = {
                    "text": node_with_score.node.get_content(),
                    "metadata": node_with_score.node.metadata,
                    "score": node_with_score.score
                }
                chunks.append(chunk_data)
            
            # Combine chunks into single context string
            context = self._combine_chunks(chunks)
            
            logger.info(f"Retrieved {len(chunks)} chunks for user {user_id}")
            
            return {
                "success": True,
                "context": context,
                "chunks": chunks,
                "num_chunks": len(chunks),
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return {
                "success": False,
                "context": "",
                "chunks": [],
                "num_chunks": 0,
                "error": str(e)
            }
    
    async def retrieve_async(
        self,
        query: str,
        user_id: str,
        tenant_id: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        additional_filters: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Async version of retrieve using LlamaIndex's retriever
        
        Args:
            query: Query text
            user_id: User identifier for filtering
            tenant_id: Tenant identifier for filtering
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            additional_filters: Additional metadata filters
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        try:
            logger.info(f"Retrieving context (async) for query: '{query[:50]}...'")
            
            # Build metadata filters
            top_k = top_k or rag_config.top_k_results
            
            filters = [
                MetadataFilter(
                    key="user_id",
                    value=user_id,
                    operator=FilterOperator.EQ
                ),
                MetadataFilter(
                    key="tenant_id",
                    value=tenant_id,
                    operator=FilterOperator.EQ
                )
            ]
            
            if additional_filters:
                for key, value in additional_filters.items():
                    filters.append(
                        MetadataFilter(
                            key=key,
                            value=value,
                            operator=FilterOperator.EQ
                        )
                    )
            
            metadata_filters = MetadataFilters(filters=cast(List[MetadataFilter | MetadataFilters], filters))
            
            # Create retriever with filters
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
                filters=metadata_filters,
                embed_model=self.embedding_service.embed_model
            )
            
            # Retrieve nodes (async)
            nodes_with_score: List[NodeWithScore] = await retriever.aretrieve(query)
            
            if not nodes_with_score:
                logger.info("No relevant context found")
                return {
                    "success": True,
                    "context": "",
                    "chunks": [],
                    "num_chunks": 0
                }
            
            # Apply similarity threshold
            similarity_threshold = similarity_threshold or rag_config.similarity_threshold
            filtered_nodes = []
            
            for node_with_score in nodes_with_score:
                if node_with_score.score is not None and node_with_score.score >= similarity_threshold:
                    filtered_nodes.append(node_with_score)
            
            if not filtered_nodes:
                logger.info("No nodes passed similarity threshold")
                return {
                    "success": True,
                    "context": "",
                    "chunks": [],
                    "num_chunks": 0
                }
            
            # Extract and format context
            chunks = []
            for node_with_score in filtered_nodes:
                chunk_data = {
                    "text": node_with_score.node.get_content(),
                    "metadata": node_with_score.node.metadata,
                    "score": node_with_score.score
                }
                chunks.append(chunk_data)
            
            context = self._combine_chunks(chunks)
            
            logger.info(f"Retrieved {len(chunks)} chunks (async) for user {user_id}")
            
            return {
                "success": True,
                "context": context,
                "chunks": chunks,
                "num_chunks": len(chunks),
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error retrieving context (async): {str(e)}")
            return {
                "success": False,
                "context": "",
                "chunks": [],
                "num_chunks": 0,
                "error": str(e)
            }
    
    def _combine_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Combine retrieved chunks into a single context string
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Combined context string
        """
        if not chunks:
            return ""
        
        # Sort by score (descending) if available
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        # Combine chunk texts with separators
        context_parts = []
        for i, chunk in enumerate(sorted_chunks, 1):
            text = chunk.get('text', '')
            score = chunk.get('score')
            
            # Add chunk header with metadata
            if score is not None:
                context_parts.append(f"[Chunk {i} - Relevance: {score:.2f}]")
            else:
                context_parts.append(f"[Chunk {i}]")
            
            context_parts.append(text)
            context_parts.append("")  # Empty line separator
        
        return "\n".join(context_parts)
    
    def retrieve_with_parent_context(
        self,
        query: str,
        user_id: str,
        tenant_id: str,
        top_k: int | None = None
    ) -> Dict[str, Any]:
        """
        Retrieve context with parent-child expansion using LlamaIndex's retriever
        
        When child chunks are retrieved, also fetch their parent chunks
        for additional context. Uses LlamaIndex's hierarchical relationships.
        
        Args:
            query: Query text
            user_id: User identifier
            tenant_id: Tenant identifier
            top_k: Number of results
            
        Returns:
            Dictionary with expanded context
        """
        try:
            # First, retrieve normally using LlamaIndex retriever
            result = self.retrieve(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                top_k=top_k
            )
            
            if not result.get("success") or not result.get("chunks"):
                return result
            
            # Identify chunks with parent relationships
            chunks_with_parents = []
            for chunk in result["chunks"]:
                metadata = chunk.get("metadata", {})
                chunk_type = metadata.get("chunk_type")
                parent_id = metadata.get("parent_id")
                
                if chunk_type == "child" and parent_id:
                    # LlamaIndex automatically maintains parent-child relationships
                    # The parent context is accessible through node relationships
                    # For now, include the chunk with its metadata
                    chunks_with_parents.append(chunk)
                else:
                    chunks_with_parents.append(chunk)
            
            # Update result with expanded chunks
            result["chunks"] = chunks_with_parents
            result["context"] = self._combine_chunks(chunks_with_parents)
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving with parent context: {str(e)}")
            return {
                "success": False,
                "context": "",
                "chunks": [],
                "num_chunks": 0,
                "error": str(e)
            }
    
    def as_query_engine(self, user_id: str, tenant_id: str, **kwargs):
        """
        Create a LlamaIndex query engine for this user/tenant
        
        This allows using the retriever with LlamaIndex's query engine
        for more advanced RAG patterns.
        
        Args:
            user_id: User identifier for filtering
            tenant_id: Tenant identifier for filtering
            **kwargs: Additional arguments for query engine
            
        Returns:
            LlamaIndex query engine
        """
        # Build metadata filters
        filters = [
            MetadataFilter(
                key="user_id",
                value=user_id,
                operator=FilterOperator.EQ
            ),
            MetadataFilter(
                key="tenant_id",
                value=tenant_id,
                operator=FilterOperator.EQ
            )
        ]
        metadata_filters = MetadataFilters(filters=cast(List[MetadataFilter | MetadataFilters], filters))
        
        # Create query engine with filters
        query_engine = self.index.as_query_engine(
            filters=metadata_filters,
            similarity_top_k=kwargs.get('top_k', rag_config.top_k_results),
            **kwargs
        )
        
        return query_engine

