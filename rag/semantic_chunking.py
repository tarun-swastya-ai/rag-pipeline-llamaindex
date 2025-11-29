"""
Semantic Chunking using LlamaIndex's SemanticSplitterNodeParser

This module provides semantic chunking for user context text.
Instead of using fixed chunk sizes, it adaptively picks breakpoints
based on embedding similarity to ensure semantically coherent chunks.

Based on Greg Kamradt's "5 levels of embedding chunking" concept:
https://youtu.be/8OJC21T2SL4?t=1933
"""
from typing import List, Dict, Any, Sequence
from loguru import logger
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode

from rag.config import rag_config
from rag.embedding_service import EmbeddingService


class SemanticChunker:
    """
    Wrapper for LlamaIndex's SemanticSplitterNodeParser
    
    Adaptively chunks text based on semantic similarity rather than fixed sizes.
    This ensures that each chunk contains semantically related sentences.
    
    Features:
    - Adaptive breakpoint detection using embedding similarity
    - Semantically coherent chunks
    - Better context preservation compared to fixed-size chunking
    """
    
    def __init__(
        self,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        embedding_service: EmbeddingService | None = None
    ):
        """
        Initialize the semantic chunker
        
        Args:
            buffer_size: Number of sentences to group together when evaluating 
                        semantic similarity (default: 1)
            breakpoint_percentile_threshold: Percentile threshold for detecting
                                            breakpoints (default: 95)
                                            Higher = fewer, larger chunks
                                            Lower = more, smaller chunks
            embedding_service: Custom embedding service instance
        """
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize LlamaIndex semantic splitter
        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=self.embedding_service.embed_model
        )
        
        logger.info(
            f"Initialized SemanticChunker with buffer_size={self.buffer_size}, "
            f"threshold={self.breakpoint_percentile_threshold}"
        )
    
    def chunk_text(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None
    ) -> List[BaseNode]:
        """
        Chunk text into semantically coherent nodes with user/tenant metadata
        
        Args:
            text: Text content to chunk
            user_id: User identifier for filtering
            tenant_id: Tenant identifier for filtering
            additional_metadata: Additional metadata to attach to nodes
            
        Returns:
            List of BaseNode objects with semantic boundaries
        """
        try:
            # Create a LlamaIndex Document
            metadata = {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "source": "user_context",
                "chunking_strategy": "semantic"
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            document = Document(
                text=text,
                metadata=metadata
            )
            
            # Parse into semantic nodes
            nodes = self.node_parser.get_nodes_from_documents([document])
            
            # Enrich nodes with additional metadata
            for i, node in enumerate(nodes):
                # Add chunk index
                node.metadata["chunk_index"] = i
                node.metadata["chunk_type"] = "semantic"
                
                # Add chunk size for reference
                node.metadata["chunk_size"] = len(node.get_content())
            
            logger.info(
                f"Chunked text into {len(nodes)} semantic nodes for user {user_id}, "
                f"tenant {tenant_id}"
            )
            
            # Log chunk size distribution
            if nodes:
                sizes = [len(node.get_content()) for node in nodes]
                avg_size = sum(sizes) / len(sizes)
                logger.info(
                    f"Semantic chunks - Avg size: {avg_size:.0f} chars, "
                    f"Min: {min(sizes)}, Max: {max(sizes)}"
                )
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            raise
    
    def get_chunk_statistics(self, nodes: List[BaseNode]) -> Dict[str, Any]:
        """
        Get statistics about the semantic chunks
        
        Args:
            nodes: List of semantic nodes
            
        Returns:
            Dictionary with chunk statistics
        """
        if not nodes:
            return {
                "num_chunks": 0,
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0,
                "total_chars": 0
            }
        
        sizes = [len(node.get_content()) for node in nodes]
        
        return {
            "num_chunks": len(nodes),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes),
            "strategy": "semantic",
            "buffer_size": self.buffer_size,
            "threshold": self.breakpoint_percentile_threshold
        }
    
    def compare_with_fixed_size(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        fixed_chunk_size: int = 512
    ) -> Dict[str, Any]:
        """
        Compare semantic chunking with fixed-size chunking
        
        Args:
            text: Text to chunk
            user_id: User identifier
            tenant_id: Tenant identifier
            fixed_chunk_size: Size for fixed chunking comparison
            
        Returns:
            Comparison statistics
        """
        from llama_index.core.node_parser import SentenceSplitter
        
        # Get semantic chunks
        semantic_nodes = self.chunk_text(text, user_id, tenant_id)
        semantic_stats = self.get_chunk_statistics(semantic_nodes)
        
        # Get fixed-size chunks for comparison
        fixed_splitter = SentenceSplitter(chunk_size=fixed_chunk_size)
        document = Document(
            text=text,
            metadata={"user_id": user_id, "tenant_id": tenant_id}
        )
        fixed_nodes = fixed_splitter.get_nodes_from_documents([document])
        fixed_sizes = [len(node.get_content()) for node in fixed_nodes]
        
        return {
            "semantic": {
                "num_chunks": semantic_stats["num_chunks"],
                "avg_size": semantic_stats["avg_size"],
                "size_std": self._calculate_std(
                    [len(n.get_content()) for n in semantic_nodes]
                )
            },
            "fixed_size": {
                "num_chunks": len(fixed_nodes),
                "avg_size": sum(fixed_sizes) / len(fixed_sizes) if fixed_sizes else 0,
                "size_std": self._calculate_std(fixed_sizes)
            },
            "comparison": {
                "chunk_reduction": len(fixed_nodes) - semantic_stats["num_chunks"],
                "semantic_coherence": "higher" if semantic_stats["num_chunks"] < len(fixed_nodes) else "similar"
            }
        }
    
    def filter_leaf_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        Filter to get leaf nodes (compatibility with pipeline)
        
        For semantic chunking, all nodes are leaf nodes since there's no hierarchy.
        This method exists for interface compatibility with HierarchicalChunker.
        
        Args:
            nodes: List of semantic nodes
            
        Returns:
            All nodes (semantic chunks don't have hierarchy)
        """
        # Semantic chunks are flat - no parent-child relationships
        # All nodes are "leaf" nodes
        return nodes
    
    def _calculate_std(self, values: Sequence[float | int]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        # Normalize to floats to ensure correct arithmetic and typing compatibility
        vals = [float(x) for x in values]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        return variance ** 0.5
    
    def adjust_threshold(self, new_threshold: int):
        """
        Adjust the breakpoint threshold dynamically
        
        Args:
            new_threshold: New percentile threshold (0-100)
                          Higher = fewer, larger chunks
                          Lower = more, smaller chunks
        """
        if not 0 <= new_threshold <= 100:
            raise ValueError("Threshold must be between 0 and 100")
        
        self.breakpoint_percentile_threshold = new_threshold
        
        # Reinitialize parser with new threshold
        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=self.embedding_service.embed_model
        )
        
        logger.info(f"Adjusted semantic chunker threshold to {new_threshold}")

