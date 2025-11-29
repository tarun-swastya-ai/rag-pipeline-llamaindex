"""
Hierarchical Chunking using LlamaIndex's HierarchicalNodeParser

This module provides hierarchical parent-child chunking for user context text.
Uses LlamaIndex's built-in HierarchicalNodeParser to create a multi-level
hierarchy of text chunks for efficient retrieval.
"""
from typing import List, Dict, Any
from loguru import logger
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship

from rag.config import rag_config


class HierarchicalChunker:
    """
    Wrapper for LlamaIndex's HierarchicalNodeParser
    
    Creates a multi-level hierarchy of text chunks:
    - Level 0 (2048 chars): Large parent chunks for broad context
    - Level 1 (512 chars): Medium chunks for focused context
    - Level 2 (128 chars): Small detail chunks for precise retrieval
    """
    
    def __init__(self, chunk_sizes: List[int] | None = None):
        """
        Initialize the hierarchical chunker
        
        Args:
            chunk_sizes: List of chunk sizes for each level (default from config)
        """
        self.chunk_sizes = chunk_sizes or rag_config.chunk_sizes
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes
        )
        logger.info(f"Initialized HierarchicalChunker with chunk sizes: {self.chunk_sizes}")
    
    def chunk_text(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None
    ) -> List[BaseNode]:
        """
        Chunk text into hierarchical nodes with user/tenant metadata
        
        Args:
            text: Text content to chunk
            user_id: User identifier for filtering
            tenant_id: Tenant identifier for filtering
            additional_metadata: Additional metadata to attach to nodes
            
        Returns:
            List of BaseNode objects with parent-child relationships
        """
        try:
            # Create a LlamaIndex Document
            metadata = {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "source": "user_context"
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            document = Document(
                text=text,
                metadata=metadata
            )
            
            # Parse into hierarchical nodes
            nodes = self.node_parser.get_nodes_from_documents([document])
            
            # Enrich nodes with additional metadata
            for i, node in enumerate(nodes):
                # Add chunk index
                node.metadata["chunk_index"] = i
                
                # Identify chunk level based on relationships
                if hasattr(node, 'relationships') and node.relationships:
                    # If node has parent, it's a child chunk
                    parent_rel = node.relationships.get(NodeRelationship.PARENT)
                    if parent_rel:
                        node.metadata["chunk_type"] = "child"
                        # parent_rel may be a list[RelatedNodeInfo] or a single RelatedNodeInfo.
                        # Safely handle both cases and extract the node_id from the first related entry.
                        if isinstance(parent_rel, list):
                            parent_id = parent_rel[0].node_id if parent_rel else None
                        else:
                            parent_id = getattr(parent_rel, "node_id", None)
                        if parent_id:
                            node.metadata["parent_id"] = parent_id
                    else:
                        node.metadata["chunk_type"] = "parent"
                else:
                    # Top-level node
                    node.metadata["chunk_type"] = "parent"
            
            logger.info(
                f"Chunked text into {len(nodes)} nodes for user {user_id}, "
                f"tenant {tenant_id}"
            )
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def get_chunk_hierarchy(self, nodes: List[BaseNode]) -> Dict[str, List[str]]:
        """
        Extract parent-child relationships from nodes
        
        Args:
            nodes: List of hierarchical nodes
            
        Returns:
            Dictionary mapping parent IDs to list of child IDs
        """
        hierarchy = {}
        
        for node in nodes:
            if node.metadata.get("chunk_type") == "child":
                parent_id = node.metadata.get("parent_id")
                if parent_id:
                    if parent_id not in hierarchy:
                        hierarchy[parent_id] = []
                    hierarchy[parent_id].append(node.node_id)
        
        return hierarchy
    
    def filter_leaf_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        Filter to get only the smallest (leaf) chunks for embedding
        
        In hierarchical chunking, we typically embed only the leaf nodes
        to avoid redundancy, then retrieve parent context when needed.
        
        Args:
            nodes: List of all hierarchical nodes
            
        Returns:
            List of leaf (smallest) nodes
        """

        leaf_nodes= get_leaf_nodes(nodes)
        
        logger.info(f"Filtered {len(leaf_nodes)} leaf nodes from {len(nodes)} total nodes")
        return leaf_nodes

