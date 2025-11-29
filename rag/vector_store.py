"""
MongoDB Vector Store Integration

Manages vector storage and retrieval in MongoDB Atlas with user/tenant filtering.
Uses LlamaIndex's MongoDBAtlasVectorSearch for efficient similarity search.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from loguru import logger
import pymongo
from pymongo import MongoClient

from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.schema import BaseNode, TextNode

from rag.config import rag_config


class VectorStore:
    """
    MongoDB Atlas Vector Store with user/tenant filtering support
    
    Stores text chunks as vectors with metadata for efficient retrieval
    filtered by user_id and tenant_id.
    """
    
    def __init__(
        self,
        mongodb_uri: str | None = None,
        db_name: str | None = None,
        collection_name: str | None = None,
        index_name: str | None = None
    ):
        """
        Initialize MongoDB vector store
        
        Args:
            mongodb_uri: MongoDB connection URI (default from config)
            db_name: Database name (default from config)
            collection_name: Collection name (default from config)
            index_name: Vector search index name (default from config)
        """
        self.mongodb_uri = mongodb_uri or rag_config.mongodb_uri
        self.db_name = db_name or rag_config.mongodb_db_name
        self.collection_name = collection_name or rag_config.vector_collection_name
        self.index_name = index_name or rag_config.vector_index_name
        
        # Initialize MongoDB client
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        # Initialize LlamaIndex MongoDB vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self.client,
            db_name=self.db_name,
            collection_name=self.collection_name,
            vector_index_name=self.index_name  # Updated from index_name (deprecated)
        )
        
        # Create indexes for efficient filtering
        self._create_indexes()
        
        logger.info(
            f"Initialized VectorStore: db={self.db_name}, "
            f"collection={self.collection_name}, index={self.index_name}"
        )
    
    def _create_indexes(self):
        """Create indexes for user_id and tenant_id filtering"""
        try:
            # Create compound index for user_id and tenant_id
            self.collection.create_index([
                ("metadata.user_id", pymongo.ASCENDING),
                ("metadata.tenant_id", pymongo.ASCENDING),
                ("metadata.timestamp", pymongo.DESCENDING)
            ])
            
            # Create index for timestamp-based queries
            self.collection.create_index([
                ("metadata.timestamp", pymongo.DESCENDING)
            ])
            
            logger.info("Created MongoDB indexes for filtering")
        except Exception as e:
            logger.warning(f"Error creating indexes (may already exist): {str(e)}")
    
    def add_nodes(
        self,
        nodes: List[BaseNode],
        user_id: str,
        tenant_id: str
    ) -> List[str]:
        """
        Add nodes to the vector store with user/tenant metadata
        
        Args:
            nodes: List of BaseNode objects to store
            user_id: User identifier
            tenant_id: Tenant identifier
            
        Returns:
            List of node IDs that were added
        """
        try:
            # Add timestamp to all nodes
            timestamp = datetime.now(timezone.utc).isoformat()
            
            for node in nodes:
                node.metadata["user_id"] = user_id
                node.metadata["tenant_id"] = tenant_id
                node.metadata["timestamp"] = timestamp
            
            # Add nodes to vector store
            node_ids = self.vector_store.add(nodes)
            
            logger.info(
                f"Added {len(nodes)} nodes for user {user_id}, tenant {tenant_id}"
            )
            
            return node_ids
            
        except Exception as e:
            logger.error(f"Error adding nodes to vector store: {str(e)}")
            raise
    
    def query(
        self,
        query_embedding: List[float],
        user_id: str,
        tenant_id: str,
        top_k: int | None = None,
        filters: Dict[str, Any] | None = None
    ) -> List[BaseNode]:
        """
        Query vector store with user/tenant filtering
        
        Args:
            query_embedding: Query vector
            user_id: User identifier for filtering
            tenant_id: Tenant identifier for filtering
            top_k: Number of results to return (default from config)
            filters: Additional metadata filters
            
        Returns:
            List of matching nodes
        """
        try:
            top_k = top_k or rag_config.top_k_results
            
            # Build metadata filters
            metadata_filters = {
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
            if filters:
                metadata_filters.update(filters)
            
            # Convert metadata_filters dict to MetadataFilters object
            from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

            metadata_filters_obj = MetadataFilters(
                filters=[
                    MetadataFilter(key=k, value=v)
                    for k, v in metadata_filters.items()
                ]
            )

            # Create vector store query
            query_obj = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=metadata_filters_obj
            )
            
            # Execute query
            result = self.vector_store.query(query_obj)
            
            num_nodes = len(result.nodes) if result.nodes is not None else 0
            logger.info(
                f"Query returned {num_nodes} nodes for user {user_id}, "
                f"tenant {tenant_id}"
            )
            
            return list(result.nodes) if result.nodes is not None else []
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise
    
    def delete_user_data(self, user_id: str, tenant_id: str) -> int:
        """
        Delete all data for a specific user
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            
        Returns:
            Number of documents deleted
        """
        try:
            result = self.collection.delete_many({
                "metadata.user_id": user_id,
                "metadata.tenant_id": tenant_id
            })
            
            logger.info(
                f"Deleted {result.deleted_count} documents for user {user_id}, "
                f"tenant {tenant_id}"
            )
            
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting user data: {str(e)}")
            raise
    
    def get_user_data_info(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Get information about stored data for a user
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with count and latest timestamp
        """
        try:
            # Count documents
            count = self.collection.count_documents({
                "metadata.user_id": user_id,
                "metadata.tenant_id": tenant_id
            })
            
            # Get latest timestamp
            latest = self.collection.find_one(
                {
                    "metadata.user_id": user_id,
                    "metadata.tenant_id": tenant_id
                },
                sort=[("metadata.timestamp", pymongo.DESCENDING)]
            )
            
            latest_timestamp = None
            if latest and "metadata" in latest:
                latest_timestamp = latest["metadata"].get("timestamp")
            
            return {
                "count": count,
                "latest_timestamp": latest_timestamp,
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error getting user data info: {str(e)}")
            raise
    
    def check_vector_index_exists(self) -> bool:
        """
        Check if the vector search index exists
        
        Returns:
            True if index exists, False otherwise
        """
        try:
            # List indexes on the collection
            indexes = list(self.collection.list_indexes())
            
            # Check if our vector index exists
            for index in indexes:
                if index.get("name") == self.index_name:
                    logger.info(f"Vector search index '{self.index_name}' exists")
                    return True
            
            logger.warning(
                f"Vector search index '{self.index_name}' not found. "
                f"Please create it manually in MongoDB Atlas UI."
            )
            return False
            
        except Exception as e:
            logger.error(f"Error checking vector index: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = self.db.command("collStats", self.collection_name)
            
            return {
                "count": stats.get("count", 0),
                "size": stats.get("size", 0),
                "storage_size": stats.get("storageSize", 0),
                "avg_obj_size": stats.get("avgObjSize", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

