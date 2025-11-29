"""
Update Service for RAG Pipeline

Handles periodic updates and on-demand refreshes of user context embeddings.
Implements dual update strategy: store on context changes + refresh if stale.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from loguru import logger
import asyncio

from rag.config import rag_config
from rag.pipeline import RAGPipeline


class UpdateService:
    """
    Service for managing user context updates
    
    Features:
    - Check if user data is stale
    - Update embeddings when context changes
    - Periodic refresh of stale data
    - Async update support
    """
    
    def __init__(self, pipeline: RAGPipeline | None = None):
        """
        Initialize update service
        
        Args:
            pipeline: Custom RAG pipeline instance
        """
        self.pipeline = pipeline or RAGPipeline()
        self.stale_threshold_hours = rag_config.stale_threshold_hours
        
        logger.info(
            f"Initialized UpdateService with stale threshold: "
            f"{self.stale_threshold_hours} hours"
        )
    
    def is_data_stale(self, user_id: str, tenant_id: str) -> bool:
        """
        Check if user's data is stale and needs refresh
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            
        Returns:
            True if data is stale or doesn't exist, False otherwise
        """
        try:
            # Get user data info
            info = self.pipeline.get_user_data_info(user_id, tenant_id)
            
            # No data exists - considered stale
            if info.get("count", 0) == 0:
                logger.info(f"No data exists for user {user_id} - considered stale")
                return True
            
            # Check timestamp
            latest_timestamp = info.get("latest_timestamp")
            
            if not latest_timestamp:
                logger.warning(f"No timestamp found for user {user_id} - considered stale")
                return True
            
            # Parse timestamp
            try:
                latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                logger.warning(f"Invalid timestamp format for user {user_id}")
                return True
            
            # Calculate age
            now = datetime.now(timezone.utc)
            age = now - latest_dt
            threshold = timedelta(hours=self.stale_threshold_hours)
            
            is_stale = age > threshold
            
            if is_stale:
                logger.info(
                    f"Data for user {user_id} is stale "
                    f"(age: {age.total_seconds() / 3600:.1f} hours)"
                )
            
            return is_stale
            
        except Exception as e:
            logger.error(f"Error checking if data is stale: {str(e)}")
            # If error, assume stale to trigger refresh
            return True
    
    def update_if_stale(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Update user context only if data is stale or force=True
        
        Args:
            text: User context text
            user_id: User identifier
            tenant_id: Tenant identifier
            additional_metadata: Additional metadata
            force: Force update even if not stale
            
        Returns:
            Dictionary with update results
        """
        try:
            # Check if update needed
            if not force and not self.is_data_stale(user_id, tenant_id):
                logger.info(f"Data for user {user_id} is fresh - skipping update")
                return {
                    "success": True,
                    "message": "Data is fresh, no update needed",
                    "updated": False,
                    "user_id": user_id,
                    "tenant_id": tenant_id
                }
            
            # Perform update
            logger.info(f"Updating user context for user {user_id}")
            
            result = self.pipeline.update_user_context(
                text=text,
                user_id=user_id,
                tenant_id=tenant_id,
                additional_metadata=additional_metadata
            )
            
            result["updated"] = result.get("success", False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in update_if_stale: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "updated": False
            }
    
    async def update_if_stale_async(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Async version of update_if_stale
        
        Args:
            text: User context text
            user_id: User identifier
            tenant_id: Tenant identifier
            additional_metadata: Additional metadata
            force: Force update even if not stale
            
        Returns:
            Dictionary with update results
        """
        try:
            # Check if update needed
            if not force and not self.is_data_stale(user_id, tenant_id):
                logger.info(f"Data for user {user_id} is fresh - skipping update")
                return {
                    "success": True,
                    "message": "Data is fresh, no update needed",
                    "updated": False,
                    "user_id": user_id,
                    "tenant_id": tenant_id
                }
            
            # Perform update (delete)
            logger.info(f"Updating user context (async) for user {user_id}")
            
            delete_result = self.pipeline.delete_user_data(user_id, tenant_id)
            
            # Ingest new data (async)
            ingest_result = await self.pipeline.ingest_user_context_async(
                text=text,
                user_id=user_id,
                tenant_id=tenant_id,
                additional_metadata=additional_metadata
            )
            
            return {
                "success": ingest_result.get("success", False),
                "message": "User context updated successfully",
                "updated": True,
                "deleted_count": delete_result.get("deleted_count", 0),
                "nodes_processed": ingest_result.get("nodes_processed", 0),
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error in update_if_stale_async: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "updated": False
            }
    
    def ingest_or_update(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Smart ingest: Update if data exists, ingest if new
        
        Args:
            text: User context text
            user_id: User identifier
            tenant_id: Tenant identifier
            additional_metadata: Additional metadata
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Check if data exists
            info = self.pipeline.get_user_data_info(user_id, tenant_id)
            data_exists = info.get("count", 0) > 0
            
            if data_exists:
                logger.info(f"Data exists for user {user_id} - updating")
                return self.pipeline.update_user_context(
                    text=text,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    additional_metadata=additional_metadata
                )
            else:
                logger.info(f"No data for user {user_id} - ingesting new")
                return self.pipeline.ingest_user_context(
                    text=text,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    additional_metadata=additional_metadata
                )
                
        except Exception as e:
            logger.error(f"Error in ingest_or_update: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "nodes_processed": 0
            }
    
    async def ingest_or_update_async(
        self,
        text: str,
        user_id: str,
        tenant_id: str,
        additional_metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Async version of ingest_or_update
        
        Args:
            text: User context text
            user_id: User identifier
            tenant_id: Tenant identifier
            additional_metadata: Additional metadata
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Check if data exists
            info = self.pipeline.get_user_data_info(user_id, tenant_id)
            data_exists = info.get("count", 0) > 0
            
            if data_exists:
                logger.info(f"Data exists for user {user_id} - updating (async)")
                # Delete old data
                delete_result = self.pipeline.delete_user_data(user_id, tenant_id)
                
                # Ingest new data
                ingest_result = await self.pipeline.ingest_user_context_async(
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
            else:
                logger.info(f"No data for user {user_id} - ingesting new (async)")
                return await self.pipeline.ingest_user_context_async(
                    text=text,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    additional_metadata=additional_metadata
                )
                
        except Exception as e:
            logger.error(f"Error in ingest_or_update_async: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "nodes_processed": 0
            }
    
    def get_update_recommendation(
        self,
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Get recommendation on whether to update user data
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with recommendation and data info
        """
        try:
            info = self.pipeline.get_user_data_info(user_id, tenant_id)
            is_stale = self.is_data_stale(user_id, tenant_id)
            
            recommendation = {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "data_exists": info.get("count", 0) > 0,
                "data_count": info.get("count", 0),
                "latest_timestamp": info.get("latest_timestamp"),
                "is_stale": is_stale,
                "should_update": is_stale,
                "recommendation": (
                    "Update recommended - data is stale or doesn't exist"
                    if is_stale
                    else "Data is fresh - no update needed"
                )
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error getting update recommendation: {str(e)}")
            return {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "error": str(e),
                "should_update": True,  # Safe default
                "recommendation": "Error checking data - update recommended"
            }

