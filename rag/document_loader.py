"""
Document Loader for Research Papers

Loads documents from filesystem (PDFs, text files, etc.) and ingests them
into the vector database alongside user context.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import BaseNode

from rag.config import rag_config
from rag.chunking import HierarchicalChunker
from rag.semantic_chunking import SemanticChunker
from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore


class DocumentLoader:
    """
    Loader for research papers and documents
    
    Supports:
    - PDF files
    - Text files (.txt, .md)
    - Word documents (.docx)
    - Recursive directory loading
    """
    
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        chunker: HierarchicalChunker | SemanticChunker | None = None
    ):
        """
        Initialize document loader
        
        Args:
            vector_store: Custom vector store instance
            embedding_service: Custom embedding service instance
            chunker: Custom chunker instance (hierarchical or semantic)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = chunker or SemanticChunker()  # Default to semantic for papers
        
        logger.info(f"Initialized DocumentLoader with {type(self.chunker).__name__}")
    
    def load_documents_from_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        required_exts: List[str] | None = None
    ) -> List[Document]:
        """
        Load documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to recursively load from subdirectories
            required_exts: List of file extensions to load (e.g., ['.pdf', '.txt'])
                          If None, loads all supported types
            
        Returns:
            List of LlamaIndex Document objects
        """
        try:
            logger.info(f"Loading documents from: {directory_path}")
            
            # Use LlamaIndex's SimpleDirectoryReader
            reader = SimpleDirectoryReader(
                input_dir=directory_path,
                recursive=recursive,
                required_exts=required_exts,
                filename_as_id=True  # Use filename as document ID
            )
            
            documents = reader.load_data()
            
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {directory_path}: {str(e)}")
            raise
    
    def ingest_documents(
        self,
        documents: List[Document],
        document_type: str = "research_paper",
        tenant_id: str = "default",
        additional_metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Ingest documents into vector database
        
        Args:
            documents: List of documents to ingest
            document_type: Type of document (e.g., 'research_paper', 'general_diet')
            tenant_id: Tenant identifier
            additional_metadata: Additional metadata to attach
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Ingesting {len(documents)} documents of type '{document_type}'")
            
            all_nodes = []
            
            for doc in documents:
                # Extract filename from metadata
                filename = doc.metadata.get('file_name', 'unknown')
                
                # Add document metadata
                doc_metadata = {
                    "document_type": document_type,
                    "tenant_id": tenant_id,
                    "source": "research_paper",
                    "filename": filename
                }
                
                if additional_metadata:
                    doc_metadata.update(additional_metadata)
                
                # Update document metadata
                doc.metadata.update(doc_metadata)
                
                # Chunk the document
                # Note: For research papers, we use a special user_id format
                user_id = f"research_paper:{document_type}"
                
                nodes = self.chunker.chunk_text(
                    text=doc.get_content(),
                    user_id=user_id,
                    tenant_id=tenant_id,
                    additional_metadata=doc_metadata
                )
                
                all_nodes.extend(nodes)
            
            if not all_nodes:
                logger.warning("No nodes created from documents")
                return {
                    "success": False,
                    "message": "No nodes created",
                    "nodes_processed": 0
                }
            
            # Filter to leaf nodes
            leaf_nodes = self.chunker.filter_leaf_nodes(all_nodes)
            
            if not leaf_nodes:
                leaf_nodes = all_nodes
            
            # Generate embeddings
            texts_to_embed = [node.get_content() for node in leaf_nodes]
            embeddings = self.embedding_service.embed_texts(texts_to_embed)
            
            # Attach embeddings
            for node, embedding in zip(leaf_nodes, embeddings):
                node.embedding = embedding
            
            # Store in MongoDB
            # Use document_type as user_id for research papers
            user_id = f"research_paper:{document_type}"
            node_ids = self.vector_store.add_nodes(
                nodes=leaf_nodes,
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            logger.info(
                f"Successfully ingested {len(leaf_nodes)} nodes from "
                f"{len(documents)} documents"
            )
            
            return {
                "success": True,
                "message": "Documents ingested successfully",
                "documents_processed": len(documents),
                "nodes_processed": len(leaf_nodes),
                "node_ids": node_ids,
                "document_type": document_type,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "documents_processed": 0,
                "nodes_processed": 0
            }
    
    def ingest_directory(
        self,
        directory_path: str,
        document_type: str | None = None,
        tenant_id: str = "default",
        recursive: bool = True,
        required_exts: List[str] | None = None,
        additional_metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Load and ingest documents from a directory in one step
        
        Args:
            directory_path: Path to directory
            document_type: Type of document (inferred from directory name if None)
            tenant_id: Tenant identifier
            recursive: Whether to load recursively
            required_exts: File extensions to load
            additional_metadata: Additional metadata
            
        Returns:
            Dictionary with ingestion results
        """
        # Infer document type from directory name if not provided
        if document_type is None:
            directory_path_obj = Path(directory_path)
            document_type = directory_path_obj.name.lower().replace(" ", "_")
        
        # Load documents
        documents = self.load_documents_from_directory(
            directory_path=directory_path,
            recursive=recursive,
            required_exts=required_exts
        )
        
        # Ingest documents
        return self.ingest_documents(
            documents=documents,
            document_type=document_type,
            tenant_id=tenant_id,
            additional_metadata=additional_metadata
        )
    
    async def ingest_directory_async(
        self,
        directory_path: str,
        document_type: str | None = None,
        tenant_id: str = "default",
        recursive: bool = True,
        required_exts: List[str] | None = None,
        additional_metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Async version of ingest_directory
        
        Args:
            directory_path: Path to directory
            document_type: Type of document
            tenant_id: Tenant identifier
            recursive: Whether to load recursively
            required_exts: File extensions to load
            additional_metadata: Additional metadata
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Infer document type from directory name if not provided
            if document_type is None:
                directory_path_obj = Path(directory_path)
                document_type = directory_path_obj.name.lower().replace(" ", "_")
            
            # Load documents (sync operation)
            documents = self.load_documents_from_directory(
                directory_path=directory_path,
                recursive=recursive,
                required_exts=required_exts
            )
            
            logger.info(f"Ingesting {len(documents)} documents (async)")
            
            all_nodes = []
            
            for doc in documents:
                filename = doc.metadata.get('file_name', 'unknown')
                
                doc_metadata = {
                    "document_type": document_type,
                    "tenant_id": tenant_id,
                    "source": "research_paper",
                    "filename": filename
                }
                
                if additional_metadata:
                    doc_metadata.update(additional_metadata)
                
                doc.metadata.update(doc_metadata)
                
                user_id = f"research_paper:{document_type}"
                
                nodes = self.chunker.chunk_text(
                    text=doc.get_content(),
                    user_id=user_id,
                    tenant_id=tenant_id,
                    additional_metadata=doc_metadata
                )
                
                all_nodes.extend(nodes)
            
            if not all_nodes:
                return {
                    "success": False,
                    "message": "No nodes created",
                    "nodes_processed": 0
                }
            
            leaf_nodes = self.chunker.filter_leaf_nodes(all_nodes)
            if not leaf_nodes:
                leaf_nodes = all_nodes
            
            # Generate embeddings (async)
            texts_to_embed = [node.get_content() for node in leaf_nodes]
            embeddings = await self.embedding_service.embed_texts_async(texts_to_embed)
            
            # Attach embeddings
            for node, embedding in zip(leaf_nodes, embeddings):
                node.embedding = embedding
            
            # Store in MongoDB
            user_id = f"research_paper:{document_type}"
            node_ids = self.vector_store.add_nodes(
                nodes=leaf_nodes,
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            logger.info(
                f"Successfully ingested {len(leaf_nodes)} nodes (async)"
            )
            
            return {
                "success": True,
                "message": "Documents ingested successfully",
                "documents_processed": len(documents),
                "nodes_processed": len(leaf_nodes),
                "node_ids": node_ids,
                "document_type": document_type,
                "tenant_id": tenant_id
            }
            
        except Exception as e:
            logger.error(f"Error in async ingestion: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "documents_processed": 0,
                "nodes_processed": 0
            }
    
    def get_document_stats(
        self,
        document_type: str,
        tenant_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Get statistics about ingested documents
        
        Args:
            document_type: Type of document
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with document statistics
        """
        user_id = f"research_paper:{document_type}"
        return self.vector_store.get_user_data_info(user_id, tenant_id)

