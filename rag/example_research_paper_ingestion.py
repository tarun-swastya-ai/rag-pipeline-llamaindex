"""
Example: Ingesting Research Papers

This file demonstrates how to ingest research papers from the Research Paper folder
into the vector database.
"""
import asyncio
from loguru import logger
from pathlib import Path

from rag.document_loader import DocumentLoader
from rag.retriever import RAGRetriever
from rag.semantic_chunking import SemanticChunker
from rag.chunking import HierarchicalChunker


# Path to research papers (adjust based on your setup)
RESEARCH_PAPER_PATH = Path("Research Paper")


# ==============================================================================
# Example 1: Ingest All Research Papers
# ==============================================================================

def example_ingest_all_research_papers():
    """
    Example: Ingest all research papers from all subdirectories
    """
    logger.info("=== Example 1: Ingest All Research Papers ===")
    
    # Create document loader with semantic chunking (better for papers)
    semantic_chunker = SemanticChunker(
        buffer_size=1,
        breakpoint_percentile_threshold=92  # Slightly lower for more granular chunks
    )
    
    doc_loader = DocumentLoader(chunker=semantic_chunker)
    
    # Ingest all documents from Research Paper folder
    result = doc_loader.ingest_directory(
        directory_path=str(RESEARCH_PAPER_PATH),
        document_type="research_paper",
        tenant_id="swastya",
        recursive=True,  # Load from all subdirectories
        required_exts=['.pdf', '.txt', '.md', '.docx']  # Supported file types
    )
    
    logger.info(f"Ingestion result: {result}")
    logger.info(f"Documents processed: {result.get('documents_processed', 0)}")
    logger.info(f"Nodes created: {result.get('nodes_processed', 0)}")
    
    return result


# ==============================================================================
# Example 2: Ingest Specific Category
# ==============================================================================

def example_ingest_general_diet():
    """
    Example: Ingest only General Diet papers
    """
    logger.info("=== Example 2: Ingest General Diet Papers ===")
    
    doc_loader = DocumentLoader(
        chunker=SemanticChunker(breakpoint_percentile_threshold=90)
    )
    
    # Ingest General Diet folder
    general_diet_path = RESEARCH_PAPER_PATH / "General Diet"
    
    result = doc_loader.ingest_directory(
        directory_path=str(general_diet_path),
        document_type="general_diet",
        tenant_id="swastya",
        recursive=True
    )
    
    logger.info(f"General Diet ingestion: {result}")
    
    return result


# ==============================================================================
# Example 3: Ingest Each Category Separately
# ==============================================================================

def example_ingest_by_category():
    """
    Example: Ingest each research category separately with custom metadata
    """
    logger.info("=== Example 3: Ingest By Category ===")
    
    doc_loader = DocumentLoader(
        chunker=SemanticChunker(breakpoint_percentile_threshold=92)
    )
    
    categories = [
        ("General Diet", "general_diet"),
        ("Diet and Health Risks", "diet_health_risks"),
        ("Biological Age", "biological_age")
    ]
    
    results = {}
    
    for folder_name, doc_type in categories:
        category_path = RESEARCH_PAPER_PATH / folder_name
        
        if not category_path.exists():
            logger.warning(f"Path not found: {category_path}")
            continue
        
        logger.info(f"\nIngesting {folder_name}...")
        
        result = doc_loader.ingest_directory(
            directory_path=str(category_path),
            document_type=doc_type,
            tenant_id="swastya",
            recursive=True,
            additional_metadata={
                "category": folder_name,
                "priority": "high"
            }
        )
        
        results[doc_type] = result
        
        logger.info(
            f"  ✓ {folder_name}: {result.get('documents_processed', 0)} documents, "
            f"{result.get('nodes_processed', 0)} chunks"
        )
    
    return results


# ==============================================================================
# Example 4: Query Research Papers
# ==============================================================================

def example_query_research_papers():
    """
    Example: Query the ingested research papers
    """
    logger.info("=== Example 4: Query Research Papers ===")
    
    retriever = RAGRetriever()
    
    # Query research papers about diet
    result = retriever.retrieve(
        query="What are the health benefits of a low-carb diet?",
        user_id="research_paper:general_diet",
        tenant_id="swastya",
        top_k=5
    )
    
    logger.info(f"\nQuery Results:")
    logger.info(f"Found {result['num_chunks']} relevant chunks")
    logger.info(f"\nContext:\n{result['context'][:500]}...")
    
    return result


# ==============================================================================
# Example 5: Query Specific Category
# ==============================================================================

def example_query_biological_age():
    """
    Example: Query only Biological Age papers
    """
    logger.info("=== Example 5: Query Biological Age Papers ===")
    
    retriever = RAGRetriever()
    
    # Query biological age papers
    result = retriever.retrieve(
        query="How is biological age calculated?",
        user_id="research_paper:biological_age",
        tenant_id="swastya",
        top_k=5
    )
    
    logger.info(f"\nBiological Age Query:")
    logger.info(f"Found {result['num_chunks']} chunks")
    
    if result['chunks']:
        for i, chunk in enumerate(result['chunks'], 1):
            metadata = chunk.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            score = chunk.get('score', 0)
            
            logger.info(f"\nChunk {i}:")
            logger.info(f"  File: {filename}")
            logger.info(f"  Score: {score:.3f}")
            logger.info(f"  Text: {chunk['text'][:200]}...")
    
    return result


# ==============================================================================
# Example 6: Query Across All Categories
# ==============================================================================

def example_query_all_categories():
    """
    Example: Query across all research paper categories
    """
    logger.info("=== Example 6: Query All Categories ===")
    
    retriever = RAGRetriever()
    
    # Query all research papers
    result = retriever.retrieve(
        query="What is the relationship between diet and chronic diseases?",
        user_id="research_paper:research_paper",  # General research paper user_id
        tenant_id="swastya",
        top_k=10,
        additional_filters={
            "source": "research_paper"
        }
    )
    
    logger.info(f"\nCross-category Query:")
    logger.info(f"Found {result['num_chunks']} chunks")
    
    return result


# ==============================================================================
# Example 7: Get Document Statistics
# ==============================================================================

def example_get_statistics():
    """
    Example: Get statistics about ingested research papers
    """
    logger.info("=== Example 7: Document Statistics ===")
    
    doc_loader = DocumentLoader()
    
    categories = ["general_diet", "diet_health_risks", "biological_age"]
    
    for category in categories:
        stats = doc_loader.get_document_stats(
            document_type=category,
            tenant_id="swastya"
        )
        
        logger.info(f"\n{category.upper()}:")
        logger.info(f"  Total chunks: {stats.get('count', 0)}")
        logger.info(f"  Last updated: {stats.get('latest_timestamp', 'N/A')}")


# ==============================================================================
# Example 8: Async Ingestion (Faster)
# ==============================================================================

async def example_async_ingestion():
    """
    Example: Ingest research papers asynchronously for better performance
    """
    logger.info("=== Example 8: Async Ingestion ===")
    
    doc_loader = DocumentLoader(
        chunker=SemanticChunker(breakpoint_percentile_threshold=92)
    )
    
    # Ingest asynchronously
    result = await doc_loader.ingest_directory_async(
        directory_path=str(RESEARCH_PAPER_PATH),
        document_type="research_paper",
        tenant_id="swastya",
        recursive=True
    )
    
    logger.info(f"Async ingestion result: {result}")
    
    return result


# ==============================================================================
# Example 9: Combined User Context + Research Papers Query
# ==============================================================================

def example_combined_query():
    """
    Example: Query both user context and research papers together
    """
    logger.info("=== Example 9: Combined Query ===")
    
    retriever = RAGRetriever()
    
    # For a specific user with health data
    user_id = "1234567890"  # User's ID
    tenant_id = "swastya"
    
    # Query user's health context
    user_result = retriever.retrieve(
        query="What are my glucose levels?",
        user_id=user_id,
        tenant_id=tenant_id,
        top_k=3
    )
    
    logger.info(f"\nUser Context: {user_result['num_chunks']} chunks")
    
    # Query research papers for recommendations
    research_result = retriever.retrieve(
        query="What diet is recommended for glucose control?",
        user_id="research_paper:general_diet",
        tenant_id=tenant_id,
        top_k=3
    )
    
    logger.info(f"Research Papers: {research_result['num_chunks']} chunks")
    
    # Combine contexts
    combined_context = f"""
    USER HEALTH DATA:
    {user_result['context']}
    
    RELEVANT RESEARCH:
    {research_result['context']}
    """
    
    logger.info(f"\nCombined context ready for AI agent!")
    
    return {
        "user_context": user_result,
        "research_context": research_result,
        "combined": combined_context
    }


# ==============================================================================
# Main execution
# ==============================================================================

if __name__ == "__main__":
    logger.info("Starting Research Paper Ingestion Examples\n")
    
    # Check if Research Paper folder exists
    if not RESEARCH_PAPER_PATH.exists():
        logger.error(f"Research Paper folder not found at: {RESEARCH_PAPER_PATH}")
        logger.info("Please update the RESEARCH_PAPER_PATH in this script")
        exit(1)
    
    # Run synchronous examples
    print("\n" + "="*80)
    example_ingest_by_category()
    
    print("\n" + "="*80)
    example_query_research_papers()
    
    print("\n" + "="*80)
    example_query_biological_age()
    
    print("\n" + "="*80)
    example_get_statistics()
    
    print("\n" + "="*80)
    example_combined_query()
    
    # Run async example
    print("\n" + "="*80)
    asyncio.run(example_async_ingestion())
    
    logger.info("\n" + "="*80)
    logger.info("All research paper examples completed!")
    logger.info("Research papers are now queryable alongside user context!")

