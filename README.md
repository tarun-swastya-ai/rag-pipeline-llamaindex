# RAG Pipeline with LlamaIndex and MongoDB

A complete Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex with MongoDB Atlas Vector Search for storing and retrieving user context with hierarchical chunking.

## Features

- **Dual Chunking Strategies**:
  - **Hierarchical**: Uses `HierarchicalNodeParser` for multi-level chunks (2048/512/128 chars)
  - **Semantic**: Uses `SemanticSplitterNodeParser` for adaptive, context-aware chunking
- **Document Ingestion**: Load and ingest research papers (PDF, TXT, DOCX) from filesystem
- **Native LlamaIndex Retriever**: Uses `VectorIndexRetriever` for optimal integration with LlamaIndex ecosystem
- **MongoDB Vector Store**: Stores embeddings in MongoDB Atlas with vector search capabilities
- **User/Tenant Filtering**: All queries filtered by `user_id` and `tenant_id` for multi-tenant support
- **Dual Update Strategy**: Store on context changes + refresh on-demand if stale
- **OpenAI Embeddings**: Uses OpenAI's `text-embedding-3-small` model
- **Query Engine Support**: Create LlamaIndex query engines for advanced RAG patterns
- **Combined Queries**: Query both user health context and research papers together
- **Async Support**: Full async/await support for all operations

## Architecture

```
┌─────────────────┐
│  User Context   │
│  (Text Input)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Semantic     │
│    Chunking     │ (SemanticSplitterNodeParser)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │
│    Service      │ (OpenAI text-embedding-3-small)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MongoDB       │
│  Vector Store   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VectorStore    │
│     Index       │ (LlamaIndex VectorStoreIndex)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ VectorIndex     │
│   Retriever     │ (LlamaIndex native retriever)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Query Engine  │ (Optional: Advanced RAG)
└─────────────────┘
```

## Why LlamaIndex's Native Retriever?

This implementation uses **LlamaIndex's `VectorIndexRetriever`** instead of manual vector store queries for several benefits:

1. **Better Integration**: Seamlessly works with LlamaIndex's ecosystem
2. **Query Engine Support**: Can create query engines for advanced RAG patterns
3. **Optimized Performance**: LlamaIndex handles query optimization internally
4. **Metadata Filtering**: Native support for complex metadata filters
5. **Future-Proof**: Benefits from LlamaIndex updates and improvements
6. **Advanced Features**: Access to response synthesizers, reranking, and more

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The following packages will be installed:

- `llama-index>=0.10.0`
- `llama-index-vector-stores-mongodb>=0.1.0`
- `llama-index-embeddings-openai>=0.1.0`
- `llama-index-core>=0.10.0`

### 2. Configure Environment Variables

Add to your `.env` file:

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DB_NAME=swastya

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
```

### 3. Create MongoDB Atlas Vector Search Index

**IMPORTANT**: You must manually create a vector search index in MongoDB Atlas:

1. Go to your MongoDB Atlas cluster
2. Navigate to **Search** → **Create Search Index**
3. Choose **Atlas Vector Search**
4. Select your database and collection (`user_context_vectors`)
5. Use the following configuration:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "metadata.user_id"
    },
    {
      "type": "filter",
      "path": "metadata.tenant_id"
    }
  ]
}
```

6. Name the index: `vector_search_index`

## Usage

### Basic User Context Ingestion

```python
from rag.pipeline import RAGPipeline

# Initialize pipeline with default hierarchical chunking
pipeline = RAGPipeline()

# Or use semantic chunking for better context preservation
from rag.semantic_chunking import SemanticChunker
semantic_chunker = SemanticChunker(
    buffer_size=1,
    breakpoint_percentile_threshold=95
)
pipeline_semantic = RAGPipeline(chunker=semantic_chunker)

# Ingest user context
result = pipeline.ingest_user_context(
    text="User's health context...",
    user_id="1234567890",
    tenant_id="default_tenant"
)
```

### Research Paper Ingestion

```python
from rag.document_loader import DocumentLoader

# Create document loader
doc_loader = DocumentLoader()

# Ingest research papers from folder
result = doc_loader.ingest_directory(
    directory_path="Research Paper",
    document_type="research_paper",
    tenant_id="swastya",
    recursive=True  # Include subdirectories
)

print(f"Ingested {result['documents_processed']} papers")
```

See [RESEARCH_PAPER_INGESTION.md](./docs/RESEARCH_PAPER_INGESTION.md) for complete guide.

### Retrieval

```python
from rag.retriever import RAGRetriever

# Initialize retriever
retriever = RAGRetriever()

# Query for relevant context using LlamaIndex's retriever
result = retriever.retrieve(
    query="What are the user's glucose levels?",
    user_id="1234567890",
    tenant_id="default_tenant",
    top_k=5
)

print(result['context'])

# Or use LlamaIndex's query engine for more advanced RAG
query_engine = retriever.as_query_engine(
    user_id="1234567890",
    tenant_id="default_tenant"
)
response = query_engine.query("What are the user's glucose levels?")
print(response)
```

### Smart Updates

```python
from rag.update_service import UpdateService

# Initialize update service
update_service = UpdateService()

# Update only if data is stale (older than 24 hours)
result = update_service.update_if_stale(
    text="Updated user context...",
    user_id="1234567890",
    tenant_id="default_tenant"
)
```

### Integration with get_user_context

```python
import asyncio
from rag.update_service import UpdateService
from src.whatsapp.utils.user_context import get_user_context

async def update_user_rag():
    update_service = UpdateService()

    # Get user context from existing function
    user_context = await get_user_context("1234567890", "default_tenant")

    # Store in RAG pipeline
    result = await update_service.ingest_or_update_async(
        text=user_context,
        user_id="1234567890",
        tenant_id="default_tenant"
    )

    return result

# Run
asyncio.run(update_user_rag())
```

## Module Overview

### `config.py`

Configuration settings for the RAG pipeline including:

- MongoDB connection settings
- OpenAI embedding model configuration
- Chunk sizes for hierarchical parsing
- Retrieval parameters

### `chunking.py`

Hierarchical text chunking using LlamaIndex's `HierarchicalNodeParser`:

- Creates 3-level hierarchy (2048/512/128 characters)
- Maintains parent-child relationships
- Filters leaf nodes for embedding

### `semantic_chunking.py` (NEW!)

Semantic text chunking using LlamaIndex's `SemanticSplitterNodeParser`:

- Adaptive breakpoints based on embedding similarity
- Semantically coherent chunks (no arbitrary splits)
- Better context preservation
- Configurable threshold and buffer size

### `embedding_service.py`

OpenAI embedding generation:

- Text-to-vector conversion
- Batch processing support
- Retry logic for reliability
- Async support

### `vector_store.py`

MongoDB Atlas Vector Search integration:

- Stores embeddings with metadata
- User/tenant filtering
- Vector similarity search
- Index management

### `pipeline.py`

Main orchestrator:

- Ingests text → chunks → embeds → stores
- Update and delete operations
- Sync and async support

### `retriever.py`

Query interface using LlamaIndex's VectorIndexRetriever:

- Native LlamaIndex retriever with VectorStoreIndex
- Similarity search with metadata filtering
- Parent-child context expansion
- Configurable top-k and threshold
- Query engine creation for advanced RAG patterns

### `update_service.py`

Smart update management:

- Checks if data is stale
- Updates only when needed
- Periodic refresh support
- Batch processing

### `example_usage.py`

Comprehensive examples demonstrating all features

## Configuration

Default configuration in `rag/config.py`:

```python
# Chunk sizes (3-level hierarchy)
chunk_sizes = [2048, 512, 128]

# Embedding model
embedding_model = "text-embedding-3-small"
embedding_dimensions = 1536

# Retrieval settings
top_k_results = 5
similarity_threshold = 0.7

# Update settings
stale_threshold_hours = 24
```

## MongoDB Document Structure

Documents stored in MongoDB:

```json
{
  "text": "chunk text content",
  "embedding": [0.123, -0.456, ...],  // 1536-dimensional vector
  "metadata": {
    "user_id": "1234567890",
    "tenant_id": "default_tenant",
    "chunk_type": "child",  // or "parent"
    "parent_id": "parent_node_id",
    "chunk_index": 0,
    "timestamp": "2024-01-01T00:00:00Z",
    "source": "user_context"
  }
}
```

## API Reference

### RAGPipeline

```python
# Ingest user context
pipeline.ingest_user_context(text, user_id, tenant_id, additional_metadata=None)

# Update existing context
pipeline.update_user_context(text, user_id, tenant_id, additional_metadata=None)

# Delete user data
pipeline.delete_user_data(user_id, tenant_id)

# Get user data info
pipeline.get_user_data_info(user_id, tenant_id)
```

### RAGRetriever

```python
# Retrieve relevant context
retriever.retrieve(query, user_id, tenant_id, top_k=5, similarity_threshold=0.7)

# Retrieve with parent context expansion
retriever.retrieve_with_parent_context(query, user_id, tenant_id, top_k=5)

# Get LlamaIndex query engine for advanced usage
query_engine = retriever.as_query_engine(user_id, tenant_id, top_k=5)
response = query_engine.query("What are the user's glucose levels?")
```

### UpdateService

```python
# Update if stale
update_service.update_if_stale(text, user_id, tenant_id, force=False)

# Ingest or update (smart choice)
update_service.ingest_or_update(text, user_id, tenant_id)

# Check if data is stale
update_service.is_data_stale(user_id, tenant_id)

# Get update recommendation
update_service.get_update_recommendation(user_id, tenant_id)
```

## Performance Considerations

- **Chunk Size**: Larger chunks (2048) provide more context but less precision
- **Top-K**: Higher values return more results but may include less relevant chunks
- **Similarity Threshold**: Higher values (>0.8) are more strict, lower (<0.6) more permissive
- **Batch Processing**: Use async methods for processing multiple users

## Troubleshooting

### "Vector index not found"

- Ensure you've created the vector search index in MongoDB Atlas
- Check that the index name matches `vector_search_index`
- Verify the index is on the `embedding` field with 1536 dimensions

### "No results returned"

- Check that data has been ingested for the user/tenant
- Lower the `similarity_threshold`
- Increase `top_k` results

### "OpenAI API errors"

- Verify `OPENAI_API_KEY` is set correctly
- Check API rate limits
- Retry logic is built-in (3 attempts with exponential backoff)
