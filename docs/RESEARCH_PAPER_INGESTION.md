## Research Paper Ingestion Guide

## Overview

The RAG pipeline now supports ingesting research papers and documents from your filesystem into the vector database, allowing you to query both user health context and research literature together.

## Quick Start

### 1. Basic Ingestion

```python
from rag.document_loader import DocumentLoader

# Create document loader with semantic chunking (recommended for papers)
doc_loader = DocumentLoader()

# Ingest all research papers
result = doc_loader.ingest_directory(
    directory_path="Research Paper",
    document_type="research_paper",
    tenant_id="swastya",
    recursive=True  # Include all subdirectories
)

print(f"Ingested {result['documents_processed']} documents")
print(f"Created {result['nodes_processed']} chunks")
```

### 2. Query Research Papers

```python
from rag.retriever import RAGRetriever

retriever = RAGRetriever()

# Query research papers
result = retriever.retrieve(
    query="What are the health benefits of a low-carb diet?",
    user_id="research_paper:general_diet",
    tenant_id="swastya",
    top_k=5
)

print(result['context'])
```

## Folder Structure

```
Research Paper/
├── General Diet/           → document_type: "general_diet"
├── Diet and Health Risks/  → document_type: "diet_health_risks"
└── Biological Age/         → document_type: "biological_age"
```

## Supported File Types

- **PDF** (`.pdf`) - Research papers, articles
- **Text** (`.txt`, `.md`) - Plain text documents, markdown
- **Word** (`.docx`) - Microsoft Word documents
- **More**: LlamaIndex supports many formats via SimpleDirectoryReader

## Ingestion Strategies

### Option 1: Ingest All at Once

```python
from rag.document_loader import DocumentLoader

doc_loader = DocumentLoader()

# Ingest entire Research Paper folder
result = doc_loader.ingest_directory(
    directory_path="Research Paper",
    document_type="research_paper",
    tenant_id="swastya",
    recursive=True
)
```

### Option 2: Ingest by Category

```python
from rag.document_loader import DocumentLoader

doc_loader = DocumentLoader()

categories = [
    ("General Diet", "general_diet"),
    ("Diet and Health Risks", "diet_health_risks"),
    ("Biological Age", "biological_age")
]

for folder, doc_type in categories:
    result = doc_loader.ingest_directory(
        directory_path=f"Research Paper/{folder}",
        document_type=doc_type,
        tenant_id="swastya",
        recursive=True
    )
    print(f"{folder}: {result['nodes_processed']} chunks")
```

### Option 3: Async Ingestion (Faster)

```python
import asyncio
from rag.document_loader import DocumentLoader

async def ingest_papers():
    doc_loader = DocumentLoader()
    
    result = await doc_loader.ingest_directory_async(
        directory_path="Research Paper",
        document_type="research_paper",
        tenant_id="swastya",
        recursive=True
    )
    
    return result

# Run async
result = asyncio.run(ingest_papers())
```

## Chunking Strategies

### Semantic Chunking (Recommended for Papers)

```python
from rag.document_loader import DocumentLoader
from rag.semantic_chunking import SemanticChunker

# Semantic chunker preserves paper context better
semantic_chunker = SemanticChunker(
    buffer_size=1,
    breakpoint_percentile_threshold=92  # Lower = more chunks
)

doc_loader = DocumentLoader(chunker=semantic_chunker)
```

### Hierarchical Chunking

```python
from rag.document_loader import DocumentLoader
from rag.chunking import HierarchicalChunker

# Hierarchical for structured documents
hierarchical_chunker = HierarchicalChunker(
    chunk_sizes=[2048, 512, 128]
)

doc_loader = DocumentLoader(chunker=hierarchical_chunker)
```

## Query Patterns

### 1. Query Specific Category

```python
from rag.retriever import RAGRetriever

retriever = RAGRetriever()

# Query only General Diet papers
result = retriever.retrieve(
    query="What foods should I eat?",
    user_id="research_paper:general_diet",
    tenant_id="swastya",
    top_k=5
)
```

### 2. Query with Filters

```python
# Query with additional metadata filters
result = retriever.retrieve(
    query="How to calculate biological age?",
    user_id="research_paper:biological_age",
    tenant_id="swastya",
    top_k=5,
    additional_filters={
        "source": "research_paper",
        "priority": "high"
    }
)
```

### 3. Combined User + Research Query

```python
# Query user's health data
user_result = retriever.retrieve(
    query="What are my glucose levels?",
    user_id="user123",
    tenant_id="swastya",
    top_k=3
)

# Query research papers for recommendations
research_result = retriever.retrieve(
    query="Best diet for glucose control?",
    user_id="research_paper:general_diet",
    tenant_id="swastya",
    top_k=3
)

# Combine for AI agent
combined_context = f"""
USER HEALTH DATA:
{user_result['context']}

RELEVANT RESEARCH:
{research_result['context']}
"""
```

## Document Metadata

Each ingested document chunk includes:

```python
{
    "user_id": "research_paper:general_diet",
    "tenant_id": "swastya",
    "document_type": "general_diet",
    "source": "research_paper",
    "filename": "diet_study_2024.pdf",
    "chunk_type": "semantic",
    "chunk_index": 0,
    "chunk_size": 850,
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## User ID Format

Research papers use a special `user_id` format:

```
research_paper:<document_type>
```

Examples:
- `research_paper:general_diet`
- `research_paper:diet_health_risks`
- `research_paper:biological_age`
- `research_paper:research_paper` (all papers)

This allows filtering by document category while keeping papers separate from user health data.

## Statistics and Monitoring

### Get Document Stats

```python
from rag.document_loader import DocumentLoader

doc_loader = DocumentLoader()

# Get stats for a category
stats = doc_loader.get_document_stats(
    document_type="general_diet",
    tenant_id="swastya"
)

print(f"Total chunks: {stats['count']}")
print(f"Last updated: {stats['latest_timestamp']}")
```

### Check Collection Stats

```python
from rag.vector_store import VectorStore

vector_store = VectorStore()
stats = vector_store.get_collection_stats()

print(f"Total documents: {stats['count']}")
print(f"Storage size: {stats['storage_size']} bytes")
```

## Best Practices

### 1. Use Semantic Chunking for Papers

Research papers benefit from semantic boundaries:
```python
semantic_chunker = SemanticChunker(
    buffer_size=1,
    breakpoint_percentile_threshold=90-95  # Adjust based on paper length
)
```

### 2. Organize by Category

Ingest each research category separately for better filtering:
```python
categories = ["general_diet", "diet_health_risks", "biological_age"]
for category in categories:
    doc_loader.ingest_directory(f"Research Paper/{category}", category, ...)
```

### 3. Add Custom Metadata

Include relevant metadata for better filtering:
```python
doc_loader.ingest_directory(
    directory_path="Research Paper/General Diet",
    document_type="general_diet",
    tenant_id="swastya",
    additional_metadata={
        "category": "nutrition",
        "priority": "high",
        "year": "2024"
    }
)
```

### 4. Use Async for Large Batches

For many documents, use async ingestion:
```python
await doc_loader.ingest_directory_async(...)
```

## Example Workflow

```python
from rag.document_loader import DocumentLoader
from rag.retriever import RAGRetriever
from rag.semantic_chunking import SemanticChunker

# Step 1: Ingest research papers (one-time setup)
chunker = SemanticChunker(breakpoint_percentile_threshold=92)
doc_loader = DocumentLoader(chunker=chunker)

result = doc_loader.ingest_directory(
    directory_path="Research Paper",
    document_type="research_paper",
    tenant_id="swastya",
    recursive=True
)

print(f"✓ Ingested {result['documents_processed']} papers")

# Step 2: Query papers
retriever = RAGRetriever()

research_context = retriever.retrieve(
    query="What are the benefits of intermittent fasting?",
    user_id="research_paper:general_diet",
    tenant_id="swastya",
    top_k=5
)

# Step 3: Combine with user health data
user_context = retriever.retrieve(
    query="User's current glucose levels",
    user_id="user123",
    tenant_id="swastya",
    top_k=3
)

# Step 4: Use in AI agent
combined = f"{user_context['context']}\n\n{research_context['context']}"
# Pass to LLM for personalized recommendations
```

## Troubleshooting

### No Documents Found

```python
# Check if path exists
from pathlib import Path
if not Path("Research Paper").exists():
    print("Research Paper folder not found!")
```

### Encoding Errors

```python
# Specify encoding for text files
doc_loader = DocumentLoader()
# SimpleDirectoryReader handles encoding automatically
```

### Large Files

```python
# For very large PDFs, consider splitting first
# Or increase chunk sizes
chunker = HierarchicalChunker(chunk_sizes=[4096, 1024, 256])
```

## Performance Tips

1. **Use Async**: Faster for multiple documents
2. **Semantic Chunking**: Better retrieval for papers
3. **Batch Ingestion**: Process categories together
4. **Monitor Progress**: Log ingestion results
5. **Index Creation**: Ensure MongoDB vector index exists

## Examples

See complete examples in:
- `rag/example_research_paper_ingestion.py`

Run examples:
```bash
python rag/example_research_paper_ingestion.py
```

## MongoDB Atlas Setup

Research papers use the same vector index as user context. Ensure your MongoDB Atlas vector index supports:

- Field: `embedding` (vector)
- Dimensions: 1536
- Filters: `metadata.user_id`, `metadata.tenant_id`, `metadata.document_type`, `metadata.source`

See `rag/SETUP_MONGODB_VECTOR_SEARCH.md` for details.

