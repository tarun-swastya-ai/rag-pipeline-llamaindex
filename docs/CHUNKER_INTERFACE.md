# Chunker Interface Compatibility

## Overview

Both `HierarchicalChunker` and `SemanticChunker` implement the same interface, making them **fully interchangeable** in the RAG pipeline.

## Common Interface

Both chunkers implement these required methods:

### 1. `chunk_text()`
```python
def chunk_text(
    text: str,
    user_id: str,
    tenant_id: str,
    additional_metadata: Dict[str, Any] | None = None
) -> List[BaseNode]
```

**Purpose**: Convert text into chunks with metadata

**Returns**: List of LlamaIndex `BaseNode` objects

### 2. `filter_leaf_nodes()`
```python
def filter_leaf_nodes(nodes: List[BaseNode]) -> List[BaseNode]
```

**Purpose**: Filter to get only leaf nodes for embedding

**Behavior**:
- **Hierarchical**: Returns only leaf nodes (smallest chunks)
- **Semantic**: Returns all nodes (no hierarchy)

## Usage in RAGPipeline

The pipeline accepts both chunkers through the `chunker` parameter:

```python
from rag.pipeline import RAGPipeline
from rag.chunking import HierarchicalChunker
from rag.semantic_chunking import SemanticChunker

# Option 1: Hierarchical (default)
pipeline1 = RAGPipeline()
pipeline1 = RAGPipeline(chunker=HierarchicalChunker())

# Option 2: Semantic
pipeline2 = RAGPipeline(chunker=SemanticChunker())

# Both use identical API
result1 = pipeline1.ingest_user_context(text, user_id, tenant_id)
result2 = pipeline2.ingest_user_context(text, user_id, tenant_id)
```

## Type Hints

The pipeline now accepts both types:

```python
from typing import Union

def __init__(
    self,
    vector_store: VectorStore | None = None,
    embedding_service: EmbeddingService | None = None,
    chunker: Union[HierarchicalChunker, SemanticChunker, None] = None
):
    ...
```

## Usage in UpdateService

The `UpdateService` also works with both chunkers:

```python
from rag.update_service import UpdateService
from rag.pipeline import RAGPipeline
from rag.semantic_chunking import SemanticChunker

# Create pipeline with semantic chunker
semantic_chunker = SemanticChunker(
    buffer_size=1,
    breakpoint_percentile_threshold=95
)
pipeline = RAGPipeline(chunker=semantic_chunker)

# UpdateService inherits the chunking strategy
update_service = UpdateService(pipeline=pipeline)

# All methods work identically
update_service.ingest_or_update(text, user_id, tenant_id)
update_service.update_if_stale(text, user_id, tenant_id, force=False)
```

## Complete Example

```python
from rag.pipeline import RAGPipeline
from rag.update_service import UpdateService
from rag.chunking import HierarchicalChunker
from rag.semantic_chunking import SemanticChunker

# Sample text
text = """
Glucose: 95 mg/dL (Normal: 70-100)
HbA1c: 5.6% (Normal: <5.7%)

The patient shows excellent metabolic health.
All markers are in normal range.
"""

user_id = "user123"
tenant_id = "tenant1"

# ============================================
# Scenario 1: Hierarchical Chunking
# ============================================
print("Using Hierarchical Chunking:")

hierarchical_chunker = HierarchicalChunker(
    chunk_sizes=[2048, 512, 128]
)
pipeline_h = RAGPipeline(chunker=hierarchical_chunker)
update_service_h = UpdateService(pipeline=pipeline_h)

result_h = update_service_h.ingest_or_update(text, user_id, tenant_id)
print(f"Nodes processed: {result_h['nodes_processed']}")

# ============================================
# Scenario 2: Semantic Chunking
# ============================================
print("\nUsing Semantic Chunking:")

semantic_chunker = SemanticChunker(
    buffer_size=1,
    breakpoint_percentile_threshold=95
)
pipeline_s = RAGPipeline(chunker=semantic_chunker)
update_service_s = UpdateService(pipeline=pipeline_s)

result_s = update_service_s.ingest_or_update(text, user_id, tenant_id)
print(f"Nodes processed: {result_s['nodes_processed']}")

# ============================================
# Both produce the same result structure!
# ============================================
assert result_h['success'] == result_s['success']
assert 'nodes_processed' in result_h
assert 'nodes_processed' in result_s
```

## Testing Compatibility

Run the compatibility test suite:

```bash
python rag/test_chunker_compatibility.py
```

This runs 6 tests confirming:
1. ✅ Hierarchical works in pipeline
2. ✅ Semantic works in pipeline
3. ✅ Hierarchical works in UpdateService
4. ✅ Semantic works in UpdateService
5. ✅ Default pipeline uses hierarchical
6. ✅ Strategies can be switched

## Key Benefits

### 1. **Polymorphism**
Switch strategies without changing any other code

### 2. **Same API**
All methods return identical structures

### 3. **Drop-in Replacement**
Change one line to switch strategies

### 4. **No Breaking Changes**
Existing code continues to work

### 5. **Flexibility**
Choose strategy based on content type

## Implementation Details

### HierarchicalChunker
```python
class HierarchicalChunker:
    def chunk_text(...) -> List[BaseNode]:
        # Uses HierarchicalNodeParser
        # Returns nodes with parent-child relationships
        
    def filter_leaf_nodes(nodes) -> List[BaseNode]:
        # Returns only leaf (smallest) nodes
        # Filters out parent nodes
```

### SemanticChunker
```python
class SemanticChunker:
    def chunk_text(...) -> List[BaseNode]:
        # Uses SemanticSplitterNodeParser
        # Returns flat list of semantic chunks
        
    def filter_leaf_nodes(nodes) -> List[BaseNode]:
        # Returns all nodes (no hierarchy)
        # All semantic chunks are "leaf" nodes
```

## Decision Matrix

| Need | Recommended Chunker |
|------|---------------------|
| Speed critical | HierarchicalChunker |
| Context preservation | SemanticChunker |
| Parent-child relationships | HierarchicalChunker |
| Natural boundaries | SemanticChunker |
| Consistent sizes | HierarchicalChunker |
| Variable sizes OK | SemanticChunker |
| Health reports | SemanticChunker |
| Large documents | HierarchicalChunker |

## Future Extensibility

To add a new chunking strategy:

1. Create new chunker class
2. Implement required methods:
   - `chunk_text(text, user_id, tenant_id, additional_metadata) -> List[BaseNode]`
   - `filter_leaf_nodes(nodes) -> List[BaseNode]`
3. Add to pipeline type hints
4. Document in this file

Example:
```python
class CustomChunker:
    def chunk_text(self, text, user_id, tenant_id, additional_metadata=None):
        # Your custom chunking logic
        return nodes
    
    def filter_leaf_nodes(self, nodes):
        # Your custom filtering logic
        return filtered_nodes

# Use in pipeline
pipeline = RAGPipeline(chunker=CustomChunker())
```

## Conclusion

Both `HierarchicalChunker` and `SemanticChunker` are **100% compatible** with:
- ✅ `RAGPipeline`
- ✅ `UpdateService`
- ✅ `RAGRetriever`
- ✅ All pipeline methods (sync and async)

Choose the strategy that best fits your use case, and switch anytime without code changes!

