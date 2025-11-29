# Semantic Chunking Strategy

## Overview

Semantic chunking is an advanced text segmentation strategy that creates chunks based on **semantic similarity** rather than fixed character counts. This ensures that each chunk contains semantically related content, leading to better retrieval accuracy.

## How It Works

### Traditional Fixed-Size Chunking
```
Text: "Glucose is 95. HbA1c is 5.6%. | User exercises daily."
      [-------Chunk 1 (50 chars)-----] | [--Chunk 2--]
```
❌ Problem: Splits related content arbitrarily

### Semantic Chunking
```
Text: "Glucose is 95. HbA1c is 5.6%." | "User exercises daily."
      [-------Chunk 1 (Labs)-------]   | [--Chunk 2 (Lifestyle)--]
```
✅ Solution: Groups semantically related sentences

## Implementation

Based on LlamaIndex's `SemanticSplitterNodeParser`, which uses embedding similarity to detect natural breakpoints in text.

### Algorithm

1. **Split text into sentences**
2. **Generate embeddings** for each sentence
3. **Calculate similarity** between adjacent sentences
4. **Detect breakpoints** where similarity drops below threshold
5. **Create chunks** at breakpoints

### Key Parameters

#### `buffer_size` (default: 1)
Number of sentences to group together when calculating similarity.
- `buffer_size=1`: Compare single sentences
- `buffer_size=2`: Compare pairs of sentences
- Higher values = smoother transitions, fewer breakpoints

#### `breakpoint_percentile_threshold` (default: 95)
Percentile threshold for detecting breakpoints (0-100).
- **Higher (95-99)**: Fewer, larger chunks - preserves more context
- **Medium (85-95)**: Balanced approach (recommended)
- **Lower (70-85)**: More, smaller chunks - better precision

## Usage Examples

### Basic Usage

```python
from rag.semantic_chunking import SemanticChunker

# Initialize semantic chunker
chunker = SemanticChunker(
    buffer_size=1,
    breakpoint_percentile_threshold=95
)

# Chunk text
nodes = chunker.chunk_text(
    text="Your health context...",
    user_id="user123",
    tenant_id="tenant1"
)

# Get statistics
stats = chunker.get_chunk_statistics(nodes)
print(f"Created {stats['num_chunks']} semantic chunks")
print(f"Average size: {stats['avg_size']} chars")
```

### Using in RAG Pipeline

```python
from rag.pipeline import RAGPipeline
from rag.semantic_chunking import SemanticChunker

# Create pipeline with semantic chunker
semantic_chunker = SemanticChunker(
    buffer_size=1,
    breakpoint_percentile_threshold=92
)

pipeline = RAGPipeline(chunker=semantic_chunker)

# Ingest as usual
result = pipeline.ingest_user_context(
    text=user_context,
    user_id="user123",
    tenant_id="tenant1"
)
```

### Adjusting Threshold Dynamically

```python
chunker = SemanticChunker()

# Start with default threshold (95)
nodes_95 = chunker.chunk_text(text, user_id, tenant_id)

# Adjust for more granular chunks
chunker.adjust_threshold(85)
nodes_85 = chunker.chunk_text(text, user_id, tenant_id)

print(f"Threshold 95: {len(nodes_95)} chunks")
print(f"Threshold 85: {len(nodes_85)} chunks")
```

## When to Use Semantic vs Hierarchical

### Use **Semantic Chunking** when:
- ✅ Content has natural semantic boundaries (reports, summaries)
- ✅ Context preservation is critical
- ✅ You want to avoid splitting related concepts
- ✅ Variable chunk sizes are acceptable
- ✅ Retrieval accuracy is prioritized over fixed structure

### Use **Hierarchical Chunking** when:
- ✅ Need consistent chunk sizes for processing
- ✅ Want multi-level context (parent-child relationships)
- ✅ Content is homogeneous
- ✅ Fixed structure simplifies downstream processing
- ✅ Performance with very large documents

## Performance Considerations

### Semantic Chunking
- **Pros**: Better retrieval accuracy, natural boundaries
- **Cons**: Requires embedding generation (slower), variable sizes

### Processing Time
- Semantic chunking is ~2-3x slower than fixed-size (due to embeddings)
- For 1000-word document: ~2-3 seconds
- Embeddings are cached, so repeated chunking is fast

### Chunk Size Distribution

Example with health context (500 words):

```
Hierarchical (512 chars):
  - 5 chunks, all ~500 chars

Semantic (threshold=95):
  - 3 chunks: 800, 600, 400 chars
  - Better semantic coherence
```

## Best Practices

1. **Start with threshold=95** for most use cases
2. **Lower threshold (85-90)** for more precise retrieval
3. **Higher threshold (96-99)** for better context preservation
4. **Use buffer_size=1** for fine-grained control
5. **Monitor chunk statistics** to tune parameters

## Comparison with Other Strategies

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Semantic** | Natural boundaries, better context | Slower, variable sizes | Health reports, summaries |
| **Hierarchical** | Multi-level, consistent sizes | May split concepts | Large documents, structured data |
| **Fixed-size** | Fast, predictable | Arbitrary splits | Simple retrieval, speed critical |

## Advanced Features

### Compare Strategies

```python
from rag.semantic_chunking import SemanticChunker

chunker = SemanticChunker()

comparison = chunker.compare_with_fixed_size(
    text=user_context,
    user_id="user123",
    tenant_id="tenant1",
    fixed_chunk_size=512
)

print(f"Semantic: {comparison['semantic']['num_chunks']} chunks")
print(f"Fixed: {comparison['fixed_size']['num_chunks']} chunks")
print(f"Reduction: {comparison['comparison']['chunk_reduction']} chunks")
```

### Get Detailed Statistics

```python
stats = chunker.get_chunk_statistics(nodes)

print(f"Strategy: {stats['strategy']}")
print(f"Chunks: {stats['num_chunks']}")
print(f"Avg size: {stats['avg_size']:.0f} chars")
print(f"Range: {stats['min_size']}-{stats['max_size']}")
print(f"Buffer: {stats['buffer_size']}")
print(f"Threshold: {stats['threshold']}")
```

## References

- [LlamaIndex Semantic Splitter](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/)
- [Greg Kamradt's 5 Levels of Chunking](https://youtu.be/8OJC21T2SL4?t=1933)
- [Embedding-based Chunking Research](https://arxiv.org/abs/2101.00027)

## Example Output

```
INPUT TEXT:
"Glucose: 95 mg/dL (Normal: 70-100). HbA1c: 5.6% (Normal: <5.7%). 
User follows low-carb diet. Exercises 4x/week. Lost 5 pounds recently."

HIERARCHICAL CHUNKS (512 chars):
Chunk 1: "Glucose: 95 mg/dL (Normal: 70-100). HbA1c: 5.6% (Normal: <5.7%). User follows..."

SEMANTIC CHUNKS (threshold=95):
Chunk 1: "Glucose: 95 mg/dL (Normal: 70-100). HbA1c: 5.6% (Normal: <5.7%)."
Chunk 2: "User follows low-carb diet. Exercises 4x/week. Lost 5 pounds recently."

✅ Semantic maintains lab results together
✅ Semantic keeps lifestyle information together
```

## Troubleshooting

### Too Many Small Chunks
→ Increase `breakpoint_percentile_threshold` (try 98)

### Too Few Large Chunks
→ Decrease `breakpoint_percentile_threshold` (try 90)

### Chunks Still Split Awkwardly
→ Increase `buffer_size` to 2 or 3

### Slow Performance
→ Consider hierarchical for very large documents
→ Use async methods: `await pipeline.ingest_user_context_async()`

