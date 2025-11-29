# MongoDB Atlas Vector Search Setup Guide

This guide walks you through setting up MongoDB Atlas Vector Search for the RAG pipeline.

## Prerequisites

- MongoDB Atlas account (free tier works fine)
- MongoDB cluster (M10+ recommended for production, M0 free tier for testing)
- Atlas cluster version 6.0.11+ or 7.0.2+

## Step-by-Step Setup

### 1. Access Atlas Search

1. Log in to [MongoDB Atlas](https://cloud.mongodb.com)
2. Navigate to your cluster
3. Click on the **"Search"** tab (in the left sidebar or top menu)

### 2. Create Search Index

1. Click **"Create Search Index"**
2. Choose **"JSON Editor"** (not Visual Editor)
3. Select your database (default: `swastya`)
4. Select your collection (default: `user_context_vectors`)
   - If collection doesn't exist yet, create it first by inserting a sample document

### 3. Configure Index

Use the following JSON configuration:

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
    },
    {
      "type": "filter",
      "path": "metadata.timestamp"
    },
    {
      "type": "filter",
      "path": "metadata.chunk_type"
    },
    {
      "type": "filter",
      "path": "metadata.source"
    }
  ]
}
```

### 4. Name the Index

**IMPORTANT**: Use exactly this name:
```
vector_search_index
```

This name must match the configuration in `rag/config.py`.

### 5. Create Index

1. Click **"Create Search Index"**
2. Wait for the index to build (usually takes 1-2 minutes)
3. Status will change from "Building" to "Active"

## Configuration Details

### Vector Field
- **Path**: `embedding`
- **Type**: `vector`
- **Dimensions**: `1536` (for OpenAI text-embedding-3-small)
- **Similarity**: `cosine` (recommended for text embeddings)

### Filter Fields
These enable filtering by metadata:
- `metadata.user_id` - Filter by user
- `metadata.tenant_id` - Filter by tenant
- `metadata.timestamp` - Filter by time
- `metadata.chunk_type` - Filter by chunk type (parent/child)
- `metadata.source` - Filter by source

## Verify Setup

Run this Python script to verify your setup:

```python
from rag.vector_store import VectorStore

# Initialize vector store
vector_store = VectorStore()

# Check if index exists
index_exists = vector_store.check_vector_index_exists()

if index_exists:
    print("✅ Vector search index is configured correctly!")
else:
    print("❌ Vector search index not found. Please create it.")

# Get collection stats
stats = vector_store.get_collection_stats()
print(f"Collection stats: {stats}")
```

## Alternative: Atlas CLI Method

If you prefer using the Atlas CLI:

```bash
# Login to Atlas
atlas auth login

# Create vector search index
atlas clusters search indexes create \
  --clusterName YOUR_CLUSTER_NAME \
  --file vector_search_index.json
```

Where `vector_search_index.json` contains:

```json
{
  "name": "vector_search_index",
  "database": "swastya",
  "collectionName": "user_context_vectors",
  "type": "vectorSearch",
  "definition": {
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
}
```

## Troubleshooting

### "Index Building Failed"
- Check that your cluster tier supports Vector Search (M10+)
- Verify cluster version is 6.0.11+ or 7.0.2+
- Ensure collection has at least one document

### "Invalid numDimensions"
- Must match your embedding model's output
- text-embedding-3-small: 1536 dimensions
- text-embedding-3-large: 3072 dimensions
- text-embedding-ada-002: 1536 dimensions

### "Cannot find index"
- Verify index name is exactly `vector_search_index`
- Check that index status is "Active" (not "Building")
- Ensure you're looking at the correct database/collection

## Performance Tips

### Index Build Time
- Small collections (< 1000 docs): 1-2 minutes
- Medium collections (1000-100k docs): 5-15 minutes
- Large collections (> 100k docs): 15-60 minutes

### Query Performance
- First query may be slower (index warm-up)
- Subsequent queries are faster (cached)
- Filter fields improve query performance
- Consider using compound filters for better efficiency

## Cost Considerations

### Free Tier (M0)
- Supports Vector Search (limited)
- Good for development/testing
- Limited to 512 MB storage

### Paid Tiers (M10+)
- Full Vector Search support
- Better performance
- No storage limits
- Recommended for production

## Additional Resources

- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)
- [LlamaIndex MongoDB Integration](https://docs.llamaindex.ai/en/stable/examples/vector_stores/MongoDBAtlasVectorSearch/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

## Support

If you encounter issues:
1. Check MongoDB Atlas status page
2. Review Atlas logs in the UI
3. Verify all configuration values match
4. Test with a small dataset first

## Next Steps

After setting up the vector search index:
1. Run `python rag/example_usage.py` to test the pipeline
2. Integrate with your application using `get_user_context()`
3. Monitor performance and adjust chunk sizes if needed
4. Set up periodic updates for stale data

