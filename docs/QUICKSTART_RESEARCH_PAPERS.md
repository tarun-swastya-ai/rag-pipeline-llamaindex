# Quick Start: Research Paper Ingestion

## 1-Minute Setup

### Step 1: Install Dependencies

Already done if you installed the RAG pipeline dependencies.

### Step 2: Ingest Research Papers

```python
from rag.document_loader import DocumentLoader

# Create loader
doc_loader = DocumentLoader()

# Ingest all papers (adjust path as needed)
result = doc_loader.ingest_directory(
    directory_path="Research Paper",
    document_type="research_paper",
    tenant_id="swastya",
    recursive=True
)

print(f"✓ Ingested {result['documents_processed']} documents")
print(f"✓ Created {result['nodes_processed']} searchable chunks")
```

### Step 3: Query Papers

```python
from rag.retriever import RAGRetriever

retriever = RAGRetriever()

# Query research papers
result = retriever.retrieve(
    query="What are the health benefits of a Mediterranean diet?",
    user_id="research_paper:research_paper",
    tenant_id="swastya",
    top_k=5
)

print(result['context'])
```

## Complete Example

```python
from rag.document_loader import DocumentLoader
from rag.retriever import RAGRetriever
from rag.semantic_chunking import SemanticChunker

# ============================================
# ONE-TIME: Ingest Research Papers
# ============================================

# Use semantic chunking for better context
chunker = SemanticChunker(breakpoint_percentile_threshold=92)
doc_loader = DocumentLoader(chunker=chunker)

# Ingest by category for better organization
categories = [
    ("Research Paper/General Diet", "general_diet"),
    ("Research Paper/Diet and Health Risks", "diet_health_risks"),
    ("Research Paper/Biological Age", "biological_age")
]

for path, doc_type in categories:
    result = doc_loader.ingest_directory(
        directory_path=path,
        document_type=doc_type,
        tenant_id="swastya",
        recursive=True
    )
    print(f"✓ {doc_type}: {result['nodes_processed']} chunks")

# ============================================
# ONGOING: Query Papers
# ============================================

retriever = RAGRetriever()

# Query specific category
diet_result = retriever.retrieve(
    query="Best foods for diabetes management?",
    user_id="research_paper:diet_health_risks",
    tenant_id="swastya",
    top_k=5
)

# Query biological age research
age_result = retriever.retrieve(
    query="How to calculate biological age?",
    user_id="research_paper:biological_age",
    tenant_id="swastya",
    top_k=5
)

# ============================================
# ADVANCED: Combine with User Data
# ============================================

# Get user's health context
user_result = retriever.retrieve(
    query="User's glucose levels",
    user_id="user123",
    tenant_id="swastya",
    top_k=3
)

# Get relevant research
research_result = retriever.retrieve(
    query="Diet recommendations for glucose control",
    user_id="research_paper:general_diet",
    tenant_id="swastya",
    top_k=3
)

# Combine for AI agent
context_for_ai = f"""
PATIENT DATA:
{user_result['context']}

RELEVANT RESEARCH:
{research_result['context']}

Please provide personalized recommendations based on the patient's
data and the latest research.
"""
# Pass to your LLM/AI agent
```

## Folder Structure

Your Research Paper folder should look like:

```
Research Paper/
├── General Diet/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
├── Diet and Health Risks/
│   ├── study1.pdf
│   ├── study2.pdf
│   └── ...
└── Biological Age/
    ├── research1.pdf
    ├── research2.pdf
    └── ...
```

## Supported File Types

✅ PDF (`.pdf`)
✅ Text (`.txt`, `.md`)
✅ Word (`.docx`)

## User ID Format

Research papers use this format:

```
research_paper:<category>
```

Examples:

- `research_paper:general_diet`
- `research_paper:diet_health_risks`
- `research_paper:biological_age`

## Async Version (Faster)

```python
import asyncio
from rag.document_loader import DocumentLoader

async def ingest():
    doc_loader = DocumentLoader()

    result = await doc_loader.ingest_directory_async(
        directory_path="Research Paper",
        document_type="research_paper",
        tenant_id="swastya",
        recursive=True
    )

    return result

# Run
result = asyncio.run(ingest())
```

## Run Example Script

```bash
# Adjust path in the script first
python rag/example_research_paper_ingestion.py
```

## Next Steps

1. ✅ Ingest your research papers (one-time)
2. ✅ Query papers using retriever
3. ✅ Combine with user health data
4. ✅ Pass to AI agent for recommendations

See full documentation:

- [RESEARCH_PAPER_INGESTION.md](RESEARCH_PAPER_INGESTION.md) - Complete guide
- [SEMANTIC_CHUNKING.md](SEMANTIC_CHUNKING.md) - Chunking strategies
- [README.md](../README.md) - Full RAG pipeline docs

## Troubleshooting

**Path not found?**

```python
from pathlib import Path
print(Path("Research Paper").absolute())
```

**No results?**

- Check ingestion was successful
- Verify user_id format: `research_paper:<category>`
- Ensure MongoDB vector index is created

**Need help?**
See [RESEARCH_PAPER_INGESTION.md](RESEARCH_PAPER_INGESTION.md) for detailed troubleshooting.
