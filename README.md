# AI-Enhanced RAG Chunking System

A sophisticated document chunking and indexing system for Retrieval-Augmented Generation (RAG) that uses AI to generate contextual metadata, hypothetical questions (HyDE), and rich search metadata for each chunk.

## Features

✨ **AI-Enhanced Context Generation**
- Automatic contextual summaries for each chunk
- Hypothetical question generation (HyDE approach)
- Entity and topic extraction
- Smart metadata enrichment

🔍 **Advanced Search Capabilities**
- Hybrid approach: context + original content embedding
- Metadata-based filtering (document, section, type, topics)
- Reference link generation for citations
- Support for multiple embedding models

📊 **Rich Metadata**
- Section hierarchy tracking
- Chunk type detection (content, code, table, list)
- Token estimates
- Source attribution
- Temporal metadata

🚀 **Flexible Architecture**
- Pluggable AI backends (Anthropic Claude by default)
- Multiple vector store support
- Async batch processing
- Easy integration with existing pipelines

## Installation

```bash
# Clone or download the project
cd rag-chunking-system

# Install dependencies
pip install -r requirements.txt

# Set up Ollama (default AI provider)
# 1. Install Ollama from https://ollama.ai
# 2. Start Ollama server
ollama serve

# 3. Pull a model (in a new terminal)
ollama pull llama3.2  # or mistral, phi3, etc.

# Optional: Configure Ollama host and model
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"

# Alternative: Use Anthropic Claude (optional)
# Uncomment anthropic in requirements.txt and install
# pip install anthropic>=0.40.0
# export ANTHROPIC_API_KEY="your-api-key-here"
```

## Quick Start

### Basic Usage

```python
from rag_chunker import EnhancedRAGChunker

# Initialize chunker with Ollama (default)
chunker = EnhancedRAGChunker(
    chunk_size=800,
    chunk_overlap=150,
    use_ai_context=True,
    ai_provider="ollama",  # default
    host="http://localhost:11434",  # optional, this is the default
    model="llama3.2"  # optional, default model
)

# Or use Anthropic Claude
# chunker = EnhancedRAGChunker(
#     chunk_size=800,
#     chunk_overlap=150,
#     use_ai_context=True,
#     ai_provider="anthropic",
#     api_key="your-api-key",  # or set ANTHROPIC_API_KEY env var
#     model="claude-sonnet-4-20250514"
# )

# Process a document
chunks = chunker.process_document(
    text=your_document_text,
    source_document="path/to/document.pdf",
    document_title="Document Title"
)

# Each chunk now has:
# - original_content: The raw text
# - contextual_summary: AI-generated context
# - hypothetical_questions: Questions this chunk answers
# - metadata: Rich metadata including entities, topics, section hierarchy
# - embedding_text: Hybrid text optimized for retrieval
```

### Building a Search Index

```python
from vector_store import RAGSearchEngine

# Create search engine
search_engine = RAGSearchEngine(
    embedding_model="all-MiniLM-L6-v2",
    use_local_embeddings=True
)

# Index your chunks
search_engine.index_chunks(chunks)

# Search with optional filters
results = search_engine.search(
    query="How does RAG work?",
    top_k=5,
    filters={"chunk_type": "content", "topics": ["RAG"]}
)

# Use results
for result in results:
    print(f"Score: {result.score}")
    print(f"Reference: {result.reference_link}")
    print(f"Summary: {result.contextual_summary}")
    print(f"Content: {result.original_content}")
```

### Complete Pipeline Example

```python
# See pipeline_example.py for a full demonstration
python pipeline_example.py
```

## Architecture

### Core Components

1. **DocumentChunker** (`rag_chunker.py`)
   - Splits documents using recursive character splitting
   - Maintains semantic coherence with overlap
   - Detects section hierarchy and chunk types
   - Generates unique chunk IDs

2. **AIContextGenerator** (`ai_context_generator.py`)
   - Calls Anthropic Claude API for context generation
   - Generates contextual summaries
   - Creates hypothetical questions (HyDE)
   - Extracts entities and topics
   - Async batch processing support

3. **VectorStore** (`vector_store.py`)
   - Embedding generation with sentence-transformers
   - In-memory vector storage (easily adaptable to Pinecone, Weaviate, etc.)
   - Metadata filtering
   - Similarity search

4. **RAGSearchEngine** (`vector_store.py`)
   - High-level search interface
   - Combines embedding and filtering
   - Returns structured results with references

### Data Models

#### EnhancedChunk
```python
{
    "original_content": str,           # Raw chunk text
    "contextual_summary": str,         # AI-generated summary
    "hypothetical_questions": list,    # Questions chunk answers
    "embedding_text": str,             # Hybrid text for embedding
    "metadata": {
        "chunk_id": str,               # Unique identifier
        "source_document": str,        # Source file path
        "document_title": str,         # Human-readable title
        "chunk_index": int,            # Position in document
        "total_chunks": int,           # Total chunks in document
        "section_hierarchy": list,     # ["Chapter 1", "Section 1.2"]
        "page_number": int,            # Optional page number
        "chunk_type": str,             # content|code|table|list
        "entities": list,              # Extracted entities
        "topics": list,                # Main topics
        "tokens_estimate": int,        # Approximate token count
        "created_at": str              # ISO timestamp
    }
}
```

## Usage Examples

### Example 1: Processing Multiple Documents

```python
from pathlib import Path
from rag_chunker import EnhancedRAGChunker

chunker = EnhancedRAGChunker(chunk_size=1000, use_ai_context=True)

all_chunks = []
for doc_path in Path("documents").glob("*.md"):
    with open(doc_path, 'r') as f:
        text = f.read()
    
    chunks = chunker.process_document(
        text=text,
        source_document=str(doc_path),
        document_title=doc_path.stem
    )
    all_chunks.extend(chunks)

print(f"Processed {len(all_chunks)} total chunks")
```

### Example 2: Filtered Search

```python
# Search only in specific sections
results = search_engine.search(
    query="machine learning algorithms",
    filters={
        "section_hierarchy": ["Chapter 3"],  # Only Chapter 3
        "chunk_type": "content"              # No code blocks
    }
)

# Search by topic
results = search_engine.search(
    query="neural networks",
    filters={
        "topics": ["deep learning", "AI"]  # Match any topic
    }
)

# Search in chunk range
results = search_engine.search(
    query="introduction",
    filters={
        "chunk_index": {"min": 0, "max": 5}  # First 5 chunks
    }
)
```

### Example 3: Reference Generation

```python
# Get reference links for citations
for chunk in chunks:
    # Generate reference link
    ref_link = chunk.get_reference_link(
        base_url="https://docs.mycompany.com/"
    )
    
    print(f"Chunk {chunk.metadata.chunk_index}:")
    print(f"  Section: {' > '.join(chunk.metadata.section_hierarchy)}")
    print(f"  Reference: {ref_link}")
    print(f"  Topics: {', '.join(chunk.metadata.topics)}")
```

### Example 4: Async Batch Processing

```python
import asyncio
from ai_context_generator import batch_generate_contexts

# Prepare chunk data
chunks_data = [
    {"content": chunk_text, "metadata": chunk_metadata}
    for chunk_text, chunk_metadata in your_chunks
]

# Generate contexts in parallel (max 5 concurrent)
contexts = asyncio.run(
    batch_generate_contexts(
        chunks_data,
        api_key="your-api-key",
        max_concurrent=5
    )
)
```

### Example 5: Save and Load

```python
from rag_chunker import EnhancedRAGChunker

chunker = EnhancedRAGChunker()

# Save chunks to JSON
chunker.save_chunks(chunks, "output/chunks.json")

# Load chunks later
loaded_chunks = chunker.load_chunks("output/chunks.json")

# Save search index
search_engine.save("output/search_index.pkl")

# Load search index
from vector_store import VectorStore
search_engine.vector_store = VectorStore.load("output/search_index.pkl")
```

## Configuration Options

### Chunking Parameters

```python
EnhancedRAGChunker(
    chunk_size=800,           # Target chunk size in characters
    chunk_overlap=150,        # Overlap between chunks
    use_ai_context=True       # Enable AI context generation
)
```

### Embedding Models

The system supports various embedding models:

**Local (Sentence Transformers):**
- `all-MiniLM-L6-v2` (default) - Fast, good quality
- `all-mpnet-base-v2` - Higher quality, slower
- `multi-qa-mpnet-base-dot-v1` - Optimized for Q&A

**API-based (optional):**
- OpenAI: `text-embedding-ada-002`
- Cohere: `embed-english-v3.0`

### AI Context Generation

The system supports two AI providers:

#### Ollama (Default - Free & Local)

```python
from rag_chunker import EnhancedRAGChunker

chunker = EnhancedRAGChunker(
    use_ai_context=True,
    ai_provider="ollama",
    host="http://localhost:11434",
    model="llama3.2"  # or mistral, phi3, gemma, etc.
)
```

**Recommended Ollama Models:**
- `llama3.2` - Good balance of speed and quality
- `mistral` - Fast and efficient
- `phi3` - Compact, good for resource-constrained environments
- `gemma:7b` - Google's Gemma model

**Setup:**
```bash
# Install Ollama from https://ollama.ai
ollama serve

# Pull a model
ollama pull llama3.2
```

#### Anthropic Claude (API-based)

```python
chunker = EnhancedRAGChunker(
    use_ai_context=True,
    ai_provider="anthropic",
    api_key="your-api-key",
    model="claude-sonnet-4-20250514"
)
```

**Note:** Requires API key and has usage costs.

## Metadata Filtering

The system supports rich metadata filtering:

```python
filters = {
    # Exact match
    "chunk_type": "content",
    "document_title": "User Guide",
    
    # List contains (OR logic)
    "topics": ["ML", "AI"],  # Match if chunk has ANY of these topics
    
    # Range queries
    "chunk_index": {"min": 0, "max": 10},
    "tokens_estimate": {"min": 100, "max": 500},
    
    # Combining filters (AND logic)
    "chunk_type": "content",
    "topics": ["technical"],
    "section_hierarchy": ["Chapter 1"]
}

results = search_engine.search(query, filters=filters)
```

## Integration with Production Vector Stores

The system is designed to easily integrate with production vector databases:

### Pinecone

```python
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-key", environment="your-env")
index = pinecone.Index("rag-index")

# Upsert chunks
for chunk in chunks:
    index.upsert([(
        chunk.metadata.chunk_id,
        embedder.embed_single(chunk.embedding_text),
        chunk.metadata.to_dict()
    )])
```

### Weaviate

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Create schema with metadata
schema = {
    "class": "Document",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "chunk_id", "dataType": ["string"]},
        {"name": "topics", "dataType": ["string[]"]},
        # ... other metadata fields
    ]
}

# Add chunks
for chunk in chunks:
    client.data_object.create(
        data_object={
            "content": chunk.embedding_text,
            **chunk.metadata.to_dict()
        },
        class_name="Document",
        vector=embedder.embed_single(chunk.embedding_text)
    )
```

## Performance Considerations

- **Chunking**: ~1000 chunks/second (without AI)
- **AI Context Generation**: ~2-5 chunks/second (with Anthropic API)
- **Embedding**: ~100-500 chunks/second (local sentence-transformers)
- **Search**: <50ms for 10k chunks (in-memory)

### Optimization Tips

1. **Batch Processing**: Use async batch processing for AI context generation
2. **Caching**: Cache embeddings and AI-generated context
3. **Parallel Processing**: Process multiple documents in parallel
4. **Chunk Size**: Larger chunks = fewer API calls but less precision
5. **Vector Store**: Use production vector DBs (Pinecone, Weaviate) for large scale

## Troubleshooting

### No API Key Warning
```
Warning: No API key provided. Context generation will use mock data.
```
**Solution**: Set `ANTHROPIC_API_KEY` environment variable or pass to constructor.

### sentence-transformers Not Installed
```
sentence-transformers not installed
```
**Solution**: `pip install sentence-transformers`

### Mock Embeddings Warning
```
Warning: Using mock embeddings
```
**Solution**: Install sentence-transformers for real semantic embeddings.

## Contributing

Contributions welcome! Key areas:

- Additional document parsers (PDF, DOCX, etc.)
- More vector store integrations
- Alternative embedding models
- Improved chunk boundary detection
- Evaluation metrics and benchmarks

## License

MIT License - feel free to use in commercial projects.

## Citation

If you use this system in research, please cite:

```bibtex
@software{ai_enhanced_rag_chunker,
  title={AI-Enhanced RAG Chunking System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rag-chunking-system}
}
```

## Further Reading

- [RAG Best Practices](https://www.anthropic.com/index/contextual-retrieval)
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [Sentence Transformers](https://www.sbert.net/)
- [Vector Database Comparison](https://www.timescale.com/blog/vector-database-basics/)
