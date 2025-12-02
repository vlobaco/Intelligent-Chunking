# RAG Chunker Examples

This directory contains example scripts demonstrating how to use the RAG Chunker library.

## Examples

### 1. quickstart.py - Interactive Quickstart

An interactive script that guides you through:
- Dependency checking
- Ollama setup verification
- Running the demo
- Processing your own documents

**Usage:**
```bash
python examples/quickstart.py
```

**What it does:**
1. Checks for required packages
2. Verifies Ollama connection
3. Optionally runs the pipeline demo
4. Lets you process custom documents
5. Provides interactive search

**When to use:**
- First time setup
- Verifying installation
- Quick document processing
- Learning the basics

### 2. pipeline_example.py - Complete RAG Pipeline

A comprehensive demonstration of the full RAG workflow:
- Document chunking with AI enhancement
- Embedding generation
- Vector search index building
- Semantic search with metadata filtering

**Usage:**
```bash
python examples/pipeline_example.py
```

**What it demonstrates:**
- AI-enhanced chunking with Ollama
- Context window usage
- Vector search setup
- Metadata filtering
- Reference link generation
- Saving/loading chunks and indices

**When to use:**
- Understanding the complete workflow
- Reference implementation
- Production pipeline template
- Testing with sample documents

## Environment Configuration

Both examples respect environment variables:

```bash
# Ollama configuration
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"

# Or use Anthropic
export AI_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your-key"
```

## Running with Different AI Providers

### Using Ollama (Default)

```bash
# Make sure Ollama is running
ollama serve

# Pull a model
ollama pull llama3.2

# Run examples
python examples/pipeline_example.py
```

### Using Anthropic Claude

```bash
# Set API key
export AI_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your-key"

# Run examples
python examples/pipeline_example.py
```

## Customizing Examples

### Modify Chunk Size

```python
chunker = EnhancedRAGChunker(
    chunk_size=1000,      # Larger chunks
    chunk_overlap=200,     # More overlap
    use_ai_context=True
)
```

### Change Context Window

```python
chunks = chunker.process_document(
    text=document_text,
    source_document="doc.md",
    document_title="Document",
    context_window=2  # More surrounding context
)
```

### Use Different Embedding Model

```python
search_engine = RAGSearchEngine(
    embedding_model="all-mpnet-base-v2",  # Higher quality
    use_local_embeddings=True
)
```

### Apply Metadata Filters

```python
results = search_engine.search(
    query="your query",
    top_k=5,
    filters={
        "chunk_type": "content",
        "topics": ["technical", "implementation"]
    }
)
```

## Sample Output

### quickstart.py

```
╔════════════════════════════════════════════════════════════════╗
║        AI-Enhanced RAG Chunking System - Quick Start           ║
╚════════════════════════════════════════════════════════════════╝

📦 Checking dependencies...
  ✓ requests
  ✓ aiohttp
  ✓ sentence-transformers
  ✓ numpy

✓ All dependencies installed!

🤖 Ollama Setup (Default AI Provider)
✓ Ollama is running at http://localhost:11434
✓ Available models: llama3.2, mistral
```

### pipeline_example.py

```
============================================================
Processing: Comprehensive Guide to RAG
============================================================

Step 1: Initializing chunker with OLLAMA provider...
Connected to Ollama at http://localhost:11434
Available models: llama3.2, mistral

Step 2: Chunking document (target size: 800 chars)...
✓ Created 15 chunks

Step 3: Sample chunk analysis:
--- Chunk 1 of 15 ---
ID: a3f2c8d91e4b
Type: content
Section: Introduction
Contextual Summary: This introductory section of the RAG guide...
```

## Creating Your Own Examples

### Basic Template

```python
from rag_chunker import EnhancedRAGChunker, RAGSearchEngine

# 1. Initialize chunker
chunker = EnhancedRAGChunker(
    chunk_size=800,
    use_ai_context=True,
    ai_provider="ollama"
)

# 2. Process document
chunks = chunker.process_document(
    text=your_text,
    source_document="path/to/doc.md",
    document_title="Document Title"
)

# 3. Build search index
search_engine = RAGSearchEngine()
search_engine.index_chunks(chunks)

# 4. Search
results = search_engine.search("query", top_k=5)

# 5. Use results
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.original_content[:200]}...")
```

### Advanced Template

```python
from rag_chunker import (
    EnhancedRAGChunker,
    RAGSearchEngine,
    TechnicalDocsConfig
)

# Use domain-specific configuration
config = TechnicalDocsConfig()

# Initialize with custom settings
chunker = EnhancedRAGChunker(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    use_ai_context=True,
    ai_provider="ollama",
    host="http://localhost:11434",
    model="llama3.2"
)

# Process with context
chunks = chunker.process_document(
    text=document,
    source_document="technical_doc.md",
    document_title="Technical Documentation",
    context_window=2  # More context for technical docs
)

# Save for later
chunker.save_chunks(chunks, "output/chunks.json")

# Build and save index
search_engine = RAGSearchEngine(
    embedding_model="all-mpnet-base-v2"
)
search_engine.index_chunks(chunks)
search_engine.save("output/index.pkl")

# Advanced search with filters
results = search_engine.search(
    query="API authentication",
    top_k=10,
    filters={
        "chunk_type": "content",
        "section_hierarchy": ["API Reference"],
        "topics": ["security", "authentication"]
    }
)
```

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'rag_chunker'`:

```bash
# Install from project root
cd ..
pip install -e .
```

### Ollama Not Running

```
Warning: Could not connect to Ollama
```

**Solution:**
```bash
ollama serve
```

### Model Not Available

```
Warning: Model 'llama3.2' not found
```

**Solution:**
```bash
ollama pull llama3.2
```

## Next Steps

1. Run `quickstart.py` to verify installation
2. Run `pipeline_example.py` to see full workflow
3. Modify examples for your use case
4. Read the [Usage Guide](../docs/USAGE_GUIDE.md) for more examples
5. Check [Architecture](../docs/ARCHITECTURE.md) for design details

## Additional Resources

- **Documentation**: `../docs/`
- **Source Code**: `../src/rag_chunker/`
- **Configuration**: `../src/rag_chunker/config.py`
- **Tests**: `../tests/` (when available)

## Contributing Examples

Have a useful example? Please contribute!

1. Create a new `.py` file
2. Add documentation at the top
3. Include usage instructions
4. Update this README
5. Submit a PR

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
