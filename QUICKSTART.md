# Quick Start Guide

Get up and running with RAG Chunker in 5 minutes.

## 1. Installation

### Install the Package

```bash
# From the project directory
pip install -e .
```

### Verify Installation

```bash
python verify_install.py
```

You should see:
```
✓ All tests passed!
Your installation is working correctly.
```

## 2. Set Up Ollama (Recommended)

Ollama provides free, local AI context generation.

### Install Ollama

Download from [https://ollama.ai](https://ollama.ai)

### Start Ollama

```bash
# In one terminal
ollama serve
```

### Pull a Model

```bash
# In another terminal
ollama pull llama3.2
```

**Alternative models:**
- `ollama pull mistral` - Fast and efficient
- `ollama pull phi3` - Compact model
- `ollama pull gemma:7b` - Google's model

## 3. Basic Usage

### Simple Example

```python
from rag_chunker import EnhancedRAGChunker, RAGSearchEngine

# Create chunker with Ollama (default)
chunker = EnhancedRAGChunker(
    chunk_size=800,
    use_ai_context=True,
    ai_provider="ollama"
)

# Process a document
text = """Your document text here..."""

chunks = chunker.process_document(
    text=text,
    source_document="document.md",
    document_title="My Document"
)

print(f"Created {len(chunks)} chunks")

# Build search index
search_engine = RAGSearchEngine()
search_engine.index_chunks(chunks)

# Search
results = search_engine.search("your query", top_k=5)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.original_content[:200]}...")
    print()
```

## 4. Run Examples

### Interactive Quickstart

```bash
python examples/quickstart.py
```

This will:
- Check dependencies
- Verify Ollama setup
- Let you process documents
- Provide interactive search

### Complete Pipeline Demo

```bash
python examples/pipeline_example.py
```

This demonstrates:
- Full RAG workflow
- AI enhancement
- Vector search
- Metadata filtering

## 5. Configuration

### Environment Variables

```bash
# Ollama settings (default)
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"

# Or use Anthropic Claude
export AI_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your-api-key"
```

### Code Configuration

```python
from rag_chunker import RAGConfig

# Customize settings
RAGConfig.CHUNK_SIZE = 1000
RAGConfig.CHUNK_OVERLAP = 200
RAGConfig.CONTEXT_WINDOW = 2  # More surrounding context
RAGConfig.OLLAMA_MODEL = "mistral"
```

## 6. Common Use Cases

### Process Multiple Documents

```python
from pathlib import Path
from rag_chunker import EnhancedRAGChunker

chunker = EnhancedRAGChunker(use_ai_context=True)

all_chunks = []
for doc_path in Path("documents").glob("*.md"):
    with open(doc_path) as f:
        text = f.read()

    chunks = chunker.process_document(
        text=text,
        source_document=str(doc_path),
        document_title=doc_path.stem
    )
    all_chunks.extend(chunks)

print(f"Processed {len(all_chunks)} total chunks")
```

### Search with Filters

```python
# Search only in specific sections
results = search_engine.search(
    query="machine learning",
    filters={
        "chunk_type": "content",
        "topics": ["AI", "ML"]
    }
)
```

### Save and Load

```python
# Save chunks
chunker.save_chunks(chunks, "output/chunks.json")

# Save search index
search_engine.save("output/index.pkl")

# Load later
from rag_chunker import VectorStore
search_engine.vector_store = VectorStore.load("output/index.pkl")
```

## 7. Troubleshooting

### Module Not Found Error

```
ModuleNotFoundError: No module named 'rag_chunker'
```

**Solution:**
```bash
pip install -e .
```

### Ollama Connection Error

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

### Import Errors After Installation

If you still get import errors after `pip install -e .`, make sure you're in the right Python environment:

```bash
# Check Python location
which python

# If using virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Then reinstall
pip install -e .
```

## 8. Next Steps

Now that you're set up:

1. **Read the docs**: Check out [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
2. **Explore examples**: Look at the example scripts in `examples/`
3. **Customize config**: Review [src/rag_chunker/config.py](src/rag_chunker/config.py)
4. **Build your app**: Integrate with your RAG application

## Need Help?

- **Documentation**: See [README.md](README.md) and files in `docs/`
- **Examples**: Check `examples/` directory
- **Issues**: Open an issue on GitHub
- **Verify**: Run `python verify_install.py`

Happy chunking! 🚀
