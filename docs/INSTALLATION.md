# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Basic Installation

### From PyPI (when published)

```bash
pip install rag-chunker
```

### From Source

```bash
git clone https://github.com/yourusername/rag-chunker.git
cd rag-chunker
pip install -e .
```

## Ollama Setup (Recommended)

RAG Chunker uses Ollama as the default AI provider for free, local context generation.

### Install Ollama

1. Visit [https://ollama.ai](https://ollama.ai)
2. Download and install for your platform
3. Start the Ollama server:

```bash
ollama serve
```

### Pull a Model

In a new terminal:

```bash
# Recommended: Good balance of speed and quality
ollama pull llama3.2

# Alternatives
ollama pull mistral      # Fast and efficient
ollama pull phi3         # Compact model
ollama pull gemma:7b     # Google's model
ollama pull qwen2.5      # Excellent for technical content
```

### Verify Installation

```python
from rag_chunker import EnhancedRAGChunker

chunker = EnhancedRAGChunker(
    use_ai_context=True,
    ai_provider="ollama"
)

# If successful, you'll see:
# "Connected to Ollama at http://localhost:11434"
# "Available models: llama3.2, ..."
```

## Alternative: Anthropic Claude

If you prefer using Anthropic's Claude API:

### Install Anthropic Package

```bash
pip install anthropic
```

### Set API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or in Python:

```python
from rag_chunker import EnhancedRAGChunker

chunker = EnhancedRAGChunker(
    use_ai_context=True,
    ai_provider="anthropic",
    api_key="your-api-key-here"
)
```

## Optional Dependencies

### Development Tools

```bash
pip install -e ".[dev]"
```

Includes:
- pytest (testing)
- black (code formatting)
- mypy (type checking)
- flake8 (linting)

### Documentation

```bash
pip install -e ".[docs]"
```

Includes Sphinx and theme for building documentation.

## Configuration

### Environment Variables

Create a `.env` file in your project:

```bash
# Ollama (default)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Or Anthropic
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key-here
```

### Custom Configuration

```python
from rag_chunker import RAGConfig

# Modify defaults
RAGConfig.OLLAMA_HOST = "http://your-server:11434"
RAGConfig.OLLAMA_MODEL = "mistral"
RAGConfig.CHUNK_SIZE = 1000
RAGConfig.CONTEXT_WINDOW = 2
```

## Troubleshooting

### Ollama Connection Error

```
Warning: Could not connect to Ollama at http://localhost:11434
```

**Solutions:**
1. Ensure Ollama is running: `ollama serve`
2. Check the port: `lsof -i :11434`
3. Set custom host: `export OLLAMA_HOST="http://your-host:11434"`

### Model Not Found

```
Warning: Model 'llama3.2' not found
```

**Solution:**
```bash
ollama pull llama3.2
```

### Import Error

```
ModuleNotFoundError: No module named 'rag_chunker'
```

**Solutions:**
1. Install the package: `pip install -e .`
2. Check Python path: Make sure you're in the right environment
3. Reinstall: `pip uninstall rag-chunker && pip install -e .`

### sentence-transformers Error

```
sentence-transformers not installed
```

**Solution:**
```bash
pip install sentence-transformers
```

## Verification

Run the quickstart to verify everything works:

```bash
python examples/quickstart.py
```

Or test programmatically:

```python
from rag_chunker import EnhancedRAGChunker, RAGSearchEngine

# Test chunking
chunker = EnhancedRAGChunker(use_ai_context=False)
chunks = chunker.process_document(
    text="Test document",
    source_document="test.txt",
    document_title="Test"
)
print(f"✓ Created {len(chunks)} chunks")

# Test search
search_engine = RAGSearchEngine()
search_engine.index_chunks(chunks)
results = search_engine.search("test", top_k=1)
print(f"✓ Search returned {len(results)} results")
```

## Next Steps

- Read the [Usage Guide](USAGE_GUIDE.md)
- Explore [Examples](../examples/)
- Review [Architecture](ARCHITECTURE.md)
- Check [Configuration Options](../src/rag_chunker/config.py)
