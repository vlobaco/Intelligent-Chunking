# Ollama Embeddings Guide

The RAG Chunker now supports **Ollama for both AI context generation AND embeddings**, keeping everything local and free.

## Quick Start

### 1. Install Ollama Models

```bash
# For AI context generation (already installed)
ollama pull llama3.2

# For embeddings (recommended)
ollama pull nomic-embed-text
```

### 2. Use in Code

```python
from rag_chunker import EnhancedRAGChunker, RAGSearchEngine

# AI context generation with Ollama
chunker = EnhancedRAGChunker(
    use_ai_context=True,
    ai_provider="ollama",
    model="llama3.2"
)

# Embeddings with Ollama (default)
search_engine = RAGSearchEngine(
    embedding_model="nomic-embed-text",
    use_ollama=True  # This is the default
)

# Process and index
chunks = chunker.process_document(text, "doc.md", "Title")
search_engine.index_chunks(chunks)
```

## Available Ollama Embedding Models

### nomic-embed-text (Recommended)
```bash
ollama pull nomic-embed-text
```
- **Dimensions**: 768
- **Performance**: Excellent quality
- **Speed**: Fast
- **Use case**: General purpose, best all-around choice

### mxbai-embed-large
```bash
ollama pull mxbai-embed-large
```
- **Dimensions**: 1024
- **Performance**: High quality
- **Speed**: Slower
- **Use case**: When quality is more important than speed

### all-minilm
```bash
ollama pull all-minilm
```
- **Dimensions**: 384
- **Performance**: Good quality
- **Speed**: Very fast
- **Use case**: When speed is critical

## Configuration

### Method 1: Direct Parameters

```python
from rag_chunker import RAGSearchEngine

search_engine = RAGSearchEngine(
    embedding_model="nomic-embed-text",
    use_ollama=True,
    ollama_host="http://localhost:11434"
)
```

### Method 2: Config Class

```python
from rag_chunker import RAGConfig

# Set defaults
RAGConfig.USE_OLLAMA_EMBEDDINGS = True
RAGConfig.EMBEDDING_MODEL = "nomic-embed-text"
RAGConfig.OLLAMA_HOST = "http://localhost:11434"

# Then use normally
search_engine = RAGSearchEngine()
```

### Method 3: Environment Variables

```bash
export OLLAMA_HOST="http://localhost:11434"
export EMBEDDING_MODEL="nomic-embed-text"
```

## Using Sentence-Transformers Instead

If you prefer sentence-transformers (requires local model download):

```python
search_engine = RAGSearchEngine(
    embedding_model="all-MiniLM-L6-v2",
    use_ollama=False  # Use sentence-transformers
)
```

## Complete Example

```python
from rag_chunker import EnhancedRAGChunker, RAGSearchEngine

# Step 1: Chunk with AI context (Ollama)
chunker = EnhancedRAGChunker(
    chunk_size=800,
    use_ai_context=True,
    ai_provider="ollama",
    model="llama3.2"
)

chunks = chunker.process_document(
    text=document_text,
    source_document="guide.md",
    document_title="User Guide"
)

print(f"Created {len(chunks)} chunks with AI context")

# Step 2: Generate embeddings and index (Ollama)
search_engine = RAGSearchEngine(
    embedding_model="nomic-embed-text",
    use_ollama=True
)

search_engine.index_chunks(chunks)
print("Indexed all chunks")

# Step 3: Search
results = search_engine.search("How do I get started?", top_k=5)

for result in results:
    print(f"\nScore: {result.score:.3f}")
    print(f"Summary: {result.contextual_summary}")
    print(f"Content: {result.original_content[:150]}...")
```

## Benefits of Ollama Embeddings

✅ **Free** - No API costs
✅ **Private** - All processing local
✅ **Fast** - No network latency
✅ **Offline** - Works without internet
✅ **Consistent** - Same model everywhere

## Performance Comparison

| Model | Dims | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **nomic-embed-text** | 768 | Fast | Excellent | **Recommended** |
| mxbai-embed-large | 1024 | Medium | Excellent | High quality needed |
| all-minilm | 384 | Very Fast | Good | Speed critical |
| all-MiniLM-L6-v2* | 384 | Fast | Good | Offline alternative |
| all-mpnet-base-v2* | 768 | Medium | Excellent | High quality offline |

*Sentence-transformers models (use_ollama=False)

## Troubleshooting

### Model Not Found

```
⚠️ Model 'nomic-embed-text' not found in Ollama
```

**Solution:**
```bash
ollama pull nomic-embed-text
```

### Ollama Not Running

```
⚠️ Ollama not available: Connection refused
```

**Solution:**
```bash
ollama serve
```

### Slow Embedding Generation

If embeddings are slow, try:
1. Use a faster model: `all-minilm`
2. Reduce chunk count
3. Use sentence-transformers instead (pre-loaded model)

### Check Available Models

```bash
ollama list
```

## Advanced: Custom Ollama Host

For remote Ollama server:

```python
search_engine = RAGSearchEngine(
    embedding_model="nomic-embed-text",
    use_ollama=True,
    ollama_host="http://your-server:11434"
)
```

## Migration from Sentence-Transformers

Already using sentence-transformers? Easy to switch:

```python
# Before
search_engine = RAGSearchEngine(
    embedding_model="all-MiniLM-L6-v2",
    use_local_embeddings=True
)

# After (Ollama)
search_engine = RAGSearchEngine(
    embedding_model="nomic-embed-text",
    use_ollama=True
)
```

Note: Existing embeddings are not compatible. You'll need to re-index.

## Best Practices

1. **Pull models first**: `ollama pull nomic-embed-text`
2. **Keep Ollama running**: `ollama serve`
3. **Use nomic-embed-text**: Best balance of speed/quality
4. **Monitor dimensions**: nomic-embed-text = 768 dims
5. **Batch carefully**: Ollama processes one at a time

## Summary

Ollama embeddings provide a **completely local, free RAG pipeline**:

- **AI Context**: Ollama (llama3.2, mistral, etc.)
- **Embeddings**: Ollama (nomic-embed-text)
- **Vector Store**: In-memory (or Pinecone/Weaviate)

Everything runs on your machine with no API costs or privacy concerns!
