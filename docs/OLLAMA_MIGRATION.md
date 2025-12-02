# Ollama Migration Summary

This document summarizes the changes made to enable Ollama as the default AI provider for the RAG Chunking System.

## Overview

The system now supports **Ollama** as the default (and recommended) AI provider for context generation, with Anthropic Claude as an optional alternative. Ollama provides free, local AI processing without API costs.

## Changes Made

### 1. Configuration ([config.py](config.py))

Added new configuration options:

```python
# AI Provider: "ollama" or "anthropic"
AI_PROVIDER: str = "ollama"

# Ollama settings
OLLAMA_HOST: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.2"

# Anthropic settings (used when AI_PROVIDER = "anthropic")
ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
```

### 2. AI Context Generator ([ai_context_generator.py](ai_context_generator.py))

- **Added `OllamaContextGenerator` class**: Integrates with Ollama API for local AI processing
- **Added factory function `create_context_generator()`**: Dynamically creates the appropriate generator based on provider
- **Auto-detection**: Checks if Ollama is running and which models are available
- **Async support**: Includes async methods for batch processing with Ollama

### 3. RAG Chunker ([rag_chunker.py](rag_chunker.py))

Updated `EnhancedRAGChunker` to support provider selection:

```python
chunker = EnhancedRAGChunker(
    chunk_size=800,
    chunk_overlap=150,
    use_ai_context=True,
    ai_provider="ollama",  # or "anthropic"
    host="http://localhost:11434",  # Ollama host
    model="llama3.2"  # Ollama model
)
```

### 4. Dependencies ([requirements.txt](requirements.txt))

- **Added**: `requests>=2.31.0` and `aiohttp>=3.9.0` for Ollama API communication
- **Made optional**: `anthropic>=0.40.0` (commented out, only needed when using Anthropic)

### 5. Documentation ([README.md](README.md))

- Updated installation instructions with Ollama setup
- Added comprehensive AI provider configuration section
- Included recommended Ollama models and setup commands
- Updated all code examples to show Ollama as default

### 6. Example Files

**[pipeline_example.py](pipeline_example.py)**:
- Updated to use Ollama by default
- Added environment variable support for easy provider switching
- Maintained backward compatibility with Anthropic

**[quickstart.py](quickstart.py)**:
- Replaced `setup_api_key()` with `setup_ollama()`
- Auto-detects Ollama installation and available models
- Updated dependency checks to include requests/aiohttp instead of anthropic

## Usage

### Quick Start with Ollama

1. **Install Ollama**:
   ```bash
   # Download from https://ollama.ai
   # Or use package manager
   ```

2. **Start Ollama server**:
   ```bash
   ollama serve
   ```

3. **Pull a model** (in a new terminal):
   ```bash
   ollama pull llama3.2  # or mistral, phi3, gemma, etc.
   ```

4. **Use in your code**:
   ```python
   from rag_chunker import EnhancedRAGChunker

   chunker = EnhancedRAGChunker(
       use_ai_context=True,
       ai_provider="ollama",  # This is now the default!
       model="llama3.2"
   )

   chunks = chunker.process_document(text, source, title)
   ```

### Using Anthropic Claude (Optional)

If you prefer Anthropic:

```python
chunker = EnhancedRAGChunker(
    use_ai_context=True,
    ai_provider="anthropic",
    api_key="your-api-key",
    model="claude-sonnet-4-20250514"
)
```

### Environment Variables

Configure via environment variables:

```bash
# For Ollama (default)
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"

# Or for Anthropic
export AI_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your-key"
```

## Recommended Ollama Models

- **`llama3.2`** - Good balance of speed and quality (recommended)
- **`mistral`** - Fast and efficient
- **`phi3`** - Compact, good for resource-constrained environments
- **`gemma:7b`** - Google's Gemma model
- **`qwen2.5`** - Excellent for technical content

## Benefits of Ollama

1. **Free**: No API costs, run unlimited generations
2. **Private**: All processing happens locally
3. **Fast**: No network latency (when running locally)
4. **Flexible**: Choose from many open-source models
5. **Offline**: Works without internet connection

## Backward Compatibility

All existing code using Anthropic will continue to work. The system gracefully handles:

- Missing Ollama installation (falls back to mock data with warnings)
- Both sync and async processing
- Environment variable configuration
- Direct parameter passing

## Testing

Test your Ollama setup:

```bash
# Run the example pipeline
python pipeline_example.py

# Or use the quickstart script
python quickstart.py
```

## Troubleshooting

### Ollama not connecting

```
Warning: Could not connect to Ollama at http://localhost:11434
```

**Solution**: Start Ollama with `ollama serve`

### Model not found

```
Warning: Model 'llama3.2' not found
```

**Solution**: Pull the model with `ollama pull llama3.2`

### Want to use custom Ollama host

```python
# For remote Ollama server
chunker = EnhancedRAGChunker(
    ai_provider="ollama",
    host="http://your-server:11434",
    model="llama3.2"
)
```

## Migration Path

If you're upgrading from the Anthropic-only version:

1. Install Ollama and pull a model
2. Update your imports (no changes needed, just optional)
3. Change `ai_provider` parameter from default to explicit if desired
4. Test with your existing documents

That's it! The system will automatically use Ollama.
