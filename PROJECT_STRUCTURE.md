# Project Structure

## Directory Layout

```
rag-chunker/
├── src/
│   └── rag_chunker/           # Main package
│       ├── __init__.py        # Package initialization and exports
│       ├── config.py          # Configuration classes
│       ├── rag_chunker.py     # Core chunking logic
│       ├── ai_context_generator.py  # AI integration (Ollama & Anthropic)
│       └── vector_store.py    # Embedding and search functionality
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System architecture
│   ├── CONTEXT_ENHANCEMENT.md # Context-aware improvements
│   ├── FILES_OVERVIEW.txt     # File descriptions
│   ├── INSTALLATION.md        # Installation guide
│   ├── OLLAMA_MIGRATION.md    # Ollama integration guide
│   ├── PROJECT_SUMMARY.md     # Project overview
│   └── USAGE_GUIDE.md         # Usage examples
│
├── examples/                  # Example scripts
│   ├── pipeline_example.py    # Complete RAG pipeline demo
│   └── quickstart.py          # Interactive quickstart script
│
├── tests/                     # Test suite (placeholder)
│
├── .gitignore                 # Git ignore rules
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # MIT License
├── MANIFEST.in                # Package manifest
├── PROJECT_STRUCTURE.md       # This file
├── README.md                  # Main documentation
├── requirements.txt           # Python dependencies
└── setup.py                   # Package setup configuration
```

## Core Modules

### `src/rag_chunker/`

The main package containing all core functionality.

#### `__init__.py`
- Package initialization
- Exports all public APIs
- Version information

#### `config.py`
- `RAGConfig`: Main configuration class
- Domain-specific presets:
  - `TechnicalDocsConfig`
  - `LegalDocsConfig`
  - `ConversationalConfig`
  - `AcademicPapersConfig`

#### `rag_chunker.py`
- `DocumentChunker`: Basic text chunking
- `ChunkMetadata`: Metadata for each chunk
- `EnhancedChunk`: Chunk with AI-generated context
- `EnhancedRAGChunker`: Main orchestrator

#### `ai_context_generator.py`
- `OllamaContextGenerator`: Local Ollama integration
- `AnthropicContextGenerator`: Anthropic Claude integration
- `create_context_generator()`: Factory function
- `batch_generate_contexts()`: Async batch processing

#### `vector_store.py`
- `EmbeddingGenerator`: Generate embeddings
- `VectorStore`: In-memory vector storage
- `SearchResult`: Search result dataclass
- `RAGSearchEngine`: High-level search interface

## Documentation

### Main Docs
- **README.md**: Primary documentation, quick start, features
- **INSTALLATION.md**: Detailed installation guide
- **USAGE_GUIDE.md**: Real-world usage examples
- **ARCHITECTURE.md**: System design and architecture

### Technical Docs
- **CONTEXT_ENHANCEMENT.md**: Context-aware AI enhancement
- **OLLAMA_MIGRATION.md**: Ollama integration details
- **PROJECT_SUMMARY.md**: Executive summary
- **FILES_OVERVIEW.txt**: File-by-file descriptions

### Community
- **CONTRIBUTING.md**: How to contribute
- **LICENSE**: MIT License

## Examples

### `pipeline_example.py`
Complete RAG pipeline demonstration:
- Document processing
- AI enhancement
- Embedding generation
- Vector search
- Result display

### `quickstart.py`
Interactive quickstart:
- Dependency checking
- Ollama setup verification
- Demo execution
- Custom document processing

## Configuration

### Development
```bash
pip install -e ".[dev]"
```

### Production
```bash
pip install rag-chunker
```

### Custom Installation
```bash
# With Anthropic support
pip install rag-chunker[anthropic]

# With all dev tools
pip install -e ".[dev,docs]"
```

## Key Features by Module

### Chunking (`rag_chunker.py`)
- Recursive text splitting
- Semantic boundary detection
- Overlap management
- Section hierarchy extraction
- Chunk type detection

### AI Enhancement (`ai_context_generator.py`)
- Contextual summaries
- Hypothetical questions (HyDE)
- Entity extraction
- Topic identification
- Surrounding context awareness
- Async batch processing

### Vector Search (`vector_store.py`)
- Local embeddings (sentence-transformers)
- In-memory vector storage
- Metadata filtering
- Similarity search
- Reference link generation

## Import Patterns

### Basic Usage
```python
from rag_chunker import EnhancedRAGChunker, RAGSearchEngine

chunker = EnhancedRAGChunker()
search_engine = RAGSearchEngine()
```

### Advanced Usage
```python
from rag_chunker import (
    EnhancedRAGChunker,
    RAGSearchEngine,
    RAGConfig,
    create_context_generator,
    OllamaContextGenerator,
)
```

### Configuration
```python
from rag_chunker import (
    RAGConfig,
    TechnicalDocsConfig,
    LegalDocsConfig,
)
```

## Data Flow

```
1. Input Document
   ↓
2. DocumentChunker (rag_chunker.py)
   - Text splitting
   - Metadata extraction
   ↓
3. AIContextGenerator (ai_context_generator.py)
   - Contextual summary
   - Hypothetical questions
   - Entity extraction
   ↓
4. EmbeddingGenerator (vector_store.py)
   - Generate embeddings
   ↓
5. VectorStore (vector_store.py)
   - Store embeddings
   - Enable search
   ↓
6. RAGSearchEngine (vector_store.py)
   - Query processing
   - Result ranking
   - Reference generation
```

## Testing

### Run Tests
```bash
pytest tests/
```

### Test Coverage
```bash
pytest --cov=src/rag_chunker tests/
```

## Building Distribution

### Source Distribution
```bash
python setup.py sdist
```

### Wheel
```bash
python setup.py bdist_wheel
```

### Both
```bash
python setup.py sdist bdist_wheel
```

## Publishing (Maintainers Only)

### Test PyPI
```bash
twine upload --repository testpypi dist/*
```

### Production PyPI
```bash
twine upload dist/*
```

## Environment Variables

Supported environment variables:
- `OLLAMA_HOST`: Ollama server URL
- `OLLAMA_MODEL`: Default Ollama model
- `ANTHROPIC_API_KEY`: Anthropic API key
- `AI_PROVIDER`: "ollama" or "anthropic"

## Development Workflow

1. **Setup**: `pip install -e ".[dev]"`
2. **Code**: Make changes in `src/rag_chunker/`
3. **Format**: `black src/`
4. **Lint**: `flake8 src/`
5. **Type Check**: `mypy src/`
6. **Test**: `pytest tests/`
7. **Document**: Update relevant docs
8. **Commit**: Follow conventional commits
9. **PR**: Submit pull request

## Support Files

- `.gitignore`: Excludes build artifacts, caches, etc.
- `MANIFEST.in`: Specifies files to include in distribution
- `setup.py`: Package metadata and dependencies
- `requirements.txt`: Runtime dependencies

## Next Steps

1. Read [README.md](README.md) for overview
2. Follow [INSTALLATION.md](docs/INSTALLATION.md) for setup
3. Check [USAGE_GUIDE.md](docs/USAGE_GUIDE.md) for examples
4. Review [ARCHITECTURE.md](docs/ARCHITECTURE.md) for design
5. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
