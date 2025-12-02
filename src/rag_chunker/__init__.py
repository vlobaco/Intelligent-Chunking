"""
RAG Chunker - AI-Enhanced Document Chunking for Retrieval-Augmented Generation

A sophisticated document chunking and indexing system that uses AI to generate
contextual metadata, hypothetical questions (HyDE), and rich search metadata for each chunk.
"""

__version__ = "1.0.0"

from .rag_chunker import (
    EnhancedRAGChunker,
    DocumentChunker,
    EnhancedChunk,
    ChunkMetadata
)

from .ai_context_generator import (
    OllamaContextGenerator,
    AnthropicContextGenerator,
    create_context_generator
)

from .vector_store import (
    RAGSearchEngine,
    VectorStore,
    EmbeddingGenerator,
    SearchResult
)

from .config import (
    RAGConfig,
    TechnicalDocsConfig,
    LegalDocsConfig,
    ConversationalConfig,
    AcademicPapersConfig
)

__all__ = [
    # Main classes
    "EnhancedRAGChunker",
    "DocumentChunker",
    "EnhancedChunk",
    "ChunkMetadata",

    # AI context generators
    "OllamaContextGenerator",
    "AnthropicContextGenerator",
    "create_context_generator",

    # Vector store and search
    "RAGSearchEngine",
    "VectorStore",
    "EmbeddingGenerator",
    "SearchResult",

    # Configuration
    "RAGConfig",
    "TechnicalDocsConfig",
    "LegalDocsConfig",
    "ConversationalConfig",
    "AcademicPapersConfig",
]
