"""
Configuration file for RAG Chunking System
Customize these settings for your use case
"""

from typing import Dict, Any, List


class RAGConfig:
    """Central configuration for the RAG chunking system"""
    
    # ==================== CHUNKING SETTINGS ====================
    
    # Target chunk size in characters
    CHUNK_SIZE: int = 800
    
    # Overlap between consecutive chunks (characters)
    CHUNK_OVERLAP: int = 150
    
    # Separators for recursive splitting (in order of preference)
    CHUNK_SEPARATORS: List[str] = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentences
        " ",     # Words
        ""       # Characters (last resort)
    ]
    
    # ==================== AI CONTEXT GENERATION ====================

    # Enable AI-powered context generation
    USE_AI_CONTEXT: bool = True

    # AI Provider: "ollama" or "anthropic"
    AI_PROVIDER: str = "ollama"

    # Ollama settings
    OLLAMA_HOST: str = "http://rachel:11434"
    OLLAMA_MODEL: str = "gpt-oss:20b"

    # Anthropic settings (used when AI_PROVIDER = "anthropic")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Maximum tokens for AI context generation
    AI_MAX_TOKENS: int = 800

    # Temperature for AI generation (0.0 = deterministic, 1.0 = creative)
    AI_TEMPERATURE: float = 0.3

    # Number of hypothetical questions to generate per chunk
    NUM_HYPOTHETICAL_QUESTIONS: int = 5

    # Maximum concurrent AI API calls for batch processing
    MAX_CONCURRENT_AI_CALLS: int = 5

    # Context window for AI generation (number of surrounding chunks to include)
    # 0 = no surrounding context, 1 = one chunk before/after, 2 = two chunks before/after, etc.
    CONTEXT_WINDOW: int = 2

    # ==================== EMBEDDING SETTINGS ====================

    # Use Ollama for embeddings (default: True)
    USE_OLLAMA_EMBEDDINGS: bool = True

    # Embedding model to use
    # Ollama models (if USE_OLLAMA_EMBEDDINGS = True):
    #   - "nomic-embed-text" (recommended, 768 dims)
    #   - "mxbai-embed-large" (large model, 1024 dims)
    #   - "all-minilm" (fast, 384 dims)
    # Sentence-transformers (if USE_OLLAMA_EMBEDDINGS = False):
    #   - "all-MiniLM-L6-v2" (fast, good quality, 384 dims)
    #   - "all-mpnet-base-v2" (better quality, 768 dims)
    #   - "multi-qa-mpnet-base-dot-v1" (optimized for Q&A, 768 dims)
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Embedding dimension (auto-detected for most models)
    EMBEDDING_DIM: int = 768  # nomic-embed-text default
    
    # ==================== SEARCH SETTINGS ====================
    
    # Default number of results to return
    DEFAULT_TOP_K: int = 5
    
    # Minimum similarity score threshold (0.0 to 1.0)
    MIN_SIMILARITY_SCORE: float = 0.0
    
    # Default base URL for reference links
    DEFAULT_BASE_URL: str = "https://docs.example.com/"
    
    # ==================== METADATA SETTINGS ====================
    
    # Extract entities using NER (requires spaCy)
    EXTRACT_ENTITIES: bool = False
    
    # Detect chunk types automatically
    DETECT_CHUNK_TYPES: bool = True
    
    # Extract section hierarchy from headers
    EXTRACT_SECTION_HIERARCHY: bool = True
    
    # Track page numbers (for PDF processing)
    TRACK_PAGE_NUMBERS: bool = False
    
    # ==================== PERFORMANCE SETTINGS ====================
    
    # Batch size for embedding generation
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Cache embeddings to disk
    CACHE_EMBEDDINGS: bool = True
    
    # Cache directory
    CACHE_DIR: str = "./cache"
    
    # ==================== OUTPUT SETTINGS ====================
    
    # Output directory for processed chunks
    OUTPUT_DIR: str = "./output"
    
    # Save chunks as JSON
    SAVE_CHUNKS_JSON: bool = True
    
    # Save search index
    SAVE_SEARCH_INDEX: bool = True
    
    # Pretty print JSON output
    JSON_INDENT: int = 2
    
    # ==================== FILTERING PRESETS ====================
    
    # Common filter combinations
    FILTER_PRESETS: Dict[str, Dict[str, Any]] = {
        "content_only": {
            "chunk_type": "content"
        },
        "code_only": {
            "chunk_type": "code"
        },
        "recent_chunks": {
            "chunk_index": {"max": 10}
        },
        "technical_content": {
            "chunk_type": "content",
            "topics": ["technical", "implementation", "API"]
        }
    }
    
    # ==================== LOGGING SETTINGS ====================
    
    # Enable verbose logging
    VERBOSE: bool = True
    
    # Log level: "DEBUG", "INFO", "WARNING", "ERROR"
    LOG_LEVEL: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export config as dictionary"""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }


# ==================== DOMAIN-SPECIFIC PRESETS ====================

class TechnicalDocsConfig(RAGConfig):
    """Optimized for technical documentation"""
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    NUM_HYPOTHETICAL_QUESTIONS = 6
    DETECT_CHUNK_TYPES = True
    EXTRACT_SECTION_HIERARCHY = True


class LegalDocsConfig(RAGConfig):
    """Optimized for legal documents"""
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
    AI_TEMPERATURE = 0.1  # More deterministic
    TRACK_PAGE_NUMBERS = True


class ConversationalConfig(RAGConfig):
    """Optimized for conversational/chat data"""
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    NUM_HYPOTHETICAL_QUESTIONS = 3
    

class AcademicPapersConfig(RAGConfig):
    """Optimized for academic papers"""
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 300
    TRACK_PAGE_NUMBERS = True
    EXTRACT_ENTITIES = True


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    import json
    
    # Example 1: Use default config
    config = RAGConfig()
    print("Default Config:")
    print(json.dumps(RAGConfig.to_dict(), indent=2))
    
    # Example 2: Use domain-specific preset
    print("\nTechnical Docs Config:")
    tech_config = TechnicalDocsConfig()
    print(f"Chunk Size: {tech_config.CHUNK_SIZE}")
    print(f"Overlap: {tech_config.CHUNK_OVERLAP}")
    
    # Example 3: Custom configuration
    custom_config = RAGConfig.from_dict({
        "CHUNK_SIZE": 500,
        "USE_AI_CONTEXT": True,
        "EMBEDDING_MODEL": "all-mpnet-base-v2",
        "DEFAULT_TOP_K": 10
    })
    print(f"\nCustom chunk size: {custom_config.CHUNK_SIZE}")
    print(f"Custom embedding model: {custom_config.EMBEDDING_MODEL}")
