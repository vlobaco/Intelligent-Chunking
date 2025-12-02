# System Architecture

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI-Enhanced RAG Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

                            INPUT DOCUMENTS
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      1. DOCUMENT CHUNKING                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  DocumentChunker                                          │  │
│  │  • Recursive character splitting                          │  │
│  │  • Semantic boundary detection                            │  │
│  │  • Section hierarchy extraction                           │  │
│  │  • Chunk type detection (content/code/table/list)         │  │
│  │  • Overlap management                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                        Basic Chunks with Metadata
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. AI CONTEXT GENERATION                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  AnthropicContextGenerator                                │  │
│  │  • Contextual summary generation                          │  │
│  │  • Hypothetical questions (HyDE)                          │  │
│  │  • Entity extraction                                      │  │
│  │  • Topic identification                                   │  │
│  │  • Async batch processing                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                        Enhanced Chunks with AI Context
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. EMBEDDING GENERATION                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  EmbeddingGenerator                                       │  │
│  │  • Hybrid text creation (context + content)               │  │
│  │  • Sentence-Transformers models                           │  │
│  │  • Batch embedding generation                             │  │
│  │  • Normalization                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                            Chunk Embeddings
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     4. VECTOR STORAGE                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  VectorStore                                              │  │
│  │  • In-memory vector index                                 │  │
│  │  • Metadata storage                                       │  │
│  │  • Chunk mapping (ID → index)                             │  │
│  │  • Persistence (save/load)                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                          Searchable Index
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      5. SEARCH & RETRIEVAL                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RAGSearchEngine                                          │  │
│  │  • Query embedding                                        │  │
│  │  • Cosine similarity search                               │  │
│  │  • Metadata filtering                                     │  │
│  │  • Result ranking                                         │  │
│  │  • Reference link generation                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                            Search Results
                                  │
                                  ▼
                      APPLICATION / USER INTERFACE
```

## Data Flow

### 1. Document Processing Flow

```
Raw Document
    │
    ├── Text Extraction
    │   └── Plain text
    │
    ├── Chunking Strategy
    │   ├── Recursive splitting by separators
    │   ├── Chunk size: ~800 chars
    │   ├── Overlap: ~150 chars
    │   └── Boundary detection
    │
    └── Metadata Generation
        ├── chunk_id (unique hash)
        ├── section_hierarchy (from headers)
        ├── chunk_type (content/code/table/list)
        ├── position (index, total)
        └── timestamps
```

### 2. AI Enhancement Flow

```
Basic Chunk
    │
    ├── Send to Claude API
    │   └── Structured prompt with chunk context
    │
    ├── AI Generates
    │   ├── Contextual Summary (2-3 sentences)
    │   ├── Hypothetical Questions (4-6 questions)
    │   ├── Entity List (5-10 entities)
    │   └── Topic List (3-5 topics)
    │
    └── Parse JSON Response
        └── Update chunk metadata
```

### 3. Embedding Creation Flow

```
Enhanced Chunk
    │
    ├── Build Hybrid Text
    │   ├── "Context: [summary]"
    │   ├── "Answers questions like: [questions]"
    │   ├── "Section: [hierarchy]"
    │   └── [original content]
    │
    ├── Generate Embedding
    │   └── Sentence-Transformer model
    │       ├── all-MiniLM-L6-v2 (384 dim)
    │       ├── all-mpnet-base-v2 (768 dim)
    │       └── Custom models
    │
    └── Normalize Vector
        └── Unit length for cosine similarity
```

### 4. Search Flow

```
User Query
    │
    ├── Query Embedding
    │   └── Same model as documents
    │
    ├── Apply Filters (optional)
    │   ├── chunk_type
    │   ├── topics
    │   ├── section_hierarchy
    │   ├── document_title
    │   └── Custom metadata
    │
    ├── Similarity Search
    │   ├── Compute cosine similarity
    │   ├── Filter by metadata
    │   └── Sort by score
    │
    ├── Top-K Selection
    │   └── Return best matches
    │
    └── Build Results
        ├── Original content
        ├── Contextual summary
        ├── Metadata
        ├── Similarity score
        └── Reference link
```

## Component Interaction

```
┌─────────────────┐
│  config.py      │  Configuration & Presets
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│ rag_chunker.py  │────▶│  ChunkMetadata       │
│                 │     │  EnhancedChunk       │
│ • DocumentChunker│     │  EnhancedRAGChunker  │
│ • Chunking logic │     └──────────────────────┘
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ ai_context_generator │  AI Enhancement
│                      │
│ • Anthropic API      │
│ • Prompt engineering │
│ • Async batching     │
└────────┬─────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────┐
│  vector_store.py    │────▶│  SearchResult    │
│                     │     └──────────────────┘
│ • EmbeddingGenerator│
│ • VectorStore       │
│ • RAGSearchEngine   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  pipeline_example   │  Complete Demo
│                     │
│ • End-to-end flow   │
│ • Usage examples    │
└─────────────────────┘
```

## Metadata Structure

```
EnhancedChunk
├── original_content: str
├── contextual_summary: str  ← AI Generated
├── hypothetical_questions: [str]  ← AI Generated (HyDE)
├── embedding_text: str  ← Hybrid for retrieval
└── metadata: ChunkMetadata
    ├── chunk_id: str  ← Unique identifier
    ├── source_document: str  ← File path
    ├── document_title: str  ← Human readable
    ├── chunk_index: int  ← Position
    ├── total_chunks: int  ← Total in doc
    ├── section_hierarchy: [str]  ← ["Ch1", "Sec1.2"]
    ├── page_number: int  ← Optional
    ├── start_char: int  ← Position in doc
    ├── end_char: int  ← Position in doc
    ├── chunk_type: str  ← content|code|table|list
    ├── entities: [str]  ← AI Generated
    ├── topics: [str]  ← AI Generated
    ├── tokens_estimate: int  ← For LLM context
    └── created_at: str  ← ISO timestamp
```

## Filter Types

```
Metadata Filters
├── Exact Match
│   └── {"chunk_type": "content"}
│
├── List Contains (OR)
│   └── {"topics": ["ML", "AI"]}
│       → Matches if chunk has ANY topic
│
├── Range Query
│   └── {"chunk_index": {"min": 0, "max": 10}}
│       → Matches if value in range
│
└── Combined (AND)
    └── {
          "chunk_type": "content",
          "topics": ["technical"],
          "section_hierarchy": ["Chapter 1"]
        }
        → Matches if ALL conditions true
```

## Extension Points

```
Custom Components
├── Document Parsers
│   ├── PDF → PyPDF2, pdfplumber
│   ├── DOCX → python-docx
│   ├── HTML → BeautifulSoup
│   └── Custom formats
│
├── Embedding Models
│   ├── Local → sentence-transformers
│   ├── OpenAI → text-embedding-ada-002
│   ├── Cohere → embed-english-v3.0
│   └── Custom models
│
├── Vector Stores
│   ├── Pinecone → Managed cloud
│   ├── Weaviate → Open source
│   ├── Qdrant → Fast, Rust-based
│   ├── Chroma → Lightweight
│   └── FAISS → Facebook's library
│
└── AI Backends
    ├── Anthropic Claude → Default
    ├── OpenAI GPT-4 → Alternative
    ├── Cohere → Alternative
    └── Local models → For privacy
```

## Performance Characteristics

```
Operation              | Time Complexity | Space Complexity
─────────────────────────────────────────────────────────
Chunking               | O(n)            | O(n)
AI Context (per chunk) | ~500ms          | O(1)
Embedding (batch)      | O(n)            | O(n × d)
Vector Search          | O(n)            | O(n × d)
Metadata Filter        | O(n)            | O(1)

where:
  n = number of chunks
  d = embedding dimension
```

## Integration Examples

### With LangChain

```python
from langchain.vectorstores import VectorStore
from langchain.embeddings import Embeddings

class CustomVectorStore(VectorStore):
    def __init__(self, search_engine):
        self.search_engine = search_engine
    
    def similarity_search(self, query, k=4):
        results = self.search_engine.search(query, top_k=k)
        return [r.original_content for r in results]
```

### With LlamaIndex

```python
from llama_index import VectorStoreIndex
from llama_index.vector_stores import SimpleVectorStore

# Convert enhanced chunks to LlamaIndex format
nodes = [chunk_to_node(chunk) for chunk in chunks]
index = VectorStoreIndex(nodes)
```

### With Haystack

```python
from haystack.document_stores import InMemoryDocumentStore

docs = [chunk_to_document(chunk) for chunk in chunks]
document_store = InMemoryDocumentStore()
document_store.write_documents(docs)
```
