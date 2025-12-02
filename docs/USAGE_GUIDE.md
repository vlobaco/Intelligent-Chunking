# Practical Usage Guide

This guide shows real-world examples of using the AI-Enhanced RAG Chunking System.

## Use Case 1: Building a Technical Documentation Search

```python
from rag_chunker import EnhancedRAGChunker
from vector_store import RAGSearchEngine
from config import TechnicalDocsConfig
from pathlib import Path

# Setup
config = TechnicalDocsConfig()
chunker = EnhancedRAGChunker(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    use_ai_context=True
)

# Process all markdown files in docs directory
all_chunks = []
for doc_file in Path("docs").rglob("*.md"):
    with open(doc_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = chunker.process_document(
        text=text,
        source_document=str(doc_file),
        document_title=doc_file.stem.replace('_', ' ').title()
    )
    all_chunks.extend(chunks)

# Build search index
search_engine = RAGSearchEngine()
search_engine.index_chunks(all_chunks)

# Save for later use
search_engine.save("docs_index.pkl")
```

## Use Case 2: Multi-Step Research Pipeline

This demonstrates how to use the system for complex research tasks that require multiple searches and synthesis.

```python
from vector_store import RAGSearchEngine

class ResearchAssistant:
    def __init__(self, index_path: str):
        self.search_engine = RAGSearchEngine()
        self.search_engine.load(index_path)
    
    def research_topic(self, main_query: str, depth: int = 2):
        """
        Perform multi-step research on a topic
        """
        results = {
            "main_query": main_query,
            "findings": [],
            "related_topics": set(),
            "sources": set()
        }
        
        # Step 1: Initial search
        initial_results = self.search_engine.search(main_query, top_k=5)
        
        for result in initial_results:
            results["findings"].append({
                "content": result.original_content,
                "summary": result.contextual_summary,
                "reference": result.reference_link,
                "score": result.score
            })
            
            # Extract related topics
            results["related_topics"].update(result.metadata.get("topics", []))
            results["sources"].add(result.metadata["document_title"])
        
        # Step 2: Explore related topics
        if depth > 1:
            for topic in list(results["related_topics"])[:3]:
                related_results = self.search_engine.search(
                    topic, 
                    top_k=2
                )
                for result in related_results:
                    results["findings"].append({
                        "content": result.original_content,
                        "summary": result.contextual_summary,
                        "reference": result.reference_link,
                        "related_to": topic
                    })
        
        return results

# Use it
assistant = ResearchAssistant("docs_index.pkl")
research = assistant.research_topic("machine learning deployment")

print(f"Found {len(research['findings'])} relevant passages")
print(f"From {len(research['sources'])} sources")
print(f"Related topics: {', '.join(research['related_topics'])}")
```

## Use Case 3: Custom Metadata Filtering for Compliance

```python
from vector_store import RAGSearchEngine
from datetime import datetime, timedelta

# Assume chunks have been enhanced with compliance metadata
search_engine = RAGSearchEngine()
search_engine.load("compliance_docs.pkl")

def search_current_policies(query: str):
    """
    Search only in current (non-deprecated) policy documents
    """
    # Calculate cutoff date (policies older than 1 year)
    cutoff_date = datetime.now() - timedelta(days=365)
    
    results = search_engine.search(
        query=query,
        top_k=5,
        filters={
            "chunk_type": "content",
            "topics": ["policy", "compliance"],
            # Custom metadata added during chunking
            "is_deprecated": False,
            "last_updated": {"min": cutoff_date.isoformat()}
        }
    )
    
    return results

def get_policy_lineage(policy_name: str):
    """
    Find all versions of a policy
    """
    results = search_engine.search(
        query=policy_name,
        top_k=20,
        filters={
            "document_title": policy_name
        }
    )
    
    # Sort by date
    sorted_results = sorted(
        results,
        key=lambda r: r.metadata.get('created_at', ''),
        reverse=True
    )
    
    return sorted_results

# Use it
current_policies = search_current_policies("data retention requirements")
all_versions = get_policy_lineage("Data Privacy Policy")
```

## Use Case 4: Hybrid Search (Vector + Keyword)

```python
from vector_store import RAGSearchEngine
import re

def hybrid_search(search_engine, query: str, top_k: int = 5):
    """
    Combine semantic search with keyword matching
    """
    # Semantic search
    semantic_results = search_engine.search(query, top_k=top_k * 2)
    
    # Keyword extraction (simple version)
    keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    if not keywords:
        keywords = [w for w in query.split() if len(w) > 3]
    
    # Boost results that contain exact keywords
    def hybrid_score(result):
        semantic_score = result.score
        
        content_lower = result.original_content.lower()
        keyword_matches = sum(
            1 for kw in keywords 
            if kw.lower() in content_lower
        )
        keyword_score = keyword_matches / max(len(keywords), 1)
        
        # Weighted combination: 70% semantic, 30% keyword
        return semantic_score * 0.7 + keyword_score * 0.3
    
    # Rerank and return top-k
    reranked = sorted(semantic_results, key=hybrid_score, reverse=True)
    return reranked[:top_k]

# Use it
search_engine = RAGSearchEngine()
search_engine.load("index.pkl")

results = hybrid_search(
    search_engine,
    "What is the GDPR compliance requirement for user data?",
    top_k=5
)
```

## Use Case 5: Batch Processing Pipeline

```python
import asyncio
from pathlib import Path
from rag_chunker import EnhancedRAGChunker
from ai_context_generator import batch_generate_contexts

async def process_document_batch(file_paths: list, api_key: str):
    """
    Process multiple documents in parallel with batched AI calls
    """
    chunker = EnhancedRAGChunker(
        chunk_size=800,
        use_ai_context=False  # We'll add context in batch
    )
    
    # Step 1: Chunk all documents
    all_basic_chunks = []
    for filepath in file_paths:
        with open(filepath, 'r') as f:
            text = f.read()
        
        chunks = chunker.chunker.chunk_text(
            text, 
            str(filepath),
            Path(filepath).stem
        )
        all_basic_chunks.extend(chunks)
    
    print(f"Created {len(all_basic_chunks)} basic chunks")
    
    # Step 2: Generate AI context in batch
    print("Generating AI context...")
    contexts = await batch_generate_contexts(
        all_basic_chunks,
        api_key=api_key,
        max_concurrent=10  # Process 10 at a time
    )
    
    # Step 3: Combine chunks with context
    from rag_chunker import EnhancedChunk
    enhanced_chunks = []
    for chunk_data, context in zip(all_basic_chunks, contexts):
        enhanced_chunk = EnhancedChunk(
            original_content=chunk_data["content"],
            contextual_summary=context["contextual_summary"],
            hypothetical_questions=context["hypothetical_questions"],
            metadata=chunk_data["metadata"]
        )
        enhanced_chunks.append(enhanced_chunk)
    
    print(f"Enhanced {len(enhanced_chunks)} chunks with AI context")
    return enhanced_chunks

# Use it
file_list = list(Path("documents").glob("*.md"))
chunks = asyncio.run(process_document_batch(
    file_list,
    api_key="your-anthropic-api-key"
))
```

## Use Case 6: Building a Citation Generator

```python
from vector_store import RAGSearchEngine

class CitationGenerator:
    def __init__(self, index_path: str):
        self.search_engine = RAGSearchEngine()
        self.search_engine.load(index_path)
    
    def generate_answer_with_citations(self, question: str):
        """
        Generate an answer with properly formatted citations
        """
        results = self.search_engine.search(question, top_k=5)
        
        # Build answer (simplified - use actual LLM here)
        answer_parts = []
        citations = []
        
        for i, result in enumerate(results, 1):
            # Extract key information
            content = result.original_content[:200]
            doc_title = result.metadata['document_title']
            section = ' > '.join(result.metadata['section_hierarchy'])
            
            # Add to answer
            answer_parts.append(
                f"According to {doc_title} ({section}): "
                f"{content}... [{i}]"
            )
            
            # Build citation
            citations.append(
                f"[{i}] {doc_title}. {section}. "
                f"Retrieved from: {result.reference_link}"
            )
        
        # Combine
        full_answer = '\n\n'.join(answer_parts)
        full_citations = '\n'.join(citations)
        
        return {
            "answer": full_answer,
            "citations": full_citations,
            "num_sources": len(results)
        }

# Use it
generator = CitationGenerator("academic_index.pkl")
result = generator.generate_answer_with_citations(
    "What are the main causes of climate change?"
)

print(result["answer"])
print("\n--- References ---")
print(result["citations"])
```

## Use Case 7: Domain-Specific Search Enhancement

```python
from config import RAGConfig

class MedicalRAGConfig(RAGConfig):
    """Custom configuration for medical documents"""
    CHUNK_SIZE = 600  # Smaller for precise citations
    CHUNK_OVERLAP = 100
    AI_TEMPERATURE = 0.1  # Very deterministic
    TRACK_PAGE_NUMBERS = True
    
    # Custom filter for medical content
    FILTER_PRESETS = {
        "clinical_trials": {
            "topics": ["clinical trial", "study", "research"]
        },
        "treatment_guidelines": {
            "topics": ["treatment", "guideline", "protocol"]
        },
        "peer_reviewed": {
            "source_type": "peer_reviewed_journal"
        }
    }

# Use custom config
from rag_chunker import EnhancedRAGChunker
from vector_store import RAGSearchEngine

config = MedicalRAGConfig()
chunker = EnhancedRAGChunker(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)

# Search with domain filters
search_engine = RAGSearchEngine()
# ... load medical literature index ...

results = search_engine.search(
    "treatment options for type 2 diabetes",
    filters=config.FILTER_PRESETS["treatment_guidelines"]
)
```

## Performance Tips

### 1. Caching for Development

```python
import pickle
from pathlib import Path

def get_or_create_chunks(doc_path: str, cache_dir: str = "./cache"):
    """Cache chunks to avoid reprocessing during development"""
    cache_path = Path(cache_dir) / f"{Path(doc_path).stem}_chunks.pkl"
    
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Create chunks
    chunker = EnhancedRAGChunker()
    with open(doc_path, 'r') as f:
        text = f.read()
    
    chunks = chunker.process_document(text, doc_path, Path(doc_path).stem)
    
    # Cache
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    return chunks
```

### 2. Progress Tracking for Large Batches

```python
from tqdm import tqdm

def process_large_corpus(file_paths: list):
    """Process with progress bar"""
    chunker = EnhancedRAGChunker()
    all_chunks = []
    
    for filepath in tqdm(file_paths, desc="Processing documents"):
        with open(filepath, 'r') as f:
            text = f.read()
        
        chunks = chunker.process_document(text, str(filepath))
        all_chunks.extend(chunks)
    
    return all_chunks
```

### 3. Incremental Index Updates

```python
def update_index_incrementally(search_engine, new_chunks):
    """Add new chunks to existing index"""
    # Index new chunks
    search_engine.index_chunks(new_chunks)
    
    # Save updated index
    search_engine.save("index.pkl")
    
    print(f"Added {len(new_chunks)} chunks to index")
    print(f"Total chunks: {search_engine.get_stats()['total_chunks']}")
```
