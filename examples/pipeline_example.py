"""
Complete RAG Pipeline Example
Demonstrates the full workflow: chunking -> AI enhancement -> embedding -> search
"""

import json
import os
from pathlib import Path

from rag_chunker import EnhancedRAGChunker, RAGSearchEngine


def process_document_with_ai(
    text: str,
    source_document: str,
    document_title: str,
    ai_provider: str = "ollama",
    ollama_host: str = "http://rachel:11434",
    ollama_model: str = "gpt-oss:20b",
    anthropic_api_key: str = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150
):
    """
    Process a document through the complete pipeline with AI enhancement

    Args:
        text: Document text
        source_document: Path/identifier for the document
        document_title: Human-readable title
        ai_provider: "ollama" (default) or "anthropic"
        ollama_host: Ollama server URL (default: http://rachel:11434)
        ollama_model: Ollama model to use (default: gpt-oss:20b)
        anthropic_api_key: Anthropic API key (only needed if ai_provider="anthropic")
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of enhanced chunks
    """
    print(f"\n{'='*60}")
    print(f"Processing: {document_title}")
    print(f"{'='*60}\n")

    # Step 1: Initialize chunker with AI context generation
    print(f"Step 1: Initializing chunker with {ai_provider.upper()} provider...")

    if ai_provider == "ollama":
        chunker = EnhancedRAGChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_ai_context=True,
            ai_provider="ollama",
            host=ollama_host,
            model=ollama_model
        )
    else:  # anthropic
        chunker = EnhancedRAGChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_ai_context=True,
            ai_provider="anthropic",
            api_key=anthropic_api_key
        )
    
    # Step 2: Chunk the document
    print(f"Step 2: Chunking document (target size: {chunk_size} chars)...")
    chunks = chunker.process_document(text, source_document, document_title)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Display sample results
    print("\nStep 3: Sample chunk analysis:")
    if chunks:
        sample = chunks[0]
        print(f"\n--- Chunk 1 of {len(chunks)} ---")
        print(f"ID: {sample.metadata.chunk_id}")
        print(f"Type: {sample.metadata.chunk_type}")
        print(f"Section: {' > '.join(sample.metadata.section_hierarchy)}")
        print(f"Tokens (est): {sample.metadata.tokens_estimate}")
        print(f"\nOriginal Content ({len(sample.original_content)} chars):")
        print(sample.original_content[:200] + "..." if len(sample.original_content) > 200 else sample.original_content)
        print(f"\nContextual Summary:")
        print(sample.contextual_summary)
        print(f"\nHypothetical Questions:")
        for q in sample.hypothetical_questions[:3]:
            print(f"  • {q}")
        print(f"\nEntities: {', '.join(sample.metadata.entities[:5])}")
        print(f"Topics: {', '.join(sample.metadata.topics)}")
        print(f"\nReference Link: {sample.get_reference_link('https://docs.company.com/')}")
    
    return chunks


def build_search_index(chunks, model_name="nomic-embed-text"):
    """
    Build a searchable index from enhanced chunks
    
    Args:
        chunks: List of EnhancedChunk objects
        model_name: Embedding model to use
        
    Returns:
        RAGSearchEngine instance
    """
    print(f"\n{'='*60}")
    print("Building Search Index")
    print(f"{'='*60}\n")
    
    print(f"Initializing search engine with model: {model_name}")
    search_engine = RAGSearchEngine(
        embedding_model=model_name,
    )
    
    print(f"Indexing {len(chunks)} chunks...")
    search_engine.index_chunks(chunks)
    
    stats = search_engine.get_stats()
    print("\n✓ Indexing complete!")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Topics: {', '.join(stats['topics'][:5])}")
    print(f"  Chunk types: {stats['chunk_types']}")
    
    return search_engine


def demonstrate_search(search_engine, queries):
    """
    Demonstrate search with various queries and filters
    
    Args:
        search_engine: RAGSearchEngine instance
        queries: List of (query, filters) tuples
    """
    print(f"\n{'='*60}")
    print("Search Demonstrations")
    print(f"{'='*60}\n")
    
    for i, (query, filters) in enumerate(queries, 1):
        print(f"\nQuery {i}: \"{query}\"")
        if filters:
            print(f"Filters: {filters}")
        
        results = search_engine.search(query, top_k=3, filters=filters)
        
        if results:
            for j, result in enumerate(results, 1):
                print(f"\n  Result {j} (score: {result.score:.3f})")
                print(f"  Section: {' > '.join(result.metadata.get('section_hierarchy', ['N/A']))}")
                print(f"  Type: {result.metadata.get('chunk_type', 'N/A')}")
                print(f"  Reference: {result.reference_link}")
                print(f"  Summary: {result.contextual_summary[:150]}...")
                print(f"  Content: {result.original_content[:100]}...")
        else:
            print("  No results found")
        
        print("\n" + "-"*60)


def main():
    """Run the complete demonstration"""
    
    # Sample document (you can replace this with real documents)
    sample_document = """
# Comprehensive Guide to Retrieval-Augmented Generation (RAG)

## Introduction

Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of large language models with external knowledge retrieval. This approach addresses key limitations of standalone LLMs, such as hallucination and knowledge cutoff dates.

### What is RAG?

RAG works by first retrieving relevant documents from a knowledge base, then using those documents as context for the language model to generate responses. This ensures that answers are grounded in factual information and can be updated without retraining the model.

## Core Components

### 1. Document Processing

The first step in RAG is processing your documents into searchable chunks. This involves:

- **Text Extraction**: Extracting text from various formats (PDF, DOCX, HTML, etc.)
- **Chunking**: Splitting documents into semantic units that preserve context
- **Metadata Enrichment**: Adding information like source, date, author, and topics

Best practices for chunking include maintaining semantic coherence, using overlapping chunks to preserve context across boundaries, and keeping chunks sized appropriately for your embedding model.

### 2. Embedding Generation

Once documents are chunked, each chunk is converted into a dense vector representation (embedding) that captures its semantic meaning. Popular embedding models include:

- OpenAI's text-embedding-ada-002
- Sentence-BERT variants (nomic-embed-text)
- Instructor embeddings
- Cohere embeddings

The choice of embedding model affects retrieval quality. Models trained on domain-specific data often perform better for specialized applications.

### 3. Vector Storage

Embeddings are stored in a vector database that enables fast similarity search. Popular options include:

- **Pinecone**: Fully managed, excellent for production
- **Weaviate**: Open-source with rich filtering
- **Qdrant**: Fast and written in Rust
- **Chroma**: Lightweight and easy to use
- **FAISS**: Facebook's library for efficient similarity search

### 4. Retrieval

When a user asks a question, the system:

1. Embeds the query using the same model as the documents
2. Performs similarity search in the vector database
3. Retrieves the top-k most similar chunks
4. Optionally applies metadata filters to narrow results

### 5. Generation

Retrieved chunks are formatted into a prompt for the LLM, which generates a response grounded in the retrieved information. This typically includes:

- System instructions about how to use the context
- The retrieved document chunks
- The user's question
- Instructions to cite sources

## Advanced Techniques

### Hybrid Search

Combining vector similarity with traditional keyword search (BM25) often yields better results than either approach alone. This helps with:

- Exact phrase matching
- Proper nouns and technical terms
- Acronyms and abbreviations

### Reranking

After initial retrieval, a reranking model can reorder results based on relevance to the specific query. Cross-encoders like MiniLM-L6-v2-reranker are popular choices.

### Query Transformation

Transforming the user's query before retrieval can improve results:

- **HyDE**: Generate a hypothetical answer, then search for similar content
- **Multi-query**: Generate multiple variations of the query
- **Query decomposition**: Break complex queries into simpler sub-queries

### Contextual Chunk Headers

Adding AI-generated context to each chunk improves retrieval by:

- Explaining what the chunk discusses
- Providing document and section context
- Including hypothetical questions the chunk answers
- Listing key entities and topics

## Implementation Challenges

### Chunk Size Optimization

Finding the right chunk size is crucial. Too small and you lose context; too large and you dilute relevance. Typical ranges:

- Short (200-400 tokens): Good for QA, may lose context
- Medium (400-800 tokens): Balanced approach for most use cases
- Long (800-1200 tokens): Better context preservation, may reduce precision

### Handling Multi-Document Context

When answers require information from multiple documents, strategies include:

- Retrieving more chunks (higher top-k)
- Iterative retrieval (follow-up searches based on initial results)
- Graph-based approaches linking related documents

### Citation and Attribution

Properly citing sources is essential for trustworthiness. Include:

- Document title and author
- Publication date
- Page numbers or section references
- Direct links to source material

### Evaluation

Measuring RAG performance requires evaluating both retrieval and generation:

- **Retrieval metrics**: Precision@k, Recall@k, MRR, NDCG
- **Generation metrics**: Faithfulness, answer relevance, completeness
- **End-to-end metrics**: Human evaluation, user satisfaction

## Best Practices

1. **Clean your data**: Remove duplicates, fix formatting, validate content
2. **Optimize chunk boundaries**: Split at semantic boundaries (paragraphs, sections)
3. **Enrich with metadata**: Add all relevant filtering dimensions
4. **Monitor and iterate**: Track metrics and improve based on failures
5. **Handle edge cases**: Empty results, low confidence, ambiguous queries
6. **Implement guardrails**: Verify citations, check for hallucinations

## Conclusion

RAG represents a practical approach to building AI systems that are both powerful and trustworthy. By grounding language models in retrieved documents, we can create applications that provide accurate, up-to-date information with proper attribution.

The key to success is careful attention to each component: document processing, embedding quality, retrieval accuracy, and generation faithfulness. With the right implementation, RAG systems can dramatically improve the utility of large language models for real-world applications.
"""
    
    # Process document with AI enhancement
    # Default: Use Ollama (free, local)
    # To use Anthropic instead, set ai_provider="anthropic" and provide anthropic_api_key

    ai_provider = os.environ.get("AI_PROVIDER", "ollama")  # or "anthropic"

    chunks = process_document_with_ai(
        text=sample_document,
        source_document="guides/rag_comprehensive_guide.md",
        document_title="Comprehensive Guide to RAG",
        ai_provider=ai_provider,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://rachel:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        chunk_size=800,
        chunk_overlap=150
    )
    
    # Build search index
    search_engine = build_search_index(chunks)
    
    # Save the chunks and index
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save chunks as JSON
    chunks_file = output_dir / "enhanced_chunks.json"
    chunks_data = [chunk.to_dict() for chunk in chunks]
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved chunks to {chunks_file}")
    
    # Save search index
    index_file = output_dir / "search_index.pkl"
    search_engine.save(str(index_file))
    
    # Demonstrate searches
    queries = [
        ("What is RAG and how does it work?", None),
        ("Best practices for chunking documents", None),
        ("What vector databases are available?", None),
        ("How do you evaluate RAG systems?", {"chunk_type": "content"}),
        ("Embedding models for RAG", None)
    ]
    
    demonstrate_search(search_engine, queries)
    
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}\n")
    print(f"Chunks saved to: {chunks_file}")
    print(f"Search index saved to: {index_file}")
    print("\nYou can now:")
    print("1. Load the chunks for further processing")
    print("2. Use the search index for queries")
    print("3. Generate references from chunk metadata")
    print("4. Apply custom filters for domain-specific retrieval")


if __name__ == "__main__":
    main()
