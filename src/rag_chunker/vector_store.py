"""
Vector Store Integration
Handles embedding generation and storage for enhanced chunks
Supports multiple vector store backends
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle


@dataclass
class SearchResult:
    """Result from a vector search"""
    chunk_id: str
    score: float
    original_content: str
    contextual_summary: str
    metadata: Dict[str, Any]
    reference_link: str


class EmbeddingGenerator:
    """
    Generate embeddings for chunks using various models
    Supports Ollama embeddings, Sentence Transformers, and API-based embeddings
    """

    def __init__(self,
                 model_name: str = "nomic-embed-text",
                 use_ollama: bool = True,
                 ollama_host: str = "http://rachel:11434"):
        """
        Initialize embedding generator

        Args:
            model_name: Model to use
                       - Ollama models: "nomic-embed-text", "mxbai-embed-large", "all-minilm"
                       - Sentence-transformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", etc.
            use_ollama: If True, use Ollama; else use sentence-transformers
            ollama_host: Ollama server URL (only used if use_ollama=True)
        """
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.ollama_host = ollama_host.rstrip('/')
        self.model = None

        if use_ollama:
            self._init_ollama()
        else:
            self._init_local_model()
    
    def _init_ollama(self):
        """Initialize Ollama embedding model"""
        import requests
        try:
            # Test connection to Ollama
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                print(f"✓ Connected to Ollama at {self.ollama_host}")
                print(f"Using embedding model: {self.model_name}")

                # Test if model is available
                if not any(self.model_name in m for m in models):
                    print(f"⚠️  Model '{self.model_name}' not found in Ollama")
                    print(f"   Available models: {', '.join(models)}")
                    print(f"   Pull it with: ollama pull {self.model_name}")
            else:
                print(f"⚠️  Could not connect to Ollama at {self.ollama_host}")
                self.model = None
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("   Make sure Ollama is running: ollama serve")
            self.model = None

    def _init_local_model(self):
        """Initialize local sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully")
        except ImportError:
            print("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if self.use_ollama:
            return self._embed_with_ollama(texts)
        elif self.model:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # Fallback to mock embeddings
            print("Warning: Using mock embeddings. Install sentence-transformers or use Ollama for real embeddings.")
            return self._generate_mock_embeddings(texts)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.embed_batch([text])[0]

    def _embed_with_ollama(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Ollama API"""
        import requests

        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = np.array(result['embedding'])
                    embeddings.append(embedding)
                else:
                    print(f"Warning: Ollama embedding failed with status {response.status_code}")
                    # Fallback to mock for this text
                    embeddings.append(self._generate_mock_embeddings([text])[0])
            except Exception as e:
                print(f"Warning: Error getting Ollama embedding: {e}")
                # Fallback to mock for this text
                embeddings.append(self._generate_mock_embeddings([text])[0])

        return np.array(embeddings)
    
    def _generate_mock_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings based on text hashing (for testing)"""
        embeddings = []
        for text in texts:
            # Simple hash-based embedding (deterministic but not semantic)
            hash_val = hash(text)
            np.random.seed(hash_val % (2**32))
            embedding = np.random.randn(384)  # Standard size for MiniLM
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        return np.array(embeddings)


class VectorStore:
    """
    Simple in-memory vector store with metadata filtering
    For production, use Pinecone, Weaviate, Qdrant, or Chroma
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.chunks = []
        self.metadata = []
        self.index_map = {}  # chunk_id -> index
    
    def add_chunks(self, 
                   chunks: List['EnhancedChunk'],
                   embeddings: np.ndarray):
        """
        Add chunks with their embeddings to the store
        
        Args:
            chunks: List of EnhancedChunk objects
            embeddings: Corresponding embeddings array
        """
        start_idx = len(self.embeddings)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            idx = start_idx + i
            
            self.embeddings.append(embedding)
            self.chunks.append(chunk)
            self.metadata.append(chunk.metadata.to_dict())
            self.index_map[chunk.metadata.chunk_id] = idx
    
    def search(self,
               query_embedding: np.ndarray,
               top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"chunk_type": "content", "topics": ["ML"]})
            
        Returns:
            List of SearchResult objects
        """
        if not self.embeddings:
            return []
        
        # Convert to numpy array
        embeddings_array = np.array(self.embeddings)
        
        # Apply filters if provided
        valid_indices = self._apply_filters(filters) if filters else list(range(len(self.embeddings)))
        
        if not valid_indices:
            return []
        
        # Compute similarity only for valid indices
        filtered_embeddings = embeddings_array[valid_indices]
        similarities = np.dot(filtered_embeddings, query_embedding)
        
        # Get top-k
        top_indices_in_filtered = np.argsort(similarities)[-top_k:][::-1]
        top_indices = [valid_indices[i] for i in top_indices_in_filtered]
        
        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(similarities[list(valid_indices).index(idx)])
            
            result = SearchResult(
                chunk_id=chunk.metadata.chunk_id,
                score=score,
                original_content=chunk.original_content,
                contextual_summary=chunk.contextual_summary,
                metadata=chunk.metadata.to_dict(),
                reference_link=chunk.get_reference_link()
            )
            results.append(result)
        
        return results
    
    def _apply_filters(self, filters: Dict[str, Any]) -> List[int]:
        """
        Apply metadata filters to get valid indices
        
        Supported filter types:
        - Exact match: {"chunk_type": "content"}
        - List contains: {"topics": ["ML", "AI"]} (match if any topic in list)
        - Range: {"chunk_index": {"min": 0, "max": 10}}
        """
        valid_indices = []
        
        for idx, metadata in enumerate(self.metadata):
            matches = True
            
            for key, value in filters.items():
                if key not in metadata:
                    matches = False
                    break
                
                # Handle different filter types
                if isinstance(value, dict) and "min" in value or "max" in value:
                    # Range filter
                    if "min" in value and metadata[key] < value["min"]:
                        matches = False
                        break
                    if "max" in value and metadata[key] > value["max"]:
                        matches = False
                        break
                elif isinstance(value, list):
                    # List contains (OR logic)
                    if isinstance(metadata[key], list):
                        if not any(item in metadata[key] for item in value):
                            matches = False
                            break
                    else:
                        if metadata[key] not in value:
                            matches = False
                            break
                else:
                    # Exact match
                    if metadata[key] != value:
                        matches = False
                        break
            
            if matches:
                valid_indices.append(idx)
        
        return valid_indices
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            "embeddings": np.array(self.embeddings),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata,
            "index_map": self.index_map,
            "embedding_dim": self.embedding_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Vector store saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VectorStore':
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Import here to avoid circular dependency
        from .rag_chunker import EnhancedChunk, ChunkMetadata
        
        store = cls(embedding_dim=data["embedding_dim"])
        store.embeddings = list(data["embeddings"])
        store.metadata = data["metadata"]
        store.index_map = data["index_map"]
        
        # Reconstruct chunks
        for chunk_data in data["chunks"]:
            metadata = ChunkMetadata(**chunk_data["metadata"])
            chunk = EnhancedChunk(
                original_content=chunk_data["original_content"],
                contextual_summary=chunk_data["contextual_summary"],
                hypothetical_questions=chunk_data["hypothetical_questions"],
                metadata=metadata,
                embedding_text=chunk_data.get("embedding_text", "")
            )
            store.chunks.append(chunk)
        
        print(f"Vector store loaded from {filepath}")
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        stats = {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_dim,
            "chunk_types": {},
            "documents": set(),
            "topics": set()
        }
        
        for metadata in self.metadata:
            # Count chunk types
            chunk_type = metadata.get("chunk_type", "unknown")
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1
            
            # Collect documents
            stats["documents"].add(metadata.get("source_document", "unknown"))
            
            # Collect topics
            for topic in metadata.get("topics", []):
                stats["topics"].add(topic)
        
        stats["total_documents"] = len(stats["documents"])
        stats["total_topics"] = len(stats["topics"])
        stats["documents"] = list(stats["documents"])
        stats["topics"] = list(stats["topics"])
        
        return stats


class RAGSearchEngine:
    """
    High-level search engine combining embedding and vector store
    """

    def __init__(self,
                 embedding_model: str = "nomic-embed-text",
                 use_ollama: bool = True,
                 ollama_host: str = "http://rachel:11434"):
        """
        Initialize RAG search engine

        Args:
            embedding_model: Model to use for embeddings
                           - Ollama: "nomic-embed-text", "mxbai-embed-large", "all-minilm"
                           - Sentence-transformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2"
            use_ollama: If True, use Ollama for embeddings; else use sentence-transformers
            ollama_host: Ollama server URL
        """
        self.embedder = EmbeddingGenerator(
            model_name=embedding_model,
            use_ollama=use_ollama,
            ollama_host=ollama_host
        )
        self.vector_store = VectorStore()
    
    def index_chunks(self, chunks: List['EnhancedChunk']):
        """
        Index enhanced chunks into the search engine
        
        Args:
            chunks: List of EnhancedChunk objects
        """
        print(f"Indexing {len(chunks)} chunks...")
        
        # Extract embedding texts
        embedding_texts = [chunk.embedding_text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(embedding_texts)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)
        
        print(f"Indexing complete. Total chunks: {len(self.vector_store.chunks)}")
    
    def search(self,
               query: str,
               top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None,
               return_with_scores: bool = True) -> List[SearchResult]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            return_with_scores: If True, include similarity scores
            
        Returns:
            List of SearchResult objects
        """
        # Embed query
        query_embedding = self.embedder.embed_single(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k, filters)
        
        return results
    
    def save(self, filepath: str):
        """Save search engine to disk"""
        self.vector_store.save(filepath)
    
    def load(self, filepath: str):
        """Load search engine from disk"""
        self.vector_store = VectorStore.load(filepath)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return self.vector_store.get_stats()


if __name__ == "__main__":
    # Example usage
    from .rag_chunker import EnhancedRAGChunker
    
    sample_text = """
# Machine Learning Fundamentals

Machine learning enables computers to learn from data without explicit programming.

## Supervised Learning

In supervised learning, we train models on labeled examples. The model learns to map inputs to outputs based on example input-output pairs.

## Unsupervised Learning  

Unsupervised learning finds patterns in unlabeled data. Common techniques include clustering and dimensionality reduction.
"""
    
    # Create chunks
    chunker = EnhancedRAGChunker(chunk_size=200, use_ai_context=False)
    chunks = chunker.process_document(
        sample_text,
        "docs/ml_basics.md",
        "Machine Learning Fundamentals"
    )
    
    # Create search engine and index
    search_engine = RAGSearchEngine(use_local_embeddings=True)
    search_engine.index_chunks(chunks)
    
    # Search
    results = search_engine.search("What is supervised learning?", top_k=2)
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Reference: {result.reference_link}")
        print(f"   Content: {result.original_content[:100]}...")
    
    # Show stats
    print("\nSearch Engine Stats:")
    print(json.dumps(search_engine.get_stats(), indent=2))
