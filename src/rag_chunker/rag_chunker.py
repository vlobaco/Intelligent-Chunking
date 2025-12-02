"""
AI-Enhanced RAG Chunking System
Generates context-rich chunks with metadata and HyDE questions for improved retrieval
"""

import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import re


@dataclass
class ChunkMetadata:
    """Metadata for each chunk to enable filtering and reference generation"""
    chunk_id: str
    source_document: str
    document_title: str
    chunk_index: int
    total_chunks: int
    section_hierarchy: List[str]  # e.g., ["Chapter 1", "Section 1.2", "Subsection 1.2.3"]
    page_number: Optional[int] = None
    start_char: int = 0
    end_char: int = 0
    created_at: str = ""
    tokens_estimate: int = 0
    chunk_type: str = "content"  # content, code, table, list, etc.
    entities: List[str] = None  # Named entities, concepts, keywords
    topics: List[str] = None  # Main topics/themes
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if self.entities is None:
            self.entities = []
        if self.topics is None:
            self.topics = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnhancedChunk:
    """A chunk with original content, AI-generated context, and metadata"""
    original_content: str
    contextual_summary: str
    hypothetical_questions: List[str]
    metadata: ChunkMetadata
    embedding_text: str = ""  # The actual text to embed (hybrid of context + content)
    
    def __post_init__(self):
        if not self.embedding_text:
            # Create hybrid text: context first, then original content
            self.embedding_text = self._build_embedding_text()
    
    def _build_embedding_text(self) -> str:
        """Build the text that will be embedded for retrieval"""
        parts = []
        
        # Add contextual summary
        if self.contextual_summary:
            parts.append(f"Context: {self.contextual_summary}")
        
        # Add hypothetical questions
        if self.hypothetical_questions:
            questions_text = " | ".join(self.hypothetical_questions)
            parts.append(f"Answers questions like: {questions_text}")
        
        # Add section hierarchy for context
        if self.metadata.section_hierarchy:
            hierarchy = " > ".join(self.metadata.section_hierarchy)
            parts.append(f"Section: {hierarchy}")
        
        # Add original content
        parts.append(self.original_content)
        
        return "\n\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_content": self.original_content,
            "contextual_summary": self.contextual_summary,
            "hypothetical_questions": self.hypothetical_questions,
            "metadata": self.metadata.to_dict(),
            "embedding_text": self.embedding_text
        }
    
    def get_reference_link(self, base_url: str = "") -> str:
        """Generate a reference link to the source document"""
        doc_path = self.metadata.source_document
        chunk_id = self.metadata.chunk_id
        page = self.metadata.page_number
        
        if page:
            return f"{base_url}{doc_path}#page={page}&chunk={chunk_id}"
        else:
            return f"{base_url}{doc_path}#chunk={chunk_id}"


class DocumentChunker:
    """Handles the basic chunking of documents"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk_text(self, 
                   text: str, 
                   source_document: str,
                   document_title: str = "") -> List[Dict[str, Any]]:
        """
        Chunk text using recursive character splitting with overlap
        Returns basic chunks with minimal metadata
        """
        if not document_title:
            document_title = source_document.split("/")[-1]
        
        chunks = self._recursive_split(text)
        
        enhanced_chunks = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = self._generate_chunk_id(source_document, idx)
            
            # Try to detect section hierarchy from the chunk
            section_hierarchy = self._extract_section_hierarchy(chunk_text, text)
            
            # Estimate token count (rough approximation)
            tokens_estimate = len(chunk_text.split()) * 1.3
            
            # Detect chunk type
            chunk_type = self._detect_chunk_type(chunk_text)
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                source_document=source_document,
                document_title=document_title,
                chunk_index=idx,
                total_chunks=len(chunks),
                section_hierarchy=section_hierarchy,
                start_char=text.find(chunk_text),
                end_char=text.find(chunk_text) + len(chunk_text),
                tokens_estimate=int(tokens_estimate),
                chunk_type=chunk_type
            )
            
            enhanced_chunks.append({
                "content": chunk_text,
                "metadata": metadata
            })
        
        return enhanced_chunks
    
    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text using different separators"""
        chunks = []
        
        for separator in self.separators:
            if separator == "":
                # Base case: split by character
                return [text[i:i + self.chunk_size] 
                       for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            
            parts = text.split(separator)
            current_chunk = []
            current_size = 0
            
            for part in parts:
                part_size = len(part)
                
                if current_size + part_size > self.chunk_size and current_chunk:
                    # Finalize current chunk
                    chunk_text = separator.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_parts = current_chunk[-2:] if len(current_chunk) > 1 else current_chunk
                    current_chunk = overlap_parts + [part]
                    current_size = sum(len(p) for p in current_chunk)
                else:
                    current_chunk.append(part)
                    current_size += part_size
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(separator.join(current_chunk))
            
            # If we got good chunks, return them
            if len(chunks) > 1 or len(chunks[0]) <= self.chunk_size * 1.5:
                return chunks
            
            # Otherwise, try next separator
            text = chunks[0] if chunks else text
            chunks = []
        
        return chunks if chunks else [text]
    
    def _generate_chunk_id(self, source_document: str, index: int) -> str:
        """Generate a unique ID for the chunk"""
        base = f"{source_document}_{index}"
        return hashlib.md5(base.encode()).hexdigest()[:12]
    
    def _extract_section_hierarchy(self, chunk_text: str, full_text: str) -> List[str]:
        """Try to extract section hierarchy from headers"""
        # Look for markdown-style headers
        headers = []
        lines = chunk_text.split('\n')
        
        for line in lines[:5]:  # Check first few lines
            if line.strip().startswith('#'):
                header = line.strip().lstrip('#').strip()
                if header:
                    headers.append(header)
        
        return headers if headers else ["Main Content"]
    
    def _detect_chunk_type(self, chunk_text: str) -> str:
        """Detect the type of content in the chunk"""
        # Simple heuristics
        if '```' in chunk_text or 'def ' in chunk_text or 'class ' in chunk_text:
            return "code"
        elif chunk_text.strip().startswith('|') or '\t' in chunk_text:
            return "table"
        elif re.match(r'^\s*[-*\d.]+\s', chunk_text.strip()):
            return "list"
        else:
            return "content"




class EnhancedRAGChunker:
    """Main orchestrator for the enhanced chunking pipeline"""

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_ai_context: bool = True,
                 ai_provider: str = "ollama",
                 **ai_kwargs):
        """
        Initialize the enhanced chunker

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            use_ai_context: Whether to use AI for context generation
            ai_provider: "ollama" (default) or "anthropic"
            **ai_kwargs: Provider-specific arguments (host, model, api_key, etc.)
        """
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.use_ai_context = use_ai_context

        if use_ai_context:
            from .ai_context_generator import create_context_generator
            self.context_generator = create_context_generator(provider=ai_provider, **ai_kwargs)
        else:
            self.context_generator = None
    
    def process_document(self,
                        text: str,
                        source_document: str,
                        document_title: str = "",
                        context_window: int = 1) -> List[EnhancedChunk]:
        """
        Process a document into enhanced chunks with AI-generated context

        Args:
            text: The document text to process
            source_document: Path/identifier for the source document
            document_title: Human-readable title
            context_window: Number of surrounding chunks to provide as context (default: 1)
                          0 = no surrounding context, 1 = one chunk before/after, etc.
        """
        # Step 1: Basic chunking
        basic_chunks = self.chunker.chunk_text(text, source_document, document_title)

        # Step 2: Enhance each chunk with AI context
        enhanced_chunks = []
        for idx, chunk_data in enumerate(basic_chunks):
            chunk_text = chunk_data["content"]
            metadata = chunk_data["metadata"]

            if self.use_ai_context and self.context_generator:
                # Gather surrounding context for better AI understanding
                surrounding_context = self._get_surrounding_context(
                    basic_chunks, idx, context_window
                )

                # Generate AI context with surrounding chunks for better understanding
                ai_context = self.context_generator.generate_context(
                    chunk_text,
                    metadata,
                    preceding_chunks=surrounding_context["preceding"],
                    following_chunks=surrounding_context["following"],
                    full_document_title=document_title
                )

                # Update metadata with AI-extracted entities and topics
                metadata.entities = ai_context.get("entities", [])
                metadata.topics = ai_context.get("topics", [])

                contextual_summary = ai_context.get("contextual_summary", "")
                hypothetical_questions = ai_context.get("hypothetical_questions", [])
            else:
                contextual_summary = ""
                hypothetical_questions = []

            # Create enhanced chunk
            enhanced_chunk = EnhancedChunk(
                original_content=chunk_text,
                contextual_summary=contextual_summary,
                hypothetical_questions=hypothetical_questions,
                metadata=metadata
            )

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def _get_surrounding_context(self, chunks: List[Dict], current_idx: int, window: int) -> Dict[str, List[str]]:
        """
        Get surrounding chunks for context

        Args:
            chunks: List of all chunk data
            current_idx: Index of current chunk
            window: Number of chunks before/after to include

        Returns:
            Dict with 'preceding' and 'following' chunk texts
        """
        preceding = []
        following = []

        # Get preceding chunks
        for i in range(max(0, current_idx - window), current_idx):
            preceding.append(chunks[i]["content"])

        # Get following chunks
        for i in range(current_idx + 1, min(len(chunks), current_idx + window + 1)):
            following.append(chunks[i]["content"])

        return {
            "preceding": preceding,
            "following": following
        }
    
    def save_chunks(self, chunks: List[EnhancedChunk], output_path: str):
        """Save enhanced chunks to a JSON file"""
        chunks_data = [chunk.to_dict() for chunk in chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    def load_chunks(self, input_path: str) -> List[EnhancedChunk]:
        """Load enhanced chunks from a JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for data in chunks_data:
            metadata = ChunkMetadata(**data["metadata"])
            chunk = EnhancedChunk(
                original_content=data["original_content"],
                contextual_summary=data["contextual_summary"],
                hypothetical_questions=data["hypothetical_questions"],
                metadata=metadata,
                embedding_text=data.get("embedding_text", "")
            )
            chunks.append(chunk)
        
        return chunks


if __name__ == "__main__":
    # Example usage
    sample_text = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the algorithm learns from labeled training data. The model makes predictions based on input-output pairs.

### Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm tries to find patterns and structure in the input data.

## Applications

Machine learning has numerous applications across industries:
- Healthcare: Disease diagnosis and drug discovery
- Finance: Fraud detection and algorithmic trading
- Transportation: Autonomous vehicles
- Retail: Recommendation systems
"""
    
    # Initialize chunker
    chunker = EnhancedRAGChunker(chunk_size=300, chunk_overlap=50, use_ai_context=False)
    
    # Process document
    chunks = chunker.process_document(
        text=sample_text,
        source_document="docs/ml_intro.md",
        document_title="Introduction to Machine Learning"
    )
    
    # Display results
    print(f"Created {len(chunks)} enhanced chunks\n")
    for i, chunk in enumerate(chunks[:2]):  # Show first 2
        print(f"=== Chunk {i+1} ===")
        print(f"ID: {chunk.metadata.chunk_id}")
        print(f"Type: {chunk.metadata.chunk_type}")
        print(f"Section: {' > '.join(chunk.metadata.section_hierarchy)}")
        print(f"Reference: {chunk.get_reference_link('https://docs.example.com/')}")
        print(f"\nOriginal Content ({len(chunk.original_content)} chars):")
        print(chunk.original_content[:200] + "...")
        print(f"\nEmbedding Text Preview:")
        print(chunk.embedding_text[:300] + "...")
        print("\n")
