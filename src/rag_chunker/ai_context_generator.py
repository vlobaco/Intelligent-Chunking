"""
AI API Integration for Context Generation
Supports Ollama (default) and Anthropic Claude for generating context, HyDE questions, and metadata for chunks
"""

import json
import os
from typing import Dict, Any, List
import re
import requests


class OllamaContextGenerator:
    """
    Ollama API integration for context generation
    Uses local Ollama server to generate context for chunks
    """

    def __init__(self, host: str = "http://rachel:11434", model: str = "llama3.2"):
        """
        Initialize with Ollama host and model

        Args:
            host: Ollama server URL (default: http://rachel:11434)
            model: Model to use for generation (e.g., "llama3.2", "mistral", "phi3")
        """
        self.host = host.rstrip('/')
        self.model = model
        self.use_mock = False

        # Test connection
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                if available_models:
                    print(f"Connected to Ollama at {self.host}")
                    print(f"Available models: {', '.join(available_models)}")
                    if model not in available_models:
                        print(f"Warning: Model '{model}' not found. Available models: {available_models}")
                        print(f"You can pull it with: ollama pull {model}")
                        self.use_mock = True
                else:
                    print("Warning: No models found in Ollama. Please pull a model first.")
                    self.use_mock = True
            else:
                print(f"Warning: Could not connect to Ollama at {self.host}")
                print("Make sure Ollama is running. Start it with: ollama serve")
                self.use_mock = True
        except Exception as e:
            print(f"Warning: Could not connect to Ollama at {self.host}: {e}")
            print("Make sure Ollama is running. Start it with: ollama serve")
            self.use_mock = True

    async def generate_context_async(self,
                                     chunk_text: str,
                                     metadata: 'ChunkMetadata',
                                     preceding_chunks: List[str] = None,
                                     following_chunks: List[str] = None,
                                     full_document_title: str = None) -> Dict[str, Any]:
        """
        Async version for batch processing

        Args:
            chunk_text: The main chunk to analyze
            metadata: Chunk metadata
            preceding_chunks: List of chunks that come before (for context)
            following_chunks: List of chunks that come after (for context)
            full_document_title: Full document title for additional context
        """
        import asyncio
        import aiohttp

        if self.use_mock:
            return self._generate_mock_context(chunk_text, metadata)

        prompt = self._build_context_prompt(
            chunk_text,
            metadata,
            preceding_chunks=preceding_chunks or [],
            following_chunks=following_chunks or [],
            full_document_title=full_document_title
        )

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 800
                    }
                }

                async with session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')
                        return self._parse_response(response_text)
                    else:
                        print(f"Error calling Ollama API: HTTP {response.status}")
                        return self._generate_mock_context(chunk_text, metadata)

        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return self._generate_mock_context(chunk_text, metadata)

    def generate_context(self,
                        chunk_text: str,
                        metadata: 'ChunkMetadata',
                        preceding_chunks: List[str] = None,
                        following_chunks: List[str] = None,
                        full_document_title: str = None) -> Dict[str, Any]:
        """
        Synchronous version - generate context for a single chunk

        Args:
            chunk_text: The main chunk to analyze
            metadata: Chunk metadata
            preceding_chunks: List of chunks that come before (for context)
            following_chunks: List of chunks that come after (for context)
            full_document_title: Full document title for additional context
        """
        if self.use_mock:
            return self._generate_mock_context(chunk_text, metadata)

        prompt = self._build_context_prompt(
            chunk_text,
            metadata,
            preceding_chunks=preceding_chunks or [],
            following_chunks=following_chunks or [],
            full_document_title=full_document_title
        )

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 800
                }
            }

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                return self._parse_response(response_text)
            else:
                print(f"Error calling Ollama API: HTTP {response.status_code}")
                return self._generate_mock_context(chunk_text, metadata)

        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return self._generate_mock_context(chunk_text, metadata)

    def _build_context_prompt(self,
                              chunk_text: str,
                              metadata: 'ChunkMetadata',
                              preceding_chunks: List[str] = None,
                              following_chunks: List[str] = None,
                              full_document_title: str = None) -> str:
        """Build the prompt for context generation with surrounding context"""
        section_path = ' > '.join(metadata.section_hierarchy) if metadata.section_hierarchy else "Unknown"
        doc_title = full_document_title or metadata.document_title

        # Build surrounding context section
        context_info = ""
        if preceding_chunks:
            context_info += "\nPRECEDING CONTEXT (what came before this chunk):\n"
            for i, chunk in enumerate(preceding_chunks, 1):
                preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                context_info += f"[Previous chunk {i}]: {preview}\n"

        if following_chunks:
            context_info += "\nFOLLOWING CONTEXT (what comes after this chunk):\n"
            for i, chunk in enumerate(following_chunks, 1):
                preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                context_info += f"[Next chunk {i}]: {preview}\n"

        return f"""You are a RAG (Retrieval-Augmented Generation) optimization expert. Your task is to analyze a text chunk and generate metadata that will improve its retrievability in a semantic search system.

DOCUMENT CONTEXT:
- Document: "{doc_title}"
- Section: {section_path}
- Chunk position: {metadata.chunk_index + 1} of {metadata.total_chunks}
- Content type: {metadata.chunk_type}
{context_info}

MAIN TEXT CHUNK TO ANALYZE:
{chunk_text}

TASK:
Using the surrounding context to better understand this chunk's role in the document, generate the following metadata in JSON format:

1. "contextual_summary": Write a concise 2-3 sentence summary that:
   - Explains what this chunk discusses and its PURPOSE in the document flow
   - Mentions how it relates to the broader document/section AND the surrounding content
   - Includes key concepts and connections that might not be explicitly stated
   - Uses the preceding/following context to understand transitions and relationships

2. "hypothetical_questions": Generate 4-6 diverse questions that this chunk answers well:
   - Include both specific and general questions
   - Use varied question formats (What, How, Why, When, etc.)
   - Think about how real users might search for this information
   - Consider questions that arise from the document flow

3. "entities": Extract 5-10 key entities/concepts:
   - Named entities (people, places, organizations)
   - Technical terms and jargon
   - Important concepts or topics
   - Product names, methodologies, etc.

4. "topics": Identify 3-5 main topics/themes:
   - High-level subject areas
   - Categories this content belongs to
   - Related fields or domains

IMPORTANT:
- Use the surrounding context to understand how this chunk fits into the larger narrative
- The contextual_summary should reflect the chunk's role in the document flow
- Respond ONLY with valid JSON. No markdown, no explanations, just the JSON object with these four keys.

Example format:
{{
  "contextual_summary": "This section explains...",
  "hypothetical_questions": ["How does X work?", "What is the purpose of Y?"],
  "entities": ["Entity1", "Entity2", "Concept1"],
  "topics": ["Topic1", "Topic2"]
}}"""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith('```'):
                # Remove first and last lines (markdown markers)
                lines = cleaned.split('\n')
                # Find the actual JSON content
                json_start = 0
                json_end = len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith('{'):
                        json_start = i
                        break
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().endswith('}'):
                        json_end = i + 1
                        break
                cleaned = '\n'.join(lines[json_start:json_end])

            # Try to extract JSON
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Validate required keys
                required_keys = ["contextual_summary", "hypothetical_questions", "entities", "topics"]
                if all(key in data for key in required_keys):
                    return data

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {response_text[:200]}...")

        # Fallback to empty structure
        return {
            "contextual_summary": "Content excerpt from the document.",
            "hypothetical_questions": ["What does this section discuss?"],
            "entities": [],
            "topics": ["General"]
        }

    def _generate_mock_context(self, chunk_text: str, metadata: 'ChunkMetadata') -> Dict[str, Any]:
        """Generate mock context when API is not available"""
        # Simple keyword extraction
        words = chunk_text.lower().split()
        # Get unique words longer than 4 characters
        keywords = list(set([w.strip('.,;:!?') for w in words if len(w) > 4]))[:8]

        section = ' > '.join(metadata.section_hierarchy) if metadata.section_hierarchy else "content"

        return {
            "contextual_summary": f"This section from '{metadata.document_title}' ({section}) discusses {', '.join(keywords[:3])} and related concepts.",
            "hypothetical_questions": [
                f"What does this section of {metadata.document_title} explain?",
                f"How does this relate to {keywords[0] if keywords else 'the topic'}?",
                "What are the key points covered here?",
                f"Why is this important in the context of {metadata.document_title}?"
            ],
            "entities": keywords,
            "topics": [metadata.chunk_type, section.split(' > ')[0] if ' > ' in section else section]
        }


class AnthropicContextGenerator:
    """
    Real implementation using Anthropic API to generate context for chunks
    """
    
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize with Anthropic API key
        
        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            model: Model to use for generation
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        
        if not self.api_key:
            print("Warning: No API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
            print("Context generation will use mock data.")
            self.use_mock = True
        else:
            self.use_mock = False
    
    async def generate_context_async(self,
                                     chunk_text: str,
                                     metadata: 'ChunkMetadata',
                                     preceding_chunks: List[str] = None,
                                     following_chunks: List[str] = None,
                                     full_document_title: str = None) -> Dict[str, Any]:
        """
        Async version for batch processing

        Args:
            chunk_text: The main chunk to analyze
            metadata: Chunk metadata
            preceding_chunks: List of chunks that come before (for context)
            following_chunks: List of chunks that come after (for context)
            full_document_title: Full document title for additional context
        """
        # Import here to make it optional
        try:
            import anthropic
        except ImportError:
            print("anthropic package not installed. Install with: pip install anthropic")
            return self._generate_mock_context(chunk_text, metadata)

        if self.use_mock:
            return self._generate_mock_context(chunk_text, metadata)

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Use shared prompt builder from OllamaContextGenerator
        temp_generator = OllamaContextGenerator(host="http://rachel:11434", model="llama3.2")
        prompt = temp_generator._build_context_prompt(
            chunk_text,
            metadata,
            preceding_chunks=preceding_chunks or [],
            following_chunks=following_chunks or [],
            full_document_title=full_document_title
        )

        try:
            message = await client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.3,  # Lower temperature for more consistent output
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = message.content[0].text
            return self._parse_response(response_text)

        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return self._generate_mock_context(chunk_text, metadata)
    
    def generate_context(self,
                        chunk_text: str,
                        metadata: 'ChunkMetadata',
                        preceding_chunks: List[str] = None,
                        following_chunks: List[str] = None,
                        full_document_title: str = None) -> Dict[str, Any]:
        """
        Synchronous version - generate context for a single chunk

        Args:
            chunk_text: The main chunk to analyze
            metadata: Chunk metadata
            preceding_chunks: List of chunks that come before (for context)
            following_chunks: List of chunks that come after (for context)
            full_document_title: Full document title for additional context
        """
        # Import here to make it optional
        try:
            import anthropic
        except ImportError:
            print("anthropic package not installed. Install with: pip install anthropic")
            return self._generate_mock_context(chunk_text, metadata)

        if self.use_mock:
            return self._generate_mock_context(chunk_text, metadata)

        client = anthropic.Anthropic(api_key=self.api_key)

        # Use shared prompt builder from OllamaContextGenerator
        # We'll need to create a temporary instance to reuse the method
        temp_generator = OllamaContextGenerator(host="http://rachel:11434", model="llama3.2")
        prompt = temp_generator._build_context_prompt(
            chunk_text,
            metadata,
            preceding_chunks=preceding_chunks or [],
            following_chunks=following_chunks or [],
            full_document_title=full_document_title
        )

        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = message.content[0].text
            return self._parse_response(response_text)

        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return self._generate_mock_context(chunk_text, metadata)
    
    def _build_context_prompt(self, chunk_text: str, metadata: 'ChunkMetadata') -> str:
        """Build the prompt for context generation"""
        section_path = ' > '.join(metadata.section_hierarchy) if metadata.section_hierarchy else "Unknown"
        
        return f"""You are a RAG (Retrieval-Augmented Generation) optimization expert. Your task is to analyze a text chunk and generate metadata that will improve its retrievability in a semantic search system.

DOCUMENT CONTEXT:
- Document: "{metadata.document_title}"
- Section: {section_path}
- Chunk position: {metadata.chunk_index + 1} of {metadata.total_chunks}
- Content type: {metadata.chunk_type}

TEXT CHUNK:
{chunk_text}

TASK:
Generate the following metadata in JSON format:

1. "contextual_summary": Write a concise 2-3 sentence summary that:
   - Explains what this chunk discusses
   - Mentions how it relates to the broader document/section
   - Includes key concepts that might not be explicitly stated in the chunk
   
2. "hypothetical_questions": Generate 4-6 diverse questions that this chunk answers well:
   - Include both specific and general questions
   - Use varied question formats (What, How, Why, When, etc.)
   - Think about how real users might search for this information
   
3. "entities": Extract 5-10 key entities/concepts:
   - Named entities (people, places, organizations)
   - Technical terms and jargon
   - Important concepts or topics
   - Product names, methodologies, etc.
   
4. "topics": Identify 3-5 main topics/themes:
   - High-level subject areas
   - Categories this content belongs to
   - Related fields or domains

IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations, just the JSON object with these four keys.

Example format:
{{
  "contextual_summary": "This section explains...",
  "hypothetical_questions": ["How does X work?", "What is the purpose of Y?"],
  "entities": ["Entity1", "Entity2", "Concept1"],
  "topics": ["Topic1", "Topic2"]
}}"""
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith('```'):
                # Remove first and last lines (markdown markers)
                lines = cleaned.split('\n')
                cleaned = '\n'.join(lines[1:-1])
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Validate required keys
                required_keys = ["contextual_summary", "hypothetical_questions", "entities", "topics"]
                if all(key in data for key in required_keys):
                    return data
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {response_text[:200]}...")
        
        # Fallback to empty structure
        return {
            "contextual_summary": "Content excerpt from the document.",
            "hypothetical_questions": ["What does this section discuss?"],
            "entities": [],
            "topics": ["General"]
        }
    
    def _generate_mock_context(self, chunk_text: str, metadata: 'ChunkMetadata') -> Dict[str, Any]:
        """Generate mock context when API is not available"""
        # Simple keyword extraction
        words = chunk_text.lower().split()
        # Get unique words longer than 4 characters
        keywords = list(set([w.strip('.,;:!?') for w in words if len(w) > 4]))[:8]
        
        section = ' > '.join(metadata.section_hierarchy) if metadata.section_hierarchy else "content"
        
        return {
            "contextual_summary": f"This section from '{metadata.document_title}' ({section}) discusses {', '.join(keywords[:3])} and related concepts.",
            "hypothetical_questions": [
                f"What does this section of {metadata.document_title} explain?",
                f"How does this relate to {keywords[0] if keywords else 'the topic'}?",
                "What are the key points covered here?",
                f"Why is this important in the context of {metadata.document_title}?"
            ],
            "entities": keywords,
            "topics": [metadata.chunk_type, section.split(' > ')[0] if ' > ' in section else section]
        }


async def batch_generate_contexts(chunks_data: List[Dict], 
                                  api_key: str = None,
                                  max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Process multiple chunks concurrently with rate limiting
    
    Args:
        chunks_data: List of dicts with 'content' and 'metadata' keys
        api_key: Anthropic API key
        max_concurrent: Maximum concurrent API calls
    
    Returns:
        List of context dictionaries
    """
    import asyncio
    
    generator = AnthropicContextGenerator(api_key=api_key)

    # Import metadata class
    from .rag_chunker import ChunkMetadata
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_chunk(chunk_data):
        async with semaphore:
            metadata = chunk_data["metadata"]
            if not isinstance(metadata, ChunkMetadata):
                metadata = ChunkMetadata(**metadata)
            
            result = await generator.generate_context_async(
                chunk_data["content"],
                metadata
            )
            return result
    
    tasks = [process_chunk(chunk) for chunk in chunks_data]
    results = await asyncio.gather(*tasks)
    
    return results


def create_context_generator(provider: str = "ollama", **kwargs):
    """
    Factory function to create the appropriate context generator based on provider

    Args:
        provider: "ollama" or "anthropic"
        **kwargs: Provider-specific arguments

    Returns:
        Context generator instance
    """
    if provider.lower() == "ollama":
        host = kwargs.get('host', os.environ.get('OLLAMA_HOST', 'http://rachel:11434'))
        model = kwargs.get('model', os.environ.get('OLLAMA_MODEL', 'llama3.2'))
        return OllamaContextGenerator(host=host, model=model)
    elif provider.lower() == "anthropic":
        api_key = kwargs.get('api_key', os.environ.get('ANTHROPIC_API_KEY'))
        model = kwargs.get('model', 'claude-sonnet-4-20250514')
        return AnthropicContextGenerator(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'anthropic'")


if __name__ == "__main__":
    # Example usage
    from .rag_chunker import ChunkMetadata

    sample_metadata = ChunkMetadata(
        chunk_id="test_001",
        source_document="docs/guide.md",
        document_title="RAG Implementation Guide",
        chunk_index=0,
        total_chunks=10,
        section_hierarchy=["Introduction", "Core Concepts"],
        chunk_type="content"
    )

    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
    by providing them with relevant context from external knowledge sources. The core idea is
    to retrieve pertinent information from a database or document collection and include it
    in the prompt, allowing the model to generate more accurate and grounded responses.
    """

    # Test Ollama (default)
    print("Testing Ollama context generator...")
    generator = create_context_generator(provider="ollama")
    context = generator.generate_context(sample_text, sample_metadata)

    print("Generated Context:")
    print(json.dumps(context, indent=2))
