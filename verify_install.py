#!/usr/bin/env python3
"""
Quick verification script to test the rag_chunker installation
"""

def main():
    print("="*60)
    print("RAG Chunker Installation Verification")
    print("="*60)

    # Test 1: Import package
    print("\n1. Testing package import...")
    try:
        import rag_chunker
        print(f"   ✓ Package imported successfully")
        print(f"   ✓ Version: {rag_chunker.__version__}")
    except ImportError as e:
        print(f"   ✗ Failed to import package: {e}")
        print("\n   Solution: Run 'pip install -e .' from the project root")
        return False

    # Test 2: Import main classes
    print("\n2. Testing main class imports...")
    try:
        from rag_chunker import (
            EnhancedRAGChunker,
            RAGSearchEngine,
            RAGConfig,
            ChunkMetadata,
            EnhancedChunk
        )
        print("   ✓ All main classes imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import classes: {e}")
        return False

    # Test 3: Import AI generators
    print("\n3. Testing AI generator imports...")
    try:
        from rag_chunker import (
            OllamaContextGenerator,
            AnthropicContextGenerator,
            create_context_generator
        )
        print("   ✓ AI generators imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import AI generators: {e}")
        return False

    # Test 4: Create chunker instance
    print("\n4. Testing chunker instantiation...")
    try:
        chunker = EnhancedRAGChunker(
            chunk_size=800,
            use_ai_context=False  # Disable AI for quick test
        )
        print("   ✓ Chunker created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create chunker: {e}")
        return False

    # Test 5: Basic chunking
    print("\n5. Testing basic chunking...")
    try:
        test_text = "This is a test document. " * 50
        chunks = chunker.process_document(
            text=test_text,
            source_document="test.txt",
            document_title="Test Document"
        )
        print(f"   ✓ Created {len(chunks)} chunks")
    except Exception as e:
        print(f"   ✗ Chunking failed: {e}")
        return False

    # Test 6: Create search engine
    print("\n6. Testing search engine...")
    try:
        search_engine = RAGSearchEngine()
        print("   ✓ Search engine created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create search engine: {e}")
        return False

    # Test 7: Index chunks
    print("\n7. Testing indexing...")
    try:
        search_engine.index_chunks(chunks)
        print(f"   ✓ Indexed {len(chunks)} chunks")
    except Exception as e:
        print(f"   ✗ Indexing failed: {e}")
        return False

    # Test 8: Search
    print("\n8. Testing search...")
    try:
        results = search_engine.search("test", top_k=3)
        print(f"   ✓ Search returned {len(results)} results")
    except Exception as e:
        print(f"   ✗ Search failed: {e}")
        return False

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nYour installation is working correctly.")
    print("\nNext steps:")
    print("  - Run: python examples/quickstart.py")
    print("  - Run: python examples/pipeline_example.py")
    print("  - Read: docs/USAGE_GUIDE.md")

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
