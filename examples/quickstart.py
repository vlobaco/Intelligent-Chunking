#!/usr/bin/env python3
"""
Quick Start Script for AI-Enhanced RAG Chunking System

This script helps you get started quickly by:
1. Installing dependencies
2. Testing the system
3. Processing your first document
"""

import sys
import subprocess
from pathlib import Path


def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                 ║
║        AI-Enhanced RAG Chunking System - Quick Start           ║
║                                                                 ║
╚════════════════════════════════════════════════════════════════╝
    """)


def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")

    required = {
        'requests': 'requests',
        'aiohttp': 'aiohttp',
        'sentence-transformers': 'sentence_transformers',
        'numpy': 'numpy'
    }

    missing = []
    for package, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)

    return missing


def install_dependencies(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"\n📥 Installing {len(packages)} missing package(s)...")
    print(f"Packages: {', '.join(packages)}")
    
    response = input("\nProceed with installation? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled.")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '--break-system-packages', *packages
        ])
        print("✓ Installation complete!")
        return True
    except subprocess.CalledProcessError:
        print("✗ Installation failed. Please install manually:")
        print(f"  pip install {' '.join(packages)} --break-system-packages")
        return False


def run_demo():
    """Run the demo pipeline"""
    print("\n🚀 Running demo pipeline...")
    print("This will process a sample document and demonstrate the system.\n")
    
    try:
        subprocess.check_call([sys.executable, 'pipeline_example.py'])
        print("\n✓ Demo completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Demo failed. Check the error messages above.")
        return False


def setup_ollama():
    """Help user set up Ollama"""
    import os
    import requests

    print("\n🤖 Ollama Setup (Default AI Provider)")
    print("This system uses Ollama for local, free AI context generation.")

    # Check if Ollama is running
    try:
        ollama_host = os.environ.get('OLLAMA_HOST', 'http://rachel:11434')
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print(f"\n✓ Ollama is running at {ollama_host}")
                print(f"✓ Available models: {', '.join([m['name'] for m in models])}")
                return True
            else:
                print(f"\n⚠️  Ollama is running but no models are installed.")
                print("   Install a model with: ollama pull llama3.2")
                return False
    except:
        pass

    print("\n⚠️  Ollama is not running or not installed.")
    print("\nTo set up Ollama:")
    print("1. Install from https://ollama.ai")
    print("2. Start the server: ollama serve")
    print("3. Pull a model: ollama pull llama3.2")
    print("\nAlternatively, use Anthropic Claude by setting AI_PROVIDER=anthropic")

    return False


def process_custom_document():
    """Interactive document processing"""
    print("\n📄 Process Your Own Document")
    
    doc_path = input("Enter path to your document (or press Enter to skip): ").strip()
    if not doc_path:
        return
    
    if not Path(doc_path).exists():
        print(f"✗ File not found: {doc_path}")
        return
    
    print(f"\n Processing: {doc_path}")
    
    # Create a simple processing script
    from rag_chunker import EnhancedRAGChunker, RAGSearchEngine
    
    try:
        # Read document
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process with Ollama (default)
        chunker = EnhancedRAGChunker(
            chunk_size=800,
            use_ai_context=True,
            ai_provider="ollama"
        )
        chunks = chunker.process_document(
            text=text,
            source_document=doc_path,
            document_title=Path(doc_path).stem
        )
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Build index
        search_engine = RAGSearchEngine()
        search_engine.index_chunks(chunks)
        
        # Save
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        search_engine.save(str(output_dir / "custom_index.pkl"))
        chunker.save_chunks(chunks, str(output_dir / "custom_chunks.json"))
        
        print(f"✓ Saved to {output_dir}/")
        
        # Interactive search
        print("\n🔍 Try searching your document:")
        while True:
            query = input("\nEnter query (or 'q' to quit): ").strip()
            if query.lower() == 'q':
                break
            
            results = search_engine.search(query, top_k=3)
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.score:.3f}")
                print(f"   {result.original_content[:150]}...")
                print(f"   [{result.reference_link}]")
        
    except Exception as e:
        print(f"✗ Error processing document: {e}")


def show_next_steps():
    """Show what to do next"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                        Next Steps                               ║
╚════════════════════════════════════════════════════════════════╝

1. 📖 Read the documentation:
   - README.md          - Overview and features
   - USAGE_GUIDE.md     - Real-world examples
   - ARCHITECTURE.md    - System design details

2. 🔧 Customize the configuration:
   - Edit config.py for your use case
   - Try domain-specific presets (TechnicalDocsConfig, etc.)

3. 🚀 Start building:
   - Process your own documents
   - Build a search interface
   - Integrate with your LLM application

4. 📚 Explore examples:
   - Check USAGE_GUIDE.md for 7 detailed use cases
   - Adapt the code to your needs

5. 🤝 Get help:
   - Check the troubleshooting section in README.md
   - Review the architecture diagrams

Happy chunking! 🎉
    """)


def main():
    """Main quick start flow"""
    print_banner()
    
    # Check and install dependencies
    missing = check_dependencies()
    if missing:
        if not install_dependencies(missing):
            print("\nPlease install dependencies manually and try again.")
            return
        print()
    else:
        print("\n✓ All dependencies installed!")
    
    # Ollama setup
    setup_ollama()
    
    # Run demo
    print("\n" + "="*60)
    response = input("\nRun the demo? (y/n): ")
    if response.lower() == 'y':
        run_demo()
    
    # Custom document
    print("\n" + "="*60)
    response = input("\nProcess your own document? (y/n): ")
    if response.lower() == 'y':
        process_custom_document()
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        sys.exit(1)
