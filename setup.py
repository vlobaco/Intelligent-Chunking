"""
Setup configuration for RAG Chunker package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="rag-chunker",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Enhanced Document Chunking for Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-chunker",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/rag-chunker/issues",
        "Documentation": "https://github.com/yourusername/rag-chunker/tree/main/docs",
        "Source Code": "https://github.com/yourusername/rag-chunker",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "anthropic": ["anthropic>=0.40.0"],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    keywords=[
        "rag",
        "retrieval-augmented-generation",
        "chunking",
        "embeddings",
        "vector-search",
        "semantic-search",
        "llm",
        "ai",
        "ollama",
        "document-processing",
    ],
    include_package_data=True,
    zip_safe=False,
)
