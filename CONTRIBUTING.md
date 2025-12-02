# Contributing to RAG Chunker

Thank you for your interest in contributing to RAG Chunker! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/rag-chunker.git
   cd rag-chunker
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in Development Mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up Ollama** (for testing)
   ```bash
   # Install from https://ollama.ai
   ollama serve
   ollama pull llama3.2
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions/classes
- Maximum line length: 100 characters

Run code formatters:
```bash
black src/
flake8 src/
mypy src/
```

## Testing

Run tests before submitting:
```bash
pytest tests/
```

## Pull Request Process

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Make your changes
3. Add tests for new functionality
4. Update documentation
5. Run tests and linters
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your fork (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what and why
- **Tests**: Include tests for new features
- **Documentation**: Update relevant docs
- **Backwards Compatibility**: Don't break existing functionality

## Areas for Contribution

- **New Features**: Additional AI providers, embedding models, vector stores
- **Documentation**: Tutorials, examples, API docs
- **Tests**: Unit tests, integration tests, benchmarks
- **Bug Fixes**: Check the issue tracker
- **Performance**: Optimization improvements
- **Examples**: Real-world use cases

## Reporting Bugs

Use GitHub Issues with:
- **Title**: Clear summary
- **Description**: Steps to reproduce
- **Expected behavior**
- **Actual behavior**
- **Environment**: Python version, OS, etc.
- **Code sample**: Minimal reproduction

## Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Describe the problem you're solving
- Provide use cases
- Suggest implementation approach (optional)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Keep discussions professional

## Questions?

Feel free to open an issue for questions or join discussions.

Thank you for contributing!
