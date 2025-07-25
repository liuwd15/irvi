# Contributing to IRVI

We welcome contributions to IRVI! This document provides guidelines for contributing.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/irvi.git
   cd irvi
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black irvi/`
- Use isort for import sorting: `isort irvi/`
- Use flake8 for linting: `flake8 irvi/`

### Testing

- Write tests for new functionality
- Run tests with: `pytest tests/`
- Ensure all tests pass before submitting a PR

### Documentation

- Add docstrings to all public functions and classes
- Update README.md if needed
- Include examples for new features

## Submitting Changes

1. Create a new branch for your feature: `git checkout -b feature-name`
2. Make your changes and commit them with descriptive messages
3. Push to your fork: `git push origin feature-name`
4. Submit a pull request with a clear description of changes

## Reporting Issues

Use GitHub Issues to report bugs or request features. Include:
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)
- Minimal code example

## Questions

For questions about using IRVI, please:
- Check the documentation first
- Search existing issues
- Open a new issue with the "question" label

Thank you for contributing!
