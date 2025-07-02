# Contributing to SFunctor

We welcome contributions to SFunctor! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the issue tracker
2. Create a new issue with a clear title and description
3. Include:
   - Steps to reproduce the problem
   - Expected behavior
   - Actual behavior
   - System information (OS, Python version, etc.)

### Submitting Changes

1. Fork the repository
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Make your changes following the code style guidelines
4. Add tests for new functionality
5. Run the test suite to ensure nothing is broken
6. Commit your changes with clear, descriptive messages
7. Push to your fork and submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep line length under 88 characters (Black default)
- Use type hints where appropriate

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Test with different Python versions if possible

### Documentation

- Update docstrings for any changed functionality
- Update README.md if adding new features
- Add examples for new functionality
- Keep CLAUDE.md updated with development notes

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sfunctor.git
   cd sfunctor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

5. Run linting:
   ```bash
   ruff check .
   black --check .
   ```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the version number in setup.py following Semantic Versioning
3. The PR will be merged once it has been reviewed and approved

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

Thank you for contributing to SFunctor!