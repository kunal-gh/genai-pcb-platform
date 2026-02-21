# Contributing to stuff-made-easy

Thank you for your interest in contributing to Stuff-made-easy! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- KiCad 7.0+
- Git

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/stuff-made-easy.git`
3. Set up the development environment:
   ```bash
   cd stuff-made-easy
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Start the development services:
   ```bash
   docker-compose up -d postgres redis
   python -m uvicorn src.main:app --reload
   ```

## 📋 Development Workflow

### Branch Naming Convention
- `feature/task-number-description` - New features
- `bugfix/issue-description` - Bug fixes
- `docs/update-description` - Documentation updates
- `refactor/component-name` - Code refactoring

### Commit Message Format
We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(nlp): add natural language prompt validation
fix(skidl): handle missing component libraries gracefully
docs(api): update endpoint documentation
test(verification): add property tests for DFM validation
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Property-based tests
pytest tests/property/ -v --hypothesis-show-statistics

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=src --cov-report=html
```

### Writing Tests

#### Unit Tests
- Test specific functions and classes in isolation
- Use descriptive test names: `test_parse_prompt_with_valid_input_returns_structured_json`
- Mock external dependencies (LLM APIs, file system, etc.)
- Test both success and failure scenarios

#### Property-Based Tests
- Use Hypothesis for testing universal properties
- Focus on invariants that should hold for all valid inputs
- Include shrinking strategies for complex data types
- Tag tests with the property they validate:
  ```python
  # Feature: genai-pcb-platform, Property 1: Natural Language Parsing Completeness
  @given(valid_prompts())
  def test_prompt_parsing_completeness(prompt):
      # Test implementation
  ```

#### Integration Tests
- Test complete workflows end-to-end
- Use realistic test data
- Verify file generation and validation
- Test error handling and recovery

### Test Data
- Store test fixtures in `tests/fixtures/`
- Use factory_boy for generating test objects
- Include sample prompts, component data, and expected outputs

## 📝 Code Style Guidelines

### Python Code Style
- Follow PEP 8 with line length of 88 characters (Black default)
- Use type hints for all function parameters and return values
- Write Google-style docstrings for all public functions and classes
- Use meaningful variable and function names

### Code Quality Tools
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### Pre-commit Hooks
Install pre-commit hooks to automatically run quality checks:
```bash
pre-commit install
```

## 🏗️ Architecture Guidelines

### Service Design
- Follow single responsibility principle
- Use dependency injection for testability
- Implement proper error handling and logging
- Design for horizontal scalability

### API Design
- Follow RESTful conventions
- Use Pydantic models for request/response validation
- Include comprehensive OpenAPI documentation
- Implement proper HTTP status codes

### Database Design
- Use SQLAlchemy ORM with proper relationships
- Include database migrations with Alembic
- Design for performance with appropriate indexes
- Follow normalization principles

## 📚 Documentation

### Code Documentation
- Write clear docstrings for all public APIs
- Include usage examples in docstrings
- Document complex algorithms and business logic
- Keep README files updated

### API Documentation
- Use OpenAPI/Swagger specifications
- Include request/response examples
- Document error responses and status codes
- Provide integration examples

## 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce the problem
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages
- Minimal code example if applicable

Use the bug report template in GitHub Issues.

## 💡 Feature Requests

For new features:
- Check existing issues and discussions first
- Provide clear use case and motivation
- Consider implementation complexity
- Discuss API design implications
- Include mockups or examples if relevant

## 🔍 Code Review Process

### Submitting Pull Requests
1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass and code quality checks succeed
4. Update documentation if needed
5. Submit a pull request with clear description

### Review Criteria
- Code follows style guidelines and best practices
- Adequate test coverage for new functionality
- Documentation is updated appropriately
- No breaking changes without discussion
- Performance implications are considered

### Review Process
- All PRs require at least one approval
- Automated checks must pass
- Address reviewer feedback promptly
- Squash commits before merging when appropriate

## 🏷️ Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are bumped
- [ ] Release notes are prepared
- [ ] Security review is completed

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different perspectives and experiences

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Discord: Real-time chat and community support
- Email: Security issues and private matters

## 📞 Getting Help

If you need help:
1. Check the documentation and FAQ
2. Search existing GitHub issues
3. Ask in GitHub Discussions
4. Join our Discord community
5. Contact maintainers directly for sensitive issues

## 🙏 Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Annual contributor highlights

Thank you for contributing to stuff-made-easy! 🎉