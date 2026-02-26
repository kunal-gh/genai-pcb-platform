# Contributing to AI-Powered PCB Design Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 16+
- Docker & Docker Compose
- Git

### Local Development

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-pcb-design.git
   cd ai-pcb-design
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start services**
   ```bash
   docker-compose up -d postgres redis
   ```

5. **Run the application**
   ```bash
   uvicorn src.main:app --reload
   ```

## Code Style

### Python

- Follow PEP 8 style guide
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use Black for code formatting
- Use isort for import sorting

```bash
# Format code
black src/
isort src/

# Check style
flake8 src/
mypy src/
```

### TypeScript/React

- Follow Airbnb style guide
- Use functional components with hooks
- Use TypeScript for type safety

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_falcon_gnn.py
```

### Writing Tests

- Write unit tests for all new functions
- Write integration tests for API endpoints
- Use property-based testing for ML components
- Aim for >80% code coverage

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes
   - `refactor:` Code refactoring
   - `test:` Test additions/changes
   - `chore:` Build process or auxiliary tool changes

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all tests pass
   - Request review from maintainers

## Project Structure

```
src/
├── api/              # FastAPI routes and schemas
├── models/           # Database models
├── services/         # Business logic
│   ├── falcon_gnn.py           # GNN implementation
│   ├── rl_routing_agent.py    # RL router
│   └── ...
└── training/         # ML training scripts
```

## ML Model Development

### Adding New Models

1. Create model file in `src/services/`
2. Add training script in `src/training/`
3. Update model registry
4. Add comprehensive tests
5. Document model architecture and usage

### Training Guidelines

- Use configuration files for hyperparameters
- Log training metrics to TensorBoard/Weights & Biases
- Save checkpoints regularly
- Document training procedures

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all functions and classes
- Update API documentation for endpoint changes
- Include examples for new features

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Suggestions for improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
