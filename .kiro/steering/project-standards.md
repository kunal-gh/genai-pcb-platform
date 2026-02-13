---
inclusion: auto
description: Development standards, code quality requirements, testing guidelines, and best practices for the GenAI PCB Design Platform
---

# Project Development Standards

## Code Quality Standards

### Python Code Standards
- **Type Hints**: All functions must include type hints for parameters and return values
- **Docstrings**: Use Google-style docstrings for all classes and functions
- **Linting**: Code must pass flake8, black, and mypy checks
- **Import Organization**: Use isort for consistent import ordering
- **Error Handling**: Explicit exception handling with custom exception classes

### Testing Requirements
- **Coverage**: Minimum 80% code coverage for all modules
- **Property-Based Testing**: Use Hypothesis for universal property validation
- **Unit Tests**: pytest with fixtures for component isolation
- **Integration Tests**: End-to-end pipeline testing with realistic data
- **Performance Tests**: Benchmark critical paths (LLM calls, file generation)

### Git Workflow
- **Branch Naming**: `feature/task-number-description`, `bugfix/issue-description`
- **Commit Messages**: Conventional commits format: `type(scope): description`
- **Pull Requests**: Require code review and passing CI checks
- **Documentation**: Update relevant docs with each feature addition

## Architecture Principles

### Microservices Design
- **Single Responsibility**: Each service handles one domain area
- **API-First**: Define OpenAPI specs before implementation
- **Async Processing**: Use message queues for long-running operations
- **Stateless Services**: Enable horizontal scaling and fault tolerance

### Data Management
- **Schema Versioning**: Use Alembic for database migrations
- **Data Validation**: Pydantic models for all API interfaces
- **Caching Strategy**: Redis for frequently accessed data
- **Backup Strategy**: Automated backups with point-in-time recovery

### Security Standards
- **Authentication**: JWT tokens with refresh mechanism
- **Authorization**: Role-based access control (RBAC)
- **Data Encryption**: Encrypt sensitive data at rest and in transit
- **Input Validation**: Sanitize all user inputs to prevent injection attacks
- **Audit Logging**: Log all security-relevant actions

## Performance Guidelines

### Response Time Targets
- **API Endpoints**: < 200ms for simple operations
- **Design Generation**: < 60 seconds for basic circuits
- **File Downloads**: < 5 seconds for typical file sizes
- **UI Interactions**: < 100ms for user feedback

### Scalability Requirements
- **Concurrent Users**: Support 10+ users in MVP, 100+ in production
- **Auto-scaling**: Kubernetes HPA based on CPU and memory usage
- **Load Balancing**: Distribute requests across service instances
- **Resource Monitoring**: Prometheus metrics with Grafana dashboards

## Documentation Standards

### Code Documentation
- **API Documentation**: Auto-generated from OpenAPI specs
- **Architecture Diagrams**: Keep C4 model diagrams updated
- **Deployment Guides**: Step-by-step production deployment instructions
- **Troubleshooting**: Common issues and resolution procedures

### User Documentation
- **Getting Started**: Quick start guide for new users
- **API Reference**: Complete endpoint documentation with examples
- **Integration Guides**: How to integrate with external tools
- **FAQ**: Address common user questions and use cases

## Monitoring and Observability

### Logging Standards
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: Appropriate use of DEBUG, INFO, WARN, ERROR
- **Sensitive Data**: Never log passwords, tokens, or personal information
- **Retention**: 30-day retention for application logs

### Metrics Collection
- **Business Metrics**: Design success rate, user engagement, error rates
- **Technical Metrics**: Response times, throughput, resource utilization
- **Custom Metrics**: Domain-specific measurements (DFM pass rate, etc.)
- **Alerting**: Proactive alerts for critical issues and degraded performance

## Deployment and Operations

### CI/CD Pipeline
- **Automated Testing**: Run full test suite on every commit
- **Security Scanning**: Vulnerability scanning for dependencies
- **Code Quality Gates**: Block deployment if quality standards not met
- **Deployment Automation**: Zero-downtime deployments with rollback capability

### Environment Management
- **Development**: Local development with Docker Compose
- **Staging**: Production-like environment for integration testing
- **Production**: High-availability setup with monitoring and alerting
- **Configuration**: Environment-specific config via environment variables