# Contributing to LLM R&D Platform

We love your input! We want to make contributing to the LLM R&D Platform as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git

### Local Development
```bash
# Clone your fork
git clone https://github.com/yourusername/LLM-RnD.git
cd LLM-RnD

# Set up backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up frontend
cd web_interface/frontend
npm install
cd ../..

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Run backend
python run_api.py

# Run frontend (in another terminal)
cd web_interface/frontend
npm start
```

## Code Style

### Python
- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters (Black formatter)
- Use docstrings for all functions and classes

```python
def example_function(param: str, optional_param: int = 10) -> dict:
    """
    Brief description of the function.
    
    Args:
        param: Description of param
        optional_param: Description of optional param
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param is invalid
    """
    pass
```

### TypeScript/React
- Use TypeScript strict mode
- Follow ESLint configuration
- Use functional components with hooks
- Use proper prop types

```typescript
interface ComponentProps {
  title: string;
  optional?: boolean;
}

const ExampleComponent: React.FC<ComponentProps> = ({ title, optional = false }) => {
  return <div>{title}</div>;
};
```

## Testing

### Backend Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_api_integration.py -v
```

### Frontend Tests
```bash
cd web_interface/frontend

# Run tests
npm test

# Run with coverage
npm test -- --coverage --watchAll=false
```

### Integration Tests
```bash
# Run end-to-end tests
python -m pytest tests/test_end_to_end.py -v
```

## Code Quality Tools

### Python
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy .
```

### TypeScript
```bash
cd web_interface/frontend

# Lint
npm run lint

# Format
npm run format

# Type check
npm run type-check
```

## Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(api): add cost tracking endpoint
fix(frontend): resolve authentication token refresh
docs(readme): update installation instructions
test(evaluation): add unit tests for metrics calculation
```

## Issue and Bug Reports

Use GitHub issues to track bugs. Write bug reports with detail, background, and sample code.

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

### Bug Report Template
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. macOS, Ubuntu]
 - Browser [e.g. chrome, safari]
 - Version [e.g. 22]

**Additional context**
Add any other context about the problem here.
```

## Feature Requests

Use GitHub issues to suggest new features.

### Feature Request Template
```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Documentation

### API Documentation
- Update OpenAPI spec in `docs/api_specification.yaml`
- Add examples for new endpoints
- Document authentication requirements

### User Documentation
- Update relevant guides in `docs/`
- Include screenshots for UI changes
- Update setup instructions if needed

### Code Documentation
- Add docstrings to all new functions/classes
- Update type hints
- Add inline comments for complex logic

## Release Process

### Version Numbering
We use [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git
- [ ] Deploy to staging
- [ ] Deploy to production

## Security

### Reporting Security Issues
Please do not report security vulnerabilities through public GitHub issues.

Instead, please send an email to security@yourcompany.com with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Best Practices
- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all user inputs
- Use HTTPS in production
- Keep dependencies updated

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

### Communication
- Use clear, concise language
- Be patient with questions
- Provide helpful feedback on pull requests
- Celebrate contributions and achievements

## Getting Help

### Documentation
- [Setup Guide](SETUP_GUIDE.md)
- [Testing Guide](TESTING_GUIDE.md)
- [API Documentation](docs/api_documentation.html)

### Community
- GitHub Discussions for questions
- Discord for real-time chat
- Stack Overflow with tag `llm-rnd-platform`

### Maintainers
- @maintainer1 - Backend/API
- @maintainer2 - Frontend/UI
- @maintainer3 - ML/Evaluation

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Annual contributor highlights

Thank you for contributing to the LLM R&D Platform! 🚀