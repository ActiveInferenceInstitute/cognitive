---
title: Contribution Guide
type: guide
status: stable
created: 2024-02-28
tags:
  - development
  - contribution
  - guidelines
  - documentation
semantic_relations:
  - type: guides
    links:
      - [[development_workflow]]
      - [[code_standards]]
      - [[testing_guide]]
      - [[documentation_standards]]
---

# Contribution Guide

This guide outlines the process and standards for contributing to our cognitive modeling framework. We welcome contributions from the community and aim to make the process as clear and straightforward as possible.

## Getting Started

### 1. Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/yourusername/cognitive.git
cd cognitive
```

3. Set up development environment:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

### 2. Development Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Run tests:
```bash
pytest tests/
```

4. Commit changes:
```bash
git add .
git commit -m "feat: description of your changes"
```

5. Push to your fork:
```bash
git push origin feature/your-feature-name
```

6. Create a pull request

## Code Standards

### 1. Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good example
def compute_free_energy(
    observations: np.ndarray,
    beliefs: Dict[str, Distribution],
    model: GenerativeModel
) -> float:
    """Compute variational free energy.
    
    Args:
        observations: Observed data
        beliefs: Current belief state
        model: Generative model
        
    Returns:
        float: Computed free energy
    """
    # Implementation
    return free_energy
```

### 2. Documentation Standards

1. **Docstrings**
   - Use Google style
   - Include types
   - Describe parameters
   - Note exceptions
   - Provide examples

2. **Comments**
   - Explain why, not what
   - Keep current
   - Use complete sentences

3. **Markdown**
   - Use headers properly
   - Include code examples
   - Add cross-references

### 3. Testing Standards

1. **Unit Tests**
```python
def test_free_energy_computation():
    # Arrange
    model = create_test_model()
    observations = generate_test_data()
    
    # Act
    result = compute_free_energy(observations, model)
    
    # Assert
    assert np.isclose(result, expected_value)
```

2. **Integration Tests**
```python
def test_agent_learning():
    # Setup environment
    env = TestEnvironment()
    agent = ActiveInferenceAgent()
    
    # Run interaction
    for _ in range(100):
        action = agent.act()
        obs = env.step(action)
        agent.update(obs)
    
    # Verify learning
    assert agent.performance() > threshold
```

## Pull Request Process

### 1. Preparation

- Update documentation
- Add tests
- Run linters
- Check coverage

### 2. Submission

1. **Title Format**
   - feat: New feature
   - fix: Bug fix
   - docs: Documentation
   - test: Testing
   - refactor: Code refactoring

2. **Description Template**
```markdown
## Description
Brief description of changes

## Changes
- Detailed list of changes
- With specific components

## Testing
- How to test changes
- Test coverage details

## Documentation
- Updated documentation
- New documentation
```

### 3. Review Process

1. **Initial Checks**
   - CI passes
   - Coverage maintained
   - Style guide followed

2. **Code Review**
   - Functionality
   - Performance
   - Security
   - Maintainability

3. **Documentation Review**
   - Completeness
   - Clarity
   - Examples
   - Cross-references

## Development Best Practices

### 1. Code Organization

```python
# Module structure
cognitive/
  ├── agents/
  │   ├── __init__.py
  │   ├── base.py
  │   └── active_inference.py
  ├── models/
  │   ├── __init__.py
  │   └── generative.py
  └── inference/
      ├── __init__.py
      └── message_passing.py
```

### 2. Error Handling

```python
class CognitiveError(Exception):
    """Base class for framework exceptions."""
    pass

class InferenceError(CognitiveError):
    """Raised when inference fails."""
    pass

def safe_inference(model, data):
    try:
        return model.infer(data)
    except ValueError as e:
        raise InferenceError(f"Inference failed: {e}")
```

### 3. Performance Considerations

- Use vectorized operations
- Implement caching where appropriate
- Profile critical sections
- Consider memory usage

## Community Guidelines

### 1. Communication

- Be respectful
- Stay professional
- Focus on code
- Welcome newcomers

### 2. Issue Reporting

```markdown
## Issue Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS:
- Python version:
- Package version:
```

### 3. Feature Requests

```markdown
## Feature Description
Clear description of proposed feature

## Use Case
Why this feature is needed

## Implementation Ideas
Possible approaches to implement

## Alternatives Considered
Other options and why they were rejected
```

## License and Copyright

- MIT License
- Copyright notices required
- Third-party licenses respected

## Questions and Support

- GitHub Issues for bugs
- Discussions for questions
- Stack Overflow for help
- Email for private matters

## Additional Resources

- [[docs/development/style_guide|Style Guide]]
- [[docs/development/testing_guide|Testing Guide]]
- [[docs/development/documentation_guide|Documentation Guide]]
- [[docs/development/release_process|Release Process]] 