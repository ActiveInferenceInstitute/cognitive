# Git Workflow Guide

## Overview
This guide outlines our Git workflow for managing cognitive modeling projects, integrating with both [[cursor_integration]] and [[obsidian_usage]].

## Repository Structure

### Core Branches
- `main` - Stable production code
- `develop` - Integration branch
- `feature/*` - Feature branches
- `release/*` - Release preparation
- `hotfix/*` - Production fixes

## Workflow Patterns

### Feature Development
1. Create feature branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/new-feature
   ```

2. Develop with [[cursor_integration]]:
   - Use AI assistance
   - Follow [[code_style]]
   - Update [[documentation_guide]]

3. Commit changes:
   ```bash
   git add .
   git commit -m "feat: add new feature
   
   - Added X functionality
   - Updated Y components
   - Related to #issue"
   ```

### Knowledge Base Changes

#### Working with Obsidian
1. Create knowledge branch:
   ```bash
   git checkout -b kb/topic-name
   ```

2. Update content:
   - Follow [[obsidian_usage]] guidelines
   - Maintain [[linking_patterns]]
   - Update [[node_types]]

3. Commit changes:
   ```bash
   git add knowledge_base/
   git commit -m "kb: update topic structure
   
   - Added new nodes
   - Updated relationships
   - Fixed broken links"
   ```

## Commit Conventions

### Message Structure
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `kb`: Knowledge base
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

### Scopes
- `agent`: Agent-related changes
- `belief`: Belief network updates
- `model`: Model implementation
- `docs`: Documentation
- `kb`: Knowledge base

## Integration Practices

### With Obsidian
- Track `.obsidian/` selectively
- Include templates
- Exclude personal settings
- See [[obsidian_usage]]

### With Cursor
- Include `.cursorrules`
- Track AI configurations
- See [[cursor_integration]]

## Branching Strategy

### Feature Branches
- Branch from: `develop`
- Merge to: `develop`
- Naming: `feature/description`

### Knowledge Base Branches
- Branch from: `main`
- Merge to: `main`
- Naming: `kb/topic`

### Release Branches
- Branch from: `develop`
- Merge to: `main` and `develop`
- Naming: `release/version`

## Code Review

### Process
1. Create Pull Request
2. Use [[cursor_integration]] for review
3. Check [[code_style]] compliance
4. Verify [[testing_guide]] coverage
5. Update [[documentation_guide]]

### Checklist
- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Knowledge base links valid
- [ ] CI/CD passes

## Conflict Resolution

### Strategy
1. Keep local branch updated:
   ```bash
   git fetch origin
   git rebase origin/develop
   ```

2. Resolve conflicts:
   - Check [[linking_patterns]]
   - Maintain [[node_types]] integrity
   - Preserve knowledge structure

## Best Practices

### Repository Maintenance
- Regular cleanup
- Archive old branches
- Update documentation
- Follow [[maintenance_guide]]

### Knowledge Management
- Consistent structure
- Clear relationships
- Updated metadata
- See [[obsidian_usage]]

### Collaboration
- Clear communication
- Regular updates
- Proper documentation
- Follow [[workflow_guides]]

## Automation

### Git Hooks
```bash
#!/bin/sh
# pre-commit hook
npm run lint
npm run test
```

### CI/CD Integration
- Automated testing
- Documentation generation
- Knowledge base validation
- See [[deployment_guide]]

## Troubleshooting

### Common Issues
- Merge conflicts
- Lost changes
- Broken links
See [[troubleshooting]] for solutions.

## Related Guides
- [[development_setup]]
- [[workflow_guides]]
- [[deployment_guide]]

## References
- [Git Documentation](https://git-scm.com/doc)
- [[code_organization]]
- [[contribution_guide]] 