# Naming Conventions Guide

---
title: Naming Conventions Guide
type: guide
status: stable
created: 2024-02-06
tags:
  - conventions
  - naming
  - standards
  - organization
semantic_relations:
  - type: implements
    links: [[documentation_standards]]
  - type: relates
    links:
      - [[knowledge_organization]]
      - [[ai_file_organization]]
---

## Overview

This guide establishes comprehensive naming conventions for all components in our cognitive modeling framework, ensuring consistency and clarity across documentation, code, and resources.

## File Naming

### 1. Documentation Files
```python
# @doc_file_patterns
doc_patterns = {
    "concepts": {
        "pattern": "{concept_name}.md",
        "example": "active_inference.md",
        "rules": {
            "lowercase": True,
            "separators": "_",
            "max_length": 50
        }
    },
    "guides": {
        "pattern": "{category}_{topic}.md",
        "example": "ai_documentation_style.md",
        "rules": {
            "category_prefix": True,
            "descriptive_name": True
        }
    },
    "templates": {
        "pattern": "{type}_template.md",
        "example": "concept_template.md",
        "rules": {
            "template_suffix": True,
            "type_prefix": True
        }
    }
}
```

### 2. Code Files
```python
# @code_file_patterns
code_patterns = {
    "implementation": {
        "pattern": "{module}_{component}.py",
        "example": "belief_updater.py",
        "rules": {
            "lowercase": True,
            "descriptive": True,
            "max_length": 40
        }
    },
    "tests": {
        "pattern": "test_{module}_{feature}.py",
        "example": "test_belief_updating.py",
        "rules": {
            "test_prefix": True,
            "match_implementation": True
        }
    },
    "utilities": {
        "pattern": "{category}_utils.py",
        "example": "matrix_utils.py",
        "rules": {
            "utils_suffix": True,
            "category_prefix": True
        }
    }
}
```

## Component Naming

### 1. Class Names
See [[code_organization]] for implementation context.

```python
# @class_patterns
class_patterns = {
    "agents": {
        "pattern": "{Type}Agent",
        "example": "ActiveInferenceAgent",
        "rules": {
            "PascalCase": True,
            "descriptive_prefix": True,
            "agent_suffix": True
        }
    },
    "models": {
        "pattern": "{Type}Model",
        "example": "BeliefModel",
        "rules": {
            "PascalCase": True,
            "model_suffix": True
        }
    },
    "components": {
        "pattern": "{Role}{Type}",
        "example": "BeliefUpdater",
        "rules": {
            "PascalCase": True,
            "role_prefix": True
        }
    }
}
```

### 2. Method Names
```python
# @method_patterns
method_patterns = {
    "actions": {
        "pattern": "{verb}_{object}",
        "example": "update_beliefs",
        "rules": {
            "snake_case": True,
            "verb_first": True
        }
    },
    "properties": {
        "pattern": "{object}_{attribute}",
        "example": "belief_state",
        "rules": {
            "snake_case": True,
            "noun_first": True
        }
    },
    "callbacks": {
        "pattern": "on_{event}",
        "example": "on_belief_update",
        "rules": {
            "on_prefix": True,
            "event_focus": True
        }
    }
}
```

## Documentation Structure

### 1. Section Headers
```python
# @section_patterns
section_patterns = {
    "main_sections": {
        "pattern": "## {Category}",
        "example": "## Overview",
        "rules": {
            "title_case": True,
            "max_words": 3
        }
    },
    "subsections": {
        "pattern": "### {Number}. {Title}",
        "example": "### 1. Implementation Details",
        "rules": {
            "numbered": True,
            "title_case": True
        }
    }
}
```

### 2. Link References
See [[linking_patterns]] for detailed linking guidelines.

```python
# @link_patterns
link_patterns = {
    "internal": {
        "pattern": "[[{category}/{name}]]",
        "example": "[[concepts/active_inference]]",
        "rules": {
            "category_prefix": True,
            "lowercase_path": True
        }
    },
    "aliased": {
        "pattern": "[[{path}|{display}]]",
        "example": "[[active_inference|Active Inference]]",
        "rules": {
            "descriptive_alias": True,
            "consistent_display": True
        }
    }
}
```

## Metadata Conventions

### 1. YAML Frontmatter
```yaml
# @frontmatter_patterns
frontmatter:
  title:
    pattern: "{Type}: {Description}"
    example: "Guide: Active Inference Implementation"
    rules:
      - title_case: true
      - max_length: 60
  
  tags:
    pattern: ["{category}", "{subcategory}", "{specific}"]
    example: ["implementation", "active-inference", "agent"]
    rules:
      - lowercase: true
      - hyphen_separator: true
  
  semantic_relations:
    pattern:
      type: "{relationship_type}"
      links: ["[[{target}]]"]
    example:
      type: "implements"
      links: ["[[active_inference]]"]
```

### 2. Code Documentation
```python
# @docstring_patterns
docstring_patterns = {
    "class": {
        "pattern": """
        {Description}
        
        See [[{concept}]] for theoretical background.
        
        Attributes:
            {name} ({type}): {description}
        """,
        "rules": {
            "theoretical_link": True,
            "attribute_docs": True
        }
    },
    "method": {
        "pattern": """
        {Description}
        
        See [[{implementation}]] for details.
        
        Args:
            {name} ({type}): {description}
        
        Returns:
            {type}: {description}
        """,
        "rules": {
            "implementation_link": True,
            "complete_signature": True
        }
    }
}
```

## Validation Rules

### 1. Naming Validation
```python
# @validation_rules
validation_rules = {
    "files": {
        "pattern_compliance": 1.0,    # 100% compliance
        "length_limits": True,
        "character_set": "[a-z0-9_-]"
    },
    "components": {
        "case_compliance": 1.0,       # 100% compliance
        "prefix_suffix": True,
        "descriptive_names": True
    },
    "documentation": {
        "section_format": True,
        "link_format": True,
        "metadata_format": True
    }
}
```

### 2. Quality Checks
See [[quality_metrics]] for implementation.

```python
# @quality_metrics
naming_quality = {
    "consistency": {
        "pattern_adherence": 0.95,    # 95% pattern compliance
        "case_consistency": 1.0,      # 100% case consistency
        "separator_usage": 1.0        # 100% separator consistency
    },
    "clarity": {
        "descriptive_names": 0.9,     # 90% descriptive quality
        "length_compliance": 0.95,    # 95% length compliance
        "abbreviation_usage": 0.8     # 80% abbreviation compliance
    }
}
```

## Implementation Details

### 1. Name Processing
```python
# @name_processor
class NameProcessor:
    """
    Process and validate names according to conventions.
    See [[validation_framework]] for validation rules.
    """
    def __init__(self):
        self.validator = NameValidator()
        self.formatter = NameFormatter()
        self.analyzer = NameAnalyzer()
    
    def process_name(self, name: str, context: Context) -> ProcessedName:
        """
        Process and validate a name.
        See [[naming_conventions]] for rules.
        """
        # Analyze context
        pattern = self._get_pattern(context)
        rules = self._get_rules(context)
        
        # Format name
        formatted_name = self.formatter.format(name, pattern)
        
        # Validate
        validation_result = self.validator.validate(
            formatted_name,
            rules
        )
        
        # Analyze
        analysis = self.analyzer.analyze(formatted_name)
        
        return ProcessedName(
            original=name,
            formatted=formatted_name,
            validation=validation_result,
            analysis=analysis
        )
    
    def _get_pattern(self, context: Context) -> Pattern:
        """Get naming pattern for context."""
        if context.type == "file":
            return self._get_file_pattern(context)
        elif context.type == "class":
            return self._get_class_pattern(context)
        elif context.type == "method":
            return self._get_method_pattern(context)
        else:
            raise ValueError(f"Unknown context type: {context.type}")
    
    def _get_rules(self, context: Context) -> List[Rule]:
        """Get validation rules for context."""
        return [
            Rule(rule_type, params)
            for rule_type, params in context.rules.items()
        ]
```

### 2. Name Validation
```python
# @name_validator
class NameValidator:
    """
    Validate names against conventions.
    See [[validation_framework]] for rules.
    """
    def __init__(self):
        self.rules = self._load_rules()
        self.patterns = self._load_patterns()
        self.checkers = self._init_checkers()
    
    def validate(self, name: str, context: Context) -> ValidationResult:
        """
        Validate a name against conventions.
        See [[quality_metrics]] for criteria.
        """
        # Get applicable rules
        rules = self._get_applicable_rules(context)
        
        # Run validations
        results = []
        for rule in rules:
            result = self._check_rule(name, rule)
            results.append(result)
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _check_rule(self, name: str, rule: Rule) -> RuleResult:
        """Check a single naming rule."""
        checker = self.checkers.get(rule.type)
        if not checker:
            raise ValueError(f"No checker for rule type: {rule.type}")
        
        return checker.check(name, rule.params)
    
    def _aggregate_results(self, results: List[RuleResult]) -> ValidationResult:
        """Aggregate rule check results."""
        return ValidationResult(
            valid=all(r.valid for r in results),
            issues=[issue for r in results for issue in r.issues],
            score=sum(r.score for r in results) / len(results)
        )
```

### 3. Name Analysis
```python
# @name_analyzer
class NameAnalyzer:
    """
    Analyze names for quality and patterns.
    See [[quality_metrics]] for criteria.
    """
    def __init__(self):
        self.metrics = QualityMetrics()
        self.patterns = PatternMatcher()
        self.analyzer = TextAnalyzer()
    
    def analyze(self, name: str) -> AnalysisResult:
        """
        Analyze name quality and characteristics.
        See [[analysis_tools]] for methods.
        """
        # Quality metrics
        quality = self._analyze_quality(name)
        
        # Pattern matching
        patterns = self._match_patterns(name)
        
        # Text analysis
        text_analysis = self._analyze_text(name)
        
        return AnalysisResult(
            quality=quality,
            patterns=patterns,
            text_analysis=text_analysis
        )
    
    def _analyze_quality(self, name: str) -> QualityMetrics:
        """Analyze name quality."""
        return self.metrics.compute_metrics(name, {
            "length": self._check_length(name),
            "clarity": self._check_clarity(name),
            "consistency": self._check_consistency(name)
        })
    
    def _match_patterns(self, name: str) -> List[Pattern]:
        """Match name against known patterns."""
        return self.patterns.find_matches(name)
```

## Implementation Examples

### 1. Name Pattern Matching
```python
# @pattern_matcher
class NamePatternMatcher:
    """
    Pattern matching implementation for names.
    See [[validation_framework]] for pattern rules.
    """
    def match_pattern(self, name: str, context: Context) -> MatchResult:
        """
        Match name against patterns.
        
        Example:
            >>> matcher = NamePatternMatcher()
            >>> result = matcher.match_pattern("BeliefUpdater", context)
            >>> print(f"Pattern match: {result.pattern}")
            Pattern match: {Role}{Type}
        """
        # Get applicable patterns
        patterns = self._get_patterns(context)
        
        # Try each pattern
        matches = []
        for pattern in patterns:
            if match := self._try_pattern(name, pattern):
                matches.append(match)
                print(f"Matched pattern: {pattern.name}")
        
        # Find best match
        best_match = self._select_best_match(matches)
        if best_match:
            print(f"Selected pattern: {best_match.pattern}")
        
        return MatchResult(
            matches=matches,
            best_match=best_match,
            confidence=self._calculate_confidence(matches)
        )
    
    def _try_pattern(self, name: str, pattern: Pattern) -> Optional[Match]:
        """
        Try matching a specific pattern.
        
        Example patterns:
        - PascalCase: BeliefUpdater
        - snake_case: belief_updater
        - kebab-case: belief-updater
        """
        # Convert pattern to regex
        regex = self._pattern_to_regex(pattern)
        
        # Try matching
        if match := re.match(regex, name):
            return Match(
                pattern=pattern,
                groups=match.groupdict(),
                score=self._calculate_match_score(match)
            )
        return None
    
    def _calculate_match_score(self, match: re.Match) -> float:
        """
        Calculate pattern match score.
        
        Example scoring:
        - Full match: 1.0
        - Partial match: 0.5-0.9
        - Weak match: < 0.5
        """
        # Base score
        score = 1.0
        
        # Adjust for group completeness
        groups = match.groupdict()
        if missing := [k for k, v in groups.items() if not v]:
            score -= len(missing) * 0.1
        
        # Adjust for pattern complexity
        score *= len(groups) / 10 + 0.5
        
        return min(max(score, 0.0), 1.0)
```

### 2. Name Formatting
```python
# @name_formatter
class NameFormatter:
    """
    Name formatting implementation.
    See [[ai_documentation_style]] for formatting rules.
    """
    def format_name(self, name: str, style: Style) -> FormattedName:
        """
        Format name according to style.
        
        Example:
            >>> formatter = NameFormatter()
            >>> result = formatter.format_name("beliefUpdater", Style.SNAKE_CASE)
            >>> print(f"Formatted: {result.formatted}")
            Formatted: belief_updater
        """
        # Normalize name
        normalized = self._normalize_name(name)
        print(f"Normalized: {normalized}")
        
        # Apply style
        formatted = self._apply_style(normalized, style)
        print(f"Formatted: {formatted}")
        
        # Validate result
        validation = self._validate_formatting(formatted, style)
        if not validation.is_valid:
            print(f"Validation issues: {validation.issues}")
        
        return FormattedName(
            original=name,
            normalized=normalized,
            formatted=formatted,
            validation=validation
        )
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize name for formatting.
        
        Example transformations:
        - beliefUpdater -> ['belief', 'updater']
        - belief_updater -> ['belief', 'updater']
        - belief-updater -> ['belief', 'updater']
        """
        # Split on case boundaries
        parts = re.findall('[A-Z][a-z]*|[a-z]+', name)
        
        # Handle special characters
        parts = [p.strip('_-') for p in parts]
        
        # Filter empty parts
        parts = [p for p in parts if p]
        
        return parts
    
    def _apply_style(self, parts: List[str], style: Style) -> str:
        """
        Apply naming style to parts.
        
        Example styles:
        - PascalCase: BeliefUpdater
        - camelCase: beliefUpdater
        - snake_case: belief_updater
        - kebab-case: belief-updater
        """
        if style == Style.PASCAL_CASE:
            return ''.join(p.capitalize() for p in parts)
        elif style == Style.CAMEL_CASE:
            return parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])
        elif style == Style.SNAKE_CASE:
            return '_'.join(p.lower() for p in parts)
        elif style == Style.KEBAB_CASE:
            return '-'.join(p.lower() for p in parts)
        else:
            raise ValueError(f"Unknown style: {style}")
```

### 3. Name Suggestion
```python
# @name_suggester
class NameSuggester:
    """
    Name suggestion implementation.
    See [[naming_conventions]] for suggestion rules.
    """
    def suggest_names(self, name: str, context: Context) -> Suggestions:
        """
        Generate name suggestions.
        
        Example:
            >>> suggester = NameSuggester()
            >>> suggestions = suggester.suggest_names("blf_upd", context)
            >>> print(f"Suggestions: {suggestions.alternatives}")
            Suggestions: ['BeliefUpdater', 'belief_updater']
        """
        # Analyze name
        analysis = self._analyze_name(name)
        print(f"Name analysis: {analysis.summary}")
        
        # Generate alternatives
        alternatives = self._generate_alternatives(analysis, context)
        print(f"Generated {len(alternatives)} alternatives")
        
        # Rank suggestions
        ranked = self._rank_suggestions(alternatives, context)
        print(f"Top suggestion: {ranked[0]}")
        
        return Suggestions(
            original=name,
            alternatives=ranked,
            analysis=analysis
        )
    
    def _analyze_name(self, name: str) -> NameAnalysis:
        """
        Analyze name for suggestion generation.
        
        Example analysis:
        - Abbreviations: blf -> belief
        - Word boundaries: blf_upd -> belief_update
        - Common patterns: mgr -> manager
        """
        # Detect abbreviations
        abbrevs = self._detect_abbreviations(name)
        
        # Find word boundaries
        boundaries = self._find_word_boundaries(name)
        
        # Match common patterns
        patterns = self._match_common_patterns(name)
        
        return NameAnalysis(
            abbreviations=abbrevs,
            boundaries=boundaries,
            patterns=patterns
        )
    
    def _generate_alternatives(self, analysis: NameAnalysis, context: Context) -> List[str]:
        """
        Generate alternative names.
        
        Example generations:
        - Expand abbreviations
        - Apply naming patterns
        - Consider context
        """
        alternatives = []
        
        # Expand abbreviations
        if analysis.abbreviations:
            alternatives.extend(self._expand_abbreviations(analysis))
        
        # Apply patterns
        alternatives.extend(self._apply_patterns(analysis, context))
        
        # Consider context
        alternatives.extend(self._context_based_suggestions(analysis, context))
        
        return alternatives
    
    def _rank_suggestions(self, suggestions: List[str], context: Context) -> List[str]:
        """
        Rank name suggestions.
        
        Example ranking criteria:
        - Pattern match score
        - Context relevance
        - Clarity score
        """
        scored_suggestions = [
            (suggestion, self._score_suggestion(suggestion, context))
            for suggestion in suggestions
        ]
        
        # Sort by score
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return [s[0] for s in scored_suggestions]
```

## Integration Components

### 1. IDE Integration
```python
# @ide_integration
class IDEIntegration:
    """
    IDE integration for naming conventions.
    See [[cursor_integration]] for details.
    """
    def __init__(self):
        self.processor = NameProcessor()
        self.suggester = NameSuggester()
        self.formatter = NameFormatter()
    
    def process_identifier(self, name: str, context: Context) -> Suggestions:
        """
        Process identifier in IDE.
        See [[ide_plugins]] for implementation.
        """
        # Validate name
        result = self.processor.process_name(name, context)
        
        # Generate suggestions
        if not result.validation.valid:
            suggestions = self.suggester.suggest(name, context)
            return Suggestions(
                original=name,
                suggestions=suggestions,
                reason=result.validation.issues
            )
        
        return Suggestions(original=name, suggestions=[])
```

### 2. Git Integration
```python
# @git_integration
class GitIntegration:
    """
    Git hooks for naming convention enforcement.
    See [[git_workflow]] for details.
    """
    def __init__(self):
        self.validator = NameValidator()
        self.reporter = ValidationReporter()
    
    def pre_commit_hook(self, changes: List[Change]) -> HookResult:
        """
        Validate names in changed files.
        See [[git_hooks]] for implementation.
        """
        # Collect names
        names = self._collect_names(changes)
        
        # Validate names
        results = [
            self.validator.validate(name, self._get_context(name))
            for name in names
        ]
        
        # Generate report
        report = self.reporter.generate_report(results)
        
        return HookResult(
            valid=all(r.valid for r in results),
            report=report
        )
```

### 3. Documentation Integration
```python
# @doc_integration
class DocumentationIntegration:
    """
    Documentation system integration.
    See [[documentation_standards]] for guidelines.
    """
    def __init__(self):
        self.processor = NameProcessor()
        self.linker = DocumentationLinker()
        self.validator = DocumentationValidator()
    
    def process_documentation(self, doc: Document) -> ProcessedDocument:
        """
        Process names in documentation.
        See [[documentation_guide]] for rules.
        """
        # Extract names
        names = self._extract_names(doc)
        
        # Process names
        processed_names = [
            self.processor.process_name(name, self._get_context(name))
            for name in names
        ]
        
        # Update documentation
        updated_doc = self._update_documentation(doc, processed_names)
        
        # Validate links
        self.validator.validate_links(updated_doc)
        
        return updated_doc
```

## Best Practices

### 1. General Guidelines
- Use descriptive names
- Maintain consistent patterns
- Follow case conventions
- Limit name length

### 2. Documentation
- Follow [[documentation_standards]]
- Use [[ai_documentation_style]]
- Implement [[linking_patterns]]
- Validate with [[quality_metrics]]

### 3. Code Style
- Follow [[code_organization]]
- Use [[implementation_patterns]]
- Maintain [[code_quality]]
- Check [[style_guide]]

## Related Documentation
- [[documentation_standards]]
- [[knowledge_organization]]
- [[code_organization]]
- [[style_guide]]

## References
- [[implementation_patterns]]
- [[quality_metrics]]
- [[validation_framework]]
- [[code_quality]] 