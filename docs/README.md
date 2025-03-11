# Cognitive Modeling Documentation

Welcome to the Cognitive Modeling documentation. This documentation is designed to be viewed in Obsidian for optimal navigation and knowledge linking.

> [!info] Documentation Network
> This documentation works in conjunction with the [[agents/agent_docs_readme|Autonomous Agent Documentation Clearinghouse]] which contains comprehensive agent-specific documentation.

## Directory Structure

```mermaid
graph TD
    A[Documentation Root] --> B[concepts/]
    A --> C[guides/]
    A --> D[api/]
    A --> E[examples/]
    A --> F[tools/]
    A --> G[templates/]
    A --> H[agents/]
    
    B --> B1[cognitive_modeling_concepts.md]
    B --> B2[active_inference.md]
    B --> B3[free_energy_principle.md]
    
    C --> C1[implementation_guides.md]
    C --> C2[installation_guide.md]
    C --> C3[quickstart_guide.md]
    
    D --> D1[api_documentation.md]
    
    E --> E1[usage_examples.md]
    E --> E2[quickstart_example.md]
    
    F --> F1[development_tools.md]
    
    G --> G1[documentation_templates.md]
    
    H --> H1[agent_docs_readme.md]
    
    style A fill:#f9d5e5,stroke:#333,stroke-width:1px
    style H fill:#d5f5e3,stroke:#333,stroke-width:2px
    style H1 fill:#d5f5e3,stroke:#333,stroke-width:2px
```

### Core Documentation
- [[concepts/cognitive_modeling_concepts|Concepts]] - Core concepts and theoretical foundations
- [[guides/implementation_guides|Guides]] - Implementation guides and tutorials
- [[api/api_documentation|API]] - API reference documentation
- [[examples/usage_examples|Examples]] - Usage examples and tutorials
- [[tools/development_tools|Tools]] - Development tools and utilities
- [[templates/documentation_templates|Templates]] - Documentation templates
- [[agents/agent_docs_readme|Agent Documentation]] - Autonomous agent documentation clearinghouse

## Getting Started

### Essential Setup
- [[guides/installation_guide|Installation Guide]]
- [[guides/configuration_guide|Configuration Guide]]
- [[guides/quickstart_guide|Quick Start Guide]]
- [[guides/environment_setup|Environment Setup]]
- [[guides/dependency_management|Dependency Management]]

### Core Concepts
- [[concepts/plain_text_benefits|Plain Text Benefits]]
- [[concepts/machine_readability|Machine Readability]]
- [[concepts/research_education|Research and Education]]
- [[concepts/active_inference|Active Inference]]
- [[concepts/free_energy_principle|Free Energy Principle]]
- [[concepts/predictive_processing|Predictive Processing]]
- [[concepts/variational_inference|Variational Inference]]
- [[concepts/message_passing|Message Passing]]

### Documentation Usage
- [[guides/obsidian_linking|Obsidian Linking Guide]]
- [[guides/package_documentation|Package Documentation]]
- [[guides/pomdp_structure|POMDP Structure]]
- [[guides/mermaid_diagrams|Mermaid Diagram Guide]]
- [[guides/documentation_navigation|Documentation Navigation]]

## Development

### Implementation
- [[guides/model_implementation|Model Implementation]]
- [[guides/system_integration|System Integration]]
- [[guides/testing_guide|Testing Guide]]
- [[guides/performance_optimization|Performance Optimization]]
- [[guides/agent_development|Agent Development]]
- [[guides/environment_creation|Environment Creation]]

### Architecture Overview

```mermaid
graph TB
    subgraph "Core Framework"
        A[Abstract Base Classes]
        B[Common Utilities]
        C[Math Libraries]
        D[Config Management]
    end
    
    subgraph "Agent Models"
        E[POMDP Agents]
        F[Continuous Agents]
        G[Hybrid Agents]
        H[LLM Agents]
    end
    
    subgraph "Environments"
        I[Simulation Environments]
        J[Real-world Interfaces]
        K[Benchmarks]
    end
    
    A --> E
    A --> F
    A --> G
    A --> H
    B --> E
    B --> F
    B --> G
    B --> H
    C --> E
    C --> F
    C --> G
    
    style A fill:#f9d5e5,stroke:#333,stroke-width:1px
    style B fill:#f9d5e5,stroke:#333,stroke-width:1px
    style C fill:#f9d5e5,stroke:#333,stroke-width:1px
    style D fill:#f9d5e5,stroke:#333,stroke-width:1px
    style E fill:#d5f5e3,stroke:#333,stroke-width:1px
    style F fill:#d5f5e3,stroke:#333,stroke-width:1px
    style G fill:#d5f5e3,stroke:#333,stroke-width:1px
    style H fill:#d5f5e3,stroke:#333,stroke-width:1px
```

### Tools
- [[tools/setup_tools|Setup Tools]]
- [[tools/development_tools|Development Tools]]
- [[tools/documentation_tools|Documentation Tools]]
- [[tools/experiment_tools|Experiment Tools]]
- [[tools/analysis_tools|Analysis Tools]]
- [[tools/visualization_tools|Visualization Tools]]
- [[tools/benchmark_tools|Benchmark Tools]]
- [[tools/simulation_tools|Simulation Tools]]

### Examples
- [[examples/quickstart_example|Quick Start Example]]
- [[examples/basic_agent|Basic Agent]]
- [[examples/active_inference_basic|Active Inference]]
- [[examples/multi_agent_system|Multi-Agent System]]
- [[examples/hierarchical_agent|Hierarchical Agent]]
- [[examples/belief_updating|Belief Updating Example]]
- [[examples/policy_selection|Policy Selection Example]]

## Contributing

### Guidelines
- [[guides/contribution_guide|Contribution Guide]]
- [[guides/documentation_guide|Documentation Guide]]
- [[guides/style_guide|Style Guide]]
- [[guides/code_standards|Code Standards]]
- [[guides/testing_guidelines|Testing Guidelines]]
- [[guides/review_process|Review Process]]

### Development Workflow

```mermaid
stateDiagram-v2
    [*] --> Development
    Development --> Testing
    Testing --> Review
    Review --> Refinement
    Refinement --> Testing
    Review --> Documentation
    Documentation --> Deployment
    Deployment --> [*]
```

### Templates
- [[templates/concept_template|Concept Template]]
- [[templates/guide_template|Guide Template]]
- [[templates/example_template|Example Template]]
- [[templates/api_template|API Documentation Template]]
- [[templates/implementation_template|Implementation Template]]
- [[templates/agent_template|Agent Template]]
- [[templates/research_template|Research Template]]

## Additional Resources

### References
- [[concepts/active_inference|Active Inference]]
- [[concepts/free_energy_principle|Free Energy Principle]]
- [[concepts/predictive_processing|Predictive Processing]]
- [[DOCUMENTATION_ROADMAP|Documentation Roadmap]]
- [[concepts/bibliography|Bibliography]]
- [[concepts/glossary|Glossary of Terms]]

### Research Topics
- [[research/current_projects|Current Research Projects]]
- [[research/publications|Publications]]
- [[research/experimental_results|Experimental Results]]
- [[research/benchmarks|Benchmark Results]]
- [[research/future_directions|Future Research Directions]]

### Related Projects
- [[related/similar_frameworks|Similar Frameworks]]
- [[related/complementary_tools|Complementary Tools]]
- [[related/partner_projects|Partner Projects]]
- [[related/community_extensions|Community Extensions]]

## Support

### Help
- [[guides/troubleshooting|Troubleshooting]]
- [[guides/faq|FAQ]]
- [[guides/support|Support]]
- [[guides/common_errors|Common Errors]]
- [[guides/performance_issues|Performance Issues]]

### Community
- [[guides/community_guide|Community Guide]]
- [[guides/discussion|Discussion]]
- [[guides/feedback|Feedback]]
- [[guides/feature_requests|Feature Requests]]
- [[guides/bug_reporting|Bug Reporting]]
- [[guides/community_extensions|Community Extensions]]

---

> [!tip] Navigation Tip
> Use Obsidian's graph view and search functionality to explore connections between concepts. The [[agents/agent_docs_readme|Agent Documentation Clearinghouse]] provides comprehensive details on agent implementations. 