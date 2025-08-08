---

title: Learning Paths Index

type: index

status: stable

created: 2024-02-07
modified: 2025-08-08

tags:

  - learning

  - paths

  - index

semantic_relations:

  - type: organizes

    links:

      - [[active_inference]]

      - [[pomdp_framework]]

      - [[swarm_intelligence]]

---

# Learning Paths Index

## Core Learning Paths

### Theoretical Foundations

- [[learning_paths/active_inference|Active Inference]]

- [[learning_paths/free_energy_principle|Free Energy Principle]]

- [[learning_paths/predictive_processing|Predictive Processing]]

### Implementation Frameworks

- [[learning_paths/pomdp_framework|POMDP Framework]]

- [[learning_paths/continuous_time|Continuous Time Framework]]

- [[learning_paths/swarm_intelligence|Swarm Intelligence]]

### Mathematical Foundations

- [[learning_paths/probability_theory|Probability Theory]]

- [[learning_paths/information_theory|Information Theory]]

- [[learning_paths/variational_methods|Variational Methods]]

## Advanced Paths

### Advanced Theory

- [[learning_paths/hierarchical_models|Hierarchical Models]]

- [[learning_paths/multi_agent_systems|Multi-Agent Systems]]

- [[learning_paths/complex_systems|Complex Systems]]

### Advanced Implementation

- [[learning_paths/advanced_pomdp|Advanced POMDP]]

- [[learning_paths/advanced_active_inference|Advanced Active Inference]]

- [[learning_paths/advanced_swarm|Advanced Swarm Systems]]

### Advanced Mathematics

- [[learning_paths/differential_geometry|Differential Geometry]]

- [[learning_paths/category_theory|Category Theory]]

- [[learning_paths/information_geometry|Information Geometry]]

## Specialization Paths

### Application Domains

- [[learning_paths/robotics|Robotics]]

- [[learning_paths/cognitive_systems|Cognitive Systems]]

- [[learning_paths/social_systems|Social Systems]]

### Research Areas

- [[learning_paths/computational_neuroscience|Computational Neuroscience]]

- [[learning_paths/artificial_life|Artificial Life]]

- [[learning_paths/evolutionary_systems|Evolutionary Systems]]

### Development Focus

- [[learning_paths/software_engineering|Software Engineering]]

- [[learning_paths/system_design|System Design]]

- [[learning_paths/performance_optimization|Performance Optimization]]

## Integration Paths

### Framework Integration

- [[learning_paths/pytorch_development|PyTorch Development]]

- [[learning_paths/tensorflow_development|TensorFlow Development]]

- [[learning_paths/jax_development|JAX Development]]

### Tool Integration

- [[learning_paths/visualization_tools|Visualization Tools]]

- [[learning_paths/analysis_tools|Analysis Tools]]

- [[learning_paths/development_tools|Development Tools]]

### System Integration

- [[learning_paths/environment_integration|Environment Integration]]

- [[learning_paths/hardware_integration|Hardware Integration]]

- [[learning_paths/cloud_deployment|Cloud Deployment]]

## Project Paths

### Basic Projects

- [[learning_paths/basic_agent|Basic Agent Development]]

- [[learning_paths/simple_environment|Simple Environment]]

- [[learning_paths/basic_simulation|Basic Simulation]]

### Advanced Projects

- [[learning_paths/ant_colony|Ant Colony Simulation]]

- [[learning_paths/robot_control|Robot Control]]

- [[learning_paths/multi_agent|Multi-Agent System]]

### Research Projects

- [[learning_paths/research_implementation|Research Implementation]]

- [[learning_paths/experiment_design|Experiment Design]]

- [[learning_paths/result_analysis|Result Analysis]]

## Learning Resources

### Documentation

- [[docs/guides/documentation|Documentation Guide]]

- [[docs/guides/api_documentation|API Documentation]]

- [[docs/guides/example_documentation|Example Documentation]]

### Tutorials

- [[tutorials/getting_started|Getting Started]]

- [[tutorials/basic_concepts|Basic Concepts]]

- [[tutorials/advanced_topics|Advanced Topics]]

### Examples

- [[examples/implementation|Implementation Examples]]

- [[examples/applications|Application Examples]]

- [[examples/research|Research Examples]]

## Assessment and Progress

### Knowledge Assessment

- [[assessment/theoretical_knowledge|Theoretical Knowledge]]

- [[assessment/practical_skills|Practical Skills]]

- [[assessment/research_capabilities|Research Capabilities]]

### Project Evaluation

- [[evaluation/implementation_quality|Implementation Quality]]

- [[evaluation/documentation_quality|Documentation Quality]]

- [[evaluation/research_quality|Research Quality]]

### Progress Tracking

- [[progress/learning_objectives|Learning Objectives]]

- [[progress/skill_development|Skill Development]]

- [[progress/project_completion|Project Completion]]

## Related Resources

### Knowledge Base

- [[knowledge_base/cognitive/learning_theory|Learning Theory]]

- [[knowledge_base/cognitive/skill_acquisition|Skill Acquisition]]

- [[knowledge_base/cognitive/expertise_development|Expertise Development]]

### Implementation Guides

- [[guides/implementation/best_practices|Best Practices]]

- [[guides/implementation/patterns|Implementation Patterns]]

- [[guides/implementation/optimization|Optimization Techniques]]

### Research Resources

- [[research/methodologies|Research Methodologies]]

- [[research/tools|Research Tools]]

- [[research/publications|Research Publications]]

## Global Quick Start

- Read a foundations trio:

  - Friston (2010) “The free-energy principle: a unified brain theory?” in Nature Reviews Neuroscience. See recommended readings at the Free Energy Principle hub: [activeinference.github.io](https://activeinference.github.io/)

  - Parr, Pezzulo, Friston (2022) “Active Inference” (Cambridge University Press). Book resources: [tejparr.github.io/Resources](https://tejparr.github.io/Resources.html)

  - Smith et al. (2022) “A step-by-step tutorial on Active Inference and its application to empirical data” (open access): [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8956124/)

- Do a hands-on notebook with a minimal agent (discrete-state) using `pymdp` docs: [pymdp RTD](https://pymdp-rtd.readthedocs.io/en/master/notebooks/active_inference_from_scratch.html)

- Join ongoing talks for context/mechanics: Active Inference Institute livestreams: [activeinference.institute/livestreams](https://www.activeinference.institute/livestreams)

### Repo-integrated quick runners

```bash
python3 /home/trim/Documents/GitHub/cognitive/Things/Generic_POMDP/generic_pomdp.py
python3 /home/trim/Documents/GitHub/cognitive/Things/BioFirm/active_inference/dispatcher.py
python3 /home/trim/Documents/GitHub/cognitive/Things/Ant_Colony/ant_colony/main.py --config /home/trim/Documents/GitHub/cognitive/Things/Ant_Colony/config/colony_config.yaml
python3 -m pytest /home/trim/Documents/GitHub/cognitive/tests/visualization/test_continuous_generic.py -q
```

## Centralized External Web Resources

### Foundations and Overviews

- Active Inference Institute (resources, education): [activeinference.org](https://www.activeinference.org/research/resources)

- Free Energy Principle recommended papers: [activeinference.github.io](https://activeinference.github.io/)

- “Active Inference: Demystified and Compared” (overview): [arXiv 1909.10863](https://arxiv.org/abs/1909.10863)

### Tutorials and Code

- Step-by-step tutorial (Smith et al., 2022): [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8956124/)

- IC2S2 Active Inference ABM tutorial (slides + code): [GitHub](https://github.com/apashea/IC2S2-Active-Inference-Tutorial)

- `pymdp` documentation: [ReadTheDocs](https://pymdp-rtd.readthedocs.io/en/master/notebooks/active_inference_from_scratch.html)

### Domain Gateways (by path)

- Robotics: survey and challenges: [arXiv 2112.01871](https://arxiv.org/abs/2112.01871); ROS 2 docs: [docs.ros.org](https://docs.ros.org/); Gazebo/Ignition: [gazebosim.org](https://gazebosim.org/)

- Social science ABM: Mesa docs: [mesa.readthedocs.io](https://mesa.readthedocs.io/); NetLogo: [ccl.northwestern.edu/netlogo](https://ccl.northwestern.edu/netlogo/); JASSS: [jasss.soc.surrey.ac.uk](https://jasss.soc.surrey.ac.uk/); ComSES: [comses.net](https://www.comses.net/)

- Security and governance: NIST AI RMF: [nist.gov/ai/rmf](https://www.nist.gov/itl/ai-risk-management-framework); EU AI Act (EUR-Lex): [eur-lex.europa.eu](https://eur-lex.europa.eu/)

- Quantum: Qiskit docs: [qiskit.org/documentation](https://qiskit.org/documentation/); Quantum cognition review (Pothos & Busemeyer, 2013): [APA/doi or summaries](https://psycnet.apa.org/record/2013-06753-001)

- Ecological and climate: Ecological Forecasting Initiative: [ecoforecast.org](https://ecoforecast.org/); NASA Earthdata: [earthdata.nasa.gov](https://www.earthdata.nasa.gov/); IPCC reports: [ipcc.ch/reports](https://www.ipcc.ch/reports/)

- Spatial web/XR: OpenXR spec (Khronos): [khronos.org/openxr](https://www.khronos.org/openxr/); WebXR MDN: [developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API); OGC standards: [ogc.org](https://www.ogc.org/)

- Edge: TinyML Foundation: [tinyml.org](https://www.tinyml.org/); NVIDIA Jetson docs: [developer.nvidia.com/embedded](https://developer.nvidia.com/embedded)

- Underwriting and risk: Society of Actuaries: [soa.org](https://www.soa.org/); CAS: [casact.org](https://www.casact.org/); NAIC model laws: [content.naic.org](https://content.naic.org/)

- Simulation/ML tooling: Gymnasium: [gymnasium.farama.org](https://gymnasium.farama.org/); PettingZoo: [pettingzoo.farama.org](https://pettingzoo.farama.org/); SimPy: [simpy.readthedocs.io](https://simpy.readthedocs.io/); PyTorch: [pytorch.org](https://pytorch.org/); JAX: [jax.readthedocs.io](https://jax.readthedocs.io/); NumPyro: [num.pyro.ai](https://num.pyro.ai/en/stable/)

