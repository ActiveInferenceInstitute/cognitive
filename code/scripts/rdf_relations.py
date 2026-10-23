"""Explicit source-to-target meanings for the cognitive RDF relation profile.

These describe author assertions between topics, not verified mechanisms, running
software, causal facts or OWL class axioms. Unknown/mixed-role assertions remain
document assertions and never acquire a SKOS super-property.
"""

TOPIC_PAGE_TYPES = frozenset(
    {
        "concept",
        "mathematical_concept",
        "biological_concept",
        "cognitive_concept",
        "knowledge_base",
        "biology",
        "mathematics",
        "philosophy",
        "systems",
        "system",
        "ontology",
        "article",
        "application",
        "integration",
        "research",
        "state_space",
    }
)
DOCUMENT_PAGE_TYPES = frozenset(
    {
        "agents",
        "index",
        "catalog",
        "overview",
        "implementation",
        "agent_config",
        "guide",
        "matrix_spec",
        "citation",
        "citation_redirect",
        "resource",
        "documentation",
        "implementation_guide",
        "glossary",
        "roadmap",
        "meta",
        "visualization",
        "verification",
        "learning_path",
        "example",
        "reference",
        "redirect",
    }
)
DOCUMENT_RELATIONS = {
    "documents": "Source document documents material presented in the target document.",
    "documented_by": "Source document names the target document as its documentation.",
    "documented_in": "Source document names the target document as its documentation location.",
    "defines": "Source document lists the target document under its definitions.",
    "guides": "Source document lists the target document as guidance material.",
    "maps": "Source document cross-references the target document in a source-authored map.",
    "aggregates": "Source document aggregates or indexes the target document.",
    "explains": "Source document offers an explanation of material in the target document.",
    "demonstrates": "Source document offers an example of material in the target document.",
    "visualizes": "Source document offers a visualization of material in the target document.",
}
CONCEPT_RELATIONS = {
    "foundation_for": "Source topic provides a conceptual foundation for the target topic.",
    "foundational_for": "Source topic provides a conceptual foundation for the target topic.",
    "foundational_to": "Source topic provides a conceptual foundation for the target topic.",
    "underpins": "Source topic provides a conceptual basis for the target topic.",
    "foundation": "Source topic draws its conceptual foundation from the target topic.",
    "mathematical_basis": "Source topic uses the target topic as its mathematical basis.",
    "foundational_theory": "Source topic draws on the target foundational theory.",
    "builds_on": "Source topic builds conceptually on the target topic.",
    "grounded_in": "Source topic is conceptually grounded in the target topic.",
    "implements": (
        "Source method or design concept implements the target specification or theory concept; "
        "no executable or real-world instance is asserted."
    ),
    "implemented_by": (
        "Source specification or theory concept is implemented conceptually by the target "
        "method or design; no executable instance is asserted."
    ),
    "extends": "Source topic extends the formulation or scope of the target topic.",
    "generalizes": "Source topic generalizes the target topic; no SKOS hierarchy is inferred.",
    "specializes": "Source topic specializes the target topic; no SKOS hierarchy is inferred.",
    "formalized_by": "Source topic uses the target topic as a formalization.",
    "integrates": "Source topic integrates the target topic into its conceptual formulation.",
    "influences": (
        "Source author asserts that the source topic influences the target topic; "
        "causation is not independently verified."
    ),
    "supports": "Source author asserts conceptual support from source topic to target topic.",
    "specifies": "Source topic supplies a specification for the target topic.",
    "uses": "Source topic uses the target method or conceptual resource.",
    "used_by": "Source topic is used by the target method or conceptual resource.",
    "applies_to": "Source topic is presented as applicable to the target topic.",
    "applied_in": "Source topic is presented as applied in the target topic.",
    "applications": "Source topic lists the target topic as an application.",
    "examples": "Source topic lists the target topic as a conceptual example, not an OWL instance.",
    "includes": "Source topic includes the target topic in its conceptual scope.",
    "operates_on": "Source method concept operates on the target conceptual structure.",
    "connects": "Source topic provides a conceptual connection to the target topic.",
    "provides": "Source topic provides a conceptual resource for the target topic.",
    "contrasts_with": "Source topic is contrasted with the target topic by the source author.",
    "prerequisite_for": "Source topic is a conceptual prerequisite for the target topic.",
    "prerequisite": "Source topic requires the target topic as a conceptual prerequisite.",
    "emerges_from": (
        "Source topic is described as emerging from the target topic; "
        "no instance-level causation is asserted."
    ),
    "analyzes": "Source method concept analyzes the target topic.",
    "part_of": "Source topic is a conceptual part of the target topic, not document containment.",
    "component_of": (
        "Source topic is a conceptual component of the target topic, not document containment."
    ),
}
