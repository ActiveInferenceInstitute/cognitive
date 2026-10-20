---
title: Source-reviewed mapping curation, 2026-10-20
description: Dated agent review of ontology correspondence candidates across the cognitive knowledge base, including retained mappings and explicit scope-based deferrals.
type: reference
status: stable
created: 2026-10-20
updated: 2026-10-20
tags:
  - knowledge_graphs
  - ontology_alignment
  - provenance
---

# Source-reviewed mapping curation, 2026-10-20

This is a dated implementation-agent review, **not independent human
curation**. It compares source-page openings, mathematical definitions and
declared scope with official target definitions and statements. Mappings
apply to the pages' conceptual subjects under the
[vocabulary contract](vocabulary.md), not to document identity or to
particular chemical, anatomical or process instances.

The first section follows every entry in the
[incoming-reference queue](alignment-review.md) as generated at cognitive
revision `f2d6ea06f096d3aff810feb8d082a7a261613302`. Later sections record
the wider priority batch. A retained record was rechecked against its source
page and current official target. A deferred decision asserts no new
mapping: a search result, alias, shared subject or absent search hit is not
proof of equivalence or of global absence from all ontologies.

## Evidence and scope policy

ChEBI, GO and STATO definitions and obsolete/replacement flags were read
through the official EMBL-EBI OLS API. ChEBI records also have direct
chemical-entry links below. NCI definitions, active status and terminology
version were read through the official EVS REST API. Every ChEBI, GO and
STATO term selected for a new assertion was non-obsolete with no replacement
reported at retrieval; every selected NCI concept was active in version
`26.09d`. Wikidata evidence came from its official entity JSON, including
English descriptions, aliases, typed statements and revision identifiers.
Wikidata does not expose an OBO-style obsolete flag; requested identifiers
resolved to the same identifiers in these captures. Description quality
varies, so aliases and broad type labels alone do not justify new matches.

The [VI author review](https://arxiv.org/abs/1601.00670v9) supplements the
short [variational Bayesian methods target](https://www.wikidata.org/wiki/Q7915794):
optimization-based distribution approximation is broader than the
specifically Bayesian target. Full raw-response SHA-256 values, source
URLs, retrieval dates, extracted definitions/status and the structured
decision manifest are retained in the companion local run directory
`daf-okn/output/graph-methods-20261020/curation-sources/`, supplied with the
implementation report. That directory is not presumed published. This
curation page's direct official links provide the public evidence.

`exactMatch` compares denotations of the core subject; it does not endorse
all explanatory claims in either page. In particular, a GO match does not
validate a Free Energy Principle account of the process, and a chemical
match does not validate claims about behavior, treatment or signalling.
`broadMatch` points to a broader target; `narrowMatch` points to a narrower
target. `relatedMatch` is associative and supplies no substitution rule.
No `owl:sameAs`, `owl:equivalentClass` or class-as-instance assertion is
introduced.

The named-molecule pages receive explicit definitions and scope notes
because their body discussions and aliases also mention whole signalling
systems. Their correspondences concern the chemical subject, not the
system, assay, precursor, receptor or drug product. Existing scientific
page bodies and discovery identifiers are preserved.

## Incoming-reference queue

| Source page and scope | Decision and official comparison | Remaining boundary |
|---|---|---|
| [`cognitive/active_inference`](../../knowledge_base/cognitive/active_inference.md): Perception, learning and policy selection by variational and expected free-energy minimization. | Retain existing reviewed record. Q17014702 describes Friston's neuroscience hypothesis, the free energy principle. [official evidence](https://www.wikidata.org/wiki/Q17014702). | Retain relatedMatch: a framework based on a principle is not identical to the principle. |
| [`cognitive/free_energy_principle`](../../knowledge_base/cognitive/free_energy_principle.md): A variational-free-energy account of persistence across self-organizing biological systems. | Retain existing reviewed record. Q17014702 states a neuroscience hypothesis attributed to Friston. [official evidence](https://www.wikidata.org/wiki/Q17014702). | Retain closeMatch: the page generalizes beyond the target's neuroscience description. |
| [`cognitive/precision_weighting`](../../knowledge_base/cognitive/precision_weighting.md): Dynamic weighting of prediction errors by their reliability in hierarchical inference. | Defer; no mapping asserted. STATO:0000416 defines precision as positive predictive value; Wikidata search returned papers rather than a definition of this mechanism. [official evidence 1](https://www.ebi.ac.uk/ols4/api/ontologies/stato/terms?obo_id=STATO:0000416), [official evidence 2](https://www.wikidata.org/w/api.php?action=wbsearchentities&search=precision%20weighting&language=en&format=json&limit=3). | Defer: positive predictive value is not inverse uncertainty or dynamic error weighting; an article about a mechanism is not the mechanism. |
| [`mathematics/information_theory`](../../knowledge_base/mathematics/information_theory.md): Mathematical quantification, storage and communication of information, with Shannon entropy and mutual information. | Retain existing reviewed record. Q131222 identifies the mathematical theory within probability and statistics. [official evidence](https://www.wikidata.org/wiki/Q131222). | Retain exactMatch for the field; applications to physical or cognitive systems do not equate it with those systems. |
| [`cognitive/predictive_coding`](../../knowledge_base/cognitive/predictive_coding.md): Hierarchical brain prediction and error correction, including belief updating. | Retain existing reviewed record. Q1315146 is explicitly a psychological term, distinct from engineering compression uses. [official evidence](https://www.wikidata.org/wiki/Q1315146). | Retain the existing exactMatch; the target's short description alone does not establish that every theory called predictive processing is identical. |
| [`cognitive/predictive_processing`](../../knowledge_base/cognitive/predictive_processing.md): A brain-function theory using ongoing sensory predictions and prediction errors for perception and learning. | Defer; no mapping asserted. Q1315146 lists predictive processing as an alias of predictive coding but defines it only as a psychological term. [official evidence](https://www.wikidata.org/wiki/Q1315146). | Defer a new mapping: an alias supplies discovery, not a definition resolving theory versus coding mechanism. |
| [`cognitive/attention_mechanisms`](../../knowledge_base/cognitive/attention_mechanisms.md): Neural and cognitive mechanisms for selective processing, suppression, resource allocation and control. | `relatedMatch` [Q6501338](https://www.wikidata.org/wiki/Q6501338). Q6501338 defines selective concentration while ignoring other information; NCIT:C41209 describes concentration on one object. [official evidence 1](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C41209?include=full). | relatedMatch to attention: mechanisms implement the attentional function; they are not interchangeable with the function or a concentration task. |
| [`cognitive/decision_making`](../../knowledge_base/cognitive/decision_making.md): Cognitive and neural selection between alternative actions using evidence and values. | Retain existing reviewed record. Q1331926 defines the cognitive process of choosing an action or belief. [official evidence](https://www.wikidata.org/wiki/Q1331926). | Retain exactMatch for decision making; the page's neural accounts are explanatory context. |
| [`mathematics/free_energy_principle`](../../knowledge_base/mathematics/free_energy_principle.md): A variational-free-energy formulation of biological self-organization and persistence. | Retain existing reviewed record. Q17014702 describes Friston's hypothesis in neuroscience. [official evidence](https://www.wikidata.org/wiki/Q17014702). | Retain closeMatch because the page includes a wider biological and mathematical scope. |
| [`cognitive/social_cognition`](../../knowledge_base/cognitive/social_cognition.md): Processing used to understand, interact with and learn from other people. | Retain existing reviewed record. Q1432778 describes information processing about social situations. [official evidence](https://www.wikidata.org/wiki/Q1432778). | Retain exactMatch for social cognition, without identifying it with any individual neural mechanism. |
| [`cognitive/working_memory`](../../knowledge_base/cognitive/working_memory.md): Temporary maintenance and manipulation of task-relevant information. | Retain existing reviewed record. Q11337567 defines limited-capacity temporary storage that makes information available for processing. [official evidence](https://www.wikidata.org/wiki/Q11337567). | Retain exactMatch; Active Inference formulas are an account of the same cognitive system, not target-endorsed theory. |
| [`cognitive/bayesian_inference`](../../knowledge_base/cognitive/bayesian_inference.md): Belief updating from evidence using Bayes' rule, with hierarchical cognitive applications. | Retain existing Wikidata record; add `closeMatch` [STATO_0000706](https://www.ebi.ac.uk/ols4/api/ontologies/stato/terms?obo_id=STATO:0000706). Q812535 defines Bayesian updating of hypothesis probability; STATO:0000706 defines prior-distribution updating for statistical parameters using Bayes' rule. [official evidence 1](https://www.wikidata.org/wiki/Q812535). | Retain Wikidata exactMatch; add STATO closeMatch because its parameter-inference paradigm is more explicit than the page's general cognitive belief updating. |
| [`mathematics/variational_inference`](../../knowledge_base/mathematics/variational_inference.md): Optimization-based approximation of probability distributions, including ELBO, KL divergence and Active Inference applications. | `narrowMatch` [Q7915794](https://arxiv.org/abs/1601.00670v9). Q7915794 denotes variational Bayesian methods; Blei, Kucukelbir and McAuliffe define VI as density approximation by optimization and discuss the Bayesian posterior case. [official evidence 1](https://www.wikidata.org/wiki/Q7915794). | narrowMatch: the Bayesian methods target is a narrower use of the source's distribution-approximation concept; neither is thermodynamic free energy. |
| [`mathematics/expected_free_energy`](../../knowledge_base/mathematics/expected_free_energy.md): A policy-dependent Active Inference objective combining preference-related risk and expected observation ambiguity. | Defer; no mapping asserted. Q17014702 denotes the free energy principle; the VI author review concerns distribution approximation, not this policy objective. [official evidence 1](https://www.wikidata.org/wiki/Q17014702), [official evidence 2](https://arxiv.org/abs/1601.00670v9). | Defer: the principle and a variational approximation method are not the expected-free-energy quantity; no reviewed external definition was found for the actual objective. |
| [`mathematics/dynamical_systems`](../../knowledge_base/mathematics/dynamical_systems.md): The mathematical theory of temporal evolution, including continuous vector fields, discrete maps and stability. | `exactMatch` [Q3174497](https://www.wikidata.org/wiki/Q3174497). Q3174497 defines the mathematics of dynamical-system behavior using differential or difference equations. | exactMatch to the theory; avoid Q638328, which denotes a dynamical system as the object studied. |
| [`mathematics/probability_theory`](../../knowledge_base/mathematics/probability_theory.md): Mathematical reasoning under uncertainty based on probability, measures and stochastic processes. | `exactMatch` [Q5862903](https://www.wikidata.org/wiki/Q5862903). Q5862903 defines the branch of mathematics concerning probability. | exactMatch to the field; not to a particular probability value, distribution or inference procedure. |
| [`cognitive/synaptic_plasticity`](../../knowledge_base/cognitive/synaptic_plasticity.md): Activity-dependent modification of synaptic strength and structure. | Retain existing reviewed record. Q1551556 describes activity-related strengthening and weakening of synapses. [official evidence](https://www.wikidata.org/wiki/Q1551556). | Retain the existing exactMatch for synaptic plasticity; do not map it to GO regulation-of-plasticity processes. The target summary emphasizes strength, while the source also discusses structural mechanisms. |
| [`cognitive/motor_control`](../../knowledge_base/cognitive/motor_control.md): Neural planning, execution and regulation of coordinated movement. | Retain existing reviewed record. Q2996165 defines movement regulation in organisms with a nervous system. [official evidence](https://www.wikidata.org/wiki/Q2996165). | Retain exactMatch; movement disorders and implementation mechanisms are contextual topics. |
| [`cognitive/neural_plasticity`](../../knowledge_base/cognitive/neural_plasticity.md): Experience-, learning- and injury-related change in brain structure and function. | `exactMatch` [Q849491](https://www.wikidata.org/wiki/Q849491). Q849491 defines the brain's ability to change throughout life. | exactMatch for the capacity; a process regulating synaptic plasticity would be a different target. |
| [`cognitive/reinforcement_learning`](../../knowledge_base/cognitive/reinforcement_learning.md): Optimization of cumulative reward using value functions and policy improvement, compared with Active Inference. | `exactMatch` [Q830687](https://www.wikidata.org/wiki/Q830687). Q830687 defines learning from action-dependent rewards or penalties to maximize cumulative reward. | exactMatch to reinforcement learning; the comparison with EFE does not identify reward and free-energy objectives. |
| [`cognitive/consciousness`](../../knowledge_base/cognitive/consciousness.md): Phenomenal experience, access consciousness and self-awareness, including altered states. | Defer; no mapping asserted. NCIT:C37929 defines being alert, oriented and responsive; Q7087 describes knowledge of internal/external existence and environment. [official evidence 1](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C37929?include=full), [official evidence 2](https://www.wikidata.org/wiki/Q7087). | Defer: clinical responsiveness and an epistemic description do not define all phenomenal and access phenomena on the page; no equivalence is established. |
| [`mathematics/differential_geometry`](../../knowledge_base/mathematics/differential_geometry.md): Geometry of manifolds, metrics, connections, curvature and differential forms, with cognitive applications. | `exactMatch` [Q188444](https://www.wikidata.org/wiki/Q188444). Q188444 defines functions and geometric structures on differentiable manifolds. | exactMatch to differential geometry; the applications do not identify it with information geometry or only Riemannian geometry. |
| [`cognitive/memory_systems`](../../knowledge_base/cognitive/memory_systems.md): Interacting neural mechanisms and cognitive systems for encoding, storing and retrieving information. | `relatedMatch` [Q492](https://www.wikidata.org/wiki/Q492). Q492 defines the mental faculties and processes of storing and retrieving information; NCIT:C37992 describes retaining and recalling past experience. [official evidence 1](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C37992?include=full). | relatedMatch: the page organizes the systems realizing memory; it is not simply a retained experience or the abstract capacity. |
| [`cognitive/prediction_error`](../../knowledge_base/cognitive/prediction_error.md): Discrepancy between predicted and actual outcomes across sensory, cognitive and reward domains. | Defer; no mapping asserted. OBI and STATO searches did not return a reviewed definition of this cross-domain cognitive quantity; Wikidata search returned articles about prediction errors. [official evidence 1](https://www.ebi.ac.uk/ols4/api/search?q=prediction%20error&ontology=obi&rows=5&queryFields=label%2Csynonym&obsoletes=false), [official evidence 2](https://www.wikidata.org/w/api.php?action=wbsearchentities&search=prediction%20error&language=en&format=json&limit=3). | Defer: a paper about prediction error or a specific statistical residual would not define the page's broader computational signal. |
| [`cognitive/swarm_intelligence`](../../knowledge_base/cognitive/swarm_intelligence.md): Emergent collective behavior of decentralized self-organized agents, in natural and computational systems. | `exactMatch` [Q863960](https://www.wikidata.org/wiki/Q863960). Q863960 defines collective behavior of decentralized, self-organized systems. | exactMatch to swarm intelligence; the target's loose smart-dust alias is not used as independent evidence. |

## Further mathematics

| Source page and scope | Decision and official comparison | Remaining boundary |
|---|---|---|
| [`mathematics/entropy`](../../knowledge_base/mathematics/entropy.md): Shannon entropy and its conditional and joint forms for discrete variables. | `exactMatch` [Q204570](https://www.wikidata.org/wiki/Q204570). Q204570 describes expected information delivered by a message. | exactMatch for information entropy; exclude thermodynamic entropy and generic disorder. |
| [`mathematics/mutual_information`](../../knowledge_base/mathematics/mutual_information.md): Dependence between random variables measured by uncertainty reduction and KL divergence from independent marginals. | `exactMatch` [Q252973](https://www.wikidata.org/wiki/Q252973). Q252973 defines a measure of dependence between two variables. | exactMatch for mutual information, not merely correlation or a generic information-flow relation. |
| [`mathematics/bayes_theorem`](../../knowledge_base/mathematics/bayes_theorem.md): The conditional-probability identity relating prior, likelihood, posterior and evidence. | `exactMatch` [Q182505](https://www.wikidata.org/wiki/Q182505). Q182505 defines Bayes' theorem as an event probability based on relevant prior conditions. | exactMatch to the theorem; do not equate the identity with the entire inference paradigm. |
| [`mathematics/markov_chain_monte_carlo`](../../knowledge_base/mathematics/markov_chain_monte_carlo.md): Sampling-based expectation approximation using a Markov chain with the desired invariant distribution. | `exactMatch` [Q1191869](https://www.wikidata.org/wiki/Q1191869). Q1191869 identifies a class of sampling and Monte Carlo algorithms that uses Markov chains. | exactMatch from the combined description and sampling/uses-Markov-chain statements, not the label alone; no STATO or EDAM equivalent was established. |
| [`mathematics/conditional_probability`](../../knowledge_base/mathematics/conditional_probability.md): Probability of an event given that another event occurred, with intersection-and-normalization formula. | `exactMatch` [Q327069](https://www.wikidata.org/wiki/Q327069). Q327069 defines event likelihood when another event is known to have occurred. | exactMatch; distinguish from conditional expectation or a conditional-probability table. |
| [`mathematics/probability_distributions`](../../knowledge_base/mathematics/probability_distributions.md): Categorical, Dirichlet and Gaussian specifications of outcome probabilities in discrete and continuous models. | `exactMatch` [STATO_0000225](https://www.ebi.ac.uk/ols4/api/ontologies/stato/terms?obo_id=STATO:0000225). STATO:0000225 defines a mathematical specification of a random variable's probabilities, using mass or density functions. | exactMatch at the distribution-concept level; no individual distribution instance is asserted. |

## Further biology and chemical messengers

| Source page and scope | Decision and official comparison | Remaining boundary |
|---|---|---|
| [`biology/serotonin`](../../knowledge_base/biology/serotonin.md): 5-hydroxytryptamine as a monoamine chemical messenger, emphasizing central nervous system roles. | `exactMatch` [CHEBI_28790](https://www.ebi.ac.uk/chebi/CHEBI:28790); `exactMatch` [NCIT_C828](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C828?include=full). CHEBI:28790 defines the 5-hydroxy derivative of tryptamine; NCIT:C828 denotes serotonin/5-hydroxytryptamine acting as a neurotransmitter and peripheral hormone. | Two independently compatible exactMatch records for the chemical concept; central signalling context does not map the whole serotonergic system. |
| [`biology/adrenaline`](../../knowledge_base/biology/adrenaline.md): Endogenous epinephrine secreted by adrenal-medullary chromaffin cells in acute stress. | `exactMatch` [CHEBI_28918](https://www.ebi.ac.uk/chebi/CHEBI:28918); `exactMatch` [NCIT_C2292](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2292?include=full). CHEBI:28918 defines the R-enantiomer secreted by adrenal glands; NCIT:C2292 is endogenous epinephrine, distinct from NCIT:C473 therapeutic epinephrine. | Two exactMatch records for endogenous adrenaline; exclude noradrenaline, racemic preparations and therapeutic products. |
| [`biology/homeostasis`](../../knowledge_base/biology/homeostasis.md): Biological maintenance of internal stability despite external perturbations. | `exactMatch` [GO_0042592](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0042592). GO:0042592 defines biological maintenance of an internal steady state despite varying external conditions. | exactMatch for the process; this does not validate the page's proposed Free Energy Principle explanation. |
| [`biology/gene_expression`](../../knowledge_base/biology/gene_expression.md): Use of gene sequence information to make a mature protein or functional RNA product. | `exactMatch` [GO_0010467](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0010467). GO:0010467 defines conversion of a gene sequence into a mature protein or RNA, including processing and maturation. | exactMatch; expression is not limited to transcription or protein synthesis alone. |
| [`biology/gene_regulation`](../../knowledge_base/biology/gene_regulation.md): Control of whether, when and how much a gene is expressed, at transcriptional and post-transcriptional steps. | `exactMatch` [GO_0010468](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0010468). GO:0010468 defines modulation of the frequency, rate or extent of gene expression, including regulation of intermediates. | exactMatch to regulation of gene expression; not to expression itself or only transcriptional control. |
| [`biology/morphogenesis`](../../knowledge_base/biology/morphogenesis.md): Generation of biological form and pattern during development. | `exactMatch` [GO_0009653](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0009653). GO:0009653 defines generation and organization of anatomical structures. | exactMatch for anatomical form generation; the FEP account is not an assertion made by GO. |
| [`biology/structural_plasticity`](../../knowledge_base/biology/structural_plasticity.md): Experience-related anatomical neural change, including synapse turnover, dendritic/axonal changes and limited neurogenesis. | `broadMatch` [Q849491](https://www.wikidata.org/wiki/Q849491). Q849491 covers brain change generally; GO:0051823 regulates synapse structural plasticity rather than denoting the broader structural capacity. [official evidence 1](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0051823). | broadMatch to neuroplasticity: the target includes functional as well as structural change; no identity with regulation of synapse remodelling. |
| [`biology/wound_healing`](../../knowledge_base/biology/wound_healing.md): Injury repair restoring tissue integrity, barrier function and structure through overlapping phases. | `exactMatch` [GO_0042060](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0042060). GO:0042060 defines events restoring integrity to damaged tissue following injury. | exactMatch; wound closure is one part rather than the complete process. |
| [`biology/biofilm_formation`](../../knowledge_base/biology/biofilm_formation.md): Microbial attachment, self-produced extracellular matrix, community maturation and dispersal. | `closeMatch` [GO_0042710](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0042710). GO:0042710 defines planktonic-to-matrix growth at liquid-air interfaces or solid substrates under liquid flow. | closeMatch: attachment and polymer-matrix formation agree, but GO's stated environmental conditions and the source's fuller life cycle differ. |
| [`biology/neuromuscular_junction`](../../knowledge_base/biology/neuromuscular_junction.md): The motor-neuron to skeletal-muscle-fibre synapse, with cholinergic transmission. | `broadMatch` [GO_0031594](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0031594). GO:0031594 defines motor-axon/muscle-fibre junctions and explicitly includes invertebrate inhibitory and excitatory cases with other transmitters. | broadMatch: GO includes junctions beyond the page's vertebrate skeletal-muscle emphasis; not a mapping to junction development. |
| [`biology/symbiosis`](../../knowledge_base/biology/symbiosis.md): Close prolonged association between unlike organisms, including mutualism, commensalism and parasitism. | `exactMatch` [Q121610](https://www.wikidata.org/wiki/Q121610). Q121610 defines close long-term biological interaction between different organisms; GO:0044403 concerns gene-product processes enabling that interaction. [official evidence 1](https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0044403). | exactMatch to the interaction concept in Wikidata; reject identity with the GO enabling process or mutualism alone. |
| [`cognitive/oxytocin`](../../knowledge_base/cognitive/oxytocin.md): The mature nine-amino-acid peptide acting as hormone and brain neuromodulator. | `exactMatch` [CHEBI_7872](https://www.ebi.ac.uk/chebi/CHEBI:7872); `exactMatch` [NCIT_C2283](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2283?include=full). CHEBI:7872 defines cyclic CYIQNCPLG peptide with uterine, milk-ejection and brain roles; NCIT:C2283 defines the nine-residue OXT product with mammary/uterine and bonding/cognition roles. [official evidence 1](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C724?include=full). | Two exactMatch records for the mature peptide; exclude gene, precursor, receptor and recombinant drug product C724. |
| [`cognitive/dopamine_system`](../../knowledge_base/cognitive/dopamine_system.md): Dopamine-synthesizing neurons, projection pathways and receptor systems across the brain. | Defer; no mapping asserted. CHEBI:18243 defines the dopamine chemical structure; Q170304 describes the chemical as hormone and neurotransmitter. [official evidence 1](https://www.ebi.ac.uk/chebi/CHEBI:18243), [official evidence 2](https://www.wikidata.org/wiki/Q170304). | Defer chemical equivalence: a neuronal/receptor system is not dopamine itself; no independently reviewed ontology definition of the whole system was established. |

## Further systems concepts

| Source page and scope | Decision and official comparison | Remaining boundary |
|---|---|---|
| [`systems/complex_systems`](../../knowledge_base/systems/complex_systems.md): An interdisciplinary theoretical treatment of interacting parts and emergent, nonlinear collective behavior. | `closeMatch` [Q2995634](https://www.wikidata.org/wiki/Q2995634). Q2995634 describes the study of interconnected systems using statistical physics, information theory and nonlinear dynamics. | closeMatch: the source emphasizes emergent-properties theory; the target describes the broader interdisciplinary study. |
| [`systems/dynamical_systems`](../../knowledge_base/systems/dynamical_systems.md): Dynamical systems theory applied to temporal behavior, stability, chaos and patterns in complex systems. | `exactMatch` [Q3174497](https://www.wikidata.org/wiki/Q3174497). Q3174497 defines the mathematical area describing complex-system behavior through differential or difference equations. | exactMatch to the theory; the source's application setting does not create identity with an individual system. |
| [`systems/cybernetics`](../../knowledge_base/systems/cybernetics.md): Study of regulation and purposeful behavior using feedback, information and communication. | `exactMatch` [Q123637](https://www.wikidata.org/wiki/Q123637). Q123637 defines communication and control based on regulatory feedback. | exactMatch to cybernetics; not all control systems or any particular feedback loop. |
| [`systems/guided_self_organization`](../../knowledge_base/systems/guided_self_organization.md): A design method that biases local self-organizing interactions through constraints, objectives and tuned rules. | `relatedMatch` [Q609408](https://www.wikidata.org/wiki/Q609408). Q609408 defines the process of creating order through local interactions. | relatedMatch: an engineering approach that guides a process is associated with, rather than identical to, the process. |
| [`systems/knowledge_graphs`](../../knowledge_base/systems/knowledge_graphs.md): Graph-structured information about entities, properties and relations, used for organization, query and reasoning. | `exactMatch` [Q33002955](https://www.wikidata.org/wiki/Q33002955). Q33002955 defines an information repository structured as an entity/relationship graph. | exactMatch to knowledge graphs generally; explicitly avoid Google's particular Knowledge Graph Q648625. |

## Further philosophy

| Source page and scope | Decision and official comparison | Remaining boundary |
|---|---|---|
| [`philosophy/philosophy_of_mind`](../../knowledge_base/philosophy/philosophy_of_mind.md): Philosophical study of mental phenomena and their relation to the physical world. | `exactMatch` [Q23407](https://www.wikidata.org/wiki/Q23407). Q23407 defines the branch studying the nature of mind. | exactMatch to the field; the source's FEP interpretation is one discussed position, not the target's definition. |
| [`philosophy/pragmatism`](../../knowledge_base/philosophy/pragmatism.md): A tradition relating meaning, truth and value to practical consequences, with classical and later variants. | Defer; no mapping asserted. Q126692 identifies only a philosophical tradition and school, without a substantive definition of practical-consequence criteria. [official evidence](https://www.wikidata.org/wiki/Q126692). | Defer: the official target description and type establish a tradition but do not resolve the page's semantic boundaries; keep its existing discovery link. |
| [`philosophy/enactivism`](../../knowledge_base/philosophy/enactivism.md): Embodied action and organism-environment interaction as constitutive of cognition, with autopoiesis and sense-making. | `closeMatch` [Q431627](https://www.wikidata.org/wiki/Q431627). Q431627 defines cognition arising through organism-environment interaction. | closeMatch: the shared constitutive-interaction account is explicit, but the source develops multiple autopoietic and philosophical commitments absent from the target summary. |
| [`philosophy/operationalism`](../../knowledge_base/philosophy/operationalism.md): Defining theoretical concepts through measurement and observation operations as a philosophical-methodological doctrine. | Defer; no mapping asserted. Q20826077 has an operationalism label and school/epistemology typing but no English definition; Q286017 describes operationalization as a measurement process. [official evidence 1](https://www.wikidata.org/wiki/Q20826077), [official evidence 2](https://www.wikidata.org/wiki/Q286017). | Defer: a doctrine is not a particular operationalization process, and a label without a definition does not establish the doctrine's scope. |

## Limits and follow-up

The review preserves existing correspondences rather than silently replacing
them. In particular, the pre-existing predictive-coding record has only a
short psychological-domain description on its official Wikidata item;
its retained status is not evidence that the separately reviewed
predictive-processing theory has the same scope. The retained synaptic
plasticity target's description emphasizes activity-related strength
changes, while the source also discusses structural mechanisms. These
differences are recorded for separate review.

Deferrals concern the specific targets and searches documented here.
They leave discovery links intact and can be revisited using a vocabulary
with a definition that resolves the stated boundary. They do not claim
that no suitable term exists anywhere. No equivalence is asserted for
precision weighting, expected free energy or a neuronal dopamine system
merely because a nearby statistical, physical or chemical term exists.

The [generated queue](alignment-review.md), [statistics](statistics.md),
RDF sample and remote join evidence must be regenerated together after
all concurrent source changes settle. This note records source decisions;
it is not a replacement for generated counts, submission checks or an
independent semantic review.
