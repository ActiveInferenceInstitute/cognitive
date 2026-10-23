"""Reviewed external subject terms shared by daf-okn and cognitive.

Exact source spellings are reviewed individually; unknown tags remain local.
These are document subject links, never class membership or identity assertions.
No files, network services or third-party packages are accessed during import.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import PurePosixPath
from urllib.parse import urlsplit

__all__ = ["KEYWORD_RECORDS", "KEYWORD_TERMS", "KeywordTerm", "build_keyword_terms", "keyword_iri"]


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"keyword {field} must be a nonempty, unpadded string")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"keyword {field} must not contain control characters")


def _http_iri(value: str) -> None:
    _text(value, "IRI")
    if re.search(r'[\s<>"{}|\\^`]', value) or re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise ValueError(f"invalid keyword IRI: {value!r}")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"keyword IRI must be absolute HTTP(S) without credentials: {value!r}")
    # Accessing port validates malformed and out-of-range port declarations.
    if parsed.port is not None:
        raise ValueError("keyword source IRIs must use their canonical port")


@dataclass(frozen=True)
class KeywordTerm:
    """One source-tag decision with official definition and review provenance.

    ``source_sha256`` binds the exact retained official response (the descriptor
    JSON for MeSH, full concept JSON for NCI, and pinned OWL release for STATO).
    MeSH evidence also names the preferred-concept response containing its scope.
    ``source_paths`` records inspected source contexts, not a complete usage list.
    Construction rejects malformed or inactive records before they enter lookup.
    """

    tag: str
    iri: str
    label: str
    definition: str
    evidence: tuple[str, ...]
    source_version: str
    source_sha256: str
    reviewed_on: str
    rationale: str
    source_paths: tuple[str, ...]
    active: bool

    def __post_init__(self) -> None:
        for field in ("tag", "label", "definition", "source_version", "rationale"):
            _text(getattr(self, field), field)
        _http_iri(self.iri)
        if self.active is not True:
            raise ValueError("keyword terms must have verified active status")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("keyword source_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.reviewed_on, str) or not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}", self.reviewed_on
        ):
            raise ValueError("keyword reviewed_on must be an ISO calendar date")
        date.fromisoformat(self.reviewed_on)
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("keyword evidence must be a nonempty tuple of official URLs")
        for url in self.evidence:
            _http_iri(url)
        ncit = re.fullmatch(r"http://purl.obolibrary.org/obo/NCIT_(C[0-9]+)", self.iri)
        mesh = re.fullmatch(r"http://id.nlm.nih.gov/mesh/(D[0-9]+)", self.iri)
        stato = re.fullmatch(r"http://purl.obolibrary.org/obo/STATO_[0-9]{7}", self.iri)
        if ncit:
            expected = f"https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/{ncit[1]}?include=full"
            if self.evidence != (expected,):
                raise ValueError("NCI keyword evidence must identify its exact official concept")
        elif mesh:
            expected = f"https://id.nlm.nih.gov/mesh/{mesh[1]}.json"
            if (
                len(self.evidence) != 2
                or self.evidence[0] != expected
                or not re.fullmatch(r"https://id.nlm.nih.gov/mesh/M[0-9]+\.json", self.evidence[1])
            ):
                raise ValueError(
                    "MeSH keyword evidence requires descriptor and preferred-concept URLs"
                )
        elif stato:
            if len(self.evidence) != 1 or not re.fullmatch(
                r"https://raw.githubusercontent.com/ISA-tools/stato/[0-9a-f]{40}/releases/latest_release/stato\.owl",
                self.evidence[0],
            ):
                raise ValueError("STATO keyword evidence must pin the official release commit")
        else:
            raise ValueError(
                "keyword target is outside the reviewed NCI, MeSH and STATO vocabularies"
            )
        if not isinstance(self.source_paths, tuple) or not self.source_paths:
            raise ValueError(
                "keyword source_paths must identify inspected knowledge-base documents"
            )
        if len(set(self.source_paths)) != len(self.source_paths):
            raise ValueError("duplicate keyword source context")
        for path in self.source_paths:
            _text(path, "source path")
            parts = PurePosixPath(path).parts
            if (
                not path.startswith("knowledge_base/")
                or not path.endswith(".md")
                or ".." in parts
                or "\\" in path
            ):
                raise ValueError(
                    "keyword source context must be a relative knowledge_base Markdown path"
                )


def build_keyword_terms(records: Iterable[KeywordTerm]) -> dict[str, str]:
    """Validate a reviewed registry and reject duplicate or conflicting tag records.

    Args:
        records: Explicit source-tag decisions, already validated at construction.
    Returns:
        Exact tag-to-IRI lookup; different reviewed spellings may share a target.
    Raises:
        TypeError: If a record is not a ``KeywordTerm``.
        ValueError: If any source spelling occurs twice, even with the same target.
    """
    terms: dict[str, str] = {}
    for record in records:
        if not isinstance(record, KeywordTerm):
            raise TypeError("keyword registry records must be KeywordTerm instances")
        if record.tag in terms:
            raise ValueError(f"duplicate keyword source tag: {record.tag!r}")
        terms[record.tag] = record.iri
    return terms


def keyword_iri(tag: str) -> str | None:
    """Return a reviewed external subject IRI for the exact source spelling.

    Args:
        tag: Original tag; no case, punctuation, whitespace or fuzzy normalization.
    Returns:
        The reviewed IRI, or ``None`` so the exporter can retain a local keyword.
    Raises:
        TypeError: If ``tag`` is not a string.
    """
    if not isinstance(tag, str):
        raise TypeError("keyword tag must be a string")
    return KEYWORD_TERMS.get(tag)


# Each record preserves its review rationale, source contexts and official bytes.
KEYWORD_RECORDS: tuple[KeywordTerm, ...] = (
    KeywordTerm(
        tag="acetylcholine",
        iri="http://purl.obolibrary.org/obo/NCIT_C77840",
        label="Acetylcholine",
        definition=(
            "A chemical made by some types of nerve cells. It is used to send messages "
            "to other cells, including other nerve cells, muscle cells, and gland "
            "cells. It is released from the nerve ending and carries signals to cells "
            "on the other side of a synapse (space between nerve cells and other "
            "cells). Acetylcholine helps control memory and the action of certain "
            "muscles. It is a type of neurotransmitter."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C77840?include=full",),
        source_version="26.09d",
        source_sha256="826dd1e277b8059e1b80a2889658002afb1e9ac91b2d84f06d6db93e99ced183",
        reviewed_on="2026-10-23",
        rationale=(
            "The neurotransmitter substance discussed in cholinergic signaling; not an "
            "acetylcholine measurement or administration."
        ),
        source_paths=(
            "knowledge_base/biology/neuromuscular_junction.md",
            "knowledge_base/cognitive/cholinergic_system.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="algorithms",
        iri="http://purl.obolibrary.org/obo/NCIT_C16275",
        label="Algorithm",
        definition=(
            "A defined procedure for solving a problem. Applied to a problem-solving "
            "procedure implemented in software to be executed by a computer."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16275?include=full",),
        source_version="26.09d",
        source_sha256="03ed0ce3b841d04bb39681ddba19d9b599ac8df0df54f657eb30bf7dfcf6f130",
        reviewed_on="2026-10-23",
        rationale=(
            "Defined computational problem-solving procedures in mathematical methods "
            "and implementations; plural source spelling is explicit."
        ),
        source_paths=(
            "knowledge_base/biology/mathematical_models_social_insects.md",
            "knowledge_base/cognitive/inference_patterns.md",
            "knowledge_base/cognitive/optimization_patterns.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="amnesia",
        iri="http://purl.obolibrary.org/obo/NCIT_C2867",
        label="Amnesia",
        definition=(
            "Systematic and extensive loss of memory caused by organic or psychological"
            " factors. The loss may be temporary or permanent, and may involve old or "
            "recent memories."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2867?include=full",),
        source_version="26.09d",
        source_sha256="9f95167b4bb267009c5fa2151ebe235af4097d9b9eaf1f7cc241f8bb30388c93",
        reviewed_on="2026-10-23",
        rationale=(
            "Loss of memory as discussed by cognitive disorder sources; not an "
            "adverse-event reporting grade."
        ),
        source_paths=(
            "knowledge_base/cognitive/anterograde_amnesia.md",
            "knowledge_base/cognitive/korsakoff_syndrome.md",
            "knowledge_base/cognitive/retrograde_amnesia.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="amygdala",
        iri="http://purl.obolibrary.org/obo/NCIT_C12440",
        label="Amygdala",
        definition=(
            "A cluster of nuclei located deep and medially within each temporal lobe of"
            " the brain. These limbic system structures serve a role in processing "
            "memory, emotional reactions and decision-making."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12440?include=full",),
        source_version="26.09d",
        source_sha256="8c110284fc961cb3a93ac154b9c87dd33be6fb464ac12d6b76a9caf504d9d01a",
        reviewed_on="2026-10-23",
        rationale=("The amygdaloid brain structure in emotion and salience sources."),
        source_paths=(
            "knowledge_base/cognitive/fear.md",
            "knowledge_base/cognitive/fear_circuit.md",
            "knowledge_base/cognitive/fear_conditioning.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="arousal",
        iri="http://id.nlm.nih.gov/mesh/D001143",
        label="Arousal",
        definition=(
            "Cortical vigilance or readiness of tone, presumed to be in response to "
            "sensory stimulation via the reticular activating system."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D001143.json",
            "https://id.nlm.nih.gov/mesh/M0001712.json",
        ),
        source_version="MeSH RDF; descriptor updated 2016-06-02",
        source_sha256="e5f20cfbcb58204be1a72f0bc1924be6b15c1328642054320c30800f68a8985b",
        reviewed_on="2026-10-23",
        rationale=(
            "Physiological and psychological activation in wakefulness, vigilance and "
            "neuromodulation; not merely an evaluation service."
        ),
        source_paths=(
            "knowledge_base/biology/adrenaline.md",
            "knowledge_base/cognitive/coma.md",
            "knowledge_base/cognitive/locus_coeruleus.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="artificial_intelligence",
        iri="http://purl.obolibrary.org/obo/NCIT_C16309",
        label="Artificial Intelligence",
        definition=(
            "Artificial intelligence is the branch of computer science that deals with "
            "writing computer programs that can solve problems creatively."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16309?include=full",),
        source_version="26.09d",
        source_sha256="6fc1ef578dde7d9ab03d0c4ac4e5fa0fa4d31dbe71b6b191510f770c29d4f89e",
        reviewed_on="2026-10-23",
        rationale=(
            "Computational systems and methods for intelligent problem solving; not "
            "biological cognition itself."
        ),
        source_paths=(
            "knowledge_base/cognitive/artificial_consciousness.md",
            "knowledge_base/cognitive/knowledge_representation.md",
            "knowledge_base/mathematics/heuristic_search.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="attention",
        iri="http://id.nlm.nih.gov/mesh/D001288",
        label="Attention",
        definition=(
            "Focusing on certain aspects of current experience to the exclusion of "
            "others. It is the act of heeding or taking notice or concentrating."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D001288.json",
            "https://id.nlm.nih.gov/mesh/M0001941.json",
        ),
        source_version="MeSH RDF; descriptor updated 2024-08-09",
        source_sha256="fb45821eb85866a3609b40ccda3653a5472cc3c78610b32f5bf81efb9f44d9fe",
        reviewed_on="2026-10-23",
        rationale=(
            "Selection and focusing of current experience across endogenous, exogenous,"
            " divided and computational attention accounts; not a single test score."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/cognitive/attention.md",
            "knowledge_base/citations/parr_friston_2017.md",
            "knowledge_base/cognitive/adhd.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="awareness",
        iri="http://purl.obolibrary.org/obo/NCIT_C54070",
        label="Awareness",
        definition=(
            "Knowing that something exists, or having knowledge or experience of a "
            "particular thing; having special interest in or experience of something "
            "and so being well informed of what is happening in that subject at the "
            "present time."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54070?include=full",),
        source_version="26.09d",
        source_sha256="f67a68f88cc24a92143b72e32da64c08f320b924f9ee87fad309f9326bbf143d",
        reviewed_on="2026-10-23",
        rationale=(
            "Awareness and knowledge of current experience in conscious and cognitive sources."
        ),
        source_paths=(
            "knowledge_base/cognitive/consciousness.md",
            "knowledge_base/cognitive/consciousness_and_awareness.md",
            "knowledge_base/cognitive/emotional_awareness.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="basal-ganglia",
        iri="http://purl.obolibrary.org/obo/NCIT_C12447",
        label="Basal Ganglia",
        definition=(
            "Clusters of neurons comprising the globus pallidus, putamen, caudate, "
            "nucleus accumbens, substantia nigra and subthalamic nucleus. They are "
            "involved with high level aspects of inhibitory motor activity in "
            "coordination with the excitation commands issued from the cerebellum."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12447?include=full",),
        source_version="26.09d",
        source_sha256="97797369b717520cb91a8220d29d2b8ec046adbcab34421e3c7693a67fb06546",
        reviewed_on="2026-10-23",
        rationale=("Explicit hyphenated brain-structure spelling in cognitive circuit sources."),
        source_paths=(
            "knowledge_base/cognitive/motor_networks.md",
            "knowledge_base/cognitive/ventral_striatum.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="basal_ganglia",
        iri="http://purl.obolibrary.org/obo/NCIT_C12447",
        label="Basal Ganglia",
        definition=(
            "Clusters of neurons comprising the globus pallidus, putamen, caudate, "
            "nucleus accumbens, substantia nigra and subthalamic nucleus. They are "
            "involved with high level aspects of inhibitory motor activity in "
            "coordination with the excitation commands issued from the cerebellum."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12447?include=full",),
        source_version="26.09d",
        source_sha256="97797369b717520cb91a8220d29d2b8ec046adbcab34421e3c7693a67fb06546",
        reviewed_on="2026-10-23",
        rationale=(
            "The group of deep cerebral nuclei discussed in motor and cognitive "
            "circuits; not a murine-only substitute."
        ),
        source_paths=(
            "knowledge_base/cognitive/bradykinesia.md",
            "knowledge_base/cognitive/frontostriatal.md",
            "knowledge_base/cognitive/movement_disorders.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="bayesian-inference",
        iri="http://purl.obolibrary.org/obo/STATO_0000706",
        label="Bayesian inference",
        definition=(
            "Statistical inference that updates prior parameter distributions using "
            "data and Bayes' rule."
        ),
        evidence=(
            "https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl",
        ),
        source_version="2026-04-20@a5910e6115217a88ee86e2666e12983ef2ef2a42",
        source_sha256="7e8f945ee30c6b24139f0064d57c2edbcfc988f3f74811cad6b5c84819ce1494",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in Bayesian updating sources; not a rule "
            "mapping generic inference to Bayesian inference."
        ),
        source_paths=(
            "knowledge_base/citations/mackay_2003.md",
            "knowledge_base/cognitive/probabilistic_programming.md",
            "knowledge_base/cognitive/variational_free_energy.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="bayesian_inference",
        iri="http://purl.obolibrary.org/obo/STATO_0000706",
        label="Bayesian inference",
        definition=(
            "Statistical inference that updates prior parameter distributions using "
            "data and Bayes' rule."
        ),
        evidence=(
            "https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl",
        ),
        source_version="2026-04-20@a5910e6115217a88ee86e2666e12983ef2ef2a42",
        source_sha256="7e8f945ee30c6b24139f0064d57c2edbcfc988f3f74811cad6b5c84819ce1494",
        reviewed_on="2026-10-23",
        rationale=(
            "Updating probability distributions from priors and data using Bayes' rule;"
            " general inference deliberately remains local."
        ),
        source_paths=(
            "knowledge_base/citations/parr_pezzulo_friston_2022.md",
            "knowledge_base/cognitive/bayesian_brain_hypothesis.md",
            "knowledge_base/cognitive/crossmodal_illusions.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="behavior",
        iri="http://purl.obolibrary.org/obo/NCIT_C16326",
        label="Behavior",
        definition=(
            "The actions or reactions of an object or organism, usually in relation to "
            "the environment or surrounding world of stimuli."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16326?include=full",),
        source_version="26.09d",
        source_sha256="23c58d2f03c68928672e0a3a66513393b4a045bbff8aeaeae1d2d28d41638967",
        reviewed_on="2026-10-23",
        rationale=("Actions and reactions of organisms or systems to their environment."),
        source_paths=(
            "knowledge_base/biology/alarm_pheromones.md",
            "knowledge_base/cognitive/action_selection.md",
            "knowledge_base/cognitive/associative_learning.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="behaviour",
        iri="http://purl.obolibrary.org/obo/NCIT_C16326",
        label="Behavior",
        definition=(
            "The actions or reactions of an object or organism, usually in relation to "
            "the environment or surrounding world of stimuli."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16326?include=full",),
        source_version="26.09d",
        source_sha256="23c58d2f03c68928672e0a3a66513393b4a045bbff8aeaeae1d2d28d41638967",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit British spelling in organism behavior and cognitive sources; not "
            "an inferred normalization rule."
        ),
        source_paths=(
            "knowledge_base/cognitive/animal_cognition.md",
            "knowledge_base/cognitive/operant_conditioning.md",
            "knowledge_base/cognitive/perseveration.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="biochemistry",
        iri="http://purl.obolibrary.org/obo/NCIT_C16337",
        label="Biochemistry",
        definition=(
            "The study of the chemical substances and vital processes occurring in "
            "living organisms."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16337?include=full",),
        source_version="26.09d",
        source_sha256="9cd322357517ee733411558754883d6edc7ed228b7723fff3106839020d5e055",
        reviewed_on="2026-10-23",
        rationale=("Chemical substances and vital processes in living organisms."),
        source_paths=(
            "knowledge_base/biology/biochemistry.md",
            "knowledge_base/biology/binding_sites.md",
            "knowledge_base/biology/enzyme_cascades.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="biodiversity",
        iri="http://id.nlm.nih.gov/mesh/D044822",
        label="Biodiversity",
        definition=(
            "The variety of all native living organisms and their various forms and "
            "interrelationships."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D044822.json",
            "https://id.nlm.nih.gov/mesh/M0026833.json",
        ),
        source_version="MeSH RDF; descriptor updated 2010-06-25",
        source_sha256="35e22d88a8d996200dffb721477dcb2972dcedb9b6a8aa2f41f595e9df225e10",
        reviewed_on="2026-10-23",
        rationale=("Diversity of living organisms in biological and ecological contexts."),
        source_paths=(
            "knowledge_base/biology/conservation_biology_social_insects.md",
            "knowledge_base/biology/ecological_impacts_social_insects.md",
            "knowledge_base/biology/functional_diversity.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="bioinformatics",
        iri="http://purl.obolibrary.org/obo/NCIT_C17964",
        label="Bioinformatics",
        definition=(
            "Bioinformatics derives knowledge from computer analysis of biological "
            "data. These can consist of the information stored in the genetic code, but"
            " also experimental results from various sources, patient statistics, and "
            "scientific literature. Research in bioinformatics includes method "
            "development for storage, retrieval, and analysis of the data. "
            "Bioinformatics is a rapidly developing branch of biology and is highly "
            "interdisciplinary, using techniques and concepts from informatics, "
            "statistics, mathematics, chemistry, biochemistry, physics, and "
            "linguistics. It has many practical applications in different areas of "
            "biology and medicine. (M. Nilges and Jens P. Linge, Unite de "
            "Bio-informatique Structurale, Institut Pasteur, Paris)"
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17964?include=full",),
        source_version="26.09d",
        source_sha256="a21923c396eb28056dc58c5eaefc61597189f479c75b4fab665f95fb5bae0e91",
        reviewed_on="2026-10-23",
        rationale=(
            "Computational treatment of biological information in biological and research contexts."
        ),
        source_paths=(
            "knowledge_base/biology/bioinformatics.md",
            "knowledge_base/biology/genetics.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="biology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16345",
        label="Biology",
        definition=("The scientific study of living organisms."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16345?include=full",),
        source_version="26.09d",
        source_sha256="ffa06b8ec18eb0fac7bf44c2d7fb8c174002420bcf81c99524fb1f105bad38ec",
        reviewed_on="2026-10-23",
        rationale=(
            "The scientific study of living organisms across biological knowledge-base sources."
        ),
        source_paths=(
            "knowledge_base/biology/AGENTS.md",
            "knowledge_base/biology/README.md",
            "knowledge_base/biology/cellular_intelligence.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="biomechanics",
        iri="http://purl.obolibrary.org/obo/NCIT_C16346",
        label="Biomechanics",
        definition=(
            "The study of the mechanics of a living body, especially of the forces "
            "exerted by muscles and gravity on the skeletal structure."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16346?include=full",),
        source_version="26.09d",
        source_sha256="9ad6d6967bcaeb8e002df85d525be911cfb3862f4482b996e0f42640cc0e3918",
        reviewed_on="2026-10-23",
        rationale=("Mechanical principles applied to biological movement and structure."),
        source_paths=(
            "knowledge_base/biology/insect_flight_dynamics.md",
            "knowledge_base/cognitive/orthoses.md",
            "knowledge_base/systems/computational_fluid_dynamics.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="biophysics",
        iri="http://purl.obolibrary.org/obo/NCIT_C16349",
        label="Biophysics",
        definition=(
            "Physics as applied to biological problems; an interdisciplinary science "
            "that applies theories and methods of the physical sciences to questions of"
            " biology."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16349?include=full",),
        source_version="26.09d",
        source_sha256="cd26ca7a33819c3d023fb5bbe8de64147f32cca0001f144d56c3737021f02cd2",
        reviewed_on="2026-10-23",
        rationale=("Physical theories and methods applied to biological problems."),
        source_paths=(
            "knowledge_base/biology/biophysics.md",
            "knowledge_base/biology/cell_biology.md",
            "knowledge_base/biology/molecular_motors.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="causality",
        iri="http://purl.obolibrary.org/obo/NCIT_C82552",
        label="Causality",
        definition=(
            "The principle that a relationship exists between a primary entity A, or "
            "cause, and the occurrence of a secondary entity B, or effect."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C82552?include=full",),
        source_version="26.09d",
        source_sha256="7d235b6834c8bb6529d3d24a0b233bf8f88ea437c3dcc59c98fc6117ad7c3f3d",
        reviewed_on="2026-10-23",
        rationale=(
            "General cause-effect relationships in causal models and philosophy; the "
            "general NCI definition applies, not its adverse-event alternate sense."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/philosophy/causality.md",
            "knowledge_base/mathematics/causal_inference.md",
            "knowledge_base/mathematics/causal_sets.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="cell-biology",
        iri="http://purl.obolibrary.org/obo/NCIT_C17992",
        label="Cell Biology",
        definition=(
            "The study of the internal workings of cells at the microscopic and molecular level."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17992?include=full",),
        source_version="26.09d",
        source_sha256="c3c04b14097e439f5707104deca129a4a68f834daad05bd12ff5560c7de3795c",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling reviewed in cell-structure contexts; same "
            "biological discipline as cell_biology."
        ),
        source_paths=(
            "knowledge_base/biology/cell_biology.md",
            "knowledge_base/biology/immunology.md",
            "knowledge_base/biology/neuroscience.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="cell_biology",
        iri="http://purl.obolibrary.org/obo/NCIT_C17992",
        label="Cell Biology",
        definition=(
            "The study of the internal workings of cells at the microscopic and molecular level."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17992?include=full",),
        source_version="26.09d",
        source_sha256="c3c04b14097e439f5707104deca129a4a68f834daad05bd12ff5560c7de3795c",
        reviewed_on="2026-10-23",
        rationale=("Cell structure and internal processes at microscopic and molecular scales."),
        source_paths=(
            "knowledge_base/biology/axon_growth.md",
            "knowledge_base/biology/cell_migration.md",
            "knowledge_base/biology/cell_proliferation.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="cerebellum",
        iri="http://purl.obolibrary.org/obo/NCIT_C12445",
        label="Cerebellum",
        definition=(
            "The portion of the brain located at the base of the skull that is "
            "responsible for balance, equilibrium and movement."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12445?include=full",),
        source_version="26.09d",
        source_sha256="f27bd8fe8aa89e0009f24dc713c85ab7421dbcb5602067e2f58cd6836a23a009",
        reviewed_on="2026-10-23",
        rationale=(
            "The cerebellar brain structure in motor and cognitive processing; not the "
            "rat-specific target."
        ),
        source_paths=(
            "knowledge_base/cognitive/cerebellar_ataxia.md",
            "knowledge_base/cognitive/dysdiadochokinesia.md",
            "knowledge_base/cognitive/dysmetria.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="cognition",
        iri="http://purl.obolibrary.org/obo/NCIT_C16446",
        label="Cognition",
        definition=(
            "The mental process of knowing, thinking, learning and judging; the "
            "psychological result of perception and learning and reasoning."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16446?include=full",),
        source_version="26.09d",
        source_sha256="e8df78dfb1193eeef0e1dca70b09ddc6d348ee3918d075c89cb828655a365410",
        reviewed_on="2026-10-23",
        rationale=(
            "Knowing, thinking, learning and judgment across cognitive models and "
            "philosophy; models of cognition remain about the same subject."
        ),
        source_paths=(
            "knowledge_base/cognitive/action_selection.md",
            "knowledge_base/cognitive/adaptation_mechanisms.md",
            "knowledge_base/cognitive/allostatic_control.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="cognitive_science",
        iri="http://id.nlm.nih.gov/mesh/D019336",
        label="Cognitive Science",
        definition=(
            "The study of the precise nature of different mental tasks and the "
            "operations of the brain that enable them to be performed, engaging "
            "branches of psychology, computer science, philosophy, and linguistics. "
            "(Random House Unabridged Dictionary, 2d ed)"
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D019336.json",
            "https://id.nlm.nih.gov/mesh/M0028784.json",
        ),
        source_version="MeSH RDF; descriptor updated 1999-01-01",
        source_sha256="6dafe2906d718e4042115eb91fb1cbb98edfb2e3b42779da2f93fd4fd095e60f",
        reviewed_on="2026-10-23",
        rationale=(
            "Interdisciplinary study of mental tasks, brain operations and computational accounts."
        ),
        source_paths=(
            "knowledge_base/cognitive/cognitive_science.md",
            "knowledge_base/citations/clark_2013.md",
            "knowledge_base/citations/friston_2012.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="communication",
        iri="http://purl.obolibrary.org/obo/NCIT_C16452",
        label="Communication",
        definition=("The exchange of information between objects, people, or groups."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16452?include=full",),
        source_version="26.09d",
        source_sha256="6541425fefedfe8c203f31d9789d7b4019dfaf74e22504b522ae1311bb8c4dbe",
        reviewed_on="2026-10-23",
        rationale=(
            "Information exchange between people, organisms or systems; the official "
            "definition explicitly permits objects and groups."
        ),
        source_paths=(
            "knowledge_base/cognitive/communication.md",
            "knowledge_base/biology/ant_communication_systems.md",
            "knowledge_base/biology/antennation.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="computation",
        iri="http://purl.obolibrary.org/obo/NCIT_C61298",
        label="Computation",
        definition=(
            "The act of calculating; determining something by mathematical or logical methods."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C61298?include=full",),
        source_version="26.09d",
        source_sha256="a7d5a3b0bd1fb38f8a5b4931d506a5edf3a42c65812258597d203b7b4b4377e0",
        reviewed_on="2026-10-23",
        rationale=(
            "Calculation by mathematical or logical methods across the mathematical and"
            " cognitive sources."
        ),
        source_paths=(
            "knowledge_base/biology/neural_circuits.md",
            "knowledge_base/cognitive/bayesian_brain.md",
            "knowledge_base/cognitive/bayesian_inference.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="computational_modeling",
        iri="http://purl.obolibrary.org/obo/NCIT_C16461",
        label="Computer Modeling",
        definition=(
            "A technique which attempts to provide an abstract model of a particular "
            "system. It utilizes a mathematical model, which attempts to predict the "
            "behavior of the system from a set of parameters and initial conditions."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16461?include=full",),
        source_version="26.09d",
        source_sha256="5bbd2204d20e510ef7f676e3da0cb873841e24ba66b66e4c82783eed0aad356b",
        reviewed_on="2026-10-23",
        rationale=(
            "Preserved: tutorial and computational psychiatry use mathematical computer"
            " models, matching the NCI definition rather than its radiation-planning "
            "alternate sense."
        ),
        source_paths=(
            "knowledge_base/citations/smith_2022.md",
            "knowledge_base/cognitive/computational_psychiatry.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="consciousness",
        iri="http://id.nlm.nih.gov/mesh/D003243",
        label="Consciousness",
        definition=("Sense of awareness of self and of the environment."),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D003243.json",
            "https://id.nlm.nih.gov/mesh/M0005030.json",
        ),
        source_version="MeSH RDF; descriptor updated 1999-01-01",
        source_sha256="24da36581f8d53e754bcd413198837191be66d8a20459dbabfa7ef0c46d4b0d2",
        reviewed_on="2026-10-23",
        rationale=(
            "Awareness of self and environment in cognitive and philosophical accounts;"
            " not merely clinical alertness and orientation."
        ),
        source_paths=(
            "knowledge_base/cognitive/consciousness.md",
            "knowledge_base/free_energy_principle/cognitive/consciousness.md",
            "knowledge_base/citations/hohwy_2013.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="coordination",
        iri="http://purl.obolibrary.org/obo/NCIT_C25462",
        label="Coordination",
        definition=("The act of bringing into common action, movement, or condition."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C25462?include=full",),
        source_version="26.09d",
        source_sha256="41de3377a1d587a228dc8a47eb311bd927475bbd1322ec57cc132b7b3c338f7a",
        reviewed_on="2026-10-23",
        rationale=(
            "Bringing individual movements or agents into common action, including "
            "collective and motor coordination."
        ),
        source_paths=(
            "knowledge_base/systems/coordination.md",
            "knowledge_base/cognitive/collective_behavior.md",
            "knowledge_base/cognitive/joint_action.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="cybernetics",
        iri="http://id.nlm.nih.gov/mesh/D003491",
        label="Cybernetics",
        definition=(
            "That branch of learning which brings together theories and studies on "
            "communication and control in living organisms and machines."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D003491.json",
            "https://id.nlm.nih.gov/mesh/M0005443.json",
        ),
        source_version="MeSH RDF; descriptor updated 2017-06-30",
        source_sha256="2b777f819271edef662a995f407694f41c39ad6da17102cdb3465523fd9ffd6d",
        reviewed_on="2026-10-23",
        rationale=(
            "Communication and control in living organisms and machines, matching "
            "biological and artificial source systems."
        ),
        source_paths=(
            "knowledge_base/systems/cybernetics.md",
            "knowledge_base/cognitive/homeostatic_control_theory.md",
            "knowledge_base/philosophy/conversation_theory.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="decision-making",
        iri="http://id.nlm.nih.gov/mesh/D003657",
        label="Decision Making",
        definition=(
            "The process of making a selective intellectual judgment when presented "
            "with several complex alternatives consisting of several variables, and "
            "usually defining a course of action or an idea."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D003657.json",
            "https://id.nlm.nih.gov/mesh/M0005721.json",
        ),
        source_version="MeSH RDF; descriptor updated 2022-07-12",
        source_sha256="14279a876bb2efaf01b93c66b1ab2c9652b5e96100681739c0ece87cea11bfdb",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in choice and action-selection sources; same "
            "reviewed subject."
        ),
        source_paths=(
            "knowledge_base/biology/quorum_sensing.md",
            "knowledge_base/citations/sutton_2018.md",
            "knowledge_base/cognitive/kahneman_tversky.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="decision_making",
        iri="http://id.nlm.nih.gov/mesh/D003657",
        label="Decision Making",
        definition=(
            "The process of making a selective intellectual judgment when presented "
            "with several complex alternatives consisting of several variables, and "
            "usually defining a course of action or an idea."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D003657.json",
            "https://id.nlm.nih.gov/mesh/M0005721.json",
        ),
        source_version="MeSH RDF; descriptor updated 2022-07-12",
        source_sha256="14279a876bb2efaf01b93c66b1ab2c9652b5e96100681739c0ece87cea11bfdb",
        reviewed_on="2026-10-23",
        rationale=(
            "Choosing among alternatives to determine action in cognitive, mathematical"
            " and multi-agent models."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/cognitive/decision_making.md",
            "knowledge_base/cognitive/action_selection.md",
            "knowledge_base/cognitive/ambivalence.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="deep_learning",
        iri="http://purl.obolibrary.org/obo/NCIT_C176258",
        label="Deep Learning",
        definition=(
            "A subset of machine learning that is part of the broader family of machine"
            " learning methodologies based on artificial neural networks. A deep neural"
            " network has multiple layers between input and output layers to "
            "progressively extract higher level features from the raw input. (After "
            "DeepAI Machine Learning Glossary and Terms)"
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C176258?include=full",),
        source_version="26.09d",
        source_sha256="d16bc613288cf5472efa0acff480cb8647aced8232362373c0de19fccf7bc563",
        reviewed_on="2026-10-23",
        rationale=(
            "Artificial multilayer neural-network learning; not biological learning in general."
        ),
        source_paths=(
            "knowledge_base/cognitive/deep_learning.md",
            "knowledge_base/cognitive/neural_architectures.md",
            "knowledge_base/free_energy_principle/implementations/neural_networks.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="ecology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16526",
        label="Ecology",
        definition=(
            "The branch of biology concerned with the relations between organisms and "
            "their environment."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16526?include=full",),
        source_version="26.09d",
        source_sha256="04eee415bad6d2c939f236cda5c67a67e0b71a349705a2cd3c33a50ea08fc7ae",
        reviewed_on="2026-10-23",
        rationale=(
            "Organism-environment relationships across ecosystems, social insects and "
            "ecological accounts of cognition."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/biology/ecology.md",
            "knowledge_base/biology/README.md",
            "knowledge_base/biology/adaptive_capacity.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="economics",
        iri="http://purl.obolibrary.org/obo/NCIT_C61331",
        label="Economics",
        definition=(
            "The branch of social science that deals with the production and "
            "distribution and consumption of goods and services and their management."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C61331?include=full",),
        source_version="26.09d",
        source_sha256="04e36a435951448d87982fc73c543a4dd59ec1b16b3d2b6944f17e1d4beeaa6e",
        reviewed_on="2026-10-23",
        rationale=(
            "Study of production, distribution and consumption and its models in the "
            "source economics pages."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/applications/economics.md",
            "knowledge_base/cognitive/attention_economy.md",
            "knowledge_base/research/opportunity_cost.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="education",
        iri="http://purl.obolibrary.org/obo/NCIT_C16529",
        label="Education",
        definition=(
            "The activities of educating or instructing or teaching; activities that "
            "impart knowledge or skill."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16529?include=full",),
        source_version="26.09d",
        source_sha256="325525b50d9f7df8ce2350b6950672a405340d4d23fd25358abae9bcce403dd5",
        reviewed_on="2026-10-23",
        rationale=(
            "Teaching and imparting knowledge or skills in research, learning and "
            "educational applications."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/applications/education.md",
            "knowledge_base/cognitive/cognitive_science_mathematics_education.md",
            "knowledge_base/cognitive/collaborative_learning.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="electrophysiology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16540",
        label="Electrophysiology",
        definition=(
            "That branch of physiology that is concerned with the electric phenomena "
            "associated with living bodies and involved in their functional activity."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16540?include=full",),
        source_version="26.09d",
        source_sha256="7a644cfc4eb7dcc3315dbd4b36cc05e3896c3963f8a961d8469ea392f1b5670a",
        reviewed_on="2026-10-23",
        rationale=("Electrical phenomena in living bodies, including neural recordings."),
        source_paths=(
            "knowledge_base/biology/ion_channels.md",
            "knowledge_base/cognitive/electrode_arrays.md",
            "knowledge_base/cognitive/erp.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="emotion",
        iri="http://purl.obolibrary.org/obo/NCIT_C54266",
        label="Emotion",
        definition=(
            "That aspect of consciousness devoted to affect or feeling; a strong "
            "feeling, aroused mental state, or intense state of drive or unrest "
            "directed toward a definite object, with physiological, somatic, and "
            "behavioral components."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54266?include=full",),
        source_version="26.09d",
        source_sha256="6be2740020a5db6e910b96c625445b2f9c655ee10a3b22306f3f765008d31d4e",
        reviewed_on="2026-10-23",
        rationale=(
            "Affect and feelings in appraisal, regulation, social interaction and "
            "active-inference accounts; no disease or particular emotion is asserted."
        ),
        source_paths=(
            "knowledge_base/biology/amygdala.md",
            "knowledge_base/biology/serotonin.md",
            "knowledge_base/citations/hesp_2021.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="entomology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16550",
        label="Entomology",
        definition=("Entomology; the branch of zoology that studies insects."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16550?include=full",),
        source_version="26.09d",
        source_sha256="8997dce1e47bfa86bd16207f62262a9c346838f5fcab7115b50e5e9305ca9381",
        reviewed_on="2026-10-23",
        rationale=("Scientific study of insects, including social-insect models and observations."),
        source_paths=(
            "knowledge_base/biology/entomology.md",
            "knowledge_base/biology/apidology.md",
            "knowledge_base/biology/insect_flight_dynamics.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="episodic_memory",
        iri="http://id.nlm.nih.gov/mesh/D061212",
        label="Memory, Episodic",
        definition=(
            "Type of declarative memory, consisting of personal memory in contrast to "
            "general knowledge."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D061212.json",
            "https://id.nlm.nih.gov/mesh/M0535198.json",
        ),
        source_version="MeSH RDF; descriptor updated 2012-01-01",
        source_sha256="cb2d682b30e028d0c8d582ae64eca2c5100d79dff34794ce88c79f524492b642",
        reviewed_on="2026-10-23",
        rationale=(
            "Memory for experienced events and their context in cognitive memory "
            "sources; immune memory is excluded."
        ),
        source_paths=(
            "knowledge_base/cognitive/cued_recall.md",
            "knowledge_base/cognitive/recollection.md",
            "knowledge_base/cognitive/scene_construction.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="ethics",
        iri="http://purl.obolibrary.org/obo/NCIT_C39796",
        label="Ethics",
        definition=(
            "The philosophical study of moral values and rules.  Ethics is a general "
            'term for what is often described as the "science of morality". The Western'
            " tradition of ethics is sometimes called moral philosophy. This is one of "
            "the three major branches of philosophy, alongside metaphysics and logic."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C39796?include=full",),
        source_version="26.09d",
        source_sha256="9fcf4021e56b2af05bc507e1e9330bb7d0cc387531b710b1add40a1227838d7b",
        reviewed_on="2026-10-23",
        rationale=(
            "Moral values and rules across the philosophical traditions represented in "
            "the source; does not identify ethics with one tradition."
        ),
        source_paths=(
            "knowledge_base/philosophy/absurdity.md",
            "knowledge_base/philosophy/african_communitarianism.md",
            "knowledge_base/philosophy/ai_ethics.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="executive-function",
        iri="http://purl.obolibrary.org/obo/NCIT_C88419",
        label="Executive Function",
        definition=("A set of mental processes that controls and manages the cognitive functions."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C88419?include=full",),
        source_version="26.09d",
        source_sha256="6098b076dff1a75147162ae96e3a4227fa87c982ab6e159dfcc213ee9f5e0dcb",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in cognitive control sources; same mental-process scope."
        ),
        source_paths=(
            "knowledge_base/cognitive/cognitive_functions.md",
            "knowledge_base/cognitive/cognitive_training.md",
            "knowledge_base/cognitive/prefrontal.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="executive_function",
        iri="http://purl.obolibrary.org/obo/NCIT_C88419",
        label="Executive Function",
        definition=("A set of mental processes that controls and manages the cognitive functions."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C88419?include=full",),
        source_version="26.09d",
        source_sha256="6098b076dff1a75147162ae96e3a4227fa87c982ab6e159dfcc213ee9f5e0dcb",
        reviewed_on="2026-10-23",
        rationale=(
            "Mental processes that manage cognitive functions; not a generic software executive."
        ),
        source_paths=(
            "knowledge_base/cognitive/adhd.md",
            "knowledge_base/cognitive/attentional_control.md",
            "knowledge_base/cognitive/cognitive_control.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="eye_movements",
        iri="http://id.nlm.nih.gov/mesh/D005133",
        label="Eye Movements",
        definition=("Voluntary or reflex-controlled movements of the eye."),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D005133.json",
            "https://id.nlm.nih.gov/mesh/M0008093.json",
        ),
        source_version="MeSH RDF; descriptor updated 2016-06-30",
        source_sha256="1d17c304deeeae90dd042c4e80f05cb92c505b799a61b6088994068aab43c83a",
        reviewed_on="2026-10-23",
        rationale=(
            "Movement of the eyes in oculomotor and perception sources; does not "
            "designate a particular saccade."
        ),
        source_paths=(
            "knowledge_base/cognitive/frontal_eye_fields.md",
            "knowledge_base/cognitive/overt_attention.md",
            "knowledge_base/cognitive/pursuit_movements.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="game-theory",
        iri="http://id.nlm.nih.gov/mesh/D005716",
        label="Game Theory",
        definition=(
            "Theoretical construct used in applied mathematics to analyze certain "
            "situations in which there is an interplay between parties that may have "
            "similar, opposed, or mixed interests. In a typical game, decision-making "
            "players, who each have their own goals, try to gain advantage over the "
            "other parties by anticipating each other's decisions; the game is finally "
            "resolved as a consequence of the players' decisions."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D005716.json",
            "https://id.nlm.nih.gov/mesh/M0008969.json",
        ),
        source_version="MeSH RDF; descriptor updated 2015-09-01",
        source_sha256="421596485602071cff39b65d3921418c22fa4f4b35656036828c2f0cc2cd7209",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in strategic-interaction contexts; same "
            "mathematical field."
        ),
        source_paths=(
            "knowledge_base/cognitive/multi_agent_active_inference.md",
            "knowledge_base/cognitive/negotiation.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="game_theory",
        iri="http://id.nlm.nih.gov/mesh/D005716",
        label="Game Theory",
        definition=(
            "Theoretical construct used in applied mathematics to analyze certain "
            "situations in which there is an interplay between parties that may have "
            "similar, opposed, or mixed interests. In a typical game, decision-making "
            "players, who each have their own goals, try to gain advantage over the "
            "other parties by anticipating each other's decisions; the game is finally "
            "resolved as a consequence of the players' decisions."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D005716.json",
            "https://id.nlm.nih.gov/mesh/M0008969.json",
        ),
        source_version="MeSH RDF; descriptor updated 2015-09-01",
        source_sha256="421596485602071cff39b65d3921418c22fa4f4b35656036828c2f0cc2cd7209",
        reviewed_on="2026-10-23",
        rationale=(
            "Mathematical analysis of interacting decision makers and their interests; "
            "not all decision theory."
        ),
        source_paths=(
            "knowledge_base/mathematics/game_theory.md",
            "knowledge_base/cognitive/cooperation.md",
            "knowledge_base/cognitive/economic_games.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="genetics",
        iri="http://purl.obolibrary.org/obo/NCIT_C16624",
        label="Genetics",
        definition=(
            "The branch of biology that deals with heredity, especially the mechanisms "
            "of hereditary transmission and the variation of inherited characteristics "
            "among similar or related organisms"
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16624?include=full",),
        source_version="26.09d",
        source_sha256="16dd35dcf1bf25c5b488134a914d6751e94a975d471b8191f3205c964733fd2b",
        reviewed_on="2026-10-23",
        rationale=(
            "Heredity and inherited variation across gene expression, epigenetics and "
            "population genetics; the discipline is the document subject."
        ),
        source_paths=(
            "knowledge_base/biology/genetics.md",
            "knowledge_base/biology/epigenetics.md",
            "knowledge_base/biology/gene_expression.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="hippocampus",
        iri="http://purl.obolibrary.org/obo/NCIT_C12444",
        label="Hippocampus",
        definition=(
            "A curved gray matter structure of the temporal lobe lying on the floor of "
            "the lateral ventricle of the brain."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12444?include=full",),
        source_version="26.09d",
        source_sha256="660b95719bbb762928d356832958d00049ebae4e7b93d8f676fad60a8302fe25",
        reviewed_on="2026-10-23",
        rationale=(
            "The named brain structure and its subregions in memory and spatial "
            "cognition; a subject link does not classify a page as brain tissue."
        ),
        source_paths=(
            "knowledge_base/biology/ltp.md",
            "knowledge_base/cognitive/amnesia.md",
            "knowledge_base/cognitive/anterograde_amnesia.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="homeostasis",
        iri="http://purl.obolibrary.org/obo/NCIT_C25989",
        label="Homeostatic Process",
        definition=(
            "The body's maintenance of a consistent internal environment within a "
            "narrow range of variation. It is important to the continuation of life."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C25989?include=full",),
        source_version="26.09d",
        source_sha256="f2afac10c6fc7b636451adbcbe1ddedd06dee02687e876c92c486618ec6e8a40",
        reviewed_on="2026-10-23",
        rationale=(
            "Maintenance of the internal physiological environment and active-inference"
            " accounts of biological homeostatic regulation."
        ),
        source_paths=(
            "knowledge_base/biology/homeostasis.md",
            "knowledge_base/free_energy_principle/biology/homeostasis.md",
            "knowledge_base/biology/allostatic_regulation.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="hormone",
        iri="http://purl.obolibrary.org/obo/NCIT_C2315",
        label="Hormone",
        definition=(
            "One of many chemicals made by glands in the body. Hormones circulate in "
            "the bloodstream and control the actions of certain cells or organs. Some "
            "hormones can also be made in the laboratory."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2315?include=full",),
        source_version="26.09d",
        source_sha256="f9bc18eeb8905c9ced92ada5b8793fe0407e690d644e50496e8b0033695a2925",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit singular spelling used for endocrine signaling substances, not a "
            "treatment regimen."
        ),
        source_paths=(
            "knowledge_base/biology/cortisol.md",
            "knowledge_base/biology/vasopressin.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="hormones",
        iri="http://purl.obolibrary.org/obo/NCIT_C2315",
        label="Hormone",
        definition=(
            "One of many chemicals made by glands in the body. Hormones circulate in "
            "the bloodstream and control the actions of certain cells or organs. Some "
            "hormones can also be made in the laboratory."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C2315?include=full",),
        source_version="26.09d",
        source_sha256="f9bc18eeb8905c9ced92ada5b8793fe0407e690d644e50496e8b0033695a2925",
        reviewed_on="2026-10-23",
        rationale=("Chemical signaling substances in endocrine contexts; not hormone treatment."),
        source_paths=(
            "knowledge_base/biology/adrenaline.md",
            "knowledge_base/biology/estrogen.md",
            "knowledge_base/biology/testosterone.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="immunology",
        iri="http://purl.obolibrary.org/obo/NCIT_C18011",
        label="Immunology",
        definition=(
            "The study of the immune system and its reaction to pathogens, as well as "
            "its malfunctions (autoimmune diseases, allergies, rejection of organ "
            "transplants)."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C18011?include=full",),
        source_version="26.09d",
        source_sha256="e6a6f55ae16461680de1822c637cfbf28d4ba9512f24af40c83d9a6a2eb22a88",
        reviewed_on="2026-10-23",
        rationale=("Study of immune-system function; not immune-function diagnostic testing."),
        source_paths=(
            "knowledge_base/biology/immunology.md",
            "knowledge_base/free_energy_principle/biology/immunology.md",
            "knowledge_base/biology/adaptive_immunity.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="information-theory",
        iri="http://id.nlm.nih.gov/mesh/D007257",
        label="Information Theory",
        definition=(
            "An interdisciplinary study dealing with the transmission of messages or "
            "signals, or the communication of information. Information theory does not "
            "directly deal with meaning or content, but with physical representations "
            "that have meaning or content. It overlaps considerably with communication "
            "theory and CYBERNETICS."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D007257.json",
            "https://id.nlm.nih.gov/mesh/M0011324.json",
        ),
        source_version="MeSH RDF; descriptor updated 2010-06-25",
        source_sha256="4b2bf2e913074c080a895ca573fae1d456acad65f135114afa317feda9cc3c7b",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in entropy, mutual information and "
            "free-energy mathematical accounts; same quantitative subject."
        ),
        source_paths=(
            "knowledge_base/citations/lindley_1956.md",
            "knowledge_base/citations/mackay_2003.md",
            "knowledge_base/cognitive/information_theory_in_control.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="information_theory",
        iri="http://id.nlm.nih.gov/mesh/D007257",
        label="Information Theory",
        definition=(
            "An interdisciplinary study dealing with the transmission of messages or "
            "signals, or the communication of information. Information theory does not "
            "directly deal with meaning or content, but with physical representations "
            "that have meaning or content. It overlaps considerably with communication "
            "theory and CYBERNETICS."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D007257.json",
            "https://id.nlm.nih.gov/mesh/M0011324.json",
        ),
        source_version="MeSH RDF; descriptor updated 2010-06-25",
        source_sha256="4b2bf2e913074c080a895ca573fae1d456acad65f135114afa317feda9cc3c7b",
        reviewed_on="2026-10-23",
        rationale=(
            "Quantitative information transmission and representations across entropy, "
            "coding and information measures; not semantic meaning by label alone."
        ),
        source_paths=(
            "knowledge_base/cognitive/free_energy_principle.md",
            "knowledge_base/cognitive/integrated_information.md",
            "knowledge_base/free_energy_principle/AGENTS.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="interoception",
        iri="http://id.nlm.nih.gov/mesh/D065812",
        label="Interoception",
        definition=(
            "The perception of and sensitivity to stimuli originating inside of the "
            "body. It is a sense of the physiological condition of the entire body."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D065812.json",
            "https://id.nlm.nih.gov/mesh/M0591076.json",
        ),
        source_version="MeSH RDF; descriptor updated 2015-01-01",
        source_sha256="bf1f091f1fd42c3ed673c417f45f95fe56b78c9a6597a5f10414938e217a56a5",
        reviewed_on="2026-10-23",
        rationale=(
            "Perception of internal bodily stimuli in physiological and active-inference sources."
        ),
        source_paths=(
            "knowledge_base/biology/allostatic_regulation.md",
            "knowledge_base/citations/hesp_2021.md",
            "knowledge_base/citations/pezzulo_2015.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="ion_channels",
        iri="http://id.nlm.nih.gov/mesh/D007473",
        label="Ion Channels",
        definition=(
            "Gated, ion-selective glycoproteins that traverse membranes. The stimulus "
            "for ION CHANNEL GATING can be due to a variety of stimuli such as LIGANDS,"
            " a TRANSMEMBRANE POTENTIAL DIFFERENCE, mechanical deformation or through "
            "INTRACELLULAR SIGNALING PEPTIDES AND PROTEINS."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D007473.json",
            "https://id.nlm.nih.gov/mesh/M0011674.json",
        ),
        source_version="MeSH RDF; descriptor updated 2016-07-01",
        source_sha256="778a097910d66cc37c90c2ddb64a857f24819902c2af483772f1f9f013ddd935",
        reviewed_on="2026-10-23",
        rationale=(
            "Membrane channels that conduct ions, as used by biological excitability "
            "and neural signaling sources."
        ),
        source_paths=(
            "knowledge_base/biology/excitability.md",
            "knowledge_base/biology/voltage_channels.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="language",
        iri="http://purl.obolibrary.org/obo/NCIT_C16779",
        label="Language",
        definition=(
            "A systematic means of communicating by the use of sounds, symbols, or gestures."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16779?include=full",),
        source_version="26.09d",
        source_sha256="6564eca1b1f8e3cb316bf64ed748aafbf9e6deda5216196e3783a92a793b3726",
        reviewed_on="2026-10-23",
        rationale=(
            "Natural-language communication and its formal accounts; source contexts "
            "concern words, grammar and meaning rather than a programming-language "
            "identifier."
        ),
        source_paths=(
            "knowledge_base/cognitive/affixes.md",
            "knowledge_base/cognitive/angular_gyrus.md",
            "knowledge_base/cognitive/aphasia.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="learning",
        iri="http://purl.obolibrary.org/obo/NCIT_C19369",
        label="Learning",
        definition=("The act, process, or experience of gaining knowledge or skill."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C19369?include=full",),
        source_version="26.09d",
        source_sha256="43f38620f61461807468719ea10e594388cb966d80d0e6272ba42d4f42b2ff97",
        reviewed_on="2026-10-23",
        rationale=(
            "Acquisition of knowledge or skill, including computational accounts of "
            "that process; not limited to a particular learning algorithm."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/cognitive/learning.md",
            "knowledge_base/biology/structural_plasticity.md",
            "knowledge_base/citations/friston_2010.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="linguistics",
        iri="http://id.nlm.nih.gov/mesh/D008037",
        label="Linguistics",
        definition=(
            "The science of language, including phonetics, phonology, morphology, "
            "syntax, semantics, pragmatics, and historical linguistics. (Random House "
            "Unabridged Dictionary, 2d ed)"
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D008037.json",
            "https://id.nlm.nih.gov/mesh/M0012527.json",
        ),
        source_version="MeSH RDF; descriptor updated 2017-06-30",
        source_sha256="e5df299193b4d63c5688617d3befcd6d20e4e43b9e41b40f879350805f0be88f",
        reviewed_on="2026-10-23",
        rationale=(
            "Scientific study of natural language and its structure, rather than a "
            "programming language."
        ),
        source_paths=(
            "knowledge_base/cognitive/semantic_roles.md",
            "knowledge_base/cognitive/syllable_structure.md",
            "knowledge_base/cognitive/word_formation.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="logic",
        iri="http://id.nlm.nih.gov/mesh/D008128",
        label="Logic",
        definition=(
            "The science that investigates the principles governing correct or reliable"
            " inference and deals with the canons and criteria of validity in thought "
            "and demonstration. This system of reasoning is applicable to any branch of"
            " knowledge or study. (Random House Unabridged Dictionary, 2d ed & Sippl, "
            "Computer Dictionary, 4th ed)"
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D008128.json",
            "https://id.nlm.nih.gov/mesh/M0012674.json",
        ),
        source_version="MeSH RDF; descriptor updated 1999-01-01",
        source_sha256="f82b7e7aafcd9da3925f80f39bc34002f667b46c501889899f3f02dc8b83da19",
        reviewed_on="2026-10-23",
        rationale=(
            "Principles of valid inference and demonstration, including mathematical "
            "logic; not Bayesian inference alone."
        ),
        source_paths=(
            "knowledge_base/philosophy/logic.md",
            "knowledge_base/cognitive/conditional_reasoning.md",
            "knowledge_base/cognitive/deductive_reasoning.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="machine-learning",
        iri="http://purl.obolibrary.org/obo/NCIT_C176231",
        label="Machine Learning",
        definition=(
            "A computing system (inspired by biological neural networks) that learns "
            "(progressively improves its ability) to do tasks by considering examples "
            "without task-specific programming. (After DeepAI Machine Learning Glossary"
            " and Terms)"
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C176231?include=full",),
        source_version="26.09d",
        source_sha256="77a38eae78aef1b5b2e239808ed2f808fc0b1f49374cc792f2b538ba8666a122",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in computational learning sources; same "
            "reviewed sense as machine_learning."
        ),
        source_paths=(
            "knowledge_base/citations/mackay_2003.md",
            "knowledge_base/cognitive/artificial_intelligence.md",
            "knowledge_base/cognitive/machine_readability.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="machine_learning",
        iri="http://purl.obolibrary.org/obo/NCIT_C176231",
        label="Machine Learning",
        definition=(
            "A computing system (inspired by biological neural networks) that learns "
            "(progressively improves its ability) to do tasks by considering examples "
            "without task-specific programming. (After DeepAI Machine Learning Glossary"
            " and Terms)"
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C176231?include=full",),
        source_version="26.09d",
        source_sha256="77a38eae78aef1b5b2e239808ed2f808fc0b1f49374cc792f2b538ba8666a122",
        reviewed_on="2026-10-23",
        rationale=("Computational learning from examples in machine-learning sources."),
        source_paths=(
            "knowledge_base/cognitive/deep_learning.md",
            "knowledge_base/cognitive/online_learning.md",
            "knowledge_base/cognitive/transfer_learning.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="mathematics",
        iri="http://purl.obolibrary.org/obo/NCIT_C16825",
        label="Mathematics",
        definition=("The study of numbers, shapes, patterns, and logical reasoning."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16825?include=full",),
        source_version="26.09d",
        source_sha256="f5d262a898bde19b5b497dc1a7f0b65787450d5cc8fcf061676b316b036a505d",
        reviewed_on="2026-10-23",
        rationale=(
            "Mathematical methods, probability, geometry and formal reasoning; includes"
            " the Markov-blanket document as a subject, not a mathematical instance."
        ),
        source_paths=(
            "knowledge_base/biology/evolutionary_dynamics.md",
            "knowledge_base/citations/amari_2000.md",
            "knowledge_base/cognitive/cognitive_science_mathematics_education.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="matrices",
        iri="http://purl.obolibrary.org/obo/STATO_0000642",
        label="matrix",
        definition=("A rectangular arrangement of numerical entries."),
        evidence=(
            "https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl",
        ),
        source_version="2026-04-20@a5910e6115217a88ee86e2666e12983ef2ef2a42",
        source_sha256="7e8f945ee30c6b24139f0064d57c2edbcfc988f3f74811cad6b5c84819ce1494",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit plural spelling in POMDP and numerical-matrix sources; no "
            "normalization is performed by lookup."
        ),
        source_paths=(
            "knowledge_base/agents/GenericPOMDP/guides/AGENTS.md",
            "knowledge_base/agents/GenericPOMDP/matrices/AGENTS.md",
            "knowledge_base/cognitive/matrix_specifications.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="matrix",
        iri="http://purl.obolibrary.org/obo/STATO_0000642",
        label="matrix",
        definition=("A rectangular arrangement of numerical entries."),
        evidence=(
            "https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl",
        ),
        source_version="2026-04-20@a5910e6115217a88ee86e2666e12983ef2ef2a42",
        source_sha256="7e8f945ee30c6b24139f0064d57c2edbcfc988f3f74811cad6b5c84819ce1494",
        reviewed_on="2026-10-23",
        rationale=(
            "Rectangular numerical arrays in POMDP matrices and their mathematical "
            "structure; not tissue matrix or a dosing unit."
        ),
        source_paths=(
            "knowledge_base/agents/GenericPOMDP/matrices/A_matrix.md",
            "knowledge_base/agents/GenericPOMDP/matrices/B_matrix.md",
            "knowledge_base/agents/GenericPOMDP/matrices/C_matrix.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="memory_consolidation",
        iri="http://id.nlm.nih.gov/mesh/D000069077",
        label="Memory Consolidation",
        definition=(
            "Neurological process involving the conversion of learned information into "
            "long-term memory."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D000069077.json",
            "https://id.nlm.nih.gov/mesh/M000604637.json",
        ),
        source_version="MeSH RDF; descriptor updated 2016-01-01",
        source_sha256="135a01cdfd0572d7fc54e2c7a9461607beba083d451a963341363416b900704b",
        reviewed_on="2026-10-23",
        rationale=(
            "Stabilization of acquired memories in the observed cognitive and neural sources."
        ),
        source_paths=(
            "knowledge_base/cognitive/late_ltp.md",
            "knowledge_base/cognitive/sleep_memory_ai.md",
            "knowledge_base/cognitive/slow_wave_sleep.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="metacognition",
        iri="http://id.nlm.nih.gov/mesh/D000067489",
        label="Metacognition",
        definition=("Awareness and understanding of one's own thought processes."),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D000067489.json",
            "https://id.nlm.nih.gov/mesh/M000599741.json",
        ),
        source_version="MeSH RDF; descriptor updated 2017-06-20",
        source_sha256="f219269f0a9e21749a4e26e2f6572b94734f65efb85b19bf8e8a0a37bd756648",
        reviewed_on="2026-10-23",
        rationale=(
            "Awareness and understanding of one's own thought processes across "
            "judgments of learning and monitoring."
        ),
        source_paths=(
            "knowledge_base/cognitive/confidence_judgment.md",
            "knowledge_base/cognitive/consciousness_theory.md",
            "knowledge_base/cognitive/feeling_of_knowing.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="metaphysics",
        iri="http://id.nlm.nih.gov/mesh/D008678",
        label="Metaphysics",
        definition=(
            "The branch of philosophy that treats of first principles, including "
            "ontology (the nature of existence or being) and cosmology (the origin and "
            "structure of the universe). (From Random House Unabridged Dictionary, 2d "
            "ed)"
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D008678.json",
            "https://id.nlm.nih.gov/mesh/M0013524.json",
        ),
        source_version="MeSH RDF; descriptor updated 1999-01-01",
        source_sha256="e04c0c344eb9673e99953ba5504bbdfad9ae1d7ae407a45088cdc5c6e6cc4dd8",
        reviewed_on="2026-10-23",
        rationale=(
            "First principles of existence and being across philosophical traditions; "
            "not a formal knowledge-representation ontology."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/philosophy/metaphysics.md",
            "knowledge_base/ontology/philosophical_ontology.md",
            "knowledge_base/philosophy/AGENTS.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="microbiology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16851",
        label="Microbiology",
        definition=(
            "The science that deals with the study of microorganisms, including algae, "
            "archaea, bacteria, fungi, protozoa, and viruses."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16851?include=full",),
        source_version="26.09d",
        source_sha256="b7bf5846d30a0d0563fc78b863d323767ebd8ab7317625db8a5f360ad2797073",
        reviewed_on="2026-10-23",
        rationale=("Study of microorganisms in the biological sources; not a diagnostic service."),
        source_paths=(
            "knowledge_base/biology/microbiology.md",
            "knowledge_base/biology/biofilm_formation.md",
            "knowledge_base/biology/symbiosis.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="model_fitting",
        iri="http://purl.obolibrary.org/obo/STATO_0000218",
        label="model fitting",
        definition=("Assessing how well a model represents an observed dataset."),
        evidence=(
            "https://raw.githubusercontent.com/ISA-tools/stato/a5910e6115217a88ee86e2666e12983ef2ef2a42/releases/latest_release/stato.owl",
        ),
        source_version="2026-04-20@a5910e6115217a88ee86e2666e12983ef2ef2a42",
        source_sha256="7e8f945ee30c6b24139f0064d57c2edbcfc988f3f74811cad6b5c84819ce1494",
        reviewed_on="2026-10-23",
        rationale=(
            "Preserved: the cited tutorial fits an active-inference model to empirical "
            "data; this is dataset-to-model fit assessment."
        ),
        source_paths=("knowledge_base/citations/smith_2022.md",),
        active=True,
    ),
    KeywordTerm(
        tag="molecular-biology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16872",
        label="Molecular Biology",
        definition=(
            "The study of biology at a molecular level. It chiefly concerns itself with"
            " understanding the interactions between the various systems of a cell, "
            "including the interrelationship of DNA, RNA and protein synthesis and "
            "learning how these interactions are regulated. The field overlaps with "
            "other areas of biology, particularly genetics and biochemistry."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16872?include=full",),
        source_version="26.09d",
        source_sha256="bba5bde43e2fdd00327db981880080ac109ae1c30792995bd232941fbe2a96cc",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in molecular and cellular sources; scope "
            "matches molecular_biology."
        ),
        source_paths=(
            "knowledge_base/biology/biochemistry.md",
            "knowledge_base/biology/cell_biology.md",
            "knowledge_base/biology/gene_expression.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="molecular_biology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16872",
        label="Molecular Biology",
        definition=(
            "The study of biology at a molecular level. It chiefly concerns itself with"
            " understanding the interactions between the various systems of a cell, "
            "including the interrelationship of DNA, RNA and protein synthesis and "
            "learning how these interactions are regulated. The field overlaps with "
            "other areas of biology, particularly genetics and biochemistry."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16872?include=full",),
        source_version="26.09d",
        source_sha256="bba5bde43e2fdd00327db981880080ac109ae1c30792995bd232941fbe2a96cc",
        reviewed_on="2026-10-23",
        rationale=(
            "Molecular-scale cell and gene processes; no individual molecule or gene is asserted."
        ),
        source_paths=(
            "knowledge_base/biology/allosteric_effects.md",
            "knowledge_base/biology/binding_sites.md",
            "knowledge_base/biology/gene_regulation.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="monte_carlo",
        iri="http://purl.obolibrary.org/obo/NCIT_C94936",
        label="Monte Carlo Method",
        definition=(
            "A computing method involving many similar individual, yet randomly "
            "differing, calculations. The individual calculations differ due to the "
            "introduction of random numbers for some input quantity or quantities."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C94936?include=full",),
        source_version="26.09d",
        source_sha256="8c6d89f8d5583473c6bcd6d82c2c7faa913fb3a76b2022e2c52161e314f6c767",
        reviewed_on="2026-10-23",
        rationale=(
            "Randomized repeated computation in MCMC, importance sampling and particle "
            "methods; not biological specimen sampling."
        ),
        source_paths=(
            "knowledge_base/mathematics/effective_sample_size.md",
            "knowledge_base/mathematics/hamiltonian_monte_carlo.md",
            "knowledge_base/mathematics/importance_sampling.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="motivation",
        iri="http://purl.obolibrary.org/obo/NCIT_C28495",
        label="Motivation",
        definition=(
            "The psychological feature that arouses an organism to action toward a "
            "desired goal; the reason for the action; that which gives purpose and "
            "direction to behavior."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C28495?include=full",),
        source_version="26.09d",
        source_sha256="46eaf3efce3d1626a570bc77baa4a5e93512899866c321398e1fc1346545ab8e",
        reviewed_on="2026-10-23",
        rationale=("Psychological drives and incentives underlying behavior."),
        source_paths=(
            "knowledge_base/cognitive/motivation.md",
            "knowledge_base/cognitive/approach_avoidance.md",
            "knowledge_base/cognitive/emotional_processing_ai.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="natural_selection",
        iri="http://id.nlm.nih.gov/mesh/D012641",
        label="Selection, Genetic",
        definition=(
            "Differential and non-random reproduction of different genotypes, operating"
            " to alter the gene frequencies within a population."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D012641.json",
            "https://id.nlm.nih.gov/mesh/M0019598.json",
        ),
        source_version="MeSH RDF; descriptor updated 2016-06-17",
        source_sha256="bc31da7dbfaf173e9351c84c89ad41b943dd03469b221b1bc056b28fc85d95bc",
        reviewed_on="2026-10-23",
        rationale=(
            "Differential reproductive success associated with heritable traits in "
            "biological evolution."
        ),
        source_paths=(
            "knowledge_base/biology/selection_pressure.md",
            "knowledge_base/free_energy_principle/biology/evolution.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="neural_plasticity",
        iri="http://id.nlm.nih.gov/mesh/D009473",
        label="Neuronal Plasticity",
        definition=(
            "The capacity of the NERVOUS SYSTEM to change its reactivity as the result "
            "of successive activations."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D009473.json",
            "https://id.nlm.nih.gov/mesh/M0014735.json",
        ),
        source_version="MeSH RDF; descriptor updated 2018-12-11",
        source_sha256="668bc3319403ea4fdf5d817415bfd058c86762054a595927c4e204cb3bc8be6f",
        reviewed_on="2026-10-23",
        rationale=(
            "Capacity of nervous systems to change in response to activation in "
            "sprouting and map-plasticity sources; ambiguous plasticity remains local."
        ),
        source_paths=(
            "knowledge_base/biology/sprouting.md",
            "knowledge_base/cognitive/map_plasticity.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="neuroanatomy",
        iri="http://purl.obolibrary.org/obo/NCIT_C16905",
        label="Neuroanatomy",
        definition=("A branch of medicine concerned with the anatomy of the nervous system."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16905?include=full",),
        source_version="26.09d",
        source_sha256="4da7cfca97d6ac1fe9d8c4350cb72169a713350d02f94d78b904592d6206ff24",
        reviewed_on="2026-10-23",
        rationale=(
            "Anatomy of the nervous system across named brain structures and cortical circuits."
        ),
        source_paths=(
            "knowledge_base/cognitive/angular_gyrus.md",
            "knowledge_base/cognitive/anterior_cingulate.md",
            "knowledge_base/cognitive/auditory_cortex.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="neuroendocrinology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16907",
        label="Neuroendocrinology",
        definition=(
            "The study of how the nervous and endocrine systems act together to "
            "regulate each other and physiological processes throughout the body."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16907?include=full",),
        source_version="26.09d",
        source_sha256="301a76da2c0af36ee3aa946e085646267f54b7987ba738685053dcd875b075c7",
        reviewed_on="2026-10-23",
        rationale=("Interactions between nervous and endocrine systems."),
        source_paths=(
            "knowledge_base/biology/estrogen.md",
            "knowledge_base/cognitive/pair_bonding.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="neuroimaging",
        iri="http://purl.obolibrary.org/obo/NCIT_C173635",
        label="Neuroimaging",
        definition=(
            "The use of various techniques to directly or indirectly image the anatomy "
            "and function of the central nervous system."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C173635?include=full",),
        source_version="26.09d",
        source_sha256="dcf35ad65cd36718bcf0b52ce3dfcb5a4173de03a3a64809a3b74be1d4add1e7",
        reviewed_on="2026-10-23",
        rationale=(
            "Imaging nervous-system structure or function; does not imply a particular modality."
        ),
        source_paths=(
            "knowledge_base/cognitive/cognitive_neuroscience.md",
            "knowledge_base/cognitive/eeg_recording.md",
            "knowledge_base/cognitive/fmri_studies.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="neurology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16908",
        label="Neurology",
        definition=(
            "A branch of medicine concerned with the development, anatomy, function, "
            "and dysfunction of the nervous system."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16908?include=full",),
        source_version="26.09d",
        source_sha256="b59e1c8fbeee187f7b0246ba940983f1a09d329c92b71390a61afd97153c594a",
        reviewed_on="2026-10-23",
        rationale=(
            "The medical discipline concerned with the nervous system; not a service "
            "billing category."
        ),
        source_paths=(
            "knowledge_base/cognitive/cerebellar_ataxia.md",
            "knowledge_base/cognitive/coma_scale.md",
            "knowledge_base/cognitive/disorders_consciousness.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="neurophysiology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16910",
        label="Neurophysiology",
        definition=(
            "A branch of neuroscience concerned with the physiology of the nervous system."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16910?include=full",),
        source_version="26.09d",
        source_sha256="446bb2f2aa7605310f9b1e9ccf83c7e05513e836c8812178dbf3a59466fdc99b",
        reviewed_on="2026-10-23",
        rationale=("Physiological functions of the nervous system."),
        source_paths=(
            "knowledge_base/biology/excitability.md",
            "knowledge_base/cognitive/hubel_wiesel.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="neuropsychology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16911",
        label="Neuropsychology",
        definition=(
            "The study of how the physiology of the brain and central nervous system "
            "are related to behavior."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16911?include=full",),
        source_version="26.09d",
        source_sha256="4fb46de49ae11b85db90ea38ee38db48577dda7c189cff132fb2003016ad414d",
        reviewed_on="2026-10-23",
        rationale=("Brain and nervous-system relations to behavior and cognitive function."),
        source_paths=(
            "knowledge_base/cognitive/agnosia.md",
            "knowledge_base/cognitive/alexia.md",
            "knowledge_base/cognitive/amnesia.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="neuroscience",
        iri="http://id.nlm.nih.gov/mesh/D009488",
        label="Neurosciences",
        definition=(
            "The scientific disciplines concerned with the embryology, anatomy, "
            "physiology, biochemistry, pharmacology, etc., of the nervous system."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D009488.json",
            "https://id.nlm.nih.gov/mesh/M0014763.json",
        ),
        source_version="MeSH RDF; descriptor updated 2009-07-06",
        source_sha256="2d483bc8c8d363686e76be996421f6e48f8dbc234a29efc49cc967c45b4a8b2a",
        reviewed_on="2026-10-23",
        rationale=(
            "The scientific disciplines studying the nervous system; the source uses a "
            "singular discipline label for MeSH Neurosciences."
        ),
        source_paths=(
            "knowledge_base/biology/neuroscience.md",
            "knowledge_base/free_energy_principle/applications/neuroscience.md",
            "knowledge_base/biology/README.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="optimization",
        iri="http://purl.obolibrary.org/obo/NCIT_C78561",
        label="Optimization",
        definition=("An activity that attempts to find the best solution to a problem."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C78561?include=full",),
        source_version="26.09d",
        source_sha256="55784efc91e7cb530d0b4b384f031f2b6b9be22f5a32298283ace44b7a486cbc",
        reviewed_on="2026-10-23",
        rationale=(
            "Finding an optimum solution across numerical, variational, biological and "
            "cognitive optimization; no algorithm-specific replacement."
        ),
        source_paths=(
            "knowledge_base/biology/fitness_landscapes.md",
            "knowledge_base/biology/foraging_optimization.md",
            "knowledge_base/biology/mathematical_models_social_insects.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="perception",
        iri="http://id.nlm.nih.gov/mesh/D010465",
        label="Perception",
        definition=(
            "The process by which the nature and meaning of sensory stimuli are "
            "recognized and interpreted."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D010465.json",
            "https://id.nlm.nih.gov/mesh/M0016252.json",
        ),
        source_version="MeSH RDF; descriptor updated 2020-06-09",
        source_sha256="b6813407955e6ea59762133189307de8f28f05bb53a44d0e7cde9ebe117bfa1c",
        reviewed_on="2026-10-23",
        rationale=(
            "Recognition and interpretation of sensory stimuli; prefer MeSH over the "
            "conscious-image-only NCI definition for computational and biological "
            "accounts."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/cognitive/perception.md",
            "knowledge_base/agents/GenericPOMDP/matrices/A_matrix.md",
            "knowledge_base/agents/GenericPOMDP/state_spaces/o_space.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="personality",
        iri="http://purl.obolibrary.org/obo/NCIT_C38002",
        label="Personality",
        definition=(
            "The complex of all the attributes--behavioral, temperamental, emotional "
            "and mental--that characterize a unique individual."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C38002?include=full",),
        source_version="26.09d",
        source_sha256="02774e8d5678433a57ee0aad3614d62b8d85b70b878fd05da5ab20c65e591a0f",
        reviewed_on="2026-10-23",
        rationale=(
            "Patterns of individual psychological characteristics, as discussed in "
            "cognitive and clinical sources."
        ),
        source_paths=(
            "knowledge_base/cognitive/personality_disorders.md",
            "knowledge_base/cognitive/temperament.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="pharmacology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16974",
        label="Pharmacology",
        definition=(
            "The study of pharmacologic agents. This includes the preparation, "
            "compounding, and dispensing of drugs, as well as the characteristics and "
            "properties of drugs, their effects on the body, and how they are "
            "metabolized and eliminated from the body."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16974?include=full",),
        source_version="26.09d",
        source_sha256="cbce7496f5770e6b87dc51be93a5a45eab3fbf3e09093635776aba90082145e4",
        reviewed_on="2026-10-23",
        rationale=("Study of drug properties and effects, not a particular therapeutic product."),
        source_paths=(
            "knowledge_base/cognitive/anesthetics.md",
            "knowledge_base/cognitive/stimulants.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="pheromones",
        iri="http://id.nlm.nih.gov/mesh/D010675",
        label="Pheromones",
        definition=(
            "Chemical substances, excreted by an organism into the environment, that "
            "elicit behavioral or physiological responses from other organisms of the "
            "same species. Perception of these chemical signals may be olfactory or by "
            "contact."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D010675.json",
            "https://id.nlm.nih.gov/mesh/M0016588.json",
        ),
        source_version="MeSH RDF; descriptor updated 2020-05-27",
        source_sha256="ca9c2a8277c7090a23840c3c07e422f23428caa16a73c01d99f7061ee11d310b",
        reviewed_on="2026-10-23",
        rationale=(
            "Chemical signals between members of the same species in social-insect "
            "communication; not all chemical signaling."
        ),
        source_paths=(
            "knowledge_base/biology/ant_communication_systems.md",
            "knowledge_base/biology/chemical_communication.md",
            "knowledge_base/biology/chemical_ecology.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="philosophy",
        iri="http://purl.obolibrary.org/obo/NCIT_C39794",
        label="Philosophy",
        definition=(
            "The field of study concerned with the rational investigation of questions "
            "about existence and knowledge and ethics."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C39794?include=full",),
        source_version="26.09d",
        source_sha256="a1959f2ec248d363f88cc91c162898f54f2702110922b34e0eed8e9e66536f5f",
        reviewed_on="2026-10-23",
        rationale=(
            "Investigation of existence, knowledge and ethics, including philosophical "
            "discussion of the free-energy principle."
        ),
        source_paths=(
            "knowledge_base/citations/andrews_2021.md",
            "knowledge_base/citations/bruineberg_2018.md",
            "knowledge_base/free_energy_principle/philosophy/AGENTS.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="physics",
        iri="http://purl.obolibrary.org/obo/NCIT_C16989",
        label="Physics",
        definition=(
            "The study of those aspects of nature which can be understood in a "
            "fundamental way in terms of elementary principles and laws; the laws and "
            "properties of matter, and the forces acting upon it."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16989?include=full",),
        source_version="26.09d",
        source_sha256="f1987010ea736e07c54f270225cf35858f90e9c15062de0d7b4d3e90a32ae380",
        reviewed_on="2026-10-23",
        rationale=("Laws of matter and energy across physical and mathematical treatments."),
        source_paths=(
            "knowledge_base/citations/da_costa_2021_bayesian.md",
            "knowledge_base/citations/friston_2019_particular.md",
            "knowledge_base/mathematics/brownian_motion.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="physiology",
        iri="http://purl.obolibrary.org/obo/NCIT_C16990",
        label="Physiology",
        definition=(
            "The study of the mechanical, physical, and biochemical functions of living organisms."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16990?include=full",),
        source_version="26.09d",
        source_sha256="7b17d6422435dc91f53b0a59ce100eb3309f194f36b189dbdbf95079cf57b758",
        reviewed_on="2026-10-23",
        rationale=("Mechanical, physical and biochemical functions of living organisms."),
        source_paths=(
            "knowledge_base/biology/physiology.md",
            "knowledge_base/biology/biomechanics.md",
            "knowledge_base/biology/plant_biology.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="population_genetics",
        iri="http://purl.obolibrary.org/obo/NCIT_C16628",
        label="Population Genetics",
        definition=(
            "The branch of genetics that focuses on gene distribution and transmission "
            "through populations."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16628?include=full",),
        source_version="26.09d",
        source_sha256="7bcf66cf8fe377bdf090215629fe7f0f6a2fa6fde6d57c28de312d2d5ab0619a",
        reviewed_on="2026-10-23",
        rationale=(
            "Gene distributions and transmission across populations, not individual genotypes."
        ),
        source_paths=(
            "knowledge_base/biology/genetic_diversity.md",
            "knowledge_base/biology/genetic_drift.md",
            "knowledge_base/biology/hardy_weinberg_equilibrium.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="prefrontal_cortex",
        iri="http://purl.obolibrary.org/obo/NCIT_C154779",
        label="Prefrontal Cortex",
        definition=(
            "The grey matter on the medial, lateral, and orbital surfaces of the "
            "anterior part of the frontal cortex, which play a role in complex "
            "cognitive control, emotion, and social behavior."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C154779?include=full",),
        source_version="26.09d",
        source_sha256="4f7a6b3ed37329a9077eaa2a559f37f93fa06878ea99be707b7d43d76b5b171e",
        reviewed_on="2026-10-23",
        rationale=(
            "The prefrontal brain region and its cognitive functions; not all frontal cortex."
        ),
        source_paths=(
            "knowledge_base/cognitive/goal_maintenance.md",
            "knowledge_base/cognitive/lateral_prefrontal.md",
            "knowledge_base/cognitive/set_shifting.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="probability",
        iri="http://purl.obolibrary.org/obo/NCIT_C54154",
        label="Probability",
        definition=("A measure of the expectation of the occurrence of a particular event."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54154?include=full",),
        source_version="26.09d",
        source_sha256="bae2dac48a446d94478169fd37614a1bc3f861acfb5bee3251eb7e53b39d7220",
        reviewed_on="2026-10-23",
        rationale=(
            "Event likelihood and its mathematical calculus in distributions, "
            "conditional independence and Markov blankets; not a specific probability "
            "distribution."
        ),
        source_paths=(
            "knowledge_base/agents/GenericPOMDP/matrices/B_matrix.md",
            "knowledge_base/cognitive/bayesian_brain.md",
            "knowledge_base/cognitive/bayesian_inference.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="problem_solving",
        iri="http://id.nlm.nih.gov/mesh/D011340",
        label="Problem Solving",
        definition=(
            "A learning situation involving more than one alternative from which a "
            "selection is made in order to attain a specific goal."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D011340.json",
            "https://id.nlm.nih.gov/mesh/M0017603.json",
        ),
        source_version="MeSH RDF; descriptor updated 1999-01-01",
        source_sha256="48c366e1fb1e236870de5ade9a1d1a965db53e331998485ff4f6c8ef8261862a",
        reviewed_on="2026-10-23",
        rationale=(
            "Selecting alternatives to reach a goal in cognitive and computational "
            "problem-solving sources."
        ),
        source_paths=(
            "knowledge_base/cognitive/difference_reduction.md",
            "knowledge_base/cognitive/divergent_thinking.md",
            "knowledge_base/cognitive/duncker_problem.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="psychiatry",
        iri="http://purl.obolibrary.org/obo/NCIT_C17026",
        label="Psychiatry",
        definition=(
            "The medical specialty that deals with the medical management of mental disorders."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17026?include=full",),
        source_version="26.09d",
        source_sha256="adddb3a9b07bd9facb178db4574e98d75b10d025a12b8a9c03708f49db89df59",
        reviewed_on="2026-10-23",
        rationale=("Medical management and computational investigation of mental disorders."),
        source_paths=(
            "knowledge_base/free_energy_principle/applications/psychiatry.md",
            "knowledge_base/cognitive/anxiety.md",
            "knowledge_base/cognitive/autism.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="psychometrics",
        iri="http://purl.obolibrary.org/obo/NCIT_C17034",
        label="Psychometrics",
        definition=(
            'The science of measuring "psychological" aspects of a person such as '
            "knowledge, skills, abilities, or personality. Measurement of these "
            "unobservable phenomena is difficult and much of the research and "
            "accumulated art of this discipline is designed to reliably define and then"
            " quantify."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17034?include=full",),
        source_version="26.09d",
        source_sha256="649385f815d63bb414ad607a0b00022e099285af7c5abeafcf8aa2a4787fd717",
        reviewed_on="2026-10-23",
        rationale=("Measurement of psychological characteristics and cognitive performance."),
        source_paths=(
            "knowledge_base/cognitive/construct_validity.md",
            "knowledge_base/cognitive/g_factor.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="quantum_mechanics",
        iri="http://purl.obolibrary.org/obo/NCIT_C19650",
        label="Quantum Mechanics",
        definition=("The branch of physics that accounts for matter at the atomic level."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C19650?include=full",),
        source_version="26.09d",
        source_sha256="cc1d1251c873a4ca4aa1ed6aad7cd9d4c4e907daffef77902a2287188e90c8d3",
        reviewed_on="2026-10-23",
        rationale=(
            "Quantum physical principles, including mathematical and computational applications."
        ),
        source_paths=(
            "knowledge_base/mathematics/quantum_mechanics.md",
            "knowledge_base/biology/quantum_biology.md",
            "knowledge_base/mathematics/berry_connection.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="reasoning",
        iri="http://purl.obolibrary.org/obo/NCIT_C86584",
        label="Reasoning",
        definition=("Thinking that is coherent and logical."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C86584?include=full",),
        source_version="26.09d",
        source_sha256="c7ac3f0c8c7a0066837237f69a5ada1074fab6d8e108dbc49c21e0790d5fe48f",
        reviewed_on="2026-10-23",
        rationale=(
            "Coherent logical thinking, including cognitive and philosophical accounts of it."
        ),
        source_paths=(
            "knowledge_base/cognitive/abductive_reasoning.md",
            "knowledge_base/cognitive/analogical_reasoning.md",
            "knowledge_base/cognitive/cognitive_functions.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="regulation",
        iri="http://purl.obolibrary.org/obo/NCIT_C48805",
        label="Regulatory Process",
        definition=(
            "A process that acts to control the rate or manner in which another process"
            " progresses or a product is formed."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C48805?include=full",),
        source_version="26.09d",
        source_sha256="02188291244ddbc0e8ecdb221bd33c0d6b84f4fe238ed7776060b49d44ba8be6",
        reviewed_on="2026-10-23",
        rationale=(
            "Biological and cognitive processes controlling other processes; explicitly"
            " reject the legal-rule sense of C68821."
        ),
        source_paths=(
            "knowledge_base/biology/gene_expression.md",
            "knowledge_base/biology/gene_regulation.md",
            "knowledge_base/biology/homeostasis.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="rehabilitation",
        iri="http://purl.obolibrary.org/obo/NCIT_C15315",
        label="Rehabilitation",
        definition=(
            "Restoration of the ability to function in a normal or near normal manner "
            "following disease, illness, or injury."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C15315?include=full",),
        source_version="26.09d",
        source_sha256="08338de391ecce97af7de3903c9ac99f887c95e0fddcf766a7ebaecb5032ab40",
        reviewed_on="2026-10-23",
        rationale=(
            "Restoration of function in clinical and cognitive sources; not a "
            "service-type identifier."
        ),
        source_paths=(
            "knowledge_base/cognitive/assistive_technology.md",
            "knowledge_base/cognitive/cochlear_implants.md",
            "knowledge_base/cognitive/exoskeletons.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="research",
        iri="http://purl.obolibrary.org/obo/NCIT_C15429",
        label="Research Activity",
        definition=(
            "Systematic investigation into a subject in order to discover facts, "
            "establish or revise a theory, or develop a plan of action based on the "
            "facts discovered."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C15429?include=full",),
        source_version="26.09d",
        source_sha256="7f9010b7ea5ea88813c940c36164c725d1b64e36b1a07b477063f2cf8e8a703a",
        reviewed_on="2026-10-23",
        rationale=(
            "Systematic investigation across research applications; use the active "
            "Research Activity concept, never retired C15319."
        ),
        source_paths=(
            "knowledge_base/cognitive/research_education.md",
            "knowledge_base/research/AGENTS.md",
            "knowledge_base/research/agi_development.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="retina",
        iri="http://purl.obolibrary.org/obo/NCIT_C12343",
        label="Retina",
        definition=(
            "A light-sensitive membrane that lines the back wall of the eyeball. The "
            "retina is continuous with the optic nerve and this way transmits optical "
            "images to the brain."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12343?include=full",),
        source_version="26.09d",
        source_sha256="c8ee453be8d7d7ed341446124d95a9529551170b71e94304f1682fb9687b4b4d",
        reviewed_on="2026-10-23",
        rationale=(
            "The sensory retinal tissue in visual processing; not a murine-specific concept."
        ),
        source_paths=(
            "knowledge_base/cognitive/light_adaptation.md",
            "knowledge_base/cognitive/visual_acuity.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="reward",
        iri="http://purl.obolibrary.org/obo/NCIT_C99754",
        label="Reward",
        definition=("Positive reinforcement for a desired action, behavior or response."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C99754?include=full",),
        source_version="26.09d",
        source_sha256="8cfc17ebe6e6524df3528615c8a9b328eaed9f9ef0bd24d30e76b2e1c80bbe6d",
        reviewed_on="2026-10-23",
        rationale=(
            "Positive reinforcement and reward-related motivation; no numeric utility "
            "value or pharmacological treatment is asserted."
        ),
        source_paths=(
            "knowledge_base/cognitive/addiction.md",
            "knowledge_base/cognitive/endogenous_opioids.md",
            "knowledge_base/cognitive/incentive_salience.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="semantics",
        iri="http://purl.obolibrary.org/obo/NCIT_C54194",
        label="Semantics",
        definition=(
            "The meaning of a string in some language; of or relating to the study of "
            "meaning and changes of meaning."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54194?include=full",),
        source_version="26.09d",
        source_sha256="587bf2ced8b160f9b87f8adfbfb9a5a3cd974609a252e499552aca180c93c27e",
        reviewed_on="2026-10-23",
        rationale=(
            "Meaning in language and formal representations, including compositional "
            "semantics and semantic information."
        ),
        source_paths=(
            "knowledge_base/cognitive/compositional_semantics.md",
            "knowledge_base/cognitive/pars_orbitalis.md",
            "knowledge_base/cognitive/predicate_argument.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="sleep",
        iri="http://purl.obolibrary.org/obo/NCIT_C73425",
        label="Sleep",
        definition=(
            "A natural and periodic state of rest during which consciousness of the "
            "world is suspended."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C73425?include=full",),
        source_version="26.09d",
        source_sha256="8f4addd58ef857d72f6293db13c4912c7c4a8370b42a01828cb10ebb8573fbb8",
        reviewed_on="2026-10-23",
        rationale=(
            "The biological periodic resting state, including sleep-related cognitive processes."
        ),
        source_paths=(
            "knowledge_base/cognitive/sleep.md",
            "knowledge_base/cognitive/rem_sleep.md",
            "knowledge_base/cognitive/sleep_memory_ai.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="social-cognition",
        iri="http://id.nlm.nih.gov/mesh/D000083282",
        label="Social Cognition",
        definition=(
            "Perception, interpretation, and categorizing self judgement of one's own "
            "social behaviors and group position and those of others."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D000083282.json",
            "https://id.nlm.nih.gov/mesh/M000678997.json",
        ),
        source_version="MeSH RDF; descriptor updated 2021-01-01",
        source_sha256="d04e4fcf4f7927bfd5442887c8bc6b77025ad8f5a82b04f12a199ec2a020fff3",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in social perception, trust and multi-agent "
            "cognitive accounts."
        ),
        source_paths=(
            "knowledge_base/cognitive/action_recognition.md",
            "knowledge_base/cognitive/multi_agent_active_inference.md",
            "knowledge_base/cognitive/negotiation.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="social_cognition",
        iri="http://id.nlm.nih.gov/mesh/D000083282",
        label="Social Cognition",
        definition=(
            "Perception, interpretation, and categorizing self judgement of one's own "
            "social behaviors and group position and those of others."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D000083282.json",
            "https://id.nlm.nih.gov/mesh/M000678997.json",
        ),
        source_version="MeSH RDF; descriptor updated 2021-01-01",
        source_sha256="d04e4fcf4f7927bfd5442887c8bc6b77025ad8f5a82b04f12a199ec2a020fff3",
        reviewed_on="2026-10-23",
        rationale=(
            "Perception and interpretation of one's own and others' social behavior, "
            "including computational accounts."
        ),
        source_paths=(
            "knowledge_base/free_energy_principle/cognitive/social_cognition.md",
            "knowledge_base/cognitive/automatic_imitation.md",
            "knowledge_base/cognitive/collaborative_learning.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="social_psychology",
        iri="http://purl.obolibrary.org/obo/NCIT_C17033",
        label="Social Psychology",
        definition=(
            "The branch of psychology that studies persons and their relationships with"
            " others and with groups and with society as a whole."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17033?include=full",),
        source_version="26.09d",
        source_sha256="188a5d85bb97e1f39bde5d6945beba82017bbaddf40849633a8a2e7cb96e71fd",
        reviewed_on="2026-10-23",
        rationale=("Psychological processes in social interaction, not all social science."),
        source_paths=(
            "knowledge_base/cognitive/causal_attribution.md",
            "knowledge_base/cognitive/social_comparison.md",
            "knowledge_base/cognitive/social_identity.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="speech_perception",
        iri="http://id.nlm.nih.gov/mesh/D013067",
        label="Speech Perception",
        definition=(
            "The process whereby an utterance is decoded into a representation in terms"
            " of linguistic units (sequences of phonetic segments which combine to form"
            " lexical and grammatical morphemes)."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D013067.json",
            "https://id.nlm.nih.gov/mesh/M0020254.json",
        ),
        source_version="MeSH RDF; descriptor updated 2020-06-22",
        source_sha256="042542bc20e22c740217ec964fd2d743ee615ba00f2dde721df8b469f77bbb6a",
        reviewed_on="2026-10-23",
        rationale=(
            "Recognition and interpretation of spoken language sounds in cognitive "
            "auditory sources."
        ),
        source_paths=(
            "knowledge_base/cognitive/cocktail_party.md",
            "knowledge_base/cognitive/distributional_learning.md",
            "knowledge_base/cognitive/lip_reading.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="statistics",
        iri="http://purl.obolibrary.org/obo/NCIT_C17167",
        label="Statistics",
        definition=(
            "A branch of applied mathematics concerned with the collection and "
            "interpretation of quantitative data and the use of probability theory to "
            "estimate population parameters."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17167?include=full",),
        source_version="26.09d",
        source_sha256="e0d3af17fdc841dc459c135eb69eec2b39539194e3757dfe6ff3bfbba7da4b2d",
        reviewed_on="2026-10-23",
        rationale=(
            "Quantitative data interpretation and probabilistic estimation; not the "
            "measured output of a particular study."
        ),
        source_paths=(
            "knowledge_base/mathematics/statistics.md",
            "knowledge_base/biology/mathematical_entomology.md",
            "knowledge_base/cognitive/collocation_analysis.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="stress",
        iri="http://purl.obolibrary.org/obo/NCIT_C35041",
        label="Stress",
        definition=(
            "The negative mental, emotional, and physical reactions that occur when "
            "environmental stressors are perceived as exceeding the individual's "
            "adaptive capacities."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C35041?include=full",),
        source_version="26.09d",
        source_sha256="9c428de347fa050f7cdfdb98950580d4f9136d7ca96864c099f8cb3222eda0f8",
        reviewed_on="2026-10-23",
        rationale=(
            "Psychological and physiological stress responses in appraisal and "
            "neuroendocrine contexts; not mechanical stress."
        ),
        source_paths=(
            "knowledge_base/biology/adrenaline.md",
            "knowledge_base/biology/cortisol.md",
            "knowledge_base/cognitive/neuroendocrine.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="synaptic_transmission",
        iri="http://id.nlm.nih.gov/mesh/D009435",
        label="Synaptic Transmission",
        definition=(
            "The communication from a NEURON to a target (neuron, muscle, or secretory "
            "cell) across a SYNAPSE. In chemical synaptic transmission, the presynaptic"
            " neuron releases a NEUROTRANSMITTER that diffuses across the synaptic "
            "cleft and binds to specific synaptic receptors, activating them. The "
            "activated receptors modulate specific ion channels and/or second-messenger"
            " systems in the postsynaptic cell. In electrical synaptic transmission, "
            "electrical signals are communicated as an ionic current flow across "
            "ELECTRICAL SYNAPSES."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D009435.json",
            "https://id.nlm.nih.gov/mesh/M0014683.json",
        ),
        source_version="MeSH RDF; descriptor updated 2016-06-23",
        source_sha256="81dc078fd362f8b5185d9ad67b658658a5e467724f8ea25def3e853f2c459927",
        reviewed_on="2026-10-23",
        rationale=(
            "Communication across synapses in nervous-system sources; not generic communication."
        ),
        source_paths=(
            "knowledge_base/biology/transmitter_release.md",
            "knowledge_base/cognitive/quantal_content.md",
            "knowledge_base/cognitive/receptor_desensitization.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="systems-biology",
        iri="http://id.nlm.nih.gov/mesh/D049490",
        label="Systems Biology",
        definition=(
            "Comprehensive, methodical analysis of complex biological systems by "
            "monitoring responses to perturbations of biological processes. Large "
            "scale, computerized collection and analysis of the data are used to "
            "develop and test models of biological systems."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D049490.json",
            "https://id.nlm.nih.gov/mesh/M0464791.json",
        ),
        source_version="MeSH RDF; descriptor updated 2008-07-08",
        source_sha256="694760389be03e05e56dc4e4eba22c2bf47db4f1d81c8146807720a194142d73",
        reviewed_on="2026-10-23",
        rationale=("Explicit hyphenated spelling in biological-system modeling contexts."),
        source_paths=(
            "knowledge_base/biology/systems_biology.md",
            "knowledge_base/biology/cell_biology.md",
            "knowledge_base/biology/developmental_systems.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="systems-theory",
        iri="http://id.nlm.nih.gov/mesh/D013598",
        label="Systems Theory",
        definition=(
            "Principles, models, and laws that apply to complex interrelationships and "
            "interdependencies of sets of linked components which form a functioning "
            "whole, a system. Any system may be composed of components which are "
            "systems in their own right (sub-systems), such as several organs within an"
            " individual organism."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D013598.json",
            "https://id.nlm.nih.gov/mesh/M0020984.json",
        ),
        source_version="MeSH RDF; descriptor updated 2004-07-07",
        source_sha256="5b6afcca66019548823525bf190d75a4e468546806ca09193c3f7c04c561e822",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in complex, adaptive and control-system "
            "sources; same systems-theory subject."
        ),
        source_paths=(
            "knowledge_base/systems/adaptive_systems.md",
            "knowledge_base/systems/control_systems.md",
            "knowledge_base/systems/message_passing_networks.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="systems_biology",
        iri="http://id.nlm.nih.gov/mesh/D049490",
        label="Systems Biology",
        definition=(
            "Comprehensive, methodical analysis of complex biological systems by "
            "monitoring responses to perturbations of biological processes. Large "
            "scale, computerized collection and analysis of the data are used to "
            "develop and test models of biological systems."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D049490.json",
            "https://id.nlm.nih.gov/mesh/M0464791.json",
        ),
        source_version="MeSH RDF; descriptor updated 2008-07-08",
        source_sha256="694760389be03e05e56dc4e4eba22c2bf47db4f1d81c8146807720a194142d73",
        reviewed_on="2026-10-23",
        rationale=(
            "Systematic analysis and modeling of interacting biological processes; not "
            "all systems theory."
        ),
        source_paths=(
            "knowledge_base/biology/README.md",
            "knowledge_base/biology/enzyme_cascades.md",
            "knowledge_base/biology/systems_medicine.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="systems_theory",
        iri="http://id.nlm.nih.gov/mesh/D013598",
        label="Systems Theory",
        definition=(
            "Principles, models, and laws that apply to complex interrelationships and "
            "interdependencies of sets of linked components which form a functioning "
            "whole, a system. Any system may be composed of components which are "
            "systems in their own right (sub-systems), such as several organs within an"
            " individual organism."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D013598.json",
            "https://id.nlm.nih.gov/mesh/M0020984.json",
        ),
        source_version="MeSH RDF; descriptor updated 2004-07-07",
        source_sha256="5b6afcca66019548823525bf190d75a4e468546806ca09193c3f7c04c561e822",
        reviewed_on="2026-10-23",
        rationale=(
            "Principles and models of interdependent components and functioning wholes;"
            " not every tag called systems."
        ),
        source_paths=(
            "knowledge_base/biology/systems_ecology.md",
            "knowledge_base/ontology/systems_ontology.md",
            "knowledge_base/systems/complex_systems.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="temporal_lobe",
        iri="http://purl.obolibrary.org/obo/NCIT_C12353",
        label="Temporal Lobe",
        definition=(
            "One of the cerebral lobes. It is located inferior to the frontal and "
            "parietal lobes and anterior to the occipital lobe."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C12353?include=full",),
        source_version="26.09d",
        source_sha256="8e7ff421303aa19853f6972ee87d413c276a8caee78749b6cba28d4b77888b8f",
        reviewed_on="2026-10-23",
        rationale=("The temporal cerebral lobe in language, memory and perception sources."),
        source_paths=(
            "knowledge_base/cognitive/primary_auditory.md",
            "knowledge_base/cognitive/wernicke_aphasia.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="theory_of_mind",
        iri="http://id.nlm.nih.gov/mesh/D056345",
        label="Theory of Mind",
        definition=(
            "The ability to attribute mental states (e.g., beliefs, desires, feelings, "
            "intentions, thoughts, etc.) to self and to others, allowing an individual "
            "to understand and infer behavior on the basis of the mental states. "
            "Difference or deficit in theory of mind is associated with ASPERGER "
            "SYNDROME; AUTISTIC DISORDER; and SCHIZOPHRENIA, etc."
        ),
        evidence=(
            "https://id.nlm.nih.gov/mesh/D056345.json",
            "https://id.nlm.nih.gov/mesh/M0526999.json",
        ),
        source_version="MeSH RDF; descriptor updated 2019-05-07",
        source_sha256="ccc3965f6fed7f12895db5071d8755f6934ede796a174da7a2c40d5eb57ed8d1",
        reviewed_on="2026-10-23",
        rationale=("Attributing mental states to oneself and others in social-cognitive sources."),
        source_paths=(
            "knowledge_base/cognitive/false_belief.md",
            "knowledge_base/cognitive/social_cognition_advanced.md",
            "knowledge_base/cognitive/social_cognition_detailed.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="thermodynamics",
        iri="http://purl.obolibrary.org/obo/NCIT_C17193",
        label="Thermodynamics",
        definition=(
            "The branch of physics concerned with the conversion of different forms of energy."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C17193?include=full",),
        source_version="26.09d",
        source_sha256="157a7cb3092ac0e2c3b9df254c4b54d30b311ba6bc61e94c8e43b9d9fd4891cd",
        reviewed_on="2026-10-23",
        rationale=(
            "Energy conversion in physical systems; this does not equate thermodynamic "
            "free energy with variational free energy."
        ),
        source_paths=(
            "knowledge_base/biology/biophysics.md",
            "knowledge_base/biology/energy_transduction.md",
            "knowledge_base/cognitive/free_energy_minimization.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="time",
        iri="http://purl.obolibrary.org/obo/NCIT_C25207",
        label="Time",
        definition=(
            "The continuum of experience in which events pass from the future through "
            "the present to the past."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C25207?include=full",),
        source_version="26.09d",
        source_sha256="1fff7ecb7750538e9ef41bedc85cd7199facceb7bf0b35a1459372fa3454fa20",
        reviewed_on="2026-10-23",
        rationale=(
            "Temporal experience and ordering discussed by mathematical and "
            "philosophical sources; not a unit or duration measurement."
        ),
        source_paths=(
            "knowledge_base/citations/friston_2012.md",
            "knowledge_base/cognitive/temporal_attention.md",
            "knowledge_base/cognitive/temporal_binding.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="vision",
        iri="http://purl.obolibrary.org/obo/NCIT_C38533",
        label="Sight",
        definition=("The faculty of vision, the perception of objects by the eye."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C38533?include=full",),
        source_version="26.09d",
        source_sha256="87cb5e259cbe0dcc2ed9c461a049765924c791a110329306f6dfc1e9b7b4cfae",
        reviewed_on="2026-10-23",
        rationale=(
            "All observed contexts concern visual perception, binocular vision or "
            "visual processing, not future plans or an organizational vision."
        ),
        source_paths=(
            "knowledge_base/cognitive/apparent_motion.md",
            "knowledge_base/cognitive/binocular_rivalry.md",
            "knowledge_base/cognitive/binocular_vision.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="visualization",
        iri="http://purl.obolibrary.org/obo/NCIT_C85439",
        label="Graphics Visualization",
        definition=("Computer methods for creating images, diagrams or animations."),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C85439?include=full",),
        source_version="26.09d",
        source_sha256="cba9972e772e27e67aee2bae4e9abeeccb4d4003a22de46ae74c33a9ffaec29f",
        reviewed_on="2026-10-23",
        rationale=(
            "Computer-generated diagrams, plots and visual environments in the observed"
            " sources; not mental imagery."
        ),
        source_paths=(
            "knowledge_base/cognitive/graph_visualization.md",
            "knowledge_base/cognitive/visualization_suite.md",
            "knowledge_base/cognitive/visualization_tools.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="working-memory",
        iri="http://purl.obolibrary.org/obo/NCIT_C88418",
        label="Working Memory",
        definition=(
            "Short term memory related to the storage, processing and recall of "
            "information required for the accomplishment of immediate cognitive tasks."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C88418?include=full",),
        source_version="26.09d",
        source_sha256="6bf99403ec87954752eab7e20e49ba1cedcf349e4f1dc32338079acf00620074",
        reviewed_on="2026-10-23",
        rationale=(
            "Explicit hyphenated spelling in cognitive working-memory sources; same "
            "immediate-task scope."
        ),
        source_paths=(
            "knowledge_base/cognitive/capacity_limits.md",
            "knowledge_base/cognitive/central_executive.md",
            "knowledge_base/cognitive/visuospatial_sketchpad.md",
        ),
        active=True,
    ),
    KeywordTerm(
        tag="working_memory",
        iri="http://purl.obolibrary.org/obo/NCIT_C88418",
        label="Working Memory",
        definition=(
            "Short term memory related to the storage, processing and recall of "
            "information required for the accomplishment of immediate cognitive tasks."
        ),
        evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C88418?include=full",),
        source_version="26.09d",
        source_sha256="6bf99403ec87954752eab7e20e49ba1cedcf349e4f1dc32338079acf00620074",
        reviewed_on="2026-10-23",
        rationale=(
            "Short-term storage, processing and recall needed for immediate cognitive "
            "tasks; does not encompass immune memory."
        ),
        source_paths=(
            "knowledge_base/citations/parr_friston_2017.md",
            "knowledge_base/cognitive/attentional_refreshing.md",
            "knowledge_base/cognitive/cognitive_inhibition.md",
        ),
        active=True,
    ),
)

KEYWORD_TERMS: dict[str, str] = build_keyword_terms(KEYWORD_RECORDS)
