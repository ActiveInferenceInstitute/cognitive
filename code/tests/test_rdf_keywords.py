"""Exact subject lookup must reuse reviewed terms and retain ambiguous tags."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
from scripts import rdf_keywords as subject_terms
from scripts.rdf_keywords import (
    KEYWORD_RECORDS,
    KEYWORD_TERMS,
    build_keyword_terms,
    keyword_iri,
)


@pytest.mark.parametrize(
    ("tag", "code"),
    [("emotion", "C54266"), ("genetics", "C16624"), ("mathematics", "C16825")],
)
def test_requested_nci_subjects(tag: str, code: str) -> None:
    """The three requested terms must resolve to their verified NCI concepts."""
    assert keyword_iri(tag) == "http://purl.obolibrary.org/obo/NCIT_" + code


def test_existing_terms_preserved() -> None:
    """Preserve the two pre-existing subject mappings byte for byte."""
    assert KEYWORD_TERMS["model_fitting"] == "http://purl.obolibrary.org/obo/STATO_0000218"
    assert keyword_iri("computational_modeling") == "http://purl.obolibrary.org/obo/NCIT_C16461"


def test_active_research_replaces_retired_homonym() -> None:
    """Use the active source record and reject NCI's retired Research concept."""
    record = next(r for r in KEYWORD_RECORDS if r.tag == "research")
    assert keyword_iri("research") == "http://purl.obolibrary.org/obo/NCIT_C15429"
    with pytest.raises(ValueError, match="verified active status"):
        replace(
            record,
            iri="http://purl.obolibrary.org/obo/NCIT_C15319",
            evidence=("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C15319?include=full",),
            active=False,
        )


@pytest.mark.parametrize(
    "tag",
    [
        "inference",
        "precision",
        "precision_weighting",
        "free_energy",
        "Emotion",
        " emotion",
        "emotion ",
        "unknown",
    ],
)
def test_unreviewed_spellings_and_ambiguous_tags_stay_local(tag: str) -> None:
    """No fuzzy, whitespace or case normalization can broaden a mapping."""
    assert keyword_iri(tag) is None


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "memory",
        "plasticity",
        "neural_networks",
        "ontology",
        "topology",
        "reading",
        "self",
        "integration",
        "adaptation",
        "graphical_models",
        "statistical_independence",
        "citation",
        "redirect",
        "active_inference",
        "variational_free_energy",
        "free-energy",
        "thermodynamic_free_energy",
    ],
)
def test_scope_collisions_and_domain_specific_subjects_stay_local(tag: str) -> None:
    """Homonymous and narrower alternatives cannot substitute for these tags."""
    assert keyword_iri(tag) is None


@pytest.mark.parametrize("tag", [None, 7, False, [], {}, b"emotion"])
def test_lookup_rejects_non_strings(tag: Any) -> None:
    """Malformed source input is an error rather than an apparently unknown tag."""
    with pytest.raises(TypeError, match="must be a string"):
        keyword_iri(tag)


@pytest.mark.parametrize("tag", sorted(KEYWORD_TERMS))
def test_each_record_round_trips_exactly(tag: str) -> None:
    """Each registered spelling resolves, while added punctuation and case do not."""
    assert keyword_iri(tag) == KEYWORD_TERMS[tag]
    assert keyword_iri(tag.upper()) is None
    assert keyword_iri(tag + " ") is None
    assert keyword_iri(" " + tag) is None
    assert keyword_iri(tag + "s-unreviewed") is None


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("information_theory", "information-theory", "http://id.nlm.nih.gov/mesh/D007257"),
        ("social_cognition", "social-cognition", "http://id.nlm.nih.gov/mesh/D000083282"),
        (
            "bayesian_inference",
            "bayesian-inference",
            "http://purl.obolibrary.org/obo/STATO_0000706",
        ),
        ("matrix", "matrices", "http://purl.obolibrary.org/obo/STATO_0000642"),
        ("behavior", "behaviour", "http://purl.obolibrary.org/obo/NCIT_C16326"),
    ],
)
def test_explicitly_reviewed_source_spellings(left: str, right: str, expected: str) -> None:
    """Aliases require their own source records and retain a reviewed target."""
    assert keyword_iri(left) == keyword_iri(right) == expected
    assert keyword_iri(left.replace("_", " ") + " ") is None


def test_review_metadata_and_accepting_controls() -> None:
    """All metadata reconstructs real records and a small valid registry passes."""
    assert build_keyword_terms(KEYWORD_RECORDS) == KEYWORD_TERMS
    for record in KEYWORD_RECORDS:
        assert record.active is True
        assert record.reviewed_on == "2026-10-23"
        assert record.evidence and record.definition and record.rationale
        assert record.source_paths
        assert replace(record) == record
    one = KEYWORD_RECORDS[0]
    assert build_keyword_terms([one]) == {one.tag: one.iri}
    assert build_keyword_terms([]) == {}
    assert replace(one, reviewed_on="2024-02-29").reviewed_on == "2024-02-29"


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_or_conflicting_source_records_are_rejected(conflicting: bool) -> None:
    """Reject repeated source spellings even if their targets happen to agree."""
    one = KEYWORD_RECORDS[0]
    other = replace(KEYWORD_RECORDS[1], tag=one.tag) if conflicting else one
    with pytest.raises(ValueError, match="duplicate keyword source tag"):
        build_keyword_terms([one, other])


@pytest.mark.parametrize("record", [None, "emotion", {}, 3])
def test_registry_rejects_unstructured_records(record: Any) -> None:
    """A malformed registry entry must never disappear from lookup silently."""
    with pytest.raises(TypeError, match="KeywordTerm"):
        build_keyword_terms([record])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tag", ""),
        ("tag", " emotion"),
        ("tag", "emotion\n"),
        ("tag", "emo\x7ftion"),
        ("tag", None),
        ("label", ""),
        ("definition", ""),
        ("rationale", ""),
        ("source_version", ""),
        ("active", False),
        ("active", 1),
        ("source_sha256", "abc"),
        ("source_sha256", "A" * 64),
        ("source_sha256", None),
        ("reviewed_on", "2026-02-30"),
        ("reviewed_on", "20261023"),
        ("reviewed_on", "2026-10-23T00:00:00"),
        ("reviewed_on", None),
        ("iri", "relative"),
        ("iri", "https://example.org/subject"),
        ("iri", "https://user:password@example.org/subject"),
        ("iri", "http://purl.obolibrary.org/obo/NCIT_C77840>"),
        ("iri", "http://purl.obolibrary.org/obo/NCIT_C77840%ZZ"),
        ("iri", "https://example.org:99999/subject"),
        ("iri", "https://example.org:443/subject"),
        ("iri", "http://purl.obolibrary.org/obo/NCIT_C77840 extra"),
        ("evidence", ()),
        ("evidence", []),
        ("evidence", ("https://example.org/source",)),
        ("evidence", ("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C54266?include=full",)),
        ("source_paths", ()),
        ("source_paths", []),
        ("source_paths", ("/knowledge_base/biology/topic.md",)),
        ("source_paths", ("knowledge_base/../secrets.md",)),
        ("source_paths", ("knowledge_base/topic.txt",)),
        ("source_paths", ("knowledge_base/biology/topic.md", "knowledge_base/biology/topic.md")),
        ("source_paths", ("knowledge_base/biology\\topic.md",)),
    ],
)
def test_malformed_or_inactive_review_records_are_rejected(field: str, value: Any) -> None:
    """Nearest valid records pass above; each real malformed variant raises."""
    with pytest.raises(ValueError):
        replace(KEYWORD_RECORDS[0], **{field: value})


def test_vocabulary_specific_evidence_controls() -> None:
    """MeSH requires its scope record; STATO requires an immutable official file."""
    mesh = next(r for r in KEYWORD_RECORDS if r.tag == "attention")
    with pytest.raises(ValueError, match="preferred-concept"):
        replace(mesh, evidence=mesh.evidence[:1])
    with pytest.raises(ValueError, match="preferred-concept"):
        replace(mesh, evidence=(mesh.evidence[0], "https://example.org/scope"))
    stato = next(r for r in KEYWORD_RECORDS if r.tag == "model_fitting")
    with pytest.raises(ValueError, match="pin the official"):
        replace(
            stato,
            evidence=(
                stato.evidence[0].replace("a5910e6115217a88ee86e2666e12983ef2ef2a42", "main"),
            ),
        )
    with pytest.raises(ValueError, match="pin the official"):
        replace(stato, evidence=stato.evidence + stato.evidence)


def test_records_are_immutable() -> None:
    """An existing review cannot be altered without creating a validated record."""
    with pytest.raises(FrozenInstanceError):
        KEYWORD_RECORDS[0].active = False


def test_import_is_standard_library_only_and_does_not_open_network(tmp_path: Path) -> None:
    """An isolated interpreter with a real rejecting audit hook imports the module."""
    script = """
import runpy
import sys
def reject_network(event, args):
    if event.startswith('socket.'):
        raise RuntimeError('network forbidden')
sys.addaudithook(reject_network)
import socket
try:
    socket.socket()
except RuntimeError as error:
    assert str(error) == 'network forbidden'
else:
    raise AssertionError('negative network control did not fail')
namespace = runpy.run_path(sys.argv[1])
assert namespace['keyword_iri']('emotion').endswith('NCIT_C54266')
assert namespace['keyword_iri']('precision_weighting') is None
print(len(namespace['KEYWORD_RECORDS']))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(Path(subject_terms.__file__).resolve())],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert int(result.stdout.strip()) == len(KEYWORD_RECORDS)
