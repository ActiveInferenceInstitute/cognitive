from datetime import date

import pytest
from scripts.rdf_alignment import checked_iri, reviewed_mappings


def record():
    return {
        "target": "http://purl.obolibrary.org/obo/CHEBI_17650",
        "relation": "exactMatch",
        "evidence": "https://www.ebi.ac.uk/chebi/CHEBI:17650",
        "rationale": "The source and target both denote cortisol, rather than cortisol secretion.",
        "reviewed_on": "2026-10-20",
    }


def test_complete_mapping_retains_evidence_and_accepts_yaml_date():
    value = record()
    value["reviewed_on"] = date(2026, 10, 20)
    result = reviewed_mappings([value])[0]
    assert result.target == value["target"]
    assert result.relation == "exactMatch"
    assert result.reviewed_on == "2026-10-20"
    assert reviewed_mappings(None) == reviewed_mappings([]) == []


@pytest.mark.parametrize(
    "value",
    [
        "Q42",
        "http:///broken",
        "https://x/a b",
        "https://x/<a>",
        "file:///a",
        "https://user@x/a",
        None,
        4,
    ],
)
def test_mapping_iri_rejects_unsafe_or_non_http_identifiers(value):
    with pytest.raises(ValueError, match="IRI"):
        checked_iri(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("relation", "sameAs"),
        ("relation", []),
        ("rationale", ""),
        ("rationale", 3),
        ("reviewed_on", "2026-02-30"),
        ("reviewed_on", "20261020"),
        ("target", "CHEBI:17650"),
        ("evidence", "not a URL"),
    ],
)
def test_incomplete_reviews_cannot_emit_mappings(field, value):
    mapping = record()
    mapping[field] = value
    with pytest.raises(ValueError):
        reviewed_mappings([mapping])


@pytest.mark.parametrize(
    "value", [{}, ["bad"], [{"target": "https://example.org/a"}], [dict(record(), extra="bad")]]
)
def test_review_record_schema_is_strict(value):
    with pytest.raises(ValueError):
        reviewed_mappings(value)


def test_duplicate_or_conflicting_targets_are_rejected():
    for relation in ("exactMatch", "broadMatch"):
        with pytest.raises(ValueError, match="conflicting"):
            reviewed_mappings([record(), dict(record(), relation=relation)])
