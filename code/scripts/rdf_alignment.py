"""Validate source-reviewed SKOS mappings without doing network I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlsplit

RELATIONS = frozenset({"exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch"})


@dataclass(frozen=True)
class Mapping:
    """A correspondence with a recorded source comparison, not a label guess."""

    target: str
    relation: str
    evidence: str
    rationale: str
    reviewed_on: str


def checked_iri(value: Any) -> str:
    """Require an absolute HTTP(S) IRI safe for RDF serialization."""
    if not isinstance(value, str) or re.search(r'[\s<>"{}|^`\\\x00-\x1f]', value):
        raise ValueError(f"invalid mapping IRI: {value!r}")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ValueError(f"invalid mapping IRI: {value!r}")
    return value


def reviewed_mappings(value: Any) -> list[Mapping]:
    """Parse ``external_mappings``; reject incomplete or conflicting assertions.

    Every entry requires target, SKOS relation name, evidence URL, rationale and
    ISO review date. The record documents a review; it cannot prove its truth.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"external_mappings must be a list, got {value!r}")
    required = {"target", "relation", "evidence", "rationale", "reviewed_on"}
    mappings: list[Mapping] = []
    targets: set[str] = set()
    for record in value:
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"mapping requires exactly {sorted(required)}: {record!r}")
        target = checked_iri(record["target"])
        evidence = checked_iri(record["evidence"])
        relation = record["relation"]
        if not isinstance(relation, str) or relation not in RELATIONS:
            raise ValueError(f"unsupported SKOS mapping relation: {relation!r}")
        rationale = record["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"mapping rationale must be nonempty: {rationale!r}")
        reviewed = str(record["reviewed_on"])
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reviewed):
            raise ValueError(f"invalid mapping review date: {reviewed!r}")
        date.fromisoformat(reviewed)
        if target in targets:
            raise ValueError(f"duplicate or conflicting mapping target: {target!r}")
        targets.add(target)
        mappings.append(Mapping(target, relation, evidence, rationale.strip(), reviewed))
    return mappings
