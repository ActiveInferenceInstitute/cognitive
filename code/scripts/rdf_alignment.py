"""Validate source-reviewed SKOS mappings without doing network I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from ipaddress import ip_address
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
    """Require an HTTP(S) IRI with a DNS/IP host, valid port and escapes."""
    if not isinstance(value, str) or re.search(
        r'[\s<>"{}|^`\\\x00-\x1f\x7f-\x9f\ud800-\udfff]', value
    ):
        raise ValueError(f"invalid mapping IRI: {value!r}")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or "@" in parsed.netloc
            or "%" in host
            or parsed.netloc.endswith(":")
            or (":" in host and not re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", parsed.netloc))
            or re.search(r"%(?![0-9a-fA-F]{2})", value)
            or (port is not None and port < 1)
        ):
            raise ValueError("invalid HTTP(S) authority, port or escapes")
        if ":" in host or re.fullmatch(r"[0-9.]+", host):
            ip_address(host)
        else:
            ascii_host = (host[:-1] if host.endswith(".") else host).encode("idna").decode("ascii")
            if len(ascii_host) > 253 or any(
                not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
                for label in ascii_host.split(".")
            ):
                raise ValueError("invalid DNS hostname")
    except ValueError as exc:
        raise ValueError(f"invalid mapping IRI: {value!r}") from exc
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
