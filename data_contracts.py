"""Versioned, provenance-preserving data contracts for PPI consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from numbers import Real
import os
import re
from typing import Any, Mapping, Optional, Sequence


PPI_SCHEMA_VERSION = 2
_ALLOWED_SOURCE_KINDS = {
    "online_snapshot",
    "offline_snapshot",
    "legacy_unverified",
    "synthetic_test",
}
_STRING_ID = re.compile(r"^(\d+)\.")
_TAXON_FIELDS = (
    "taxonomy_id",
    "ncbiTaxonId",
    "ncbi_taxon_id",
    "taxon_id",
)
_ENDPOINT_TAXON_FIELDS = {
    "a": ("taxonomy_id_a", "taxonomy_id_A", "ncbiTaxonId_A", "taxon_a"),
    "b": ("taxonomy_id_b", "taxonomy_id_B", "ncbiTaxonId_B", "taxon_b"),
}
_ONLINE_FIELDS = {
    "stringId_A", "stringId_B", "preferredName_A", "preferredName_B",
    "score", "combined_score", "ncbiTaxonId", "ncbi_taxon_id", "taxonomy_id",
    "taxonomy_id_A", "taxonomy_id_B", "taxon_id", "taxon_a", "taxon_b",
    "nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore",
    "retrieved_at", "source_kind", "source_record_id", "source_name", "provenance",
}
_OFFLINE_FIELDS = {
    "protein_a", "protein_b", "source_native_id_a", "source_native_id_b",
    "preferredName_A", "preferredName_B", "display_name_a", "display_name_b",
    "score", "combined_score", "ncbiTaxonId", "ncbi_taxon_id", "taxonomy_id",
    "taxonomy_id_A", "taxonomy_id_B", "taxonomy_id_a", "taxonomy_id_b",
    "taxon_id", "taxon_a", "taxon_b", "retrieved_at", "source_kind",
    "source_record_id", "source_name", "provenance",
}
_ENVELOPE_FIELDS = {
    "schema_version", "interactions", "source_kind", "alias_mappings",
    "retrieved_at", "source_uri", "source_sha256",
}
_TIME_FIELDS = frozenset({"retrieved_at", "created_at", "generated_at", "timestamp"})


def _plain(value: Any) -> Any:
    """Convert supported values to deterministic JSON data, marking non-finite floats."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            marker = "nan"
        else:
            marker = "+infinity" if value > 0 else "-infinity"
        return {"__nonfinite_float__": marker}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__unsupported_type__": f"{type(value).__module__}.{type(value).__qualname__}"}


def _without_fields(value: Any, excluded: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_fields(item, excluded)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [_without_fields(item, excluded) for item in value]
    return value


def canonical_json_sha256(value: Any, exclude_fields: Sequence[str] = ()) -> str:
    """Hash canonical UTF-8 JSON, with optional recursively excluded fields."""
    normalized = _without_fields(_plain(value), frozenset(exclude_fields))
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_taxonomy(value: Any) -> Optional[int]:
    if type(value) is int and value > 0:
        return value
    return None


def _record_taxonomy(record: Mapping[str, Any], side: str, native_id: str) -> tuple[Optional[int], Optional[str]]:
    explicit_values = []
    for field in _ENDPOINT_TAXON_FIELDS[side]:
        if field in record:
            value = _strict_taxonomy(record[field])
            if value is None:
                return None, field
            explicit_values.append((value, field))
    shared_values = []
    for field in _TAXON_FIELDS:
        if field in record:
            value = _strict_taxonomy(record[field])
            if value is None:
                return None, field
            shared_values.append((value, field))

    values = explicit_values + shared_values
    if values and any(value != values[0][0] for value, _ in values[1:]):
        return None, values[-1][1]

    parsed = _STRING_ID.match(native_id)
    parsed_taxonomy = int(parsed.group(1)) if parsed else None
    if parsed_taxonomy is not None and values and parsed_taxonomy != values[0][0]:
        return None, values[0][1]
    if values:
        return values[0][0], None
    if parsed_taxonomy is not None and parsed_taxonomy > 0:
        return parsed_taxonomy, None
    return None, "taxonomy_id"


def _score_value(record: Mapping[str, Any]) -> tuple[Optional[float], Optional[str]]:
    candidates = []
    if "score" in record:
        raw_score = record["score"]
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            return None, "score"
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return None, "score"
        candidates.append((score, "score"))
    if "combined_score" in record:
        raw_score = record["combined_score"]
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            return None, "combined_score"
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1000.0:
            return None, "combined_score"
        candidates.append((score / 1000.0, "combined_score"))
    if not candidates:
        return None, "score"
    if any(score != candidates[0][0] for score, _ in candidates[1:]):
        return None, "score"
    score = candidates[0][0]
    return (0.0 if score == 0 else score), None


def _score_issue_code(record: Mapping[str, Any], field: Optional[str]) -> str:
    if field is None:
        return "invalid_score"
    if field not in record:
        return "missing_score"
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, Real):
        return "invalid_score_type"
    if not math.isfinite(float(value)):
        return "non_finite_score"
    maximum = 1000.0 if field == "combined_score" else 1.0
    if not 0.0 <= float(value) <= maximum:
        return "score_out_of_range"
    if "score" in record and "combined_score" in record:
        score = record["score"]
        combined = record["combined_score"]
        if not isinstance(score, bool) and isinstance(score, Real) and math.isfinite(float(score)):
            if float(score) != float(combined) / 1000.0:
                return "conflicting_score_fields"
    return "invalid_score"


def _display_name(record: Mapping[str, Any], side: str) -> tuple[str, Optional[str]]:
    fields = (
        ("preferredName_A", "display_name_a") if side == "a"
        else ("preferredName_B", "display_name_b")
    )
    for field in fields:
        if field in record and record[field] is not None:
            if not isinstance(record[field], str):
                return "", field
            return record[field].strip(), None
    return "", None


@dataclass(frozen=True)
class CanonicalPpiEndpoint:
    taxonomy_id: int
    source_native_id: str
    display_name: str = ""

    @property
    def identity_key(self) -> tuple[int, str]:
        return (self.taxonomy_id, self.source_native_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "taxonomy_id": self.taxonomy_id,
            "source_native_id": self.source_native_id,
            "identity_key": [self.taxonomy_id, self.source_native_id],
            "display_name": self.display_name,
        }


@dataclass(frozen=True)
class PpiRecordProvenance:
    source_kind: str
    source_record_hash: str
    record_index: int
    source_fields: tuple[str, ...]
    input_endpoint_a: str
    input_endpoint_b: str
    mapping_sources: tuple[str, ...] = ()
    mapping_hashes: tuple[str, ...] = ()
    retrieved_at: Optional[str] = None


@dataclass(frozen=True)
class CanonicalPpiEdge:
    endpoint_a: CanonicalPpiEndpoint
    endpoint_b: CanonicalPpiEndpoint
    score: float
    provenance: tuple[PpiRecordProvenance, ...]
    semantic_sha256: str

    @property
    def source_record_hashes(self) -> tuple[str, ...]:
        return tuple(item.source_record_hash for item in self.provenance)

    @property
    def edge_key(self) -> tuple[tuple[int, str], tuple[int, str]]:
        return (self.endpoint_a.identity_key, self.endpoint_b.identity_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_a": self.endpoint_a.to_dict(),
            "endpoint_b": self.endpoint_b.to_dict(),
            "score": self.score,
            "source_record_hashes": list(self.source_record_hashes),
            "provenance": [asdict(item) for item in self.provenance],
            "semantic_sha256": self.semantic_sha256,
            "resolution_status": "resolved",
        }


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    record_index: Optional[int]
    source_kind: str
    source_record_hash: str
    field: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalPpiResult:
    schema_version: int
    status: str
    edges: tuple[CanonicalPpiEdge, ...]
    issues: tuple[ValidationIssue, ...]
    input_count: int
    accepted_record_count: int
    semantic_sha256: str
    provenance_sha256: str
    input_sha256: str

    @property
    def node_count(self) -> int:
        return len({endpoint.identity_key for edge in self.edges for endpoint in (edge.endpoint_a, edge.endpoint_b)})

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def unresolved_count(self) -> int:
        return sum(issue.code == "unresolved_alias" for issue in self.issues)

    def to_dict(self, include_edges: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "status": self.status,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "input_count": self.input_count,
            "accepted_record_count": self.accepted_record_count,
            "rejected_record_count": self.input_count - self.accepted_record_count,
            "unresolved_count": self.unresolved_count,
            "semantic_sha256": self.semantic_sha256,
            "provenance_sha256": self.provenance_sha256,
            "input_sha256": self.input_sha256,
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if include_edges:
            result["edges"] = [edge.to_dict() for edge in self.edges]
        return result


def _source_for_record(record: Mapping[str, Any], requested: Optional[str]) -> str:
    source = requested or record.get("source_kind")
    if source:
        return source if isinstance(source, str) else ""
    if "stringId_A" in record and "stringId_B" in record:
        return "online_snapshot"
    if "source_native_id_a" in record and "source_native_id_b" in record:
        return "offline_snapshot"
    return "legacy_unverified"


def _resolve_endpoint(
    record: Mapping[str, Any],
    side: str,
    alias_mappings: Mapping[str, Any],
) -> tuple[Optional[str], str, tuple[str, ...], tuple[str, ...], Optional[str]]:
    suffix = "A" if side == "a" else "B"
    direct_fields = (f"stringId_{suffix}", f"source_native_id_{side}")
    for field in direct_fields:
        if field in record:
            value = record[field]
            if not isinstance(value, str) or not value or value.strip() != value:
                return None, "", (), (), field
            return value, "", (), (), None

    alias_field = f"protein_{side}"
    alias = record.get(alias_field)
    if not isinstance(alias, str) or not alias or alias.strip() != alias:
        return None, "", (), (), alias_field
    mapping = alias_mappings.get(alias)
    if mapping is None:
        return None, alias, (), (), alias_field
    if not isinstance(mapping, Mapping):
        return None, alias, (), (), alias_field
    native_id = mapping.get("source_native_id")
    source = mapping.get("mapping_source")
    if (
        mapping.get("verified") is not True
        or not isinstance(native_id, str)
        or not native_id
        or native_id.strip() != native_id
        or not isinstance(source, str)
        or not source.strip()
    ):
        return None, alias, (), (), alias_field
    mapping_taxonomy = _strict_taxonomy(mapping.get("taxonomy_id"))
    if mapping_taxonomy is None:
        return None, alias, (), (), "taxonomy_id"
    return (
        native_id,
        alias,
        (source.strip(),),
        (canonical_json_sha256(mapping),),
        None,
    )


def _semantic_payload(
    endpoint_a: CanonicalPpiEndpoint,
    endpoint_b: CanonicalPpiEndpoint,
    score: float,
) -> dict[str, Any]:
    return {
        "schema_version": PPI_SCHEMA_VERSION,
        "endpoint_a": list(endpoint_a.identity_key),
        "endpoint_b": list(endpoint_b.identity_key),
        "score": score,
    }


def canonicalize_ppi(
    records: Any,
    *,
    source_kind: Optional[str] = None,
    alias_mappings: Optional[Mapping[str, Any]] = None,
) -> CanonicalPpiResult:
    """Normalize raw PPI records without turning display aliases into identities.

    Every input row is represented either by accepted edge provenance or by at
    least one ValidationIssue. Reversed duplicates merge provenance; conflicting
    weights for the same unordered pair are rejected as a group.
    """
    if isinstance(records, CanonicalPpiResult):
        return records
    if isinstance(records, list) and records and all(isinstance(item, CanonicalPpiEdge) for item in records):
        input_hash = canonical_json_sha256([edge.to_dict() for edge in records])
        return _build_result(records, [], len(records), len(records), input_hash)
    mappings = alias_mappings if isinstance(alias_mappings, Mapping) else {}
    source_hash = canonical_json_sha256(records)
    envelope_source_kind = None
    source_schema_version = None
    root_issues: list[ValidationIssue] = []

    if isinstance(records, Mapping) and "interactions" in records:
        envelope = records
        unknown_envelope = sorted(set(envelope) - _ENVELOPE_FIELDS)
        if unknown_envelope:
            root_issues.append(ValidationIssue(
                "unknown_fields",
                f"Unknown PPI envelope fields: {', '.join(unknown_envelope)}",
                None,
                source_kind or "legacy_unverified",
                source_hash,
                unknown_envelope[0],
            ))
        source_schema_version = envelope.get("schema_version")
        if source_schema_version is not None and source_schema_version != PPI_SCHEMA_VERSION:
            rows = envelope.get("interactions")
            rows = rows if isinstance(rows, list) else []
            issues = [
                ValidationIssue(
                    "unsupported_schema_version",
                    f"Unsupported PPI schema version {source_schema_version!r}; expected {PPI_SCHEMA_VERSION}.",
                    index,
                    source_kind or str(envelope.get("source_kind") or "legacy_unverified"),
                    canonical_json_sha256(row),
                    "schema_version",
                )
                for index, row in enumerate(rows)
            ]
            if not issues:
                issues.append(ValidationIssue(
                    "unsupported_schema_version",
                    f"Unsupported PPI schema version {source_schema_version!r}; expected {PPI_SCHEMA_VERSION}.",
                    None,
                    source_kind or "legacy_unverified",
                    source_hash,
                    "schema_version",
                ))
            return _build_result([], root_issues + issues, len(rows), 0, source_hash)
        if source_kind is None:
            envelope_source_kind = envelope.get("source_kind")
        if alias_mappings is None and isinstance(envelope.get("alias_mappings"), Mapping):
            mappings = envelope["alias_mappings"]
        rows = envelope.get("interactions")
        if not isinstance(rows, list):
            issue = ValidationIssue(
                "invalid_collection",
                "PPI interactions must be a list.",
                None,
                source_kind or "legacy_unverified",
                source_hash,
                "interactions",
            )
            return _build_result([], root_issues + [issue], 0, 0, source_hash)
    elif isinstance(records, list):
        rows = records
    else:
        issue = ValidationIssue(
            "invalid_collection",
            "PPI input must be a list or a versioned object containing interactions.",
            None,
            source_kind or "legacy_unverified",
            source_hash,
            None,
        )
        return _build_result([], root_issues + [issue], 0, 0, source_hash)

    requested_source = source_kind or envelope_source_kind
    input_rows = []
    issues = list(root_issues)
    candidate_records = []
    input_count = len(rows)

    for index, row in enumerate(rows):
        row_hash = canonical_json_sha256(row)
        if not isinstance(row, Mapping):
            issues.append(ValidationIssue(
                "invalid_record_type",
                "Each PPI record must be an object.",
                index,
                requested_source or "legacy_unverified",
                row_hash,
                None,
            ))
            input_rows.append(_plain(row))
            continue

        current_source = _source_for_record(row, requested_source)
        input_rows.append(_plain(row))
        if current_source not in _ALLOWED_SOURCE_KINDS:
            issues.append(ValidationIssue(
                "unknown_source_kind",
                f"Unsupported source_kind {current_source!r}.",
                index,
                current_source,
                row_hash,
                "source_kind",
            ))
            continue
        allowed = _ONLINE_FIELDS | _OFFLINE_FIELDS
        unknown = sorted(set(row) - allowed)
        if unknown:
            issues.append(ValidationIssue(
                "unknown_fields",
                f"Unknown PPI fields: {', '.join(unknown)}",
                index,
                current_source,
                row_hash,
                unknown[0],
            ))
            continue

        resolved = []
        unresolved = False
        row_error = None
        mapping_sources = []
        mapping_hashes = []
        original_endpoints = []
        for side in ("a", "b"):
            native_id, original_id, sources, mapping_hashes_for_endpoint, field = _resolve_endpoint(
                row, side, mappings
            )
            original_endpoints.append(original_id or native_id or "")
            if native_id is None:
                alias_field = f"protein_{side}"
                has_alias = isinstance(row.get(alias_field), str) and bool(row.get(alias_field))
                code = "unresolved_alias" if has_alias and field == alias_field else (
                    "untrusted_alias_mapping" if has_alias else "missing_endpoint"
                )
                message = (
                    f"Alias in {alias_field} has no verified source-native ID mapping."
                    if code == "unresolved_alias"
                    else f"Endpoint could not be resolved from {field or alias_field}."
                )
                issues.append(ValidationIssue(code, message, index, current_source, row_hash, field))
                unresolved = True
                row_error = True
                break

            taxonomy, taxonomy_field = _record_taxonomy(row, side, native_id)
            if original_id:
                mapping = mappings[original_id]
                mapping_taxonomy = _strict_taxonomy(mapping.get("taxonomy_id"))
                row_has_taxonomy = any(
                    field in row
                    for field in _TAXON_FIELDS + _ENDPOINT_TAXON_FIELDS[side]
                )
                if not row_has_taxonomy and taxonomy is None:
                    taxonomy = mapping_taxonomy
                    taxonomy_field = None
                elif taxonomy is not None and mapping_taxonomy != taxonomy:
                    taxonomy = None
                    taxonomy_field = "taxonomy_id"
            if taxonomy is None:
                code = "taxonomy_conflict" if taxonomy_field and taxonomy_field != "taxonomy_id" else "invalid_taxonomy"
                issues.append(ValidationIssue(
                    code,
                    "Endpoint taxonomy is absent, invalid, or conflicts with its source-native ID or mapping.",
                    index,
                    current_source,
                    row_hash,
                    taxonomy_field or "taxonomy_id",
                ))
                row_error = True
                break
            display_name, display_error = _display_name(row, side)
            if display_error:
                issues.append(ValidationIssue(
                    "invalid_display_name",
                    "Display names must be strings and cannot define endpoint identity.",
                    index,
                    current_source,
                    row_hash,
                    display_error,
                ))
                row_error = True
                break
            resolved.append(CanonicalPpiEndpoint(taxonomy, native_id, display_name))
            mapping_sources.extend(sources)
            mapping_hashes.extend(mapping_hashes_for_endpoint)

        if row_error:
            continue
        if resolved[0].taxonomy_id != resolved[1].taxonomy_id:
            issues.append(ValidationIssue(
                "taxonomy_conflict",
                "PPI endpoints have different taxonomy IDs.",
                index,
                current_source,
                row_hash,
                "taxonomy_id",
            ))
            continue

        score, score_field = _score_value(row)
        if score is None:
            code = _score_issue_code(row, score_field)
            issues.append(ValidationIssue(
                code,
                "PPI score must be a finite numeric value in [0, 1] (or combined_score in [0, 1000]).",
                index,
                current_source,
                row_hash,
                score_field,
            ))
            continue

        endpoint_a, endpoint_b = resolved
        if endpoint_b.identity_key < endpoint_a.identity_key:
            endpoint_a, endpoint_b = endpoint_b, endpoint_a
        provenance = PpiRecordProvenance(
            source_kind=current_source,
            source_record_hash=row_hash,
            record_index=index,
            source_fields=tuple(sorted(str(field) for field in row)),
            input_endpoint_a=original_endpoints[0],
            input_endpoint_b=original_endpoints[1],
            mapping_sources=tuple(sorted(mapping_sources)),
            mapping_hashes=tuple(sorted(mapping_hashes)),
            retrieved_at=row.get("retrieved_at") if isinstance(row.get("retrieved_at"), str) else None,
        )
        candidate_records.append((endpoint_a, endpoint_b, score, provenance))

    by_pair: dict[tuple[tuple[int, str], tuple[int, str]], list[tuple[CanonicalPpiEndpoint, CanonicalPpiEndpoint, float, PpiRecordProvenance]]] = {}
    for candidate in candidate_records:
        endpoint_a, endpoint_b, _, _ = candidate
        pair_key = (endpoint_a.identity_key, endpoint_b.identity_key)
        by_pair.setdefault(pair_key, []).append(candidate)

    merged_edges = []
    accepted_record_count = 0
    for pair_key in sorted(by_pair):
        pair_records = by_pair[pair_key]
        scores = {item[2] for item in pair_records}
        if len(scores) > 1:
            for _, _, _, provenance in pair_records:
                issues.append(ValidationIssue(
                    "conflicting_duplicate_score",
                    "Records for the same unordered PPI pair have conflicting scores; all are quarantined.",
                    provenance.record_index,
                    provenance.source_kind,
                    provenance.source_record_hash,
                    "score",
                ))
            continue

        accepted_record_count += len(pair_records)
        first_a, first_b, score, _ = pair_records[0]
        names_a = sorted({item[0].display_name for item in pair_records if item[0].display_name})
        names_b = sorted({item[1].display_name for item in pair_records if item[1].display_name})
        endpoint_a = CanonicalPpiEndpoint(first_a.taxonomy_id, first_a.source_native_id, names_a[0] if names_a else "")
        endpoint_b = CanonicalPpiEndpoint(first_b.taxonomy_id, first_b.source_native_id, names_b[0] if names_b else "")
        provenance = tuple(sorted(
            (item[3] for item in pair_records),
            key=lambda item: (item.source_kind, item.source_record_hash, item.record_index),
        ))
        semantic_hash = canonical_json_sha256(_semantic_payload(endpoint_a, endpoint_b, score))
        merged_edges.append(CanonicalPpiEdge(endpoint_a, endpoint_b, score, provenance, semantic_hash))

    return _build_result(merged_edges, issues, input_count, accepted_record_count, source_hash)


def _build_result(
    edges: Sequence[CanonicalPpiEdge],
    issues: Sequence[ValidationIssue],
    input_count: int,
    accepted_record_count: int,
    input_sha256: str,
) -> CanonicalPpiResult:
    sorted_edges = tuple(sorted(edges, key=lambda edge: (edge.edge_key, edge.score)))
    sorted_issues = tuple(sorted(
        issues,
        key=lambda issue: (
            -1 if issue.record_index is None else issue.record_index,
            issue.code,
            issue.field or "",
        ),
    ))
    semantic_payload = [
        _semantic_payload(edge.endpoint_a, edge.endpoint_b, edge.score)
        for edge in sorted_edges
    ]
    provenance_payload = [
        {
            "semantic_sha256": edge.semantic_sha256,
            "provenance": [asdict(item) for item in edge.provenance],
        }
        for edge in sorted_edges
    ]
    if input_count == 0 and not sorted_issues:
        status = "empty"
    elif sorted_edges and sorted_issues:
        status = "partial"
    elif sorted_edges:
        status = "ready"
    elif sorted_issues and all(issue.code == "unresolved_alias" for issue in sorted_issues):
        status = "unresolved"
    else:
        status = "invalid"
    return CanonicalPpiResult(
        schema_version=PPI_SCHEMA_VERSION,
        status=status,
        edges=sorted_edges,
        issues=sorted_issues,
        input_count=input_count,
        accepted_record_count=accepted_record_count,
        semantic_sha256=canonical_json_sha256(semantic_payload),
        provenance_sha256=canonical_json_sha256(provenance_payload),
        input_sha256=input_sha256,
    )


def read_ppi_records(path: os.PathLike[str] | str) -> Any:
    """Read UTF-8 JSON or UTF-8-SIG CSV PPI data, preserving non-ASCII IDs."""
    filename = os.fspath(path)
    extension = os.path.splitext(filename)[1].lower()
    if extension == ".json":
        with open(filename, "r", encoding="utf-8-sig") as stream:
            return json.load(stream)
    if extension == ".csv":
        with open(filename, "r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        numeric_fields = {
            "score", "combined_score", "taxonomy_id", "ncbiTaxonId", "ncbi_taxon_id",
            "taxon_id", "taxonomy_id_a", "taxonomy_id_b", "taxonomy_id_A", "taxonomy_id_B",
            "ncbiTaxonId_A", "ncbiTaxonId_B", "taxon_a", "taxon_b",
        }
        for row in rows:
            for field in numeric_fields.intersection(row):
                value = row[field]
                if value == "":
                    continue
                try:
                    if field in {
                        "taxonomy_id", "ncbiTaxonId", "ncbi_taxon_id", "taxon_id",
                        "taxonomy_id_a", "taxonomy_id_b", "taxonomy_id_A", "taxonomy_id_B",
                        "ncbiTaxonId_A", "ncbiTaxonId_B", "taxon_a", "taxon_b",
                    }:
                        if re.fullmatch(r"[0-9]+", value) is None:
                            continue
                        row[field] = int(value)
                    else:
                        row[field] = float(value)
                except (TypeError, ValueError):
                    continue
        return rows
    raise ValueError("PPI input file must end in .json or .csv")
