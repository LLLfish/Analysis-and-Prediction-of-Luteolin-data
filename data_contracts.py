"""Versioned, provenance-preserving data contracts for PPI consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
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


def _plain(value: Any, _depth: int = 0, _active: Optional[set[int]] = None) -> Any:
    """Convert values to deterministic JSON data with cycle and depth guards."""
    if _depth > 64:
        return {"__depth_limit__": f"{type(value).__module__}.{type(value).__qualname__}"}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            marker = "nan"
        else:
            marker = "+infinity" if value > 0 else "-infinity"
        return {"__nonfinite_float__": marker}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    active = _active if _active is not None else set()
    value_id = id(value)
    if value_id in active:
        return {"__cycle__": f"{type(value).__module__}.{type(value).__qualname__}"}
    active.add(value_id)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: _plain(getattr(value, item.name), _depth + 1, active)
                for item in fields(value)
            }
        if isinstance(value, Mapping):
            return {
                str(key): _plain(item, _depth + 1, active)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_plain(item, _depth + 1, active) for item in value]
        return {"__unsupported_type__": f"{type(value).__module__}.{type(value).__qualname__}"}
    finally:
        active.discard(value_id)


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


def _adapt_string_source_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize the STRING API's decimal-string taxonomy at the raw-source edge."""
    if (
        "stringId_A" not in record
        or "stringId_B" not in record
        or "ncbiTaxonId" not in record
        or not isinstance(record["ncbiTaxonId"], str)
        or re.fullmatch(r"[0-9]+", record["ncbiTaxonId"]) is None
    ):
        return record
    adapted = dict(record)
    adapted["ncbiTaxonId"] = int(record["ncbiTaxonId"])
    return adapted


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
    mapping_hashes: tuple[str, ...] = ()

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
        return sum(issue.code in {"unresolved_alias", "untrusted_alias_mapping"} for issue in self.issues)

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
    if requested is not None:
        return requested if isinstance(requested, str) else ""
    if "source_kind" in record:
        value = record["source_kind"]
        return value if isinstance(value, str) else ""
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical_edge_error(edge: Any) -> Optional[str]:
    if not isinstance(edge, CanonicalPpiEdge):
        return "value is not a CanonicalPpiEdge"
    endpoints = (edge.endpoint_a, edge.endpoint_b)
    for endpoint in endpoints:
        if not isinstance(endpoint, CanonicalPpiEndpoint):
            return "endpoint is not a CanonicalPpiEndpoint"
        if _strict_taxonomy(endpoint.taxonomy_id) is None:
            return "endpoint taxonomy_id must be a positive integer"
        if (
            not isinstance(endpoint.source_native_id, str)
            or not endpoint.source_native_id
            or endpoint.source_native_id.strip() != endpoint.source_native_id
        ):
            return "endpoint source_native_id must be a non-empty stable string"
        if not isinstance(endpoint.display_name, str):
            return "endpoint display_name must be a string"
    if edge.endpoint_a.taxonomy_id != edge.endpoint_b.taxonomy_id:
        return "canonical edge endpoints have conflicting taxonomy IDs"
    if edge.endpoint_b.identity_key < edge.endpoint_a.identity_key:
        return "canonical edge endpoints are not in stable order"
    if isinstance(edge.score, bool) or not isinstance(edge.score, Real):
        return "canonical edge score must be numeric"
    score = float(edge.score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return "canonical edge score must be finite and in [0, 1]"
    if not _is_sha256(edge.semantic_sha256):
        return "canonical edge semantic hash is invalid"
    if not isinstance(edge.provenance, tuple) or not edge.provenance:
        return "canonical edge must retain at least one source provenance record"
    provenance_keys = []
    for item in edge.provenance:
        if not isinstance(item, PpiRecordProvenance):
            return "canonical edge provenance has an invalid record type"
        if not isinstance(item.source_kind, str) or item.source_kind not in _ALLOWED_SOURCE_KINDS:
            return "canonical edge provenance source_kind is invalid"
        if not _is_sha256(item.source_record_hash):
            return "canonical edge source_record_hash is invalid"
        if type(item.record_index) is not int or item.record_index < 0:
            return "canonical edge provenance record_index is invalid"
        if (
            not isinstance(item.source_fields, tuple)
            or any(not isinstance(field, str) for field in item.source_fields)
            or tuple(sorted(item.source_fields)) != item.source_fields
        ):
            return "canonical edge provenance source_fields must be sorted strings"
        if not item.input_endpoint_a or not item.input_endpoint_b:
            return "canonical edge provenance must retain both raw endpoint values"
        if not isinstance(item.input_endpoint_a, str) or not isinstance(item.input_endpoint_b, str):
            return "canonical edge raw endpoint values must be strings"
        if (
            not isinstance(item.mapping_sources, tuple)
            or any(not isinstance(source, str) or not source for source in item.mapping_sources)
        ):
            return "canonical edge mapping_sources must be non-empty strings"
        if not isinstance(item.mapping_hashes, tuple) or any(
            not _is_sha256(mapping_hash) for mapping_hash in item.mapping_hashes
        ):
            return "canonical edge mapping_hashes must be SHA-256 values"
        if item.retrieved_at is not None and not isinstance(item.retrieved_at, str):
            return "canonical edge retrieved_at must be a string or null"
        provenance_keys.append((item.source_kind, item.source_record_hash, item.record_index))
    if provenance_keys != sorted(provenance_keys):
        return "canonical edge provenance is not in stable order"
    expected_hash = canonical_json_sha256(
        _semantic_payload(edge.endpoint_a, edge.endpoint_b, score)
    )
    if edge.semantic_sha256 != expected_hash:
        return "canonical edge semantic hash does not match its contents"
    return None


def _invalid_canonical_result(value: Any, message: str) -> CanonicalPpiResult:
    source_hash = canonical_json_sha256(value)
    raw_count = getattr(value, "input_count", 0)
    input_count = raw_count if type(raw_count) is int and raw_count >= 0 else 0
    issue = ValidationIssue(
        "invalid_canonical_result",
        message,
        None,
        "legacy_unverified",
        source_hash,
        None,
    )
    return _build_result([], [issue], input_count, 0, source_hash)


def _validate_canonical_result(value: CanonicalPpiResult) -> CanonicalPpiResult:
    if type(value.schema_version) is not int or value.schema_version != PPI_SCHEMA_VERSION:
        return _invalid_canonical_result(value, "Canonical PPI result has an unsupported schema version.")
    if not isinstance(value.status, str) or value.status not in {"ready", "partial", "empty", "unresolved", "invalid"}:
        return _invalid_canonical_result(value, "Canonical PPI result has an invalid status.")
    if type(value.input_count) is not int or value.input_count < 0:
        return _invalid_canonical_result(value, "Canonical PPI result input_count is invalid.")
    if (
        type(value.accepted_record_count) is not int
        or value.accepted_record_count < 0
        or value.accepted_record_count > value.input_count
    ):
        return _invalid_canonical_result(value, "Canonical PPI result accepted_record_count is invalid.")
    if not isinstance(value.edges, tuple) or len(value.edges) > value.accepted_record_count:
        return _invalid_canonical_result(value, "Canonical PPI result edges are malformed.")
    if not isinstance(value.issues, tuple):
        return _invalid_canonical_result(value, "Canonical PPI result issue ledger is malformed.")
    if not all(_is_sha256(item) for item in (value.semantic_sha256, value.provenance_sha256, value.input_sha256)):
        return _invalid_canonical_result(value, "Canonical PPI result hashes are invalid.")

    for index, edge in enumerate(value.edges):
        error = _canonical_edge_error(edge)
        if error:
            return _invalid_canonical_result(value, f"Canonical PPI result edge {index} is invalid: {error}.")
    for issue in value.issues:
        if not isinstance(issue, ValidationIssue):
            return _invalid_canonical_result(value, "Canonical PPI result contains an invalid issue type.")
        if (
            not isinstance(issue.code, str)
            or not issue.code
            or not isinstance(issue.message, str)
            or not isinstance(issue.source_kind, str)
            or not _is_sha256(issue.source_record_hash)
            or (issue.record_index is not None and (
                type(issue.record_index) is not int
                or issue.record_index < 0
                or issue.record_index >= value.input_count
            ))
            or (issue.field is not None and not isinstance(issue.field, str))
            or not isinstance(issue.mapping_hashes, tuple)
            or any(not _is_sha256(mapping_hash) for mapping_hash in issue.mapping_hashes)
        ):
            return _invalid_canonical_result(value, "Canonical PPI result contains a malformed issue record.")

    expected = _build_result(
        value.edges,
        value.issues,
        value.input_count,
        value.accepted_record_count,
        value.input_sha256,
    )
    if (
        value.edges != expected.edges
        or value.issues != expected.issues
        or value.status != expected.status
        or value.semantic_sha256 != expected.semantic_sha256
        or value.provenance_sha256 != expected.provenance_sha256
    ):
        return _invalid_canonical_result(value, "Canonical PPI result metadata does not match its contents.")
    return value


def _canonicalize_edge_sequence(edges: list[CanonicalPpiEdge]) -> CanonicalPpiResult:
    input_hash = canonical_json_sha256(edges)
    raw_count = max(len(edges), sum(
        len(edge.provenance) if isinstance(edge, CanonicalPpiEdge) and isinstance(edge.provenance, tuple) else 1
        for edge in edges
    ))
    issues = []
    valid = []
    for index, edge in enumerate(edges):
        error = _canonical_edge_error(edge)
        if error:
            issues.append(ValidationIssue(
                "invalid_canonical_edge",
                error,
                index,
                "legacy_unverified",
                canonical_json_sha256(edge),
                None,
            ))
        else:
            valid.append((index, edge))

    by_pair = {}
    for index, edge in valid:
        by_pair.setdefault(edge.edge_key, []).append((index, edge))
    unique_edges = []
    accepted_count = 0
    for pair_key in sorted(by_pair):
        pair_records = by_pair[pair_key]
        if len(pair_records) > 1:
            code = "conflicting_duplicate_score" if len({item.score for _, item in pair_records}) > 1 else "duplicate_canonical_edge"
            for index, edge in pair_records:
                issues.append(ValidationIssue(
                    code,
                    "Canonical edge input repeats an unordered endpoint pair; duplicate entries are quarantined.",
                    index,
                    edge.provenance[0].source_kind,
                    edge.provenance[0].source_record_hash,
                    "score" if code == "conflicting_duplicate_score" else None,
                ))
            continue
        edge = pair_records[0][1]
        unique_edges.append(edge)
        accepted_count += len(edge.provenance)
    return _build_result(unique_edges, issues, raw_count, accepted_count, input_hash)


def _metadata_failure(records: Any, errors: Sequence[tuple[str, str, Optional[str]]]) -> CanonicalPpiResult:
    source_hash = canonical_json_sha256(records)
    if isinstance(records, Mapping) and isinstance(records.get("interactions"), list):
        rows = records["interactions"]
    elif isinstance(records, list):
        rows = records
    else:
        rows = []
    issues = []
    for code, message, field in errors:
        if rows:
            issues.extend(
                ValidationIssue(code, message, index, "legacy_unverified", canonical_json_sha256(row), field)
                for index, row in enumerate(rows)
            )
        else:
            issues.append(ValidationIssue(code, message, None, "legacy_unverified", source_hash, field))
    return _build_result([], issues, len(rows), 0, source_hash)


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
    metadata_errors = []
    if source_kind is not None:
        if not isinstance(source_kind, str):
            metadata_errors.append(("invalid_source_kind", "source_kind must be a string.", "source_kind"))
        elif source_kind not in _ALLOWED_SOURCE_KINDS:
            metadata_errors.append(("unknown_source_kind", f"Unsupported source_kind {source_kind!r}.", "source_kind"))
    if alias_mappings is not None and not isinstance(alias_mappings, Mapping):
        metadata_errors.append(("invalid_alias_mappings", "alias_mappings must be an object keyed by alias.", "alias_mappings"))
    if metadata_errors:
        return _metadata_failure(records, metadata_errors)

    if isinstance(records, CanonicalPpiResult):
        return _validate_canonical_result(records)
    if isinstance(records, list) and records and all(isinstance(item, CanonicalPpiEdge) for item in records):
        return _canonicalize_edge_sequence(records)

    source_hash = canonical_json_sha256(records)
    envelope_source_kind = None
    root_issues: list[ValidationIssue] = []
    if isinstance(records, Mapping) and "interactions" in records:
        envelope = records
        envelope_errors = []
        if "schema_version" not in envelope:
            envelope_errors.append(("missing_schema_version", "PPI envelopes must declare schema_version=2.", "schema_version"))
        elif type(envelope["schema_version"]) is not int or envelope["schema_version"] != PPI_SCHEMA_VERSION:
            envelope_errors.append((
                "unsupported_schema_version",
                f"Unsupported PPI schema version {envelope['schema_version']!r}; expected integer {PPI_SCHEMA_VERSION}.",
                "schema_version",
            ))
        if "source_kind" in envelope:
            envelope_kind = envelope["source_kind"]
            if not isinstance(envelope_kind, str):
                envelope_errors.append(("invalid_source_kind", "Envelope source_kind must be a string.", "source_kind"))
            elif envelope_kind not in _ALLOWED_SOURCE_KINDS:
                envelope_errors.append(("unknown_source_kind", f"Unsupported envelope source_kind {envelope_kind!r}.", "source_kind"))
            elif source_kind is not None and envelope_kind != source_kind:
                envelope_errors.append(("conflicting_source_kind", "Explicit source_kind conflicts with the PPI envelope.", "source_kind"))
            else:
                envelope_source_kind = envelope_kind
        if "alias_mappings" in envelope and not isinstance(envelope["alias_mappings"], Mapping):
            envelope_errors.append(("invalid_alias_mappings", "Envelope alias_mappings must be an object keyed by alias.", "alias_mappings"))
        if envelope_errors:
            return _metadata_failure(records, envelope_errors)

        unknown_envelope = sorted(set(envelope) - _ENVELOPE_FIELDS)
        if unknown_envelope:
            root_issues.append(ValidationIssue(
                "unknown_fields",
                f"Unknown PPI envelope fields: {', '.join(unknown_envelope)}",
                None,
                source_kind or envelope_source_kind or "legacy_unverified",
                source_hash,
                unknown_envelope[0],
            ))
        mappings = alias_mappings if alias_mappings is not None else envelope.get("alias_mappings", {})
        rows = envelope.get("interactions")
        if not isinstance(rows, list):
            return _metadata_failure(records, [("invalid_collection", "PPI interactions must be a list.", "interactions")])
    elif isinstance(records, list):
        mappings = alias_mappings if alias_mappings is not None else {}
        rows = records
    else:
        return _metadata_failure(records, [("invalid_collection", "PPI input must be a list or a versioned object containing interactions.", None)])

    for alias, mapping in mappings.items():
        if not isinstance(alias, str) or not alias or alias.strip() != alias or not isinstance(mapping, Mapping):
            return _metadata_failure(records, [("invalid_alias_mappings", "Alias mappings must use non-empty string keys and object values.", "alias_mappings")])

    requested_source = source_kind if source_kind is not None else envelope_source_kind
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

        input_rows.append(_plain(row))
        row = _adapt_string_source_record(row)
        row_source = row.get("source_kind")
        if "source_kind" in row:
            if not isinstance(row_source, str):
                issues.append(ValidationIssue(
                    "invalid_source_kind",
                    "Record source_kind must be a string.",
                    index,
                    "legacy_unverified",
                    row_hash,
                    "source_kind",
                ))
                continue
            if row_source not in _ALLOWED_SOURCE_KINDS:
                issues.append(ValidationIssue(
                    "unknown_source_kind",
                    f"Unsupported source_kind {row_source!r}.",
                    index,
                    row_source,
                    row_hash,
                    "source_kind",
                ))
                continue
            if requested_source is not None and row_source != requested_source:
                issues.append(ValidationIssue(
                    "conflicting_source_kind",
                    "Record source_kind conflicts with the explicit source origin.",
                    index,
                    requested_source,
                    row_hash,
                    "source_kind",
                ))
                continue
        current_source = _source_for_record(row, requested_source)
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
                alias = row.get(alias_field)
                has_alias = isinstance(alias, str) and bool(alias)
                mapping_supplied = has_alias and alias in mappings
                if not has_alias:
                    code = "missing_endpoint"
                    message = f"Endpoint could not be resolved from {field or alias_field}."
                elif mapping_supplied:
                    code = "untrusted_alias_mapping"
                    message = f"Alias in {alias_field} has a mapping without verified stable identity and taxonomy provenance."
                else:
                    code = "unresolved_alias"
                    message = f"Alias in {alias_field} has no verified source-native ID mapping."
                mapping_hashes = (canonical_json_sha256(mappings[alias]),) if mapping_supplied else ()
                issues.append(ValidationIssue(code, message, index, current_source, row_hash, field, mapping_hashes))
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
                    provenance.mapping_hashes,
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
    elif sorted_issues and all(issue.code in {"unresolved_alias", "untrusted_alias_mapping"} for issue in sorted_issues):
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
        provenance_sha256=canonical_json_sha256({"accepted": provenance_payload, "issues": [asdict(issue) for issue in sorted_issues]}),
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


@dataclass(frozen=True)
class ProteinRecord:
    """Shared identity and complete-sequence record for later feature consumers."""

    accession: Optional[str]
    taxonomy_id: Optional[int]
    raw_sequence: Optional[str]
    full_sequence: Optional[str]
    sequence_length: Optional[int]
    sequence_sha256: Optional[str]
    sequence_status: str
    sequence_status_reason: Optional[str]
    identity_status: str
    source_kind: str
    source_provenance: list[dict[str, Any]] = field(default_factory=list)
    component_evidence: dict[str, Any] = field(default_factory=dict)
    returned_accession: Optional[str] = None
    returned_taxonomy_id: Optional[int] = None


_PROTEIN_STATUSES = {"verified", "missing", "invalid", "error", "unverified"}
_PROTEIN_IDENTITY_STATUSES = {"verified", "mismatch", "invalid", "unverified"}
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_protein_record(record: ProteinRecord) -> None:
    """Raise when a shared protein record breaks identity or full-sequence invariants."""
    if type(record) is not ProteinRecord:
        raise TypeError("protein must be the shared data_contracts.ProteinRecord")
    if record.sequence_status not in _PROTEIN_STATUSES:
        raise ValueError("unsupported protein sequence_status")
    if record.identity_status not in _PROTEIN_IDENTITY_STATUSES:
        raise ValueError("unsupported protein identity_status")
    if record.source_kind not in {"online_snapshot", "legacy_unverified", "synthetic_test"}:
        raise ValueError("unsupported protein source_kind")
    if record.accession is not None:
        if not isinstance(record.accession, str) or re.fullmatch(r"[A-Za-z0-9]{6,10}", record.accession) is None:
            raise ValueError("protein accession must be a 6-10 character UniProt-style identifier")
    if record.taxonomy_id is not None and _strict_taxonomy(record.taxonomy_id) is None:
        raise TypeError("canonical protein taxonomy_id must be a positive integer")
    if record.returned_accession is not None and not isinstance(record.returned_accession, str):
        raise TypeError("returned_accession must preserve the raw returned text")
    if record.returned_taxonomy_id is not None and _strict_taxonomy(record.returned_taxonomy_id) is None:
        raise TypeError("returned_taxonomy_id must be a positive integer")
    if record.raw_sequence is not None and not isinstance(record.raw_sequence, str):
        raise TypeError("raw_sequence must be text or null")
    if record.full_sequence is None:
        if record.raw_sequence is not None:
            raise ValueError("raw_sequence cannot be preserved without its normalized full_sequence")
        if record.sequence_length is not None or record.sequence_sha256 is not None:
            raise ValueError("sequence length/hash require a preserved full_sequence")
    else:
        if not isinstance(record.full_sequence, str):
            raise TypeError("full_sequence must be text or null")
        if record.raw_sequence is None:
            raise ValueError("full_sequence requires its preserved raw_sequence")
        if record.full_sequence != record.full_sequence.upper() or any(char.isspace() for char in record.full_sequence):
            raise ValueError("full_sequence must be uppercase and whitespace-free")
        normalized_raw = "".join(char for char in record.raw_sequence if not char.isspace()).upper()
        if normalized_raw != record.full_sequence:
            raise ValueError("normalized raw_sequence does not match full_sequence")
        if type(record.sequence_length) is not int or record.sequence_length != len(record.full_sequence):
            raise ValueError("sequence_length does not match full_sequence")
        expected_hash = hashlib.sha256(record.full_sequence.encode("utf-8")).hexdigest()
        if record.sequence_sha256 != expected_hash:
            raise ValueError("sequence_sha256 does not match full_sequence")
    if record.identity_status == "verified":
        if record.returned_accession is not None and record.returned_accession != record.accession:
            raise ValueError("returned accession does not match verified protein identity")
        if record.returned_taxonomy_id is not None and record.returned_taxonomy_id != record.taxonomy_id:
            raise ValueError("returned taxonomy does not match verified protein identity")
    if not isinstance(record.source_provenance, list) or any(not isinstance(item, dict) for item in record.source_provenance):
        raise TypeError("source_provenance must be a list of objects")
    if not isinstance(record.component_evidence, dict):
        raise TypeError("component_evidence must be an object")
    for source in record.source_provenance:
        if not _is_sha256(source.get("raw_sha256")):
            raise ValueError("protein raw source provenance requires a SHA-256")
        if not (isinstance(source.get("source_path"), str) or isinstance(source.get("source_url"), str)):
            raise ValueError("protein source provenance requires a path or URL")
    if record.sequence_status == "verified":
        if record.identity_status != "verified" or record.accession is None or record.taxonomy_id is None:
            raise ValueError("verified sequence requires a verified accession and taxonomy")
        if record.returned_accession != record.accession or record.returned_taxonomy_id != record.taxonomy_id:
            raise ValueError("verified sequence requires returned accession and taxonomy to match")
        if record.source_kind == "legacy_unverified":
            raise ValueError("legacy_unverified source cannot produce a verified sequence")
        if record.full_sequence is None or not record.source_provenance:
            raise ValueError("verified sequence requires full sequence and raw source provenance")
        if not record.full_sequence or any(residue not in _STANDARD_AMINO_ACIDS for residue in record.full_sequence):
            raise ValueError("verified sequence contains a nonstandard amino-acid residue")
    elif not record.sequence_status_reason:
        raise ValueError("non-verified sequence status requires a reason")


_FEATURE_STATUSES = {
    "ready",
    "partial",
    "skipped_untrained",
    "unavailable",
    "invalid_sequence",
    "unsupported_length",
    "unverified_source",
    "error",
    "empty",
}


@dataclass(frozen=True)
class StageStatus:
    """Machine-readable readiness for one pipeline stage."""

    status: str
    reason: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _FEATURE_STATUSES:
            raise ValueError(f"unsupported stage status: {self.status}")
        if self.status != "ready" and not self.reason:
            raise ValueError("non-ready stage status requires a reason")
        if not isinstance(self.details, dict):
            raise TypeError("stage status details must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "details": dict(self.details)}


@dataclass(frozen=True)
class FeatureResult:
    """Per-protein features tied to a verified entity bundle and sequence hash."""

    status: str
    reason: Optional[str] = None
    features: dict[str, Any] = field(default_factory=dict)
    accession: Optional[str] = None
    taxonomy_id: Optional[int] = None
    sequence_length: Optional[int] = None
    sequence_sha256: Optional[str] = None
    entity_version: Optional[str] = None
    entity_bundle_hash: Optional[str] = None
    source_kind: Optional[str] = None
    model_revision: Optional[str] = None
    pooling_policy: Optional[str] = None
    cache_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in _FEATURE_STATUSES:
            raise ValueError(f"unsupported feature status: {self.status}")
        if self.status != "ready" and not self.reason:
            raise ValueError("non-ready feature result requires a reason")
        if not isinstance(self.features, dict):
            raise TypeError("feature result features must be an object")
        if self.status != "ready" and self.features:
            raise ValueError("non-ready feature result cannot contain features")
        if self.status == "ready":
            if not self.accession or type(self.taxonomy_id) is not int:
                raise ValueError("ready feature result requires a protein identity")
            if not self.sequence_sha256 or not self.entity_version or not self.entity_bundle_hash:
                raise ValueError("ready feature result requires sequence and entity bundle provenance")
            if not self.model_revision or not self.pooling_policy or not self.cache_key:
                raise ValueError("ready feature result requires model and pooling cache provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "features": dict(self.features),
            "accession": self.accession,
            "taxonomy_id": self.taxonomy_id,
            "sequence_length": self.sequence_length,
            "sequence_sha256": self.sequence_sha256,
            "entity_version": self.entity_version,
            "entity_bundle_hash": self.entity_bundle_hash,
            "source_kind": self.source_kind,
            "model_revision": self.model_revision,
            "pooling_policy": self.pooling_policy,
            "cache_key": self.cache_key,
        }
