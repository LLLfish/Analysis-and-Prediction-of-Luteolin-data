"""Deterministic grouping and label decisions over source activity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PairLabelGroup:
    pair_key: str
    split_group_id: str
    molecule_chembl_id: str
    protein_accession: str
    taxonomy_id: int
    evidence_ids: tuple[str, ...]
    records_json: tuple[str, ...]
    group_sha256: str


@dataclass(frozen=True)
class FilteringLedger:
    input_count: int = 0
    counts: Mapping[str, int] | None = None
    sha256: str = ""


def group_evidence(activity_rows: Any) -> tuple[tuple[PairLabelGroup, ...], FilteringLedger]:
    """Group ActivityEvidence records by native molecule/protein identity."""
    return (), FilteringLedger()
