"""Versioned, fail-closed preflight for the first formal ChEMBL cohort.

Collection is intentionally not implemented here until a source-only candidate
selection receipt is sealed.  This module provides the resource gate used by
that later entry point and keeps the development pilot controller untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import artifact_store
import dataset_builder
import evidence_builder


APL_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = APL_ROOT / "configs" / "corpus_formal_v1.json"
MAX_ESM_TOKENS = 1024
SPECIAL_TOKEN_RESERVE = 2
MAX_SUPPORTED_RESIDUES = MAX_ESM_TOKENS - SPECIAL_TOKEN_RESERVE


class FormalCorpusError(ValueError):
    """A formal protocol or resource gate failed closed."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FormalCorpusError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FormalCorpusError(f"expected a JSON object: {path}")
    return value, raw


def _safe_new_root(value: Any) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise FormalCorpusError("output path must be an absolute GUID directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FormalCorpusError("output path must be absolute")
    try:
        artifact_store._assert_no_reparse_components(path)
    except Exception as exc:
        raise FormalCorpusError("output path contains a symlink/reparse point") from exc
    path = path.resolve()
    if path.name != path.name.lower() or len(path.name) != 32:
        raise FormalCorpusError("output directory name must be a lowercase 32-character GUID")
    try:
        import uuid
        if uuid.UUID(hex=path.name).hex != path.name:
            raise ValueError
    except ValueError as exc:
        raise FormalCorpusError("output directory name must be a lowercase 32-character GUID") from exc
    protected = [APL_ROOT / name for name in (
        "data", "framework_output", "offline_data.json", ".model_cache", "model_cache",
        "features", "history", "output_dashboard.html",
    )]
    for item in protected:
        target = item.resolve()
        if path == target or target in path.parents or path in target.parents:
            raise FormalCorpusError("output root intersects a protected project path")
    if path.exists() or path.is_symlink():
        raise FormalCorpusError("output root already exists")
    # Reject any existing symlink/reparse component to avoid escaping the requested root.
    cursor = path
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise FormalCorpusError("output path contains a symlink/reparse point")
        cursor = cursor.parent
    return path


def _verify_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    config, _ = _read_object(path.resolve(strict=True))
    base_path = APL_ROOT / "configs" / "corpus_v1.json"
    base, base_raw = _read_object(base_path)
    if config.get("protocol_version") != "corpus-formal-v1.0" or config.get("base_corpus_sha256") != _sha(base_raw):
        raise FormalCorpusError("formal config protocol/base corpus binding mismatch")
    if config.get("base_corpus_path") != "configs/corpus_v1.json":
        raise FormalCorpusError("formal config must bind the immutable corpus_v1.json")
    bindings = {
        "label_policy": base["label_policy"]["path"],
        "split_policy": base["split_policy"]["path"],
        "support_policy": base["support_policy"]["path"],
    }
    for key, relative in bindings.items():
        target = APL_ROOT / relative
        if not target.is_file() or config.get(f"{key}_sha256") != _sha(target.read_bytes()):
            raise FormalCorpusError(f"formal config {key} hash mismatch")
        if config.get(f"{key}_path") != relative:
            raise FormalCorpusError(f"formal config {key} path mismatch")
    budget = config.get("budget")
    if not isinstance(budget, Mapping) or budget.get("status") != "measured_pre_registered":
        raise FormalCorpusError("formal budget decision is not registered")
    decision_path = APL_ROOT / str(budget.get("decision_path", ""))
    if not decision_path.is_file() or _sha(decision_path.read_bytes()) != budget.get("decision_sha256"):
        raise FormalCorpusError("formal budget decision missing or hash mismatch")
    decision, _ = _read_object(decision_path)
    if decision.get("decision_version") != config["protocol_version"] or decision.get("base_corpus_sha256") != config["base_corpus_sha256"]:
        raise FormalCorpusError("budget decision is not bound to the formal corpus protocol")
    caps = decision.get("caps")
    expected = {"molecules": 128, "targets": 64, "chembl_requests": 2048,
                "chembl_activity_pages": 256, "chembl_activity_rows": 20000,
                "chembl_response_bytes_per_request": 8_388_608,
                "chembl_response_bytes_total": 536_870_912,
                "chembl_wall_seconds": 1800, "request_timeout_seconds": 30,
                "uniprot_requests": 128, "uniprot_response_bytes_per_request": 8_388_608,
                "uniprot_response_bytes_total": 134_217_728,
                "builder_molecule_pairs": 8128, "builder_protein_pairs": 2016,
                "builder_alignment_cells": 2_200_000_000, "builder_wall_seconds": 900,
                "auditor_molecule_pairs": 8128, "auditor_protein_pairs": 2016,
                "auditor_alignment_cells": 2_200_000_000, "auditor_wall_seconds": 900}
    if caps != expected:
        raise FormalCorpusError("formal budget caps differ from the signed decision")
    if config.get("sequence_policy") != {
        "esm_max_tokens": MAX_ESM_TOKENS,
        "reserved_special_tokens": SPECIAL_TOKEN_RESERVE,
        "maximum_supported_full_sequence_residues": MAX_SUPPORTED_RESIDUES,
        "tokenizer_validation": "not_claimed; conservative max_tokens minus BOS/EOS reserve",
        "retention": "keep complete source sequence and receipts; record unsupported_length; never truncate",
    }:
        raise FormalCorpusError("formal sequence applicability policy mismatch")
    return config, decision


def _require_formal_online_snapshot(bundle: Mapping[str, Any]) -> None:
    if bundle.get("source_kind") != "online_snapshot":
        raise FormalCorpusError("formal preflight requires an online_snapshot entity bundle")
    for collection in ("molecules", "proteins", "activities", "source_provenance"):
        rows = bundle.get(collection)
        if not isinstance(rows, list):
            raise FormalCorpusError(f"formal source bundle omitted {collection}")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("source_kind") != "online_snapshot":
                raise FormalCorpusError(f"formal source bundle contains non-online_snapshot {collection}[{index}]")
    if not bundle["molecules"] or not bundle["proteins"] or not bundle["activities"] or not bundle["source_provenance"]:
        raise FormalCorpusError("formal source bundle must contain molecules, proteins, activities, and raw receipts")


def _sequence_applicability(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supported: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in bundle.get("proteins", []):
        if not isinstance(row, Mapping):
            continue
        accession, sequence = row.get("accession"), row.get("full_sequence")
        label = f"{row.get('taxonomy_id')}|{accession}"
        length = len(sequence) if isinstance(sequence, str) else None
        item = {"entity_id": label, "accession": accession, "sequence_length": length,
                "sequence_sha256": row.get("sequence_sha256"), "source_kind": row.get("source_kind")}
        if row.get("sequence_status") != "verified" or row.get("identity_status") != "verified":
            excluded.append({**item, "reason": row.get("sequence_status_reason") or "sequence_or_identity_unverified"})
        elif length is None or length == 0:
            excluded.append({**item, "reason": "full_sequence_missing"})
        elif length > MAX_SUPPORTED_RESIDUES:
            excluded.append({**item, "reason": "unsupported_length_full_sequence_retained"})
        else:
            supported.append(item)
    return supported, excluded


def _workload(molecule_count: int, protein_lengths: Sequence[int], caps: Mapping[str, Any],
              cells_per_second: float) -> dict[str, Any]:
    molecule_pairs = molecule_count * (molecule_count - 1) // 2
    protein_count = len(protein_lengths)
    protein_pairs = protein_count * (protein_count - 1) // 2
    if molecule_pairs > caps["builder_molecule_pairs"] or protein_pairs > caps["builder_protein_pairs"]:
        return {"status": "size_limit", "molecule_pairs": molecule_pairs, "protein_pairs": protein_pairs,
                "alignment_cells": None, "max_single_alignment_cells": None,
                "builder_estimated_seconds": None, "auditor_estimated_seconds": None,
                "reason": "pair_count_exceeds_registered_budget_before_alignment"}
    cells = sum(left * right for i, left in enumerate(protein_lengths) for right in protein_lengths[i + 1:])
    maximum_pair = max((left * right for i, left in enumerate(protein_lengths) for right in protein_lengths[i + 1:]), default=0)
    builder_seconds = cells / cells_per_second
    auditor_seconds = cells / cells_per_second
    over = (cells > caps["builder_alignment_cells"] or cells > caps["auditor_alignment_cells"]
            or builder_seconds > caps["builder_wall_seconds"] or auditor_seconds > caps["auditor_wall_seconds"])
    return {"status": "size_limit" if over else "within_budget", "molecule_pairs": molecule_pairs,
            "protein_pairs": protein_pairs, "alignment_cells": cells,
            "max_single_alignment_cells": maximum_pair, "builder_estimated_seconds": builder_seconds,
            "auditor_estimated_seconds": auditor_seconds,
            "reason": "alignment_or_time_budget_exceeded" if over else None}


def preflight(config_path: Any, entities_path: Any, output_dir: Any) -> dict[str, Any]:
    """Verify a source bundle and calculate pair/cell/time costs before alignment."""
    started = time.perf_counter()
    config, decision = _verify_config(Path(config_path))
    bundle_path = Path(entities_path).expanduser().resolve(strict=True)
    bundle = evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)
    _require_formal_online_snapshot(bundle)
    root = _safe_new_root(output_dir)
    molecules, excluded_molecules = dataset_builder._molecule_entities(bundle)
    usable_proteins, excluded_verified_proteins = dataset_builder._protein_entities(bundle)
    supported_proteins, unsupported_by_length = _sequence_applicability(bundle)
    excluded_proteins = excluded_verified_proteins + [item for item in unsupported_by_length
                                                       if item.get("reason") == "unsupported_length_full_sequence_retained"]
    caps = decision["caps"]
    # The full source bundle is exactly what the strict split builder/auditor would consume.
    # ESM length applicability never removes a sequence from duplicate/neighbor auditing.
    lengths = [len(item["full_sequence"]) for item in usable_proteins]
    workload = _workload(len(molecules), lengths, caps, decision["measurement"]["alignment_cells_per_second"])
    status = "size_limit" if (len(molecules) > caps["molecules"] or len(usable_proteins) > caps["targets"]
                              or workload["status"] == "size_limit") else "within_budget"
    report = {
        "schema_version": 1, "protocol_version": config["protocol_version"],
        "status": status, "reason": None if status == "within_budget" else "declared_scope_exceeds_registered_entity_or_alignment_budget",
        "training_ready": False,
        "support_assessment": "not inferred from development-only pilot; formal split support remains unassessed",
        "base_corpus_sha256": config["base_corpus_sha256"], "budget_decision_sha256": config["budget"]["decision_sha256"],
        "entity_bundle": {"path": str(bundle_path), "content_sha256": bundle["content_sha256"],
                          "source_kind": bundle.get("source_kind")},
        "counts": {"source_molecules": len(bundle.get("molecules", [])), "usable_molecules": len(molecules),
                   "excluded_molecules": len(excluded_molecules), "source_proteins": len(bundle.get("proteins", [])),
                   "split_usable_full_sequence_proteins": len(usable_proteins),
                   "length_supported_proteins": len(supported_proteins), "unsupported_or_unverified_proteins": len(excluded_proteins)},
        "sequence_applicability": {"max_full_sequence_residues": MAX_SUPPORTED_RESIDUES,
                                   "supported": supported_proteins, "excluded": excluded_proteins},
        "workload": {"molecule_pairs": workload["molecule_pairs"], "protein_pairs": workload["protein_pairs"],
                     "alignment_cells": workload["alignment_cells"], "max_single_alignment_cells": workload["max_single_alignment_cells"],
                     "alignment_cells_measure": "sum(len(sequence_i)*len(sequence_j)) over each unique full-sequence pair",
                     "builder_estimated_seconds": workload["builder_estimated_seconds"], "auditor_estimated_seconds": workload["auditor_estimated_seconds"],
                     "workload_status": workload["status"], "workload_reason": workload["reason"],
                     "measurement_source": decision["measurement"]["source"]},
        "caps": dict(caps),
        "measurement": {"preflight_seconds": time.perf_counter() - started,
                        "generated_utc": datetime.now(timezone.utc).isoformat()},
    }
    root.mkdir(parents=True, exist_ok=False)
    (root / "preflight.json").write_bytes(_json_bytes(report))
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight a new formal source-verified corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="calculate all-pairs resource cost before any builder/auditor run")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--entities", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True, help="new absolute 32-character GUID directory")
    expand = commands.add_parser("expand", help="formal collection remains gated until a sealed quality-only source selection exists")
    expand.add_argument("--config", type=Path, required=True)
    expand.add_argument("--out", type=Path, required=True)
    expand.add_argument("--allow-online", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.config, args.entities, args.out)
            code = 0 if result["status"] == "within_budget" else 1
        else:
            _verify_config(args.config)
            raise FormalCorpusError("formal expansion requires a sealed source-quality selection receipt")
    except Exception as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
