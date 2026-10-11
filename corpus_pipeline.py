"""Bounded, source-bound ChEMBL pilot selection and trace driver."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import artifact_store
import dataset_builder
import evidence_builder
import label_policy

_APL_ROOT = Path(__file__).resolve().parent
_CORPUS_PATH = _APL_ROOT / "configs" / "corpus_v1.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GUID = re.compile(r"^[0-9a-fA-F]{32}$")


class CorpusPipelineError(ValueError):
    """A corpus source, policy, budget, or output contract failed closed."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {token}")))
    if not isinstance(value, dict):
        raise CorpusPipelineError(f"JSON root must be an object: {path}")
    return value, raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_new_root(value: Any, *, require_guid: bool = True) -> Path:
    if value is None or not os.fspath(value).strip():
        raise CorpusPipelineError("an explicit output path is required")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise CorpusPipelineError("output path must be absolute")
    if require_guid and _GUID.fullmatch(candidate.name) is None:
        raise CorpusPipelineError("output directory name must be a new 32-character GUID")
    root = artifact_store._validate_output_root(candidate, apl_root=_APL_ROOT)
    if root.exists():
        raise CorpusPipelineError(f"refusing to overwrite existing output directory: {root}")
    root.mkdir(parents=True, exist_ok=False)
    artifact_store._assert_no_reparse_components(root)
    return root


def _load_base_contract(path: Path = _CORPUS_PATH) -> tuple[dict[str, Any], bytes]:
    corpus, raw = _read_json(path)
    if corpus.get("protocol_version") != "corpus-v1":
        raise CorpusPipelineError("unsupported corpus protocol version")
    if corpus.get("molecule_selection", {}).get("selected_molecule_chembl_ids"):
        raise CorpusPipelineError("base pre-registration must remain immutable; use a separate run config")
    return corpus, raw


def _verify_bound_policies(corpus: Mapping[str, Any]) -> tuple[Path, Path]:
    label_ref = corpus.get("label_policy")
    split_ref = corpus.get("split_policy")
    if not isinstance(label_ref, Mapping) or not isinstance(split_ref, Mapping):
        raise CorpusPipelineError("corpus policy bindings are missing")
    label_path = (_APL_ROOT / str(label_ref.get("path", ""))).resolve()
    split_path = (_APL_ROOT / str(split_ref.get("path", ""))).resolve()
    for path, binding, key in ((label_path, label_ref, "sha256"), (split_path, split_ref, "sha256")):
        if _APL_ROOT not in path.parents:
            raise CorpusPipelineError("bound policy path escapes APL repository")
        if not path.is_file() or _sha(path.read_bytes()) != binding.get(key):
            raise CorpusPipelineError(f"bound policy file hash mismatch: {path.name}")
    label_policy._load_policy(label_path)
    return split_path, label_path


def _bounded_get(url: str, *, timeout: float, max_response_bytes: int):
    return evidence_builder._bounded_get_json(url, timeout=timeout, max_response_bytes=max_response_bytes)


def _save_response(root: Path, relative: str, source_id: str, url: str, body: bytes) -> dict[str, Any]:
    saved, digest = evidence_builder._write_raw_bytes(root, relative, body)
    return {"source_id": source_id, "source_kind": "online_snapshot", "source_path": saved, "raw_sha256": digest, "source_url": url}


def _selection_attempt_elapsed(ledger: Mapping[str, Any], maximum_request_seconds: float) -> tuple[float, float]:
    """Charge an interrupted pending call its full bounded timeout on recovery."""
    elapsed = ledger.get("elapsed_seconds", 0.0)
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
            or type(maximum_request_seconds) not in (int, float)
            or not math.isfinite(maximum_request_seconds) or maximum_request_seconds <= 0):
        raise CorpusPipelineError("selection ledger has invalid elapsed time or request timeout")
    attempts = ledger.get("attempts", [])
    if not isinstance(attempts, list) or any(not isinstance(row, Mapping) for row in attempts):
        raise CorpusPipelineError("selection request attempt ledger is malformed")
    pending_count = sum(row.get("status") == "pending" for row in attempts)
    if type(ledger.get("request_count")) is not int or ledger["request_count"] < pending_count:
        raise CorpusPipelineError("pending requests do not reconcile with the ledger counter")
    pending_charge = pending_count * float(maximum_request_seconds)
    return float(elapsed) + pending_charge, pending_charge


def _run_persisted_selection_attempt(ledger_path: Path, ledger: dict[str, Any], attempt: dict[str, Any], operation: Any) -> Any:
    """Persist a counted pending selection request before transport and close it on every outcome."""
    request_started = time.monotonic()
    previous_elapsed = float(ledger.get("elapsed_seconds", 0.0))
    ledger["request_count"] = int(ledger.get("request_count", 0)) + 1
    attempt["request_index"] = ledger["request_count"]
    attempt["request_count"] = 1
    attempt["status"] = "pending"
    attempt["started_utc"] = _utc_now()
    ledger.setdefault("attempts", []).append(attempt)
    ledger["status"] = "in_progress"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_bytes(_json_bytes(ledger))
    try:
        result = operation()
        attempt["status"] = "succeeded"
        attempt["completed_utc"] = _utc_now()
        if isinstance(result, tuple) and len(result) >= 3 and isinstance(result[1], bytes):
            attempt["response_bytes"] = len(result[1])
            attempt["response_sha256"] = result[2]
        ledger["status"] = "in_progress"
        return result
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["completed_utc"] = _utc_now()
        attempt["error_type"] = type(exc).__name__
        ledger["status"] = "failed"
        raise
    finally:
        ledger["elapsed_seconds"] = round(previous_elapsed + time.monotonic() - request_started, 3)
        ledger_path.write_bytes(_json_bytes(ledger))


def _ingest_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep trace reports compact; raw response bodies live only in verified snapshots."""
    reasons = result.get("incomplete_reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    provenance = result.get("source_provenance", [])
    return {
        key: result.get(key) for key in (
            "status", "source_kind", "pagination_status", "metadata_status", "overall_status",
            "activity_count", "molecule_count", "protein_count", "exclusion_count",
            "request_count", "page_count", "budgets", "uniprot_budget", "activity_filters", "entity_version", "content_sha256",
        ) if key in result
    } | {"incomplete_reason_count": len(reasons), "incomplete_reasons_sample": reasons[:30],
         "source_receipt_count": len(provenance) if isinstance(provenance, list) else None}


def select_pilot(
    output_dir: Any,
    *,
    limit: int = 3,
    max_requests: int = 128,
    max_response_bytes: int = 8_388_608,
    timeout: float = 30,
    http_get: Any = None,
    prior_discovery: Any = None,
) -> dict[str, Any]:
    """Pre-register quality-only rules, then discover stable-ID pilot candidates.

    The activity search filters only endpoint, species and assay confidence. The
    selection loop never reads activity values, pChEMBL values, or labels.
    """
    if type(limit) is not int or not 1 <= limit <= 10:
        raise CorpusPipelineError("pilot selection limit must be 1..10")
    if type(max_requests) is not int or max_requests < 3:
        raise CorpusPipelineError("selection max_requests must be at least three")
    if type(max_response_bytes) is not int or max_response_bytes < 1:
        raise CorpusPipelineError("max_response_bytes must be positive")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise CorpusPipelineError("timeout must be positive")
    corpus, corpus_raw = _load_base_contract()
    caps = corpus["pilot"]["whole_batch_caps"]
    if max_requests > caps["chembl_requests"] or max_response_bytes > caps["max_response_bytes"]:
        raise CorpusPipelineError("selection request/response limits may not exceed the registered whole-batch caps")
    split_path, label_path = _verify_bound_policies(corpus)
    root = _safe_new_root(output_dir)
    started = _utc_now()
    prior = None
    prior_receipts: list[dict[str, Any]] = []
    prior_requests = prior_pages = prior_rows = prior_bytes = 0
    prior_snapshot = None
    prior_protocol_sha256 = None
    if prior_discovery is not None:
        prior_path = Path(prior_discovery).expanduser().resolve(strict=True)
        prior, _prior_raw = _read_json(prior_path)
        if prior.get("status") != "no_eligible_candidates" or prior.get("base_corpus_sha256") != _sha(corpus_raw):
            raise CorpusPipelineError("prior discovery receipt is not a matching failed attempt")
        prior_requests = prior.get("request_count")
        prior_pages = prior.get("activity_pages")
        prior_rows = prior.get("activity_rows")
        prior_bytes = prior.get("response_bytes")
        if any(type(value) is not int or value < 0 for value in (prior_requests, prior_pages, prior_rows, prior_bytes)):
            raise CorpusPipelineError("prior discovery budget counters are invalid")
        prior_snapshot = prior.get("source_snapshot")
        prior_protocol_sha256 = prior.get("protocol_sha256")
        for index, receipt in enumerate(prior.get("source_receipts", [])):
            if not isinstance(receipt, Mapping) or receipt.get("source_kind") != "online_snapshot":
                raise CorpusPipelineError("prior discovery receipt contains an invalid source")
            source = (prior_path.parent / str(receipt.get("source_path", ""))).resolve()
            if prior_path.parent.resolve() not in source.parents or not source.is_file():
                raise CorpusPipelineError("prior discovery raw source is missing or escapes its directory")
            raw = source.read_bytes()
            if _sha(raw) != receipt.get("raw_sha256"):
                raise CorpusPipelineError("prior discovery raw source hash mismatch")
            saved, digest = evidence_builder._write_raw_bytes(root, f"raw/prior-selection/{index:04d}-{source.name}", raw)
            prior_receipts.append({**dict(receipt), "source_path": saved, "raw_sha256": digest})
    if prior_requests >= max_requests:
        raise CorpusPipelineError("prior discovery already exhausted the selection request cap")
    protocol = {
        "schema_version": 1,
        "purpose": "development_only_pilot_selection",
        "registered_utc_before_queries": started,
        "base_corpus_sha256": _sha(corpus_raw),
        "label_policy_sha256": _sha(label_path.read_bytes()),
        "split_policy_sha256": _sha(split_path.read_bytes()),
        "query_rule": {
            "endpoint": "Ki", "target_organism": "Homo sapiens", "target_tax_id": 9606,
            "minimum_assay_confidence_score": 8, "target_type": "SINGLE PROTEIN",
            "molecule_structure_required": True, "selection_order": "ascending ChEMBL molecule ID",
            "forbidden_selection_fields": ["standard_value", "pchembl_value", "active_inactive_counts", "validation_or_test_outcomes"],
            "holdout": "pilot is development-only and cannot tune or set formal holdout",
        },
        "pilot_limit": limit,
        "budget": {"max_requests_total_including_prior": max_requests, "max_response_bytes_per_request": max_response_bytes, "timeout_seconds": timeout},
        "prior_discovery": {"receipt_sha256": _sha(_prior_raw) if prior_discovery is not None else None,
                            "requests": prior_requests, "activity_pages": prior_pages,
                            "activity_rows": prior_rows, "response_bytes": prior_bytes},
    }
    (root / "selection_protocol.json").write_bytes(_json_bytes(protocol))
    receipts: list[dict[str, Any]] = list(prior_receipts)
    request_count = prior_requests
    response_bytes = prior_bytes

    def get_json(url: str, relative: str, source_id: str) -> dict[str, Any]:
        nonlocal request_count, response_bytes
        if request_count >= max_requests:
            raise CorpusPipelineError("selection discovery request budget exhausted")
        payload, body, digest = _bounded_get(url, timeout=timeout, max_response_bytes=max_response_bytes)
        request_count += 1
        response_bytes += len(body)
        receipts.append(_save_response(root, relative, source_id, url, body))
        if receipts[-1]["raw_sha256"] != digest:
            raise CorpusPipelineError("selection response hash changed during write")
        return payload

    status_url = f"{evidence_builder.CHEMBL_API_ROOT}/status.json"
    if prior is None:
        status = get_json(status_url, "raw/selection/status.json", "chembl_status")
    else:
        status_receipt = next((item for item in prior_receipts if item["source_id"] == "chembl_status"), None)
        if status_receipt is None:
            raise CorpusPipelineError("prior discovery receipt lacks the ChEMBL release boundary")
        status = json.loads((root / status_receipt["source_path"]).read_text(encoding="utf-8"))
    snapshot = status.get("chembl_db_version") or status.get("database_version") or f"rest-status-sha256:{next(r['raw_sha256'] for r in receipts if r['source_id'] == 'chembl_status')}"
    query = urlencode({
        "standard_type": "Ki", "target_organism": "Homo sapiens",
        "order_by": "molecule_chembl_id",
        "limit": min(100, corpus["pilot"]["whole_batch_caps"]["activity_rows"] - prior_rows), "offset": 0,
        "only": "activity_id,molecule_chembl_id,assay_chembl_id,target_chembl_id,target_organism,target_tax_id,standard_type",
    })
    activity_url = f"{evidence_builder.CHEMBL_API_ROOT}/activity.json?{query}"
    activity_search = get_json(activity_url, "raw/selection/ki_candidates.json", "chembl_selection_activity_query")
    rows = activity_search.get("activities")
    if not isinstance(rows, list):
        raise CorpusPipelineError("ChEMBL selection search omitted activities list")
    # This is the only row projection used for selection. Measurement values are never copied.
    by_molecule: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        molecule_id, target_id, assay_id = row.get("molecule_chembl_id"), row.get("target_chembl_id"), row.get("assay_chembl_id")
        if (row.get("standard_type") == "Ki" and row.get("target_organism") == "Homo sapiens"
                and row.get("target_tax_id") in (9606, "9606")
                and isinstance(molecule_id, str) and evidence_builder.CHEMBL_ID.fullmatch(molecule_id)
                and isinstance(target_id, str) and evidence_builder.CHEMBL_ID.fullmatch(target_id)
                and isinstance(assay_id, str) and evidence_builder.CHEMBL_ID.fullmatch(assay_id)):
            by_molecule.setdefault(molecule_id, []).append({"target_chembl_id": target_id, "assay_chembl_id": assay_id})
    selected: list[str] = []
    verified_targets: dict[str, dict[str, Any]] = {}
    verified_assays: dict[str, dict[str, Any]] = {}
    for molecule_id in sorted(by_molecule):
        if len(selected) >= limit:
            break
        mol_url = f"{evidence_builder.CHEMBL_API_ROOT}/molecule/{molecule_id}.json"
        molecule = get_json(mol_url, f"raw/selection/molecule-{molecule_id}.json", f"chembl_molecule:{molecule_id}")
        if molecule.get("molecule_chembl_id") != molecule_id:
            continue
        structures = molecule.get("molecule_structures")
        if not isinstance(structures, Mapping) or not (structures.get("canonical_smiles") or structures.get("standard_inchi")):
            continue
        valid_target = False
        for candidate in sorted(by_molecule[molecule_id], key=lambda item: (item["target_chembl_id"], item["assay_chembl_id"])):
            target_id, assay_id = candidate["target_chembl_id"], candidate["assay_chembl_id"]
            if target_id not in verified_targets:
                target_url = f"{evidence_builder.CHEMBL_API_ROOT}/target/{target_id}.json"
                verified_targets[target_id] = get_json(
                    target_url, f"raw/selection/target-{target_id}.json", f"chembl_target:{target_id}",
                )
            target = verified_targets[target_id]
            if (target.get("target_chembl_id") != target_id or target.get("target_type") != "SINGLE PROTEIN"
                    or target.get("organism") != "Homo sapiens" or target.get("tax_id") not in (9606, "9606")):
                continue
            if assay_id not in verified_assays:
                assay_url = f"{evidence_builder.CHEMBL_API_ROOT}/assay/{assay_id}.json"
                verified_assays[assay_id] = get_json(
                    assay_url, f"raw/selection/assay-{assay_id}.json", f"chembl_assay:{assay_id}",
                )
            assay = verified_assays[assay_id]
            confidence = assay.get("confidence_score")
            if isinstance(confidence, str) and confidence.isdigit():
                confidence = int(confidence)
            if (assay.get("assay_chembl_id") != assay_id or assay.get("target_chembl_id") != target_id
                    or type(confidence) is not int or confidence < 8):
                continue
            components = target.get("target_components")
            if isinstance(components, list) and len(components) == 1:
                target_receipt = next((r for r in receipts if r["source_id"] == f"chembl_target:{target_id}"), None)
                if target_receipt is None:
                    raise CorpusPipelineError("verified target has no raw source receipt")
                resolved = evidence_builder.resolve_target_components(
                    target, source_kind="online_snapshot", source_path=target_receipt["source_path"],
                    source_sha256=target_receipt["raw_sha256"],
                    source_url=f"{evidence_builder.CHEMBL_API_ROOT}/target/{target_id}.json",
                )
                if len(resolved) == 1 and resolved[0].get("resolution_status") == "verified" and resolved[0].get("taxonomy_id") == 9606:
                    valid_target = True
                    break
        if valid_target:
            selected.append(molecule_id)
    if not selected:
        failed = {
            "schema_version": 1, "status": "no_eligible_candidates", "purpose": "development_only_pilot_selection",
            "base_corpus_sha256": _sha(corpus_raw), "source_snapshot": str(snapshot),
            "registered_utc_before_queries": started, "completed_utc": _utc_now(),
            "request_count": request_count, "activity_pages": prior_pages + 1,
            "activity_rows": prior_rows + len(rows), "response_bytes": response_bytes,
            "protocol_sha256": _sha(_json_bytes(protocol)), "prior_protocol_sha256": prior_protocol_sha256,
            "source_receipts": receipts, "reason_code": "no_source_verified_candidate_met_preregistered_quality_rule",
        }
        (root / "discovery_attempt_receipt.json").write_bytes(_json_bytes(failed))
        raise CorpusPipelineError("bounded source query found no source-verified human single-protein Ki candidates")
    cutoff = started
    selection = {
        "schema_version": 1, "purpose": "development_only_pilot_selection",
        "base_corpus_sha256": _sha(corpus_raw), "source": "ChEMBL REST",
        "source_snapshot": str(snapshot), "source_status_sha256": next(r["raw_sha256"] for r in receipts if r["source_id"] == "chembl_status"),
        "capture_cutoff_utc": cutoff, "molecule_ids": selected,
        "selection_rule": protocol["query_rule"], "selection_limit": limit,
        "selection_request_count": request_count, "selection_activity_pages": prior_pages + 1,
        "selection_activity_rows": prior_rows + len(rows), "selection_response_bytes": response_bytes,
        "prior_protocol_sha256": prior_protocol_sha256,
        "source_receipts": receipts,
        "protocol_sha256": _sha(_json_bytes(protocol)),
    }
    selection_bytes = _json_bytes(selection)
    (root / "selection_receipt.json").write_bytes(selection_bytes)
    run_config = {
        "schema_version": 1, "purpose": "bounded_development_pilot",
        "base_corpus_path": "configs/corpus_v1.json", "base_corpus_sha256": _sha(corpus_raw),
        "label_policy_path": str(corpus["label_policy"]["path"]), "label_policy_sha256": _sha(label_path.read_bytes()),
        "split_policy_path": str(corpus["split_policy"]["path"]), "split_policy_sha256": _sha(split_path.read_bytes()),
        "selection_receipt_path": "selection_receipt.json", "selection_receipt_sha256": _sha(selection_bytes),
        "selected_molecule_chembl_ids": selected, "source_snapshot": str(snapshot),
        "capture_cutoff_utc": cutoff, "selection_request_count": request_count,
        "selection_activity_pages": prior_pages + 1, "selection_activity_rows": prior_rows + len(rows),
        "pilot_budget": corpus["pilot"]["whole_batch_caps"],
    }
    (root / "run_config.json").write_bytes(_json_bytes(run_config))
    return {"status": "selected", "selection_path": str(root / "run_config.json"), "molecule_ids": selected,
            "source_snapshot": str(snapshot), "selection_request_count": request_count,
            "selection_activity_pages": prior_pages + 1, "selection_activity_rows": prior_rows + len(rows),
            "selection_response_bytes": response_bytes, "receipt_sha256": _sha(selection_bytes)}


def _pilot_budget_v1_1_path() -> Path:
    return _APL_ROOT.parent / ".planning" / "phases" / "02-real-corpus-readiness" / "02-02-PILOT-BUDGET-v1.1.json"


def _pilot_budget_v1_2_path() -> Path:
    return _APL_ROOT.parent / ".planning" / "phases" / "02-real-corpus-readiness" / "02-02-PILOT-BUDGET-v1.2.json"


def _copy_receipts(source_root: Path, output_root: Path, receipts: Sequence[Mapping[str, Any]], destination_prefix: str) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("source_path"), str):
            raise CorpusPipelineError("source receipt is malformed")
        source = (source_root / receipt["source_path"]).resolve()
        try:
            source.relative_to(source_root.resolve())
        except ValueError as exc:
            raise CorpusPipelineError("source receipt path escapes its evidence root") from exc
        artifact_store._assert_no_reparse_components(source)
        if not source.is_file():
            raise CorpusPipelineError("source receipt raw file is missing")
        raw = source.read_bytes()
        if _sha(raw) != receipt.get("raw_sha256"):
            raise CorpusPipelineError("source receipt raw SHA-256 mismatch")
        name = Path(receipt["source_path"]).name
        saved, digest = evidence_builder._write_raw_bytes(output_root, f"raw/{destination_prefix}/{index:04d}-{name}", raw)
        copied.append({**dict(receipt), "source_path": saved, "raw_sha256": digest})
    return copied


def _verify_boundary_attempt(trace_root: Path, boundary: Mapping[str, Any]) -> dict[str, Any]:
    request_count = boundary.get("request_count")
    if type(request_count) is not int or request_count < 1:
        raise CorpusPipelineError("source-boundary request attempt count is invalid")
    result = {"request_count": request_count, "source_url": boundary.get("source_url"),
              "source_snapshot": boundary.get("source_snapshot", boundary.get("selection_snapshot")),
              "selection_snapshot": boundary.get("selection_snapshot"), "status": boundary.get("status", "succeeded")}
    ledger_path_value = boundary.get("attempt_ledger_path")
    if ledger_path_value is not None:
        ledger_path = (trace_root / str(ledger_path_value)).resolve()
        try:
            ledger_path.relative_to(trace_root.resolve())
        except ValueError as exc:
            raise CorpusPipelineError("source-boundary attempt ledger escapes its trace root") from exc
        artifact_store._assert_no_reparse_components(ledger_path)
        if not ledger_path.is_file():
            raise CorpusPipelineError("source-boundary attempt ledger is missing")
        ledger_raw = ledger_path.read_bytes()
        if _sha(ledger_raw) != boundary.get("attempt_ledger_sha256"):
            raise CorpusPipelineError("source-boundary attempt ledger hash mismatch")
        ledger, _ = _read_json(ledger_path)
        if (ledger.get("source_url") != result["source_url"] or ledger.get("request_count") != request_count
                or ledger.get("status") not in {"pending", "failed", "succeeded"}):
            raise CorpusPipelineError("source-boundary attempt ledger does not reconcile")
        result.update({"attempt_ledger_raw": ledger_raw, "attempt_ledger_status": ledger["status"]})
    source_path_value = boundary.get("source_path")
    if source_path_value is not None:
        source_path = (trace_root / str(source_path_value)).resolve()
        try:
            source_path.relative_to(trace_root.resolve())
        except ValueError as exc:
            raise CorpusPipelineError("source-boundary raw path escapes its trace root") from exc
        artifact_store._assert_no_reparse_components(source_path)
        if not source_path.is_file() or _sha(source_path.read_bytes()) != boundary.get("raw_sha256"):
            raise CorpusPipelineError("source-boundary raw response hash mismatch")
        result["source_raw"] = source_path.read_bytes()
        result["source_raw_sha256"] = boundary["raw_sha256"]
    elif result.get("status") == "succeeded":
        raise CorpusPipelineError("successful source-boundary request has no raw response receipt")
    if result.get("status") == "failed" and result.get("attempt_ledger_status") not in {"failed", "pending"}:
        raise CorpusPipelineError("failed source-boundary report conflicts with its attempt ledger")
    return result


def prepare_pilot_v1_1(selection_config_path: Any, prior_trace_root: Any, output_dir: Any) -> dict[str, Any]:
    """Reconcile the sealed v1 attempt, then bind a narrowly scoped v1.1 retry config."""
    selection_config_path = Path(selection_config_path).expanduser().resolve(strict=True)
    old_config, corpus, selection, _split, _label = _load_run_config(selection_config_path)
    prior_root = Path(prior_trace_root).expanduser().resolve(strict=True)
    decision_path = _pilot_budget_v1_1_path().resolve(strict=True)
    decision, decision_raw = _read_json(decision_path)
    if decision.get("decision_version") != "pilot-v1.1" or decision.get("base_config_sha256") != old_config.get("base_corpus_sha256"):
        raise CorpusPipelineError("pilot-v1.1 decision does not bind this corpus-v1 base config")
    if decision.get("selected_molecule_chembl_ids") != old_config.get("selected_molecule_chembl_ids"):
        raise CorpusPipelineError("pilot-v1.1 decision changes the selected real molecule IDs")
    filters = {"standard_type": "Ki", "target_organism": "Homo sapiens"}
    if (decision.get("activity_filters") != filters or decision.get("relation_filter") is not None
            or decision.get("numeric_value_or_class_filter") is not False):
        raise CorpusPipelineError("pilot-v1.1 changes scientific endpoint or filters on relation/value/class")
    caps = decision.get("cumulative_caps")
    cap_fields = {"chembl_requests", "activity_pages", "activity_rows", "per_request_timeout_seconds",
                  "max_response_bytes", "new_attempt_wall_seconds", "uniprot_requests_total",
                  "uniprot_max_response_bytes", "uniprot_total_response_bytes"}
    if not isinstance(caps, Mapping) or set(caps) != cap_fields:
        raise CorpusPipelineError("pilot-v1.1 cumulative budget schema is invalid")
    if any(type(value) is not int or value <= 0 for value in caps.values()):
        raise CorpusPipelineError("pilot-v1.1 caps must be positive integers")
    partial_path = prior_root / "bundle" / "partial_ingest.json"
    if partial_path.is_file():
        partial, partial_raw = _read_json(partial_path)
        if partial.get("status") != "incomplete" or partial.get("source_kind") != "online_snapshot":
            raise CorpusPipelineError("prior capture is not the expected failed online attempt")
        prior_sources = partial.get("source_provenance")
        if not isinstance(prior_sources, list):
            raise CorpusPipelineError("prior capture lacks raw source receipts")
    else:
        partial = {"status": "no_capture_started", "source_kind": "online_snapshot",
                   "request_count": 0, "page_count": 0, "activity_count": 0,
                   "incomplete_reasons": ["source release boundary failed before collector start"],
                   "source_provenance": []}
        partial_raw = b""
        prior_sources = []
    actual_page_receipts = [row for row in prior_sources if str(row.get("source_id", "")).startswith("chembl_activity:")]
    actual_pages = 0
    actual_rows = 0
    for receipt in actual_page_receipts:
        if not isinstance(receipt, Mapping):
            raise CorpusPipelineError("prior activity receipt is malformed")
        raw_path = (partial_path.parent / str(receipt.get("source_path", ""))).resolve()
        try:
            raw_path.relative_to(partial_path.parent.resolve())
        except ValueError as exc:
            raise CorpusPipelineError("prior activity raw path escapes its capture root") from exc
        artifact_store._assert_no_reparse_components(raw_path)
        raw = raw_path.read_bytes()
        if _sha(raw) != receipt.get("raw_sha256"):
            raise CorpusPipelineError("prior activity raw page hash mismatch")
        page, _ = _read_json(raw_path)
        rows = page.get("activities")
        if not isinstance(rows, list):
            raise CorpusPipelineError("prior activity raw page has no activities list")
        actual_pages += 1
        actual_rows += len(rows)
    if (actual_pages != partial.get("page_count", 0) or actual_rows != partial.get("activity_count", 0)
            or len(actual_page_receipts) != actual_pages):
        raise CorpusPipelineError("prior capture page/row counters do not reconcile to raw page receipts")
    previous_request_count = partial.get("request_count")
    if type(previous_request_count) is not int or previous_request_count < 0:
        raise CorpusPipelineError("prior request count is missing or invalid")
    report_path = prior_root / "trace_report.json"
    prior_report, prior_report_raw = _read_json(report_path)
    boundary = prior_report.get("source_boundary")
    if not isinstance(boundary, Mapping):
        raise CorpusPipelineError("prior trace lacks a source-boundary request receipt")
    if boundary.get("source_snapshot", boundary.get("selection_snapshot")) != selection.get("source_snapshot"):
        raise CorpusPipelineError("prior trace source-boundary release differs from selection")
    boundary_attempt = _verify_boundary_attempt(prior_root, boundary)
    base_prior = old_config.get("prior_usage")
    if not isinstance(base_prior, Mapping):
        base_prior = {"chembl_requests": selection["selection_request_count"],
                      "activity_pages": selection["selection_activity_pages"],
                      "activity_rows": selection["selection_activity_rows"],
                      "uniprot_requests": 0, "uniprot_response_bytes": 0}
    old_usage = {
        "chembl_requests": base_prior["chembl_requests"] + previous_request_count + boundary_attempt["request_count"],
        "activity_pages": base_prior["activity_pages"] + actual_pages,
        "activity_rows": base_prior["activity_rows"] + actual_rows,
        "uniprot_requests": base_prior.get("uniprot_requests", 0) + int(partial.get("uniprot_budget", {}).get("requests", 0)),
        "uniprot_response_bytes": base_prior.get("uniprot_response_bytes", 0) + int(partial.get("uniprot_budget", {}).get("response_bytes", 0)),
    }
    expected_usage = decision.get("cumulative_prior_usage", {})
    if any(type(old_usage[key]) is not int or old_usage[key] != expected_usage.get(key)
           for key in old_usage):
        raise CorpusPipelineError(f"pilot-v1.1 prior usage does not match verified receipts: {old_usage}")
    if (old_usage["chembl_requests"] >= caps["chembl_requests"]
            or old_usage["activity_pages"] >= caps["activity_pages"]
            or old_usage["activity_rows"] >= caps["activity_rows"]):
        raise CorpusPipelineError("pilot-v1.1 remaining cumulative budget is empty")

    root = _safe_new_root(output_dir)
    copied_selection_sources = _copy_receipts(selection_config_path.parent, root,
                                               selection["source_receipts"], "prior-selection")
    selection_copy = {**selection, "source_receipts": copied_selection_sources}
    selection_bytes = _json_bytes(selection_copy)
    (root / "selection_receipt.json").write_bytes(selection_bytes)
    copied_prior_sources = _copy_receipts(partial_path.parent, root, prior_sources, "prior-capture")
    prior_capture = {
        "schema_version": 1, "status": "incomplete", "source_kind": "online_snapshot",
        "request_count": previous_request_count, "activity_pages": actual_pages, "activity_rows": actual_rows,
        "incomplete_reasons": partial.get("incomplete_reasons", []),
        "raw_receipt_count": len(copied_prior_sources), "source_provenance": copied_prior_sources,
        "partial_ingest_sha256": _sha(partial_raw), "prior_trace_report_sha256": _sha(prior_report_raw),
    }
    prior_capture_bytes = _json_bytes(prior_capture)
    (root / "prior_capture_receipt.json").write_bytes(prior_capture_bytes)
    saved_boundary = None
    boundary_hash = None
    if "source_raw" in boundary_attempt:
        saved_boundary, boundary_hash = evidence_builder._write_raw_bytes(
            root, "raw/prior-capture/source-boundary-status.json", boundary_attempt["source_raw"],
        )
    saved_attempt = None
    attempt_hash = None
    if "attempt_ledger_raw" in boundary_attempt:
        saved_attempt, attempt_hash = evidence_builder._write_raw_bytes(
            root, "raw/prior-capture/source-boundary-attempt.json", boundary_attempt["attempt_ledger_raw"],
        )
    (root / "budget_decision_v1_1.json").write_bytes(decision_raw)
    run_config = {
        "schema_version": 1, "purpose": "bounded_development_pilot_v1.1",
        "base_corpus_path": "configs/corpus_v1.json", "base_corpus_sha256": _sha(_CORPUS_PATH.read_bytes()),
        "label_policy_path": str(corpus["label_policy"]["path"]), "label_policy_sha256": _sha(_label.read_bytes()),
        "split_policy_path": str(corpus["split_policy"]["path"]), "split_policy_sha256": corpus["split_policy"]["sha256"],
        "selection_receipt_path": "selection_receipt.json", "selection_receipt_sha256": _sha(selection_bytes),
        "selected_molecule_chembl_ids": old_config["selected_molecule_chembl_ids"],
        "source_snapshot": old_config["source_snapshot"], "capture_cutoff_utc": old_config["capture_cutoff_utc"],
        "selection_request_count": selection_copy["selection_request_count"],
        "selection_activity_pages": selection_copy["selection_activity_pages"],
        "selection_activity_rows": selection_copy["selection_activity_rows"],
        "activity_filters": filters, "pilot_budget": dict(caps),
        "budget_decision_path": str(decision_path.relative_to(_APL_ROOT.parent)).replace("\\", "/"),
        "budget_decision_sha256": _sha(decision_raw),
        "prior_capture_receipt_path": "prior_capture_receipt.json",
        "prior_capture_receipt_sha256": _sha(prior_capture_bytes),
        "prior_source_boundary": {"source_path": saved_boundary, "raw_sha256": boundary_hash,
                                  "attempt_ledger_path": saved_attempt, "attempt_ledger_sha256": attempt_hash,
                                  "status": boundary_attempt.get("status"), "request_count": boundary_attempt["request_count"]},
        "prior_usage": old_usage, "prior_trace_report_sha256": _sha(prior_report_raw),
    }
    run_bytes = _json_bytes(run_config)
    (root / "run_config_v1_1.json").write_bytes(run_bytes)
    return {"status": "ready_to_trace", "config_path": str(root / "run_config_v1_1.json"),
            "molecule_ids": run_config["selected_molecule_chembl_ids"], "prior_usage": old_usage,
            "remaining_budget": {key: caps[key] - old_usage[metric] for key, metric in (
                ("chembl_requests", "chembl_requests"), ("activity_pages", "activity_pages"),
                ("activity_rows", "activity_rows"))},
            "budget_decision_sha256": run_config["budget_decision_sha256"]}


def _activity_selection_fields(row: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    """Project only preregistered discovery fields; activity values/classes are never read."""
    molecule = row.get("molecule_chembl_id")
    target = row.get("target_chembl_id")
    assay = row.get("assay_chembl_id")
    relation = row.get("standard_relation")
    if all(isinstance(item, str) and evidence_builder.CHEMBL_ID.fullmatch(item)
           for item in (molecule, target, assay)) and isinstance(relation, str):
        return molecule, target, assay, relation
    return None


def _target_metadata_eligible(payload: Mapping[str, Any]) -> bool:
    components = payload.get("target_components")
    return (
        payload.get("target_type") == "SINGLE PROTEIN"
        and payload.get("organism") == "Homo sapiens"
        and payload.get("tax_id") == 9606
        and isinstance(components, list) and len(components) == 1
        and isinstance(components[0], Mapping)
        and components[0].get("component_type") == "PROTEIN"
        and isinstance(components[0].get("accession"), str)
        and re.fullmatch(r"[A-Z0-9]{6,10}", components[0]["accession"]) is not None
    )


def _cohort_target_ids(chosen: Any, expected_molecules: Sequence[str], maximum_targets: int) -> list[str]:
    """Require exactly one metadata choice per fixed molecule; shared targets are valid."""
    if not isinstance(chosen, list) or len(chosen) != len(expected_molecules):
        raise CorpusPipelineError("v1.2 cohort must cover every fixed molecule exactly once")
    molecules, targets = [], []
    for row in chosen:
        if not isinstance(row, Mapping):
            raise CorpusPipelineError("v1.2 cohort choice must be an object")
        molecule, target = row.get("molecule_chembl_id"), row.get("target_chembl_id")
        if any(not isinstance(value, str) or not evidence_builder.CHEMBL_ID.fullmatch(value)
               for value in (molecule, target)):
            raise CorpusPipelineError("v1.2 cohort has an invalid native molecule or target ID")
        molecules.append(molecule)
        targets.append(target)
    if len(set(molecules)) != len(molecules) or set(molecules) != set(expected_molecules):
        raise CorpusPipelineError("v1.2 cohort must cover every fixed molecule exactly once")
    target_ids = sorted(set(targets))
    if type(maximum_targets) is not int or not 1 <= len(target_ids) <= maximum_targets:
        raise CorpusPipelineError("v1.2 cohort exceeds the registered target maximum")
    return target_ids


def prepare_pilot_v1_2(selection_config_path: Any, prior_trace_root: Any, output_dir: Any, *,
                       http_get: Any = None, prior_selection_attempts: Sequence[Any] = ()) -> dict[str, Any]:
    """Seal a metadata-only <=2-target cohort from immutable v1.1 pages before activity capture."""
    config_path = Path(selection_config_path).expanduser().resolve(strict=True)
    old_config, corpus, old_selection, _split, _label = _load_run_config(config_path)
    if old_config.get("purpose") != "bounded_development_pilot_v1.1":
        raise CorpusPipelineError("v1.2 target selection requires the sealed v1.1 run config")
    prior_root = Path(prior_trace_root).expanduser().resolve(strict=True)
    report, report_raw = _read_json(prior_root / "trace_report.json")
    decision_path = _pilot_budget_v1_2_path().resolve(strict=True)
    decision, decision_raw = _read_json(decision_path)
    if (decision.get("decision_version") != "pilot-v1.2"
            or decision.get("base_config_sha256") != old_config.get("base_corpus_sha256")
            or decision.get("selected_molecule_chembl_ids") != old_config.get("selected_molecule_chembl_ids")):
        raise CorpusPipelineError("pilot-v1.2 decision does not bind the original fixed molecule IDs")
    usage = report.get("total_budget_usage")
    expected_prior = decision.get("cumulative_prior_usage", {})
    if not isinstance(usage, Mapping) or any(usage.get(key) != expected_prior.get(key) for key in
            ("chembl_requests", "activity_pages", "activity_rows", "uniprot_requests", "uniprot_response_bytes")):
        raise CorpusPipelineError("pilot-v1.2 prior attempt does not reconcile to sealed failed trace totals")
    partial_path = prior_root / "bundle" / "partial_ingest.json"
    partial, partial_raw = _read_json(partial_path)
    provenance = partial.get("source_provenance")
    if partial.get("status") != "incomplete" or not isinstance(provenance, list):
        raise CorpusPipelineError("pilot-v1.2 requires the immutable failed v1.1 raw page receipts")
    raw_pages: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    page_refs: dict[str, dict[str, Any]] = {}
    for receipt in provenance:
        if not isinstance(receipt, Mapping) or receipt.get("source_kind") != "online_snapshot":
            raise CorpusPipelineError("v1.1 raw receipt kind is invalid")
        raw_path = (prior_root / "bundle" / str(receipt.get("source_path", ""))).resolve()
        if prior_root not in raw_path.parents or not raw_path.is_file() or _sha(raw_path.read_bytes()) != receipt.get("raw_sha256"):
            raise CorpusPipelineError("v1.1 raw page receipt hash/path validation failed")
        if not str(receipt.get("source_id", "")).startswith("chembl_activity:"):
            continue
        page, _ = _read_json(raw_path)
        for row in page.get("activities", []):
            projected = _activity_selection_fields(row) if isinstance(row, Mapping) else None
            if projected is None or projected[0] not in decision["selected_molecule_chembl_ids"]:
                continue
            molecule, target, assay, relation = projected
            raw_pages.setdefault((molecule, target, assay), []).append((relation, receipt["source_id"], receipt["raw_sha256"]))
            page_refs[receipt["source_id"]] = dict(receipt)
    candidates: dict[str, list[tuple[str, str, str]]] = {}
    for molecule, target, assay in raw_pages:
        # Exact relation is selection evidence only; no value or activity class is consulted.
        if any(row[0] == "=" for row in raw_pages[(molecule, target, assay)]):
            candidates.setdefault(molecule, []).append((target, assay, "="))
    for values in candidates.values():
        values.sort(key=lambda item: (item[0], item[1]))

    cached: dict[tuple[str, str], tuple[dict[str, Any], bytes, dict[str, Any]]] = {}
    carried_metadata_receipts: list[dict[str, Any]] = []
    carried_request_count = 0
    carried_elapsed = 0.0
    carried_attempts: list[dict[str, Any]] = []
    for attempt_value in prior_selection_attempts:
        attempt_root = Path(attempt_value).expanduser().resolve(strict=True)
        cohort_path = attempt_root / "target_cohort_receipt.json"
        pending_charge = 0.0
        if cohort_path.is_file():
            attempt, attempt_raw = _read_json(cohort_path)
            if attempt.get("decision_version") != "pilot-v1.2" or type(attempt.get("metadata_request_count")) is not int:
                raise CorpusPipelineError("carried target-selection attempt is not a v1.2 receipt")
            request_count = attempt["metadata_request_count"]
            elapsed, pending_charge = _selection_attempt_elapsed(
                {"request_count": request_count,
                 "elapsed_seconds": attempt.get("selection_elapsed_seconds", 0.0), "attempts": []},
                decision["cumulative_caps"]["per_request_timeout_seconds"],
            )
            attempt_sources = attempt.get("source_receipts", [])
        else:
            ledger_path = attempt_root / "target-selection" / "attempt-ledger.json"
            if not ledger_path.is_file():
                raise CorpusPipelineError("carried failed target-selection attempt has no persistent request ledger")
            attempt, attempt_raw = _read_json(ledger_path)
            if attempt.get("decision_sha256") != _sha(decision_raw) or type(attempt.get("request_count")) is not int:
                raise CorpusPipelineError("carried failed target-selection ledger does not bind v1.2 decision")
            request_count = attempt["request_count"]
            elapsed, pending_charge = _selection_attempt_elapsed(
                attempt, decision["cumulative_caps"]["per_request_timeout_seconds"],
            )
            attempt_sources = list(attempt.get("source_receipts", [])) + [row["source_receipt"] for row in attempt.get("attempts", [])
                               if isinstance(row, Mapping) and row.get("status") == "succeeded"
                               and isinstance(row.get("source_receipt"), Mapping)]
            carried_attempts.extend(item for item in attempt.get("prior_selection_attempts", [])
                                    if isinstance(item, Mapping))
        carried_request_count += request_count
        carried_elapsed += elapsed
        carried_attempts.append({"root": str(attempt_root), "receipt_sha256": _sha(attempt_raw),
                                 "request_count": request_count,
                                 "pending_wall_charge_seconds": pending_charge})
        for source in attempt_sources:
            if not isinstance(source, Mapping):
                continue
            raw_path = (attempt_root / str(source.get("source_path", ""))).resolve()
            if attempt_root not in raw_path.parents or not raw_path.is_file():
                continue
            raw = raw_path.read_bytes()
            if _sha(raw) != source.get("raw_sha256"):
                raise CorpusPipelineError("carried target-selection raw receipt hash mismatch")
            sid = str(source.get("source_id", ""))
            for kind in ("target", "assay"):
                match = re.fullmatch(rf"chembl_{kind}:(CHEMBL[0-9]+)", sid)
                if match:
                    payload, _ = _read_json(raw_path)
                    cached[(kind, match.group(1))] = (payload, raw, {**dict(source), "_source_root": str(attempt_root)})
                    if source.get("source_url"):
                        carried_metadata_receipts.append(dict(source))
    for source in old_selection.get("source_receipts", []):
        if not isinstance(source, Mapping):
            continue
        source_path = (config_path.parent / str(source.get("source_path", ""))).resolve()
        if config_path.parent.resolve() not in source_path.parents or not source_path.is_file():
            continue
        raw = source_path.read_bytes()
        if _sha(raw) != source.get("raw_sha256"):
            raise CorpusPipelineError("original selection metadata raw hash mismatch")
        sid = str(source.get("source_id", ""))
        for kind in ("target", "assay"):
            match = re.fullmatch(rf"chembl_{kind}:(CHEMBL[0-9]+)", sid)
            if match:
                payload, _ = _read_json(source_path)
                cached[(kind, match.group(1))] = (payload, raw, {**dict(source), "_source_root": str(config_path.parent)})
    for receipt in provenance:
        sid = str(receipt.get("source_id", ""))
        for kind in ("target", "assay"):
            match = re.fullmatch(rf"chembl_{kind}:(CHEMBL[0-9]+)", sid)
            if match:
                raw_path = (prior_root / "bundle" / str(receipt["source_path"])).resolve()
                raw = raw_path.read_bytes()
                payload, _ = _read_json(raw_path)
                cached[(kind, match.group(1))] = (payload, raw, {**dict(receipt), "_source_root": str(prior_root / "bundle")})

    caps = decision["cumulative_caps"]
    base_requests = expected_prior["chembl_requests"]
    metadata_requests = carried_request_count
    metadata_receipts: list[dict[str, Any]] = list(carried_metadata_receipts)
    chosen: list[dict[str, Any]] = []
    deadline = time.monotonic() + caps["new_attempt_wall_seconds"] - carried_elapsed
    selection_started = time.monotonic() - carried_elapsed
    attempt_ledger_path: Path | None = None
    attempt_ledger = {"schema_version": 1, "decision_sha256": _sha(decision_raw),
                      "request_count": carried_request_count, "attempts": [],
                      "source_receipts": list(carried_metadata_receipts),
                      "prior_selection_attempts": list(carried_attempts),
                      "elapsed_seconds": carried_elapsed, "status": "in_progress"}

    def metadata(kind: str, identifier: str) -> dict[str, Any] | None:
        nonlocal metadata_requests
        key = (kind, identifier)
        if key in cached:
            return cached[key][0]
        if base_requests + metadata_requests + 2 > caps["chembl_requests"]:
            raise CorpusPipelineError("v1.2 cumulative budget must reserve one release check and one activity capture request")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CorpusPipelineError("v1.2 600-second wall budget expired during target selection")
        url = f"{evidence_builder.CHEMBL_API_ROOT}/{kind}/{identifier}.json"
        metadata_requests += 1
        def fetch_metadata():
            payload, raw, digest = _bounded_get(url, timeout=min(caps["per_request_timeout_seconds"], remaining),
                                                 max_response_bytes=caps["max_response_bytes"])
            if payload.get(f"{kind}_chembl_id") != identifier:
                raise CorpusPipelineError(f"{kind} metadata response ID mismatch for {identifier}")
            source = _save_response(root, f"raw/target-selection/{kind}-{identifier}.json",
                                    f"chembl_{kind}:{identifier}", url, raw)
            if source["raw_sha256"] != digest:
                raise CorpusPipelineError("target selection raw response hash changed after save")
            return payload, raw, digest, source
        payload, raw, digest, source = _run_persisted_selection_attempt(
            attempt_ledger_path, attempt_ledger,
            {"kind": kind, "identifier": identifier, "source_url": url}, fetch_metadata,
        )
        attempt_ledger["elapsed_seconds"] = round(time.monotonic() - selection_started, 3)
        metadata_receipts.append(source)
        cached[key] = (payload, raw, source)
        attempt_ledger["source_receipts"].append(source)
        attempt_ledger_path.write_bytes(_json_bytes(attempt_ledger))
        return payload

    root = _safe_new_root(output_dir)
    attempt_ledger_path = root / "target-selection" / "attempt-ledger.json"
    attempt_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_ledger_path.write_bytes(_json_bytes(attempt_ledger))
    for molecule in decision["selected_molecule_chembl_ids"]:
        for target_id, assay_id, relation in candidates.get(molecule, []):
            target_payload = metadata("target", target_id)
            if not isinstance(target_payload, Mapping) or target_payload.get("target_chembl_id") != target_id or not _target_metadata_eligible(target_payload):
                continue
            assay_payload = metadata("assay", assay_id)
            if (not isinstance(assay_payload, Mapping) or assay_payload.get("assay_chembl_id") != assay_id
                    or assay_payload.get("target_chembl_id") != target_id
                    or type(assay_payload.get("confidence_score")) is not int
                    or assay_payload["confidence_score"] < 8):
                continue
            chosen.append({"molecule_chembl_id": molecule, "target_chembl_id": target_id,
                           "assay_chembl_id": assay_id, "standard_relation": relation,
                           "target_eligible": True, "assay_confidence_score": assay_payload["confidence_score"],
                           "target_source_sha256": cached[("target", target_id)][2]["raw_sha256"],
                           "assay_source_sha256": cached[("assay", assay_id)][2]["raw_sha256"]})
            break
    try:
        target_ids = _cohort_target_ids(
            chosen, decision["selected_molecule_chembl_ids"], decision["pilot_target_cohort"]["maximum_targets"],
        )
    except CorpusPipelineError:
        attempt_ledger.update({"status": "no_eligible_target", "elapsed_seconds": round(time.monotonic() - selection_started, 3),
                               "completed_utc": _utc_now()})
        attempt_ledger_path.write_bytes(_json_bytes(attempt_ledger))
        raise
    attempt_ledger.update({"status": "selection_complete", "elapsed_seconds": round(time.monotonic() - selection_started, 3),
                           "completed_utc": _utc_now()})
    attempt_ledger_path.write_bytes(_json_bytes(attempt_ledger))
    selected_metadata: list[dict[str, Any]] = []
    for row in chosen:
        for kind, identifier in (("target", row["target_chembl_id"]), ("assay", row["assay_chembl_id"])):
            payload, raw, source = cached[(kind, identifier)]
            if source.get("source_path", "").startswith("raw/") and (root / source["source_path"]).is_file():
                selected_metadata.append(dict(source))
                continue
            old_path = (Path(source["_source_root"]) / str(source["source_path"])).resolve()
            if not old_path.is_file() or _sha(old_path.read_bytes()) != source.get("raw_sha256"):
                raise CorpusPipelineError("selected cached metadata source receipt no longer verifies")
            copied = evidence_builder._write_raw_bytes(root, f"raw/target-selection/cached-{kind}-{identifier}.json", raw)
            selected_metadata.append({key: value for key, value in dict(source).items() if not key.startswith("_")} | {
                "source_path": copied[0], "raw_sha256": copied[1],
            })
    candidate_page_receipts: list[dict[str, Any]] = []
    for source_id, source in sorted(page_refs.items()):
        old_path = (prior_root / "bundle" / str(source["source_path"])).resolve()
        raw = old_path.read_bytes()
        copied = evidence_builder._write_raw_bytes(root, f"raw/target-selection/pages/{len(candidate_page_receipts):04d}.json", raw)
        candidate_page_receipts.append({key: value for key, value in dict(source).items() if key != "source_path"} | {
            "source_path": copied[0], "raw_sha256": copied[1],
        })
    all_selection_sources = _copy_receipts(config_path.parent, root, old_selection.get("source_receipts", []), "prior-selection")
    all_selection_sources.extend(selected_metadata)
    all_selection_sources.extend(candidate_page_receipts)
    all_selection_sources.extend(
        item for item in metadata_receipts
        if isinstance(item.get("source_path"), str) and (root / item["source_path"]).is_file()
    )
    cohort_receipt = {
        "schema_version": 1, "decision_version": "pilot-v1.2", "source_snapshot": old_config["source_snapshot"],
        "molecule_ids": decision["selected_molecule_chembl_ids"], "selected_targets": chosen,
        "target_chembl_ids": target_ids, "activity_filters": decision["activity_filters"],
        "metadata_request_count": metadata_requests, "candidate_page_receipt_sha256": sorted({
            digest for entries in raw_pages.values() for relation, _sid, digest in entries if relation == "="}),
        "source_receipts": all_selection_sources,
        "prior_trace_report_sha256": _sha(report_raw), "prior_partial_ingest_sha256": _sha(partial_raw),
        "prior_selection_attempts": carried_attempts,
        "request_attempt_ledger_path": "target-selection/attempt-ledger.json",
        "request_attempt_ledger_sha256": _sha(attempt_ledger_path.read_bytes()),
        "selection_rule": decision["pilot_target_cohort"]["selection_rule"],
        "selection_elapsed_seconds": round(time.monotonic() - selection_started, 3), "created_utc": _utc_now(),
    }
    cohort_raw = _json_bytes(cohort_receipt)
    (root / "target_cohort_receipt.json").write_bytes(cohort_raw)
    filters = {**decision["activity_filters"], "target_chembl_id__in": target_ids}
    prior_usage = {key: expected_prior[key] for key in ("chembl_requests", "activity_pages", "activity_rows", "uniprot_requests", "uniprot_response_bytes")}
    prior_usage["chembl_requests"] += metadata_requests
    selection_copy = {**old_selection, "selection_request_count": old_selection["selection_request_count"] + metadata_requests,
                      "source_receipts": all_selection_sources}
    selection_raw = _json_bytes(selection_copy)
    (root / "selection_receipt.json").write_bytes(selection_raw)
    (root / "budget_decision_v1_2.json").write_bytes(decision_raw)
    run_config = {
        "schema_version": 1, "purpose": "bounded_development_pilot_v1.2",
        "base_corpus_path": "configs/corpus_v1.json", "base_corpus_sha256": _sha(_CORPUS_PATH.read_bytes()),
        "label_policy_path": str(corpus["label_policy"]["path"]), "label_policy_sha256": _sha((_APL_ROOT / str(corpus["label_policy"]["path"])).read_bytes()),
        "split_policy_path": str(corpus["split_policy"]["path"]), "split_policy_sha256": corpus["split_policy"]["sha256"],
        "selection_receipt_path": "selection_receipt.json", "selection_receipt_sha256": _sha(selection_raw),
        "selected_molecule_chembl_ids": decision["selected_molecule_chembl_ids"], "source_snapshot": old_config["source_snapshot"],
        "capture_cutoff_utc": _utc_now(), "selection_request_count": old_selection["selection_request_count"] + metadata_requests,
        "selection_activity_pages": old_selection["selection_activity_pages"], "selection_activity_rows": old_selection["selection_activity_rows"],
        "activity_filters": filters, "pilot_budget": caps, "prior_usage": prior_usage,
        "budget_decision_path": str(decision_path.relative_to(_APL_ROOT.parent)).replace("\\", "/"),
        "budget_decision_sha256": _sha(decision_raw), "target_cohort_receipt_path": "target_cohort_receipt.json",
        "target_cohort_receipt_sha256": _sha(cohort_raw), "v1_2_prior_usage": dict(expected_prior),
        "v1_1_trace_root": str(prior_root), "v1_1_trace_report_sha256": _sha(report_raw),
    }
    (root / "run_config_v1_2.json").write_bytes(_json_bytes(run_config))
    return {"status": "ready_to_trace", "config_path": str(root / "run_config_v1_2.json"),
            "target_cohort_receipt_sha256": _sha(cohort_raw), "selected_target_chembl_ids": target_ids,
            "metadata_request_count": metadata_requests, "prior_usage": prior_usage,
            "remaining_chembl_requests": caps["chembl_requests"] - prior_usage["chembl_requests"]}


def _load_run_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    config, config_raw = _read_json(path)
    corpus, corpus_raw = _load_base_contract()
    if _sha(corpus_raw) != config.get("base_corpus_sha256"):
        raise CorpusPipelineError("run config no longer binds the exact corpus pre-registration bytes")
    if config.get("base_corpus_path") != "configs/corpus_v1.json":
        raise CorpusPipelineError("run config must bind the canonical base corpus config")
    split_path, label_path = _verify_bound_policies(corpus)
    if (_sha(split_path.read_bytes()) != config.get("split_policy_sha256")
            or str(split_path.relative_to(_APL_ROOT)).replace("\\", "/") != config.get("split_policy_path")):
        raise CorpusPipelineError("run config split policy hash/path mismatch")
    if (_sha(label_path.read_bytes()) != config.get("label_policy_sha256")
            or str(label_path.relative_to(_APL_ROOT)).replace("\\", "/") != config.get("label_policy_path")):
        raise CorpusPipelineError("run config label policy hash/path mismatch")
    ids = config.get("selected_molecule_chembl_ids")
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or any(not isinstance(value, str) or evidence_builder.CHEMBL_ID.fullmatch(value) is None for value in ids)):
        raise CorpusPipelineError("run config requires a non-empty unique real ChEMBL ID list")
    receipt_path = (path.parent / str(config.get("selection_receipt_path", ""))).resolve()
    if path.parent.resolve() not in receipt_path.parents or not receipt_path.is_file():
        raise CorpusPipelineError("selection receipt is missing or escapes its run directory")
    selection, selection_raw = _read_json(receipt_path)
    if _sha(selection_raw) != config.get("selection_receipt_sha256"):
        raise CorpusPipelineError("selection receipt SHA-256 mismatch")
    if selection.get("base_corpus_sha256") != config.get("base_corpus_sha256") or selection.get("molecule_ids") != ids:
        raise CorpusPipelineError("selection receipt does not bind the run IDs and base protocol")
    for field in ("selection_request_count", "selection_activity_pages", "selection_activity_rows"):
        if type(selection.get(field)) is not int or selection[field] < 0 or selection.get(field) != config.get(field):
            raise CorpusPipelineError(f"selection receipt {field} does not match the run config")
    if not isinstance(config.get("source_snapshot"), str) or not config["source_snapshot"].strip():
        raise CorpusPipelineError("source snapshot boundary is pending")
    cutoff = config.get("capture_cutoff_utc")
    if not isinstance(cutoff, str) or not cutoff.endswith("Z"):
        raise CorpusPipelineError("capture UTC boundary must be explicit and end with Z")
    for receipt in selection.get("source_receipts", []):
        if not isinstance(receipt, Mapping) or receipt.get("source_kind") != "online_snapshot":
            raise CorpusPipelineError("selection source receipt is not source-verified online_snapshot")
        raw_path = (receipt_path.parent / str(receipt.get("source_path", ""))).resolve()
        if receipt_path.parent.resolve() not in raw_path.parents or not raw_path.is_file() or _sha(raw_path.read_bytes()) != receipt.get("raw_sha256"):
            raise CorpusPipelineError("selection raw response receipt verification failed")
    if config.get("purpose") == "bounded_development_pilot_v1.1":
        decision_path = _pilot_budget_v1_1_path().resolve(strict=True)
        relative_decision = str(decision_path.relative_to(_APL_ROOT.parent)).replace("\\", "/")
        if config.get("budget_decision_path") != relative_decision:
            raise CorpusPipelineError("run config does not point to the canonical pilot-v1.1 decision")
        decision, decision_raw = _read_json(decision_path)
        if _sha(decision_raw) != config.get("budget_decision_sha256"):
            raise CorpusPipelineError("pilot-v1.1 decision SHA-256 mismatch")
        if (decision.get("decision_version") != "pilot-v1.1"
                or decision.get("base_config_sha256") != config.get("base_corpus_sha256")
                or decision.get("selected_molecule_chembl_ids") != ids
                or decision.get("activity_filters") != config.get("activity_filters")
                or config.get("activity_filters") != {"standard_type": "Ki", "target_organism": "Homo sapiens"}
                or decision.get("relation_filter") is not None
                or decision.get("numeric_value_or_class_filter") is not False):
            raise CorpusPipelineError("pilot-v1.1 scientific scope or molecule selection mismatch")
        caps = decision.get("cumulative_caps")
        if not isinstance(caps, Mapping) or caps != config.get("pilot_budget"):
            raise CorpusPipelineError("pilot-v1.1 cumulative cap binding mismatch")
        prior_path = (path.parent / str(config.get("prior_capture_receipt_path", ""))).resolve()
        if path.parent.resolve() not in prior_path.parents or not prior_path.is_file():
            raise CorpusPipelineError("prior capture receipt is missing or escapes the run directory")
        prior, prior_raw = _read_json(prior_path)
        if _sha(prior_raw) != config.get("prior_capture_receipt_sha256"):
            raise CorpusPipelineError("prior capture receipt SHA-256 mismatch")
        prior_sources = prior.get("source_provenance")
        if not isinstance(prior_sources, list):
            raise CorpusPipelineError("prior capture receipt omitted raw source receipts")
        prior_pages = prior_rows = 0
        for source in prior_sources:
            if not isinstance(source, Mapping) or source.get("source_kind") != "online_snapshot":
                raise CorpusPipelineError("prior capture source receipt kind is invalid")
            raw_path = (path.parent / str(source.get("source_path", ""))).resolve()
            if path.parent.resolve() not in raw_path.parents or not raw_path.is_file():
                raise CorpusPipelineError("prior capture raw path is missing or escapes its run directory")
            raw = raw_path.read_bytes()
            if _sha(raw) != source.get("raw_sha256"):
                raise CorpusPipelineError("prior capture raw source hash mismatch")
            if str(source.get("source_id", "")).startswith("chembl_activity:"):
                page, _ = _read_json(raw_path)
                if not isinstance(page.get("activities"), list):
                    raise CorpusPipelineError("prior activity page is missing its activities list")
                prior_pages += 1
                prior_rows += len(page["activities"])
        if prior_pages != prior.get("activity_pages") or prior_rows != prior.get("activity_rows"):
            raise CorpusPipelineError("prior capture counters do not reconcile to raw pages")
        boundary = config.get("prior_source_boundary")
        if not isinstance(boundary, Mapping) or type(boundary.get("request_count")) is not int:
            raise CorpusPipelineError("prior source boundary receipt is incomplete")
        _verify_boundary_attempt(path.parent, boundary)
        computed_prior = {
            "chembl_requests": selection.get("selection_request_count") + prior.get("request_count") + boundary["request_count"],
            "activity_pages": selection.get("selection_activity_pages") + prior_pages,
            "activity_rows": selection.get("selection_activity_rows") + prior_rows,
            "uniprot_requests": 0, "uniprot_response_bytes": 0,
        }
        expected_prior = decision.get("cumulative_prior_usage", {})
        if (computed_prior != config.get("prior_usage")
                or any(computed_prior.get(key) != expected_prior.get(key) for key in computed_prior)
                or expected_prior.get("status") != "must_reconcile_against_saved_receipts_before_retry"):
            raise CorpusPipelineError("v1.1 cumulative prior usage differs from raw receipt reconciliation")
    elif config.get("purpose") == "bounded_development_pilot_v1.2":
        decision_path = _pilot_budget_v1_2_path().resolve(strict=True)
        relative_decision = str(decision_path.relative_to(_APL_ROOT.parent)).replace("\\", "/")
        decision, decision_raw = _read_json(decision_path)
        if (config.get("budget_decision_path") != relative_decision
                or _sha(decision_raw) != config.get("budget_decision_sha256")
                or decision.get("decision_version") != "pilot-v1.2"
                or decision.get("base_config_sha256") != config.get("base_corpus_sha256")
                or decision.get("selected_molecule_chembl_ids") != ids
                or decision.get("activity_filters") != {key: config.get("activity_filters", {}).get(key)
                                                          for key in ("standard_type", "target_organism")}
                or decision.get("relation_filter") is not None
                or decision.get("numeric_value_or_class_filter") is not False):
            raise CorpusPipelineError("pilot-v1.2 decision/hash/scientific boundary mismatch")
        caps = decision.get("cumulative_caps")
        if not isinstance(caps, Mapping) or caps != config.get("pilot_budget"):
            raise CorpusPipelineError("pilot-v1.2 cumulative cap binding mismatch")
        cohort_path = (path.parent / str(config.get("target_cohort_receipt_path", ""))).resolve()
        if path.parent.resolve() not in cohort_path.parents or not cohort_path.is_file():
            raise CorpusPipelineError("v1.2 target cohort receipt is missing or escapes its run root")
        cohort, cohort_raw = _read_json(cohort_path)
        if _sha(cohort_raw) != config.get("target_cohort_receipt_sha256"):
            raise CorpusPipelineError("v1.2 target cohort receipt SHA-256 mismatch")
        target_ids = cohort.get("target_chembl_ids")
        chosen = cohort.get("selected_targets")
        if (cohort.get("decision_version") != "pilot-v1.2" or cohort.get("molecule_ids") != ids
                or not isinstance(chosen, list) or len(chosen) != len(ids)
                or not isinstance(target_ids, list) or target_ids != sorted(set(target_ids))
                or config.get("activity_filters") != {"standard_type": "Ki", "target_organism": "Homo sapiens",
                                                       "target_chembl_id__in": target_ids}):
            raise CorpusPipelineError("v1.2 target cohort or query scope is malformed")
        if _cohort_target_ids(chosen, ids, decision["pilot_target_cohort"]["maximum_targets"]) != target_ids:
            raise CorpusPipelineError("v1.2 target cohort IDs do not match the per-molecule choices")
        verified_sources: dict[str, tuple[dict[str, Any], bytes]] = {}
        for receipt in cohort.get("source_receipts", []):
            if not isinstance(receipt, Mapping) or receipt.get("source_kind") != "online_snapshot":
                raise CorpusPipelineError("v1.2 source receipt is not online_snapshot")
            raw_path = (path.parent / str(receipt.get("source_path", ""))).resolve()
            if path.parent.resolve() not in raw_path.parents or not raw_path.is_file():
                raise CorpusPipelineError("v1.2 source raw path is missing or escapes the run root")
            raw = raw_path.read_bytes()
            if _sha(raw) != receipt.get("raw_sha256"):
                raise CorpusPipelineError("v1.2 raw source SHA-256 mismatch")
            verified_sources[str(receipt.get("source_id"))] = (_read_json(raw_path)[0], raw)
        exact_pairs: set[tuple[str, str, str]] = set()
        for sid, (page, _raw) in verified_sources.items():
            if not sid.startswith("chembl_activity:"):
                continue
            for row in page.get("activities", []):
                projected = _activity_selection_fields(row) if isinstance(row, Mapping) else None
                if projected is not None and projected[3] == "=":
                    exact_pairs.add(projected[:3])
        for selection_row in chosen:
            if (not isinstance(selection_row, Mapping) or selection_row.get("target_eligible") is not True
                    or selection_row.get("standard_relation") != "="
                    or (selection_row.get("molecule_chembl_id"), selection_row.get("target_chembl_id"), selection_row.get("assay_chembl_id")) not in exact_pairs):
                raise CorpusPipelineError("v1.2 selected target lacks exact-relation raw row evidence")
            target_key = f"chembl_target:{selection_row['target_chembl_id']}"
            assay_key = f"chembl_assay:{selection_row['assay_chembl_id']}"
            if target_key not in verified_sources or assay_key not in verified_sources:
                raise CorpusPipelineError("v1.2 selected target/assay lacks source-verified raw metadata")
            target = verified_sources[target_key][0]
            assay = verified_sources[assay_key][0]
            if (not _target_metadata_eligible(target)
                    or assay.get("target_chembl_id") != selection_row["target_chembl_id"]
                    or type(assay.get("confidence_score")) is not int or assay["confidence_score"] < 8):
                raise CorpusPipelineError("v1.2 selected target metadata no longer satisfies the fixed quality rule")
        prior_base = decision.get("cumulative_prior_usage", {})
        receipt_prior = config.get("v1_2_prior_usage")
        if receipt_prior != prior_base:
            raise CorpusPipelineError("v1.2 config prior attempt ledger differs from the authorized decision")
        expected = {key: prior_base[key] for key in
                    ("chembl_requests", "activity_pages", "activity_rows", "uniprot_requests", "uniprot_response_bytes")}
        expected["chembl_requests"] += cohort.get("metadata_request_count", -1)
        if config.get("prior_usage") != expected:
            raise CorpusPipelineError("v1.2 cumulative usage does not include all target-selection metadata requests")
        if config.get("selection_request_count") != selection.get("selection_request_count"):
            raise CorpusPipelineError("v1.2 selection receipt request count mismatch")
    return config, corpus, selection, split_path, label_path


def _make_runtime_split_policy(output_root: Path, base_policy_path: Path, decision: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    source, raw = _read_json(base_policy_path)
    caps = decision.get("pilot_split_runtime_caps")
    if not isinstance(caps, Mapping) or set(caps) != {"molecule_pairs", "protein_pairs", "alignment_cells", "preflight_required", "on_excess"}:
        raise CorpusPipelineError("pilot split runtime cap schema is invalid")
    if (any(type(caps.get(key)) is not int or caps[key] < 0 for key in ("molecule_pairs", "protein_pairs", "alignment_cells"))
            or caps.get("preflight_required") is not True or caps.get("on_excess") != "size_limit; no automatic threshold/seed change"):
        raise CorpusPipelineError("pilot split runtime caps are not fail-closed")
    updated = json.loads(json.dumps(source))
    before = json.loads(json.dumps(source))
    updated["budgets"]["molecule_pairs"] = caps["molecule_pairs"]
    updated["budgets"]["protein_pairs"] = caps["protein_pairs"]
    updated["budgets"]["alignment_cells"] = caps["alignment_cells"]
    for key in source:
        if key != "budgets" and updated[key] != before[key]:
            raise CorpusPipelineError("runtime split policy changed scientific rules outside the resource budget")
    path = output_root / "policies" / "split_policy_pilot_v1_1.json"
    path.parent.mkdir(parents=True, exist_ok=False)
    payload = _json_bytes(updated)
    path.write_bytes(payload)
    return path, {"base_file_sha256": _sha(raw), "runtime_file_sha256": _sha(payload),
                  "runtime_semantic_sha256": evidence_builder.canonical_json_sha256(updated),
                  "policy_version": updated["policy_version"], "policy": updated}


def _preflight_workload(bundle_path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    bundle = evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)
    molecules, excluded_molecules = dataset_builder._molecule_entities(bundle)
    proteins, excluded_proteins = dataset_builder._protein_entities(bundle)
    molecule_pairs = len(molecules) * (len(molecules) - 1) // 2
    protein_pairs = len(proteins) * (len(proteins) - 1) // 2
    if molecule_pairs > policy["budgets"]["molecule_pairs"] or protein_pairs > policy["budgets"]["protein_pairs"]:
        return {"status": "size_limit", "molecule_count": len(molecules), "protein_count": len(proteins),
                "molecule_pairs": molecule_pairs, "protein_pairs": protein_pairs, "alignment_cells": None,
                "excluded_molecules": len(excluded_molecules), "excluded_proteins": len(excluded_proteins),
                "reason": "pair_count_exceeds_runtime_budget_before_alignment"}
    alignment_cells = sum(
        len(left["full_sequence"]) * len(right["full_sequence"])
        for index, left in enumerate(proteins) for right in proteins[index + 1:]
    )
    return {"status": "ready" if alignment_cells <= policy["budgets"]["alignment_cells"] else "size_limit",
            "molecule_count": len(molecules), "protein_count": len(proteins),
            "molecule_pairs": molecule_pairs, "protein_pairs": protein_pairs,
            "alignment_cells": alignment_cells, "alignment_cells_measure": policy["budgets"]["alignment_cells_measure"],
            "excluded_molecules": len(excluded_molecules), "excluded_proteins": len(excluded_proteins),
            "caps": dict(policy["budgets"]),
            "reason": None if alignment_cells <= policy["budgets"]["alignment_cells"] else "alignment_cells_exceed_runtime_budget_before_alignment"}


def trace(config_path: Any, output_dir: Any, *, allow_online: bool = False) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve(strict=True)
    config, corpus, selection, split_path, label_path = _load_run_config(config_file)
    root = _safe_new_root(output_dir)
    trace_started = time.monotonic()
    caps = config.get("pilot_budget", corpus["pilot"]["whole_batch_caps"])
    prior_usage = config.get("prior_usage") or {
        "chembl_requests": selection["selection_request_count"],
        "activity_pages": selection["selection_activity_pages"],
        "activity_rows": selection["selection_activity_rows"],
        "uniprot_requests": 0, "uniprot_response_bytes": 0,
    }
    decision = (
        _read_json(_pilot_budget_v1_1_path())[0] if config.get("purpose") == "bounded_development_pilot_v1.1"
        else _read_json(_pilot_budget_v1_2_path())[0] if config.get("purpose") == "bounded_development_pilot_v1.2"
        else None
    )
    if not allow_online:
        manifest = config.get("verified_manifest_path")
        if not isinstance(manifest, str) or not Path(manifest).is_file():
            raise CorpusPipelineError("trace is offline by default; provide a source-verified manifest or explicit --allow-online")
        ingest = evidence_builder.ingest_local_manifest(manifest, root / "bundle")
    else:
        selection_requests = selection["selection_request_count"]
        selection_pages = selection["selection_activity_pages"]
        selection_rows = selection["selection_activity_rows"]
        timeout_seconds = caps.get("per_request_timeout_seconds", caps.get("request_timeout_seconds"))
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise CorpusPipelineError("pilot request timeout cap is invalid")
        if any(type(value) is not int or value < 0 for value in (selection_requests, selection_pages, selection_rows)):
            raise CorpusPipelineError("selection budget counters are invalid")
        page_cap = caps.get("activity_pages", caps.get("chembl_pages"))
        if (caps.get("chembl_requests", 0) - prior_usage.get("chembl_requests", selection_requests) < 2
                or page_cap - prior_usage.get("activity_pages", selection_pages) < 1
                or caps.get("activity_rows", 0) - prior_usage.get("activity_rows", selection_rows) < 1):
            report = {"schema_version": 1, "status": "blocked", "reason": "insufficient_budget_for_release_check_and_capture",
                      "source_boundary": {"status": "skipped_preflight", "request_count": 0,
                                          "selection_snapshot": config["source_snapshot"]},
                      "prior_usage": prior_usage,
                      "total_budget_usage": {**{key: prior_usage.get(key, 0) for key in
                          ("chembl_requests", "activity_pages", "activity_rows", "uniprot_requests", "uniprot_response_bytes")},
                          "caps": {"chembl_requests": caps.get("chembl_requests"), "activity_pages": page_cap,
                                   "activity_rows": caps.get("activity_rows"),
                                   "uniprot_requests": caps.get("uniprot_requests_total", caps.get("uniprot_requests")),
                                   "uniprot_response_bytes": caps.get("uniprot_total_response_bytes", caps.get("uniprot_response_bytes"))}},
                      "output_root": str(root)}
            (root / "trace_report.json").write_bytes(_json_bytes(report))
            return report
        status_url = f"{evidence_builder.CHEMBL_API_ROOT}/status.json"
        attempt_path = root / "source-boundary" / "status-request-attempt.json"
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        status_attempt = {"source_url": status_url, "request_count": 1, "status": "pending", "started_utc": _utc_now()}
        attempt_path.write_bytes(_json_bytes(status_attempt))
        try:
            status, status_body, status_hash = _bounded_get(
                status_url, timeout=timeout_seconds, max_response_bytes=caps["max_response_bytes"],
            )
            status_saved, status_saved_hash = evidence_builder._write_raw_bytes(root, "source-boundary/status.json", status_body)
            if status_hash != status_saved_hash:
                raise CorpusPipelineError("trace source-boundary status hash changed")
            current_snapshot = status.get("chembl_db_version") or status.get("database_version") or f"rest-status-sha256:{status_hash}"
            status_attempt.update({"status": "succeeded", "completed_utc": _utc_now(),
                                  "source_path": status_saved, "raw_sha256": status_hash,
                                  "response_bytes": len(status_body), "source_snapshot": current_snapshot})
            attempt_path.write_bytes(_json_bytes(status_attempt))
        except Exception as exc:
            status_attempt.update({"status": "failed", "completed_utc": _utc_now(),
                                  "error_type": type(exc).__name__})
            attempt_path.write_bytes(_json_bytes(status_attempt))
            report = {"schema_version": 1, "status": "blocked", "reason": "chembl_release_boundary_request_failed",
                      "source_boundary": {**status_attempt, "attempt_ledger_path": str(attempt_path.relative_to(root)).replace("\\", "/"),
                                          "attempt_ledger_sha256": _sha(attempt_path.read_bytes()),
                                          "selection_snapshot": config["source_snapshot"]},
                      "prior_usage": prior_usage,
                      "total_budget_usage": {"chembl_requests": prior_usage["chembl_requests"] + 1,
                                              "activity_pages": prior_usage["activity_pages"],
                                              "activity_rows": prior_usage["activity_rows"],
                                              "uniprot_requests": prior_usage.get("uniprot_requests", 0),
                                              "uniprot_response_bytes": prior_usage.get("uniprot_response_bytes", 0),
                                              "caps": {"chembl_requests": caps["chembl_requests"],
                                                       "activity_pages": caps.get("activity_pages", caps.get("chembl_pages")),
                                                       "activity_rows": caps["activity_rows"],
                                                       "per_request_timeout_seconds": timeout_seconds,
                                                       "uniprot_requests": caps.get("uniprot_requests_total", caps.get("uniprot_requests")),
                                                       "uniprot_response_bytes": caps.get("uniprot_total_response_bytes", caps.get("uniprot_response_bytes")),
                                                       "new_attempt_wall_seconds": caps.get("new_attempt_wall_seconds")}},
                      "output_root": str(root)}
            (root / "trace_report.json").write_bytes(_json_bytes(report))
            return report
        report_boundary = {"source_url": status_url, "source_snapshot": current_snapshot,
                           "source_path": status_saved, "raw_sha256": status_hash,
                           "selection_snapshot": config["source_snapshot"], "request_count": 1,
                           "response_bytes": len(status_body),
                           "attempt_ledger_path": str(attempt_path.relative_to(root)).replace("\\", "/"),
                           "attempt_ledger_sha256": _sha(attempt_path.read_bytes()), "attempt_status": "succeeded"}
        if str(current_snapshot) != str(config["source_snapshot"]):
            report = {"schema_version": 1, "status": "blocked", "reason": "chembl_release_changed_after_selection",
                      "source_boundary": report_boundary, "prior_usage": prior_usage,
                      "total_budget_usage": {"chembl_requests": prior_usage["chembl_requests"] + 1,
                          "activity_pages": prior_usage["activity_pages"], "activity_rows": prior_usage["activity_rows"],
                          "uniprot_requests": prior_usage.get("uniprot_requests", 0),
                          "uniprot_response_bytes": prior_usage.get("uniprot_response_bytes", 0),
                          "caps": {"chembl_requests": caps["chembl_requests"], "activity_pages": page_cap,
                                   "activity_rows": caps["activity_rows"],
                                   "uniprot_requests": caps.get("uniprot_requests_total", caps.get("uniprot_requests")),
                                   "uniprot_response_bytes": caps.get("uniprot_total_response_bytes", caps.get("uniprot_response_bytes"))}},
                      "output_root": str(root)}
            (root / "trace_report.json").write_bytes(_json_bytes(report))
            return report
        remaining = caps["chembl_requests"] - prior_usage["chembl_requests"] - 1
        page_cap = caps.get("activity_pages", caps.get("chembl_pages"))
        remaining_pages = page_cap - prior_usage["activity_pages"]
        remaining_rows = caps["activity_rows"] - prior_usage["activity_rows"]
        wall_cap = caps.get("new_attempt_wall_seconds")
        cohort_selection_elapsed = 0.0
        if config.get("purpose") == "bounded_development_pilot_v1.2":
            cohort, _ = _read_json(config_file.parent / config["target_cohort_receipt_path"])
            cohort_selection_elapsed = float(cohort.get("selection_elapsed_seconds", 0.0))
        wall_remaining = None if wall_cap is None else wall_cap - cohort_selection_elapsed - (time.monotonic() - trace_started)
        if remaining < 1 or remaining_pages < 1 or remaining_rows < 1:
            raise CorpusPipelineError("verified prior usage leaves no cumulative pilot budget")
        if wall_remaining is not None and wall_remaining <= 0:
            raise CorpusPipelineError("pilot new-attempt wall budget is exhausted before capture")
        ingest = evidence_builder.fetch_chembl_bounded(
            config["selected_molecule_chembl_ids"], root / "bundle",
            max_pages=remaining_pages, max_records=remaining_rows, timeout=timeout_seconds,
            max_requests=remaining, max_response_bytes=caps["max_response_bytes"], per_request_timeout=True,
            total_timeout_seconds=wall_remaining,
            activity_filters=config.get("activity_filters", {"standard_type": "Ki", "target_organism": "Homo sapiens"}),
            hydrate_uniprot=True, max_uniprot_requests=caps["uniprot_requests_total"],
            uniprot_max_response_bytes=caps["uniprot_max_response_bytes"],
            uniprot_total_response_bytes=caps["uniprot_total_response_bytes"],
        )
    report: dict[str, Any] = {
        "schema_version": 1, "status": "blocked", "run_config_sha256": _sha(config_file.read_bytes()),
        "base_corpus_sha256": config["base_corpus_sha256"], "selection_receipt_sha256": config["selection_receipt_sha256"],
        "source_snapshot": config["source_snapshot"], "capture_cutoff_utc": config["capture_cutoff_utc"],
        "molecule_ids": config["selected_molecule_chembl_ids"], "ingest": _ingest_summary(ingest),
        "source_boundary": report_boundary if allow_online else {"source_snapshot": config["source_snapshot"], "manifest_path": config.get("verified_manifest_path")},
        "selection_budget": {"requests": selection["selection_request_count"], "pages": selection["selection_activity_pages"],
                             "activity_rows": selection["selection_activity_rows"]},
        "prior_usage": prior_usage,
        "activity_filters": config.get("activity_filters", {}),
        "budget_decision_sha256": config.get("budget_decision_sha256"),
        "label_policy_path": str(label_path), "label_policy_sha256": _sha(label_path.read_bytes()),
        "split_policy_path": str(split_path), "split_policy_sha256": _sha(split_path.read_bytes()),
        "output_root": str(root),
    }
    if allow_online:
        total_usage = {
            "chembl_requests": prior_usage["chembl_requests"] + report_boundary["request_count"] + int(ingest.get("request_count", 0)),
            "activity_pages": prior_usage["activity_pages"] + int(ingest.get("page_count", 0)),
            "activity_rows": prior_usage["activity_rows"] + int(ingest.get("activity_count", 0)),
            "uniprot_requests": prior_usage.get("uniprot_requests", 0) + int(ingest.get("uniprot_budget", {}).get("requests", 0)),
            "uniprot_response_bytes": prior_usage.get("uniprot_response_bytes", 0) + int(ingest.get("uniprot_budget", {}).get("response_bytes", 0)),
        }
        page_cap = caps.get("activity_pages", caps.get("chembl_pages"))
        report["total_budget_usage"] = {**total_usage, "caps": {
            "chembl_requests": caps["chembl_requests"], "activity_pages": page_cap,
            "activity_rows": caps["activity_rows"], "per_request_timeout_seconds": timeout_seconds,
            "uniprot_requests": caps["uniprot_requests_total"],
            "uniprot_response_bytes": caps["uniprot_total_response_bytes"],
            "new_attempt_wall_seconds": caps.get("new_attempt_wall_seconds"),
        }, "new_attempt_elapsed_seconds": round(time.monotonic() - trace_started + cohort_selection_elapsed, 3)}
        if (total_usage["chembl_requests"] > caps["chembl_requests"]
                or total_usage["activity_pages"] > page_cap
                or total_usage["activity_rows"] > caps["activity_rows"]
                or total_usage["uniprot_requests"] > caps["uniprot_requests_total"]
                or total_usage["uniprot_response_bytes"] > caps["uniprot_total_response_bytes"]):
            report["reason"] = "whole_batch_request_or_data_budget_exceeded"
            (root / "trace_report.json").write_bytes(_json_bytes(report))
            return report
        report["budget_decision_sha256"] = config.get("budget_decision_sha256")
    bundle_path = root / "bundle" / "entities.json"
    if ingest.get("status") != "complete" or not bundle_path.is_file():
        report["reason"] = "chembl_or_metadata_capture_incomplete"
        (root / "trace_report.json").write_bytes(_json_bytes(report))
        return report
    bundle = evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)
    if {row.get("molecule_chembl_id") for row in bundle["molecules"]} != set(config["selected_molecule_chembl_ids"]):
        report["reason"] = "captured_molecule_ids_do_not_match_selection"
        (root / "trace_report.json").write_bytes(_json_bytes(report))
        return report
    real_ki = [row for row in bundle["activities"] if row.get("standard_type") == "Ki"
               and row.get("target_type") == "SINGLE PROTEIN" and row.get("target_organism") == "Homo sapiens"
               and row.get("target_taxonomy_id") == 9606 and row.get("metadata_status") == "complete"
               and row.get("pagination_status") == "complete" and row.get("source_kind") == "online_snapshot"]
    report["real_verified_human_single_protein_ki_rows"] = len(real_ki)
    if not real_ki:
        report["reason"] = "no_verified_real_human_single_protein_ki_row"
        (root / "trace_report.json").write_bytes(_json_bytes(report))
        return report
    runtime_split_path = split_path
    if decision is not None:
        runtime_split_path, runtime_split = _make_runtime_split_policy(root, split_path, decision)
        preflight = _preflight_workload(bundle_path, runtime_split["policy"])
        report["runtime_split_policy"] = {key: value for key, value in runtime_split.items() if key != "policy"}
        report["split_workload_preflight"] = preflight
        if preflight.get("status") != "ready":
            report["status"] = "size_limit"
            report["reason"] = preflight.get("reason")
            (root / "trace_report.json").write_bytes(_json_bytes(report))
            return report
        if caps.get("new_attempt_wall_seconds") is not None and time.monotonic() - trace_started >= caps["new_attempt_wall_seconds"]:
            report["status"] = "blocked"
            report["reason"] = "pilot_wall_budget_exhausted_before_split"
            (root / "trace_report.json").write_bytes(_json_bytes(report))
            return report
    from dataset_builder import publish_frozen_dataset
    freeze_root = root / "frozen"
    version = "pilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frozen = publish_frozen_dataset(
        bundle_path, freeze_root, version_id=version,
        policy_path=runtime_split_path, label_policy_path=label_path,
    )
    frozen_data = frozen.to_dict()
    report["frozen"] = frozen_data
    if frozen.status not in {"complete", "insufficient"} or frozen.audit_report is None or frozen.audit_report.get("status") != "clean":
        report["reason"] = "freeze_or_independent_audit_not_clean"
        (root / "trace_report.json").write_bytes(_json_bytes(report))
        return report
    report["training_ready"] = bool(frozen.audit_report.get("training_ready"))
    report["status"] = "complete" if report["training_ready"] else "not_ready"
    report["reason"] = None if report["training_ready"] else "pilot_support_or_sequence_gate_insufficient"
    report["frozen_manifest"] = str(Path(frozen.output_dir) / "manifest.json")
    report["audit_status"] = frozen.audit_report.get("status")
    (root / "trace_report.json").write_bytes(_json_bytes(report))
    return report


def rebuild_v1_2_from_real_capture(config_path: Any, prior_trace_root: Any, output_dir: Any) -> dict[str, Any]:
    """Re-derive target component facts and audit a sealed real v1.2 capture without new requests."""
    config_file = Path(config_path).expanduser().resolve(strict=True)
    config, corpus, selection, split_path, label_path = _load_run_config(config_file)
    if config.get("purpose") != "bounded_development_pilot_v1.2":
        raise CorpusPipelineError("rebuild-v1.2 requires a hash-verified v1.2 run config")
    previous_root = Path(prior_trace_root).expanduser().resolve(strict=True)
    prior_report, prior_report_raw = _read_json(previous_root / "trace_report.json")
    usage = prior_report.get("total_budget_usage")
    if not isinstance(usage, Mapping):
        raise CorpusPipelineError("prior v1.2 trace has no cumulative request ledger")
    old_audit = prior_report.get("frozen", {}).get("audit_report", {})
    old_findings = old_audit.get("findings", []) if isinstance(old_audit, Mapping) else []
    if (prior_report.get("ingest", {}).get("status") != "complete"
            or old_audit.get("status") != "violation" or not old_findings
            or any(item.get("code") != "unmapped_activity_evidence"
                   or not all(source.get("reason") == "component_species_not_human"
                              for source in item.get("source_evidence", [])) for item in old_findings)):
        raise CorpusPipelineError("prior v1.2 capture does not match the narrow known component-species derivation failure")
    caps = config["pilot_budget"]
    cap_bindings = {"chembl_requests": "chembl_requests", "activity_pages": "activity_pages",
                    "activity_rows": "activity_rows", "uniprot_requests": "uniprot_requests_total",
                    "uniprot_response_bytes": "uniprot_total_response_bytes"}
    if any(usage.get(key, 0) > caps.get(cap_key, 0) for key, cap_key in cap_bindings.items()):
        raise CorpusPipelineError("prior v1.2 attempt exceeded an authorized cumulative cap")
    source_bundle_path = previous_root / "bundle" / "entities.json"
    if not source_bundle_path.is_file():
        raise CorpusPipelineError("prior v1.2 capture has no complete verified entity bundle")
    source_bundle = evidence_builder.load_entity_bundle(source_bundle_path, verify_sources=True)
    if source_bundle.get("source_kind") != "online_snapshot":
        raise CorpusPipelineError("fixture or non-online bundle cannot enter the real v1.2 rebuild")
    filters = config["activity_filters"]
    for activity in source_bundle["activities"]:
        if activity.get("query", {}).get("filters") != filters:
            raise CorpusPipelineError("prior activity evidence does not match the exact v1.2 target cohort scope")
    root = _safe_new_root(output_dir)
    bundle_root = root / "bundle"
    bundle_root.mkdir()
    for receipt in source_bundle["source_provenance"]:
        source_path = evidence_builder._safe_relative_file(source_bundle_path.parent, receipt.get("source_path"))
        artifact_store._assert_no_reparse_components(source_path)
        raw = source_path.read_bytes()
        if _sha(raw) != receipt.get("raw_sha256"):
            raise CorpusPipelineError("source capture raw SHA changed while rebuilding")
        relative = Path(str(receipt["source_path"]))
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        artifact_store._assert_no_reparse_components(destination)
        if _sha(destination.read_bytes()) != receipt["raw_sha256"]:
            raise CorpusPipelineError("source raw bytes changed during exact-copy rebuild")
    activities = []
    for activity in source_bundle["activities"]:
        target_type, organism, taxonomy, components = evidence_builder._target_fields(
            activity.get("raw_activity_record", {}), activity.get("target_record"),
        )
        rebuilt = dict(activity)
        rebuilt.update({"target_type": target_type, "target_organism": organism,
                        "target_taxonomy_id": taxonomy, "target_components": components})
        activities.append(rebuilt)
    ingest = evidence_builder.write_entity_bundle(
        bundle_root, activities=activities, source_provenance=source_bundle["source_provenance"],
        molecules=source_bundle["molecules"], proteins=source_bundle["proteins"],
        exclusions=source_bundle.get("exclusions", []), source_kind="online_snapshot",
    )
    bundle_path = bundle_root / "entities.json"
    bundle = evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)
    real_ki = [row for row in bundle["activities"] if row.get("standard_type") == "Ki"
               and row.get("target_type") == "SINGLE PROTEIN" and row.get("target_organism") == "Homo sapiens"
               and row.get("target_taxonomy_id") == 9606 and row.get("metadata_status") == "complete"
               and row.get("pagination_status") == "complete" and row.get("source_kind") == "online_snapshot"]
    report: dict[str, Any] = {
        "schema_version": 1, "status": "blocked", "reason": "freeze_or_independent_audit_not_clean",
        "run_config_sha256": _sha(config_file.read_bytes()), "base_corpus_sha256": config["base_corpus_sha256"],
        "selection_receipt_sha256": config["selection_receipt_sha256"], "target_cohort_receipt_sha256": config["target_cohort_receipt_sha256"],
        "budget_decision_sha256": config["budget_decision_sha256"], "source_snapshot": config["source_snapshot"],
        "capture_cutoff_utc": config["capture_cutoff_utc"], "molecule_ids": config["selected_molecule_chembl_ids"],
        "activity_filters": filters, "prior_attempt_report_sha256": _sha(prior_report_raw),
        "ingest": _ingest_summary(ingest), "real_verified_human_single_protein_ki_rows": len(real_ki),
        "source_boundary": prior_report.get("source_boundary"), "prior_usage": config["prior_usage"],
        "total_budget_usage": usage, "output_root": str(root),
    }
    if not real_ki:
        report["reason"] = "rebuilt_capture_not_complete_real_human_single_protein_ki"
        (root / "trace_report.json").write_bytes(_json_bytes(report))
        return report
    runtime_path, runtime = _make_runtime_split_policy(root, split_path, _read_json(_pilot_budget_v1_2_path())[0])
    preflight = _preflight_workload(bundle_path, runtime["policy"])
    report["runtime_split_policy"] = {key: value for key, value in runtime.items() if key != "policy"}
    report["split_workload_preflight"] = preflight
    if preflight.get("status") != "ready":
        report.update({"status": "size_limit", "reason": preflight.get("reason")})
        (root / "trace_report.json").write_bytes(_json_bytes(report))
        return report
    frozen = dataset_builder.publish_frozen_dataset(
        bundle_path, root / "frozen", version_id="pilot-rebuild-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        policy_path=runtime_path, label_policy_path=label_path,
    )
    report["frozen"] = frozen.to_dict()
    if frozen.audit_report is None or frozen.audit_report.get("status") != "clean":
        report["reason"] = "freeze_or_independent_audit_not_clean"
    else:
        report["status"] = "complete" if frozen.audit_report.get("training_ready") else "not_ready"
        report["training_ready"] = bool(frozen.audit_report.get("training_ready"))
        report["reason"] = None if report["training_ready"] else "pilot_support_or_sequence_gate_insufficient"
    (root / "trace_report.json").write_bytes(_json_bytes(report))
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded source-verified real corpus pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="pre-register quality-only selection and discover real ChEMBL IDs")
    select.add_argument("--out", required=True, type=Path, help="new absolute GUID directory")
    select.add_argument("--limit", type=int, default=3)
    select.add_argument("--max-requests", type=int, default=128)
    select.add_argument("--max-response-bytes", type=int, default=8_388_608)
    select.add_argument("--timeout", type=float, default=30)
    select.add_argument("--prior-discovery", type=Path, help="failed prior bounded attempt whose budget and receipts must be carried forward")
    prep = commands.add_parser("prepare-v1.1", help="verify prior receipts and bind the authorized pilot-v1.1 cumulative retry budget")
    prep.add_argument("--config", required=True, type=Path, help="v1 selection/run config")
    prep.add_argument("--prior-trace-root", required=True, type=Path, help="sealed prior trace GUID root")
    prep.add_argument("--out", required=True, type=Path, help="new absolute GUID directory")
    prep_v12 = commands.add_parser("prepare-v1.2", help="verify sealed failed captures, metadata-select a tiny target cohort, and bind the v1.2 retry budget")
    prep_v12.add_argument("--config", required=True, type=Path, help="sealed v1.1 run config")
    prep_v12.add_argument("--prior-trace-root", required=True, type=Path, help="sealed v1.1 trace GUID root")
    prep_v12.add_argument("--out", required=True, type=Path, help="new absolute GUID directory")
    prep_v12.add_argument("--prior-selection-attempt", action="append", default=[], type=Path,
                          help="sealed failed v1.2 target-selection attempt; include all prior query attempts")
    trace_parser = commands.add_parser("trace", help="trace a pilot through evidence, Ki policy, split and independent audit")
    trace_parser.add_argument("--config", required=True, type=Path)
    trace_parser.add_argument("--out", required=True, type=Path, help="new absolute GUID directory")
    trace_parser.add_argument("--allow-online", action="store_true", help="allow the user-authorized bounded ChEMBL/UniProt requests")
    rebuild = commands.add_parser("rebuild-v1.2", help="re-audit a sealed complete v1.2 raw capture after deterministic source-derived fixes; issues no requests")
    rebuild.add_argument("--config", required=True, type=Path)
    rebuild.add_argument("--prior-trace-root", required=True, type=Path)
    rebuild.add_argument("--out", required=True, type=Path, help="new absolute GUID directory")
    preflight = commands.add_parser("preflight", help="offline formal corpus pair/cell/time resource preflight")
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument("--entities", required=True, type=Path)
    preflight.add_argument("--out", required=True, type=Path, help="new absolute GUID directory")
    expand = commands.add_parser("expand", help="run the pre-registered bounded formal corpus candidate workflow")
    expand.add_argument("--config", required=True, type=Path)
    expand.add_argument("--out", required=True, type=Path, help="new absolute GUID directory")
    expand.add_argument("--allow-online", action="store_true", help="allow the authorized bounded ChEMBL/UniProt requests")
    expand.add_argument("--resume-from", type=Path, help="continue this same formal budget batch from its latest registered ledger root")
    for name in ("freeze",):
        gated = commands.add_parser(name, help=f"{name} is blocked until a formal source scope is captured and audited")
        gated.add_argument("--config", required=True, type=Path)
        gated.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "select":
            result = select_pilot(args.out, limit=args.limit, max_requests=args.max_requests,
                                  max_response_bytes=args.max_response_bytes, timeout=args.timeout,
                                  prior_discovery=args.prior_discovery)
            code = 0
        elif args.command == "prepare-v1.1":
            result = prepare_pilot_v1_1(args.config, args.prior_trace_root, args.out)
            code = 0
        elif args.command == "prepare-v1.2":
            result = prepare_pilot_v1_2(args.config, args.prior_trace_root, args.out,
                                        prior_selection_attempts=args.prior_selection_attempt)
            code = 0
        elif args.command == "trace":
            result = trace(args.config, args.out, allow_online=args.allow_online)
            code = 0 if result.get("status") in {"complete", "not_ready"} else 1
        elif args.command == "rebuild-v1.2":
            result = rebuild_v1_2_from_real_capture(args.config, args.prior_trace_root, args.out)
            code = 0 if result.get("status") in {"complete", "not_ready"} else 1
        elif args.command in {"preflight", "expand"}:
            import formal_corpus
            if args.command == "preflight":
                result = formal_corpus.preflight(args.config, args.entities, args.out)
                code = 0 if result.get("status") == "within_budget" else 1
            else:
                result = formal_corpus.expand(args.config, args.out, allow_online=args.allow_online,
                                              resume_from=args.resume_from)
                code = 0 if result.get("status") in {
                    "candidate_audited_not_ready", "selection_window_exhausted", "size_limit"} else 1
        else:
            config, corpus, _selection, _split, _label = _load_run_config(args.config.resolve(strict=True))
            gate = corpus.get("formal_expansion_budget", {})
            if gate.get("status") != "approved" or gate.get("formal_collection_allowed") is not True:
                raise CorpusPipelineError(f"{args.command} rejected: formal expansion budget is pending")
            raise CorpusPipelineError(f"{args.command} requires the dedicated formal corpus workflow")
    except Exception as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
