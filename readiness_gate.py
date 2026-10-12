"""Fail-closed, read-only training admission diagnosis for the formal corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from artifact_store import _assert_no_reparse_components
from feature_snapshot import verify_derived_snapshot
from feature_snapshot import _canonical_bytes
from leak_audit import audit_split
from evidence_builder import load_entity_bundle
from data_contracts import ProteinRecord
from protein_features import ProteinFeatureExtractor, build_feature_cache_key


ROOT = Path(__file__).resolve().parent
FORMAL_CONFIG = ROOT / "configs" / "corpus_formal_v1.json"
EXPECTED = {
    "config": "addfa131f3850b037e8edc81c160f2bfb320e24b0c0d9ee1b053b92a0dce5b05",
    "decision": "cd0887b1f550ba35ba42661cc696f15c52070443ae1bfe01faffdddcd368ff46",
    "label": "817587aae18bd85d5986db44d5383b014cc88689a7574eb1479e606c3be92082",
    "support": "883ab850c3f0f77ccce0dff09dc5a6406445768d57dc21d7749ccc26d5ca7bb5",
    "base_corpus": "84f12e741db0543dbcf3190d3c3894b99038eaa9da7bcc31e46ece7848b1afe1",
    "original_split": "7b390e73aa8392bfb0bb53a771b4d938385e34dadc7db86162dfca542bf65fde",
}
LABEL_HASH = EXPECTED["label"]
SUPPORT_HASH = EXPECTED["support"]
MAX_AUDIT_CELLS = 2_200_000_000
MAX_AUDIT_SECONDS = 900
PRIOR_MEASURED_CELLS = 1_608_611_159
PRIOR_MEASURED_SECONDS = 645.6823531000118
GLOBAL_CELLS_CAP = 6_856_000_000
GLOBAL_SECONDS_CAP = 2_820
FORMAL_RECEIPT_SHA256 = "6c480572fc09a497064fdb8d242a01d3a902f00981793db9c7c31e01731cdc8e"
FORMAL_MANIFEST_SHA256 = "959155da68689e580ad25ad5aea831783d1f4c323a82cecc3a53eec50822659c"
EXPECTED_FEATURE_MANIFEST_SHA256 = "46d13b05db30c15e39cc4b455a5ab3ec1a0faf6265017b32301b08212da0c1c5"
EXPECTED_MODEL_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
EXPECTED_MODEL_ARTIFACT_SHA256 = "2e3251586898deaed82236033fdc7099647f3d6af30952318bb576e14861fb5b"
EXPECTED_FEATURE_STATUS_COUNTS = {"ready": 41, "unsupported_length": 3}
EXPECTED_UNSUPPORTED_ACCESSIONS = {"P00519", "P00533", "P08069"}
EXPECTED_FULL_PASS_CELLS = 257_672_979
PROTECTED = (
    "data", "framework_output", "offline_data.json", ".model_cache",
    "model_cache", "features", "history", "output_dashboard.html",
)


class ContractError(ValueError):
    """Input bytes or bindings do not satisfy the frozen formal contract."""


def _bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"expected regular file: {path}")
    return path.read_bytes()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(_bytes(path))


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _bytes(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value, raw


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha(payload)


def _resolve_manifest_ref(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


def _verify_protocol(config_path: Path, formal_receipt_path: Path, manifest_path: Path,
                     manifest: Mapping[str, Any], *, protocol_root: Path = ROOT) -> dict[str, str]:
    config, config_raw = _json(config_path)
    receipt, _ = _json(formal_receipt_path)
    if _file_sha(formal_receipt_path) != FORMAL_RECEIPT_SHA256:
        raise ContractError("formal receipt bytes are not the immutable 02-03 acceptance receipt")
    config_sha = _sha(config_raw)
    if config_sha != EXPECTED["config"] or config_path.name != "corpus_formal_v1.json":
        raise ContractError("formal corpus config bytes/version are not the registered immutable protocol")
    if receipt.get("config_sha256") != config_sha or receipt.get("decision_sha256") != EXPECTED["decision"]:
        raise ContractError("formal receipt config/decision binding mismatch")
    if receipt.get("protocol_version") != config.get("protocol_version"):
        raise ContractError("formal receipt protocol version mismatch")
    if (receipt.get("status") != "candidate_audited_not_ready" or receipt.get("training_ready") is not False
            or receipt.get("support_sufficient_for_minimal_baseline_attempt") is not False
            or receipt.get("source_snapshot") != "ChEMBL_37"):
        raise ContractError("formal receipt does not describe the registered blocked real-source candidate")
    for receipt_name, hash_key in (("selection_receipt.json", "selection_receipt_sha256"),
                                   ("development_exclusion_receipt.json", "development_exclusion_receipt_sha256")):
        bound_file = formal_receipt_path.parent / receipt_name
        if receipt.get(hash_key) != _file_sha(bound_file):
            raise ContractError(f"formal {receipt_name} bytes do not match their receipt binding")
    if (receipt.get("independent_audit", {}).get("status") != "clean"
            or receipt.get("independent_audit", {}).get("exit_code") != 0):
        raise ContractError("formal receipt lacks a successful independent audit")
    decision_path = protocol_root / config["budget"]["decision_path"]
    decision_raw = _bytes(decision_path)
    if _sha(decision_raw) != EXPECTED["decision"]:
        raise ContractError("registered budget decision bytes changed")
    base_path = protocol_root / config["base_corpus_path"]
    if _file_sha(base_path) != EXPECTED["base_corpus"] or config.get("base_corpus_sha256") != EXPECTED["base_corpus"]:
        raise ContractError("base corpus protocol binding mismatch")
    for label, relkey, hashkey in (("label", "label_policy_path", "label_policy_sha256"),
                                   ("support", "support_policy_path", "support_policy_sha256")):
        path = protocol_root / config[relkey]
        actual = _file_sha(path)
        if actual != EXPECTED[label] or config.get(hashkey) != actual:
            raise ContractError(f"registered {label} policy bytes changed")
        if receipt.get(f"{label}_policy_sha256") != actual:
            raise ContractError(f"formal receipt {label} policy binding mismatch")
    # The runtime budget variant may differ only in the three registered caps.
    frozen_split_path = _resolve_manifest_ref(manifest_path, manifest["policies"]["split"]["path"])
    frozen_split, frozen_raw = _json(frozen_split_path)
    if _sha(frozen_raw) != manifest["policies"]["split"].get("file_sha256"):
        raise ContractError("frozen split policy file hash mismatch")
    registered_split, _ = _json(protocol_root / config["split_policy_path"])
    runtime_science = json.loads(json.dumps(frozen_split))
    registered_science = json.loads(json.dumps(registered_split))
    runtime_science.pop("budgets", None)
    registered_science.pop("budgets", None)
    if runtime_science != registered_science:
        raise ContractError("frozen split scientific policy differs from registered policy")
    runtime_budgets = frozen_split.get("budgets", {})
    if (runtime_budgets.get("molecule_pairs") != 8128 or runtime_budgets.get("protein_pairs") != 2016
            or runtime_budgets.get("alignment_cells") != MAX_AUDIT_CELLS):
        raise ContractError("frozen runtime budget variant is not the registered formal cap")
    if receipt.get("frozen_manifest_sha256") != _file_sha(manifest_path):
        raise ContractError("formal receipt manifest byte hash mismatch")
    if _file_sha(manifest_path) != FORMAL_MANIFEST_SHA256:
        raise ContractError("manifest is not the immutable 02-03 formal cohort")
    if receipt.get("frozen_manifest_semantic_sha256") != manifest.get("semantic_sha256"):
        raise ContractError("formal receipt manifest semantic hash mismatch")
    bundle_path = _resolve_manifest_ref(manifest_path, manifest["entities"]["path"])
    bundle_file_sha = _file_sha(bundle_path)
    bundle_semantic_sha = manifest["entities"].get("content_sha256")
    if bundle_file_sha != manifest["entities"].get("file_sha256"):
        raise ContractError("entity bundle file bytes do not match the manifest")
    if (bundle_semantic_sha != receipt.get("derived_bundle_sha256")
            or bundle_semantic_sha != receipt.get("capture_bundle_sha256")):
        raise ContractError("entity semantic bundle binding mismatch")
    if manifest.get("source_readiness_contract_version") is None or manifest.get("source_readiness_receipt_ref") is None:
        raise ContractError("formal source-readiness receipt is missing")
    if manifest.get("audit") != {"exit_code": 0, "finding_count": 0, "status": "clean"}:
        raise ContractError("frozen manifest does not claim the required clean independent audit")
    return {"corpus_config_sha256": config_sha, "budget_decision_sha256": EXPECTED["decision"],
            "base_corpus_sha256": EXPECTED["base_corpus"], "label_policy_sha256": EXPECTED["label"],
            "support_policy_sha256": EXPECTED["support"], "manifest_sha256": _file_sha(manifest_path),
            "manifest_semantic_sha256": str(manifest.get("semantic_sha256")), "entity_bundle_file_sha256": bundle_file_sha,
            "entity_bundle_sha256": str(bundle_semantic_sha),
            "selection_receipt_sha256": receipt["selection_receipt_sha256"],
            "development_exclusion_receipt_sha256": receipt["development_exclusion_receipt_sha256"]}


def _support(audit: Mapping[str, Any], manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    counts = audit.get("class_support")
    if not isinstance(counts, dict):
        raise ContractError("independent audit did not return class support")
    split = manifest["split"]
    assignments = split.get("pair_assignments")
    if not isinstance(assignments, list):
        raise ContractError("frozen pair assignments are missing")
    mol_members = split.get("molecule_memberships", {})
    prot_members = split.get("protein_memberships", {})
    cluster_counts = {name: {"molecule": set(), "protein": set()} for name in ("train", "val", "test")}
    for row in assignments:
        if not isinstance(row, dict):
            raise ContractError("malformed pair assignment")
        part = row.get("split")
        if part not in cluster_counts:
            continue
        if not isinstance(row.get("molecule_cluster_id"), str) or not isinstance(row.get("protein_cluster_id"), str):
            raise ContractError("pair assignment lacks independently auditable cluster IDs")
        cluster_counts[part]["molecule"].add(row["molecule_cluster_id"])
        cluster_counts[part]["protein"].add(row["protein_cluster_id"])
    minimums = policy["minimum_labeled_examples_per_class"]
    molecule_min = policy["minimum_distinct_connected_molecule_clusters"]
    protein_min = policy["minimum_distinct_connected_protein_clusters"]
    observed: dict[str, Any] = {}
    met = True
    for part in ("train", "val", "test"):
        part_counts = counts.get(part)
        if not isinstance(part_counts, dict):
            raise ContractError("independent audit support omits a split")
        observed[part] = {"active": part_counts.get("active"), "inactive": part_counts.get("inactive"),
                          "molecule_clusters": len(cluster_counts[part]["molecule"]),
                          "protein_clusters": len(cluster_counts[part]["protein"])}
        for label in ("active", "inactive"):
            if type(part_counts.get(label)) is not int or part_counts[label] < minimums[part]:
                met = False
        for kind, floor in (("molecule", molecule_min[part]), ("protein", protein_min[part])):
            if floor is not None and observed[part][f"{kind}_clusters"] < floor:
                met = False
    thresholds = {"examples_per_class": minimums, "molecule_clusters": molecule_min, "protein_clusters": protein_min}
    return {"counts": observed, "thresholds": thresholds}, met


def _verify_support_policy(path: os.PathLike[str] | str) -> dict[str, Any]:
    raw = _bytes(Path(path).resolve(strict=True))
    if _sha(raw) != SUPPORT_HASH:
        raise ContractError("support policy bytes are not the registered frozen policy")
    try:
        policy = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"registered support policy is invalid: {exc}") from exc
    if not isinstance(policy, dict):
        raise ContractError("registered support policy must be a JSON object")
    return policy


def _feature_gate(feature_path: Path, manifest: Mapping[str, Any], bundle: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    feature, raw = _json(feature_path)
    if _sha(raw) != EXPECTED_FEATURE_MANIFEST_SHA256:
        raise ContractError("feature manifest bytes are not the final 02-04 inference snapshot")
    bundle_hash = bundle.get("content_sha256")
    if feature.get("entity_bundle_hash") != bundle_hash or feature.get("entity_version") != manifest["entities"].get("entity_version"):
        raise ContractError("feature manifest is bound to a different corpus")
    entities = feature.get("entities")
    if not isinstance(entities, list) or len(entities) != 44:
        raise ContractError("feature manifest must contain all 44 formal proteins")
    expected = {row.get("accession"): row for row in entities}
    if len(expected) != 44 or set(expected) != {row.get("accession") for row in feature.get("entities", [])}:
        raise ContractError("feature manifest has duplicate or missing protein accessions")
    # Verify every row is from the exact cohort, revision, weights and pooling contract.
    model = feature.get("model", {})
    if model.get("external_revision_verified") is not False or not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", ""))):
        raise ContractError("feature model revision provenance is invalid or overstated")
    if (model.get("model_id") != "facebook/esm2_t6_8M_UR50D"
            or model.get("revision") != EXPECTED_MODEL_REVISION
            or model.get("model_artifact_sha256") != EXPECTED_MODEL_ARTIFACT_SHA256):
        raise ContractError("feature model does not match the locally verified 02-04 snapshot")
    conv = model.get("source_conversion_receipt")
    if not isinstance(conv, str):
        raise ContractError("feature manifest lacks derived model conversion receipt path")
    conv_path = Path(conv)
    if not conv_path.is_absolute():
        conv_path = feature_path.parent / conv_path
    verify_derived_snapshot(conv_path.parent)
    if _file_sha(conv_path) != model.get("source_conversion_receipt_sha256"):
        raise ContractError("feature model conversion receipt hash mismatch")
    provenance = ProteinFeatureExtractor._local_model_provenance(conv_path.parent, model_name=model["model_id"])
    if (not isinstance(provenance, Mapping)
            or provenance.get("revision") != model.get("revision")
            or provenance.get("model_artifact_sha256") != model.get("model_artifact_sha256")
            or provenance.get("config_sha256") != model.get("config_sha256")
            or provenance.get("tokenizer_sha256") != model.get("tokenizer_sha256")
            or provenance.get("weights_sha256") != model.get("weights_sha256")):
        raise ContractError("derived local model file inventory does not match feature-manifest provenance")
    feature_ready = True
    protein_records = {row["accession"]: ProteinRecord(**row) for row in bundle.get("proteins", [])}
    if set(protein_records) != set(expected):
        raise ContractError("feature manifest accessions differ from the verified source bundle")
    for accession, row in expected.items():
        if row.get("entity_bundle_hash") != bundle_hash or row.get("model_revision") != model.get("revision"):
            raise ContractError("feature row corpus/model binding mismatch")
        if row.get("model_artifact_sha256") != model.get("model_artifact_sha256"):
            raise ContractError("feature row model artifact binding mismatch")
        if row.get("status") == "ready":
            if row.get("finite") is not True or row.get("embedding_dim") != model.get("embedding_dim"):
                raise ContractError("ready feature row is non-finite or has a dimension mismatch")
            embedding = row.get("features", {}).get("esm_embedding")
            if (not isinstance(embedding, list) or len(embedding) != model.get("embedding_dim")
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) for value in embedding)):
                raise ContractError("ready feature row does not contain the finite registered embedding")
            protein = protein_records[accession]
            expected_key = build_feature_cache_key(
                protein, model_id=model["model_id"], model_revision=model["revision"],
                model_artifact_sha256=model["model_artifact_sha256"], pooling_policy=row["pooling_policy"],
                entity_version=bundle["entity_version"], entity_bundle_hash=bundle_hash,
            )
            if row.get("cache_key") != expected_key:
                raise ContractError(f"feature cache key does not match verified identity/model binding: {accession}")
            cache_path = feature_path.parent / "cache" / f"{expected_key}.json"
            cache_raw = _bytes(cache_path)
            if _sha(cache_raw) != row.get("cache_file_sha256"):
                raise ContractError(f"feature cache bytes hash mismatch: {accession}")
            cache_entry = json.loads(cache_raw.decode("utf-8"))
            cached_result = cache_entry.get("result")
            if (not isinstance(cached_result, dict)
                    or cache_entry.get("result_sha256") != _canonical_hash(cached_result)
                    or cached_result.get("features") != row.get("features")
                    or cached_result.get("sequence_sha256") != row.get("sequence_sha256")
                    or cached_result.get("model_artifact_sha256") != model.get("model_artifact_sha256")
                    or cached_result.get("cache_key") != expected_key):
                raise ContractError(f"feature cache contents do not match the feature manifest: {accession}")
        elif row.get("status") == "unsupported_length":
            feature_ready = False
        else:
            raise ContractError(f"feature row has unresolved/unknown status: {accession}")
    expected_long = EXPECTED_UNSUPPORTED_ACCESSIONS
    unsupported = {a for a, row in expected.items() if row.get("status") == "unsupported_length"}
    if unsupported != expected_long:
        raise ContractError("unsupported-length records differ from the registered complete-cohort result")
    if feature.get("status_counts") != EXPECTED_FEATURE_STATUS_COUNTS:
        raise ContractError("feature manifest status totals differ from the verified 44-protein run")
    return {"status": "ready" if feature_ready else "blocked", "feature_manifest_sha256": _sha(raw),
            "status_counts": feature.get("status_counts"), "unsupported_accessions": sorted(unsupported),
            "model_revision": model["revision"], "model_artifact_sha256": model["model_artifact_sha256"],
            "conversion_receipt_sha256": _file_sha(conv_path)}, feature_ready


def _budget_blocked(bindings: Mapping[str, str], manifest: Mapping[str, Any], receipt_path: Path,
                    feature: Mapping[str, Any], alignment_cells: int, reason: str) -> tuple[dict[str, Any], int]:
    report = {
        "schema_version": 1, "corpus_version": manifest["version_id"], "protocol_version": "corpus-formal-v1.0",
        "hashes": {**bindings, "formal_receipt_sha256": _file_sha(receipt_path),
                   "feature_manifest_sha256": feature["feature_manifest_sha256"]},
        "tool_gate": {"status": "blocked", "audit_status": "not_run_within_registered_budget",
                      "audit_exit_code": None, "audit_findings": None,
                      "preflight_alignment_cells": alignment_cells, "alignment_cell_cap": MAX_AUDIT_CELLS,
                      "prior_measured_cells": PRIOR_MEASURED_CELLS,
                      "cumulative_measured_cells": PRIOR_MEASURED_CELLS + alignment_cells,
                      "global_alignment_cell_cap": GLOBAL_CELLS_CAP,
                      "prior_measured_seconds": PRIOR_MEASURED_SECONDS,
                      "global_compute_wall_cap_seconds": GLOBAL_SECONDS_CAP,
                      "budget_reason": reason},
        "support_gate": {"status": "unverified", "observed_support": None,
                         "thresholds": "registered support policy retained; audit not completed"},
        "esm_gate": {**feature, "status": feature["status"]},
        "status": "blocked", "reasons": ["registered_resource_budget_insufficient_for_audit", reason],
        "unverified": ["independent_audit_not_run_within_registered_budget",
                       "support_not_recomputed_without_independent_audit",
                       "external_model_revision_not_verified_remotely"],
    }
    return report, 4


def evaluate(manifest_path: os.PathLike[str] | str, feature_path: os.PathLike[str] | str,
             support_policy_path: os.PathLike[str] | str, formal_receipt_path: os.PathLike[str] | str,
             corpus_config_path: os.PathLike[str] | str) -> tuple[dict[str, Any], int]:
    manifest_file = Path(manifest_path).resolve(strict=True)
    manifest, _ = _json(manifest_file)
    receipt_file = Path(formal_receipt_path).resolve(strict=True)
    config_file = Path(corpus_config_path).resolve(strict=True)
    bindings = _verify_protocol(config_file, receipt_file, manifest_file, manifest)
    policy = _verify_support_policy(support_policy_path)
    if _sha(_bytes(ROOT / json.loads(_bytes(config_file).decode("utf-8"))["support_policy_path"])) != SUPPORT_HASH:
        raise ContractError("explicit support policy differs from formal config binding")
    bundle_path = _resolve_manifest_ref(manifest_file, manifest["entities"]["path"])
    bundle = load_entity_bundle(bundle_path, verify_sources=True)
    proteins = bundle.get("proteins", [])
    feature, esm_ok = _feature_gate(Path(feature_path).resolve(strict=True), manifest, bundle)
    feature_rows = {row["accession"]: row for row in json.loads(Path(feature_path).read_text(encoding="utf-8"))["entities"]}
    protein_by_accession = {row["accession"]: row for row in proteins}
    if set(feature_rows) != set(protein_by_accession):
        raise ContractError("feature rows do not cover the source-verified frozen proteins")
    for accession, protein in protein_by_accession.items():
        row = feature_rows[accession]
        if (row.get("taxonomy_id") != protein.get("taxonomy_id")
                or row.get("sequence_length") != protein.get("sequence_length")
                or row.get("sequence_sha256") != protein.get("sequence_sha256")):
            raise ContractError(f"feature identity/sequence binding differs from frozen source: {accession}")
    alignment_cells = sum(
        int(left["sequence_length"]) * int(right["sequence_length"])
        for index, left in enumerate(proteins) for right in proteins[index + 1:]
    )
    if alignment_cells != EXPECTED_FULL_PASS_CELLS:
        raise ContractError("independent audit preflight differs from the measured 44-protein full pass")
    cumulative_cells = PRIOR_MEASURED_CELLS + alignment_cells
    if alignment_cells > MAX_AUDIT_CELLS or cumulative_cells > GLOBAL_CELLS_CAP:
        return _budget_blocked(bindings, manifest, receipt_file, feature, alignment_cells,
                               "single_pass_or_global_alignment_cell_cap_exceeded")
    audit_start = time.monotonic()
    audit = audit_split(manifest_file, budget_overrides={"molecule_pairs": 8128, "protein_pairs": 2016,
                                                        "alignment_cells": MAX_AUDIT_CELLS})
    audit_elapsed = time.monotonic() - audit_start
    if audit.status in {"size_limit", "dependency_missing"} or audit.exit_code == 4:
        return _budget_blocked(bindings, manifest, receipt_file, feature, alignment_cells,
                               f"independent_audit_{audit.status}")
    if audit.status != "clean" or audit.exit_code != 0:
        raise ContractError(f"independent corpus audit rejected input: {audit.status}: {audit.error}")
    cumulative_seconds = PRIOR_MEASURED_SECONDS + audit_elapsed
    budget_ok = audit_elapsed <= MAX_AUDIT_SECONDS and cumulative_seconds <= GLOBAL_SECONDS_CAP
    if audit.entity_bundle_hash != manifest["entities"].get("content_sha256"):
        raise ContractError("audit entity bundle binding differs from frozen manifest")
    support, support_ok = _support(audit.to_dict(), manifest, policy)
    tool_ok = bool(manifest.get("audit", {}).get("status") == "clean"
                   and manifest.get("source_readiness_receipt_ref") and budget_ok)
    status, reasons, exit_code = _admission_status(tool_ok, support_ok, esm_ok)
    if not budget_ok:
        reasons.append("independent_audit_exceeded_registered_900s_or_global_wall_cap")
    report = {"schema_version": 1, "corpus_version": manifest["version_id"], "protocol_version": "corpus-formal-v1.0",
              "hashes": {**bindings, "formal_receipt_sha256": _file_sha(receipt_file),
                         "support_policy_sha256": SUPPORT_HASH, "feature_manifest_sha256": feature["feature_manifest_sha256"]},
              "tool_gate": {"status": "ready" if tool_ok else "blocked", "audit_status": audit.status,
                            "budget_status": "within_budget" if budget_ok else "exceeded",
                            "audit_exit_code": audit.exit_code, "audit_findings": len(audit.findings),
                            "independent_audit_seconds": round(audit_elapsed, 6),
                            "preflight_alignment_cells": alignment_cells, "alignment_cell_cap": MAX_AUDIT_CELLS,
                            "prior_measured_cells": PRIOR_MEASURED_CELLS, "cumulative_measured_cells": cumulative_cells,
                            "global_alignment_cell_cap": GLOBAL_CELLS_CAP,
                            "prior_measured_seconds": PRIOR_MEASURED_SECONDS,
                            "cumulative_measured_seconds": round(cumulative_seconds, 6),
                            "global_compute_wall_cap_seconds": GLOBAL_SECONDS_CAP,
                            "molecule_pair_cap": 8128, "protein_pair_cap": 2016,
                            "audit_wall_cap_seconds": MAX_AUDIT_SECONDS},
              "support_gate": {"status": "ready" if support_ok else "blocked", "observed_support": support["counts"],
                               "thresholds": support["thresholds"]},
              "esm_gate": {**feature, "status": "ready" if esm_ok else "blocked"},
              "status": status, "reasons": reasons, "unverified": ["external_model_revision_not_verified_remotely"]}
    return report, exit_code


def resolve_new_output(path: os.PathLike[str] | str) -> Path:
    requested = Path(os.path.expanduser(os.fspath(path)))
    if not requested.is_absolute():
        raise ContractError("output directory must be absolute")
    requested = Path(os.path.abspath(requested))
    _assert_no_reparse_components(requested)
    if re.fullmatch(r"[0-9a-f]{32}", requested.name) is None:
        raise ContractError("output directory name must be a lowercase GUID")
    apl_root = ROOT.resolve()
    for name in PROTECTED:
        protected = (apl_root / name).resolve(strict=False)
        if requested == protected or protected in requested.parents or requested in protected.parents:
            raise ContractError("output root intersects a protected project path")
    if requested.exists() or requested.is_symlink():
        raise ContractError("refusing to overwrite existing output directory")
    return requested


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--support-policy", required=True)
    parser.add_argument("--formal-receipt", required=True)
    parser.add_argument("--corpus-config", default=str(FORMAL_CONFIG))
    parser.add_argument("--out", required=True)
    return parser


def _admission_status(tool_ok: bool, support_ok: bool, esm_ok: bool) -> tuple[str, list[str], int]:
    reasons = []
    if not tool_ok:
        reasons.append("source_or_contract_audit_not_clean")
    if not support_ok:
        reasons.append("registered_class_or_cluster_support_floors_not_met")
    if not esm_ok:
        reasons.append("whole_cohort_ESM_has_known_unsupported_lengths")
    status = "ready" if tool_ok and support_ok and esm_ok else "blocked"
    return status, reasons, 0 if status == "ready" else 4


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_argument_parser().parse_args(argv)
        output = resolve_new_output(args.out)
        report, code = evaluate(args.manifest, args.features, args.support_policy, args.formal_receipt, args.corpus_config)
        output.mkdir(parents=True)
        try:
            target = output / "training_admission.json"
            payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            with target.open("xb") as stream:
                stream.write(payload)
        except Exception:
            raise
        print(json.dumps({"status": report["status"], "out": str(output), "exit_code": code}, ensure_ascii=False))
        return code
    except ContractError as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "rejected", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
