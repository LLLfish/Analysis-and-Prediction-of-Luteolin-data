"""Create immutable, source-bound snapshots of local ESM protein features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping
from importlib import metadata as importlib_metadata

from data_contracts import FeatureResult, ProteinRecord
from evidence_builder import load_entity_bundle
from protein_features import ProteinFeatureExtractor, build_feature_cache_key
from artifact_store import _assert_no_reparse_components


MODEL_ID = "facebook/esm2_t6_8M_UR50D"
MODEL_DIR_NAME = MODEL_ID.replace("/", "--")
MAX_SUPPORTED_RESIDUES = 1022
POOLING_VERSION = "residue-mask-mean-v1"
_PROTECTED_NAMES = (
    "data", "framework_output", "offline_data.json", ".model_cache",
    "model_cache", "features", "history", "output_dashboard.html",
)
_MODEL_FILES = {
    ".gitattributes", "config.json", "model.safetensors", "pytorch_model.bin",
    "README.md", "special_tokens_map.json", "tf_model.h5",
    "tokenizer_config.json", "vocab.txt",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def resolve_new_output(path: os.PathLike[str] | str) -> Path:
    unresolved = Path(os.path.expanduser(os.fspath(path)))
    if not unresolved.is_absolute():
        raise ValueError("output directory must be absolute")
    unresolved = Path(os.path.abspath(unresolved))
    _assert_no_reparse_components(unresolved)
    requested = unresolved
    try:
        uuid.UUID(hex=requested.name)
    except (ValueError, AttributeError) as exc:
        raise ValueError("output directory name must be a 32-character GUID") from exc
    if requested.name != requested.name.lower() or len(requested.name) != 32:
        raise ValueError("output directory name must be a lowercase 32-character GUID")
    resolved = requested.resolve(strict=False)
    apl_root = Path(__file__).resolve().parent
    for name in _PROTECTED_NAMES:
        protected = (apl_root / name).resolve(strict=False)
        if _is_under(requested, protected) or _is_under(protected, requested) or _is_under(resolved, protected) or _is_under(protected, resolved):
            raise ValueError("output root intersects a protected project path")
    if requested.exists() or requested.is_symlink() or resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {requested}")
    return resolved


def _assert_safe_model_destination(source: Path, destination: os.PathLike[str] | str) -> Path:
    unresolved = Path(os.path.expanduser(os.fspath(destination)))
    if not unresolved.is_absolute():
        raise ValueError("derived model destination must be absolute")
    unresolved = Path(os.path.abspath(unresolved))
    _assert_no_reparse_components(unresolved)
    if unresolved.name != MODEL_DIR_NAME:
        raise ValueError(f"derived model directory must be named {MODEL_DIR_NAME}")
    if re.fullmatch(r"model-[0-9a-f]{32}", unresolved.parent.name) is None:
        raise ValueError("derived model must be inside a new model-<32-character-guid> directory")
    resolved = unresolved.resolve(strict=False)
    source_resolved = source.resolve(strict=True)
    if (_is_under(unresolved, source_resolved) or _is_under(source_resolved, unresolved)
            or _is_under(resolved, source_resolved) or _is_under(source_resolved, resolved)):
        raise ValueError("derived model destination intersects its source snapshot")
    apl_root = Path(__file__).resolve().parent
    for name in _PROTECTED_NAMES:
        protected = (apl_root / name).resolve(strict=False)
        if any(_is_under(candidate, protected) or _is_under(protected, candidate)
               for candidate in (unresolved, resolved)):
            raise ValueError("derived model destination intersects a protected project path")
    if unresolved.exists() or unresolved.is_symlink() or resolved.exists():
        raise FileExistsError("derived model destination already exists")
    return resolved


def derive_revision_snapshot(
    source_dir: os.PathLike[str] | str,
    inventory_path: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Copy an already verified local cache into a new, revision-bearing snapshot."""
    source = Path(source_dir).resolve(strict=True)
    destination_path = _assert_safe_model_destination(source, destination)
    inventory_file = Path(inventory_path).resolve(strict=True)
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    rows = inventory.get("files") if isinstance(inventory, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("model inventory has no verified file rows")
    by_name = {row.get("file"): row for row in rows if isinstance(row, dict)}
    if set(by_name) != _MODEL_FILES:
        raise ValueError("model inventory file set is incomplete or unexpected")
    revisions = {row.get("revision") for row in rows}
    if len(revisions) != 1:
        raise ValueError("source model files do not share one recorded revision")
    revision = revisions.pop()
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("source model revision is invalid")

    cache_metadata = source / ".cache" / "huggingface" / "download"
    verified_rows: list[dict[str, Any]] = []
    for name in sorted(_MODEL_FILES):
        row = by_name[name]
        source_file = source / name
        metadata_file = cache_metadata / f"{name}.metadata"
        if not source_file.is_file() or not metadata_file.is_file():
            raise ValueError(f"source model file or its local HF metadata is missing: {name}")
        actual_sha = _sha256(source_file)
        metadata_lines = metadata_file.read_text(encoding="utf-8").splitlines()
        if (actual_sha != row.get("sha256") or len(metadata_lines) < 2
                or metadata_lines[0] != revision or metadata_lines[1] != row.get("etag")
                or row.get("metadata_matches_bytes") is not True):
            raise ValueError(f"source model bytes or local revision/ETag metadata mismatch: {name}")
        verified_rows.append({
            "file": name, "bytes": source_file.stat().st_size,
            "sha256": actual_sha, "revision": revision,
            "etag": metadata_lines[1], "etag_kind": row.get("etag_kind"),
        })

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.mkdir()
    try:
        shutil.copyfile(inventory_file, destination_path / "source_inventory.json")
        for name in sorted(_MODEL_FILES):
            shutil.copyfile(source / name, destination_path / name)
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        derived_config = dict(source_config)
        derived_config["_commit_hash"] = revision
        config_bytes = (json.dumps(derived_config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        (destination_path / "config.json").write_bytes(config_bytes)
        receipt = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "revision": revision,
            "revision_origin": "verified local Hugging Face download metadata; not newly remote-verified",
            "external_revision_verified": False,
            "source_dir": str(source),
            "source_inventory_sha256": _sha256(inventory_file),
            "source_config_sha256": _sha256(source / "config.json"),
            "derived_config_sha256": _sha256(destination_path / "config.json"),
            "files": verified_rows,
        }
        (destination_path / "conversion_receipt.json").write_bytes(
            json.dumps(receipt, ensure_ascii=False, indent=2).encode("utf-8")
        )
        return receipt
    except Exception:
        # Keep partial evidence for diagnosis; never silently replace it.
        raise


def verify_derived_snapshot(model_dir: os.PathLike[str] | str) -> dict[str, Any]:
    """Recheck the local inventory, conversion receipt, and derived model bytes."""
    root = Path(model_dir)
    receipt_path = root / "conversion_receipt.json"
    inventory_path = root / "source_inventory.json"
    if not receipt_path.is_file() or not inventory_path.is_file():
        raise ValueError("revision-bearing model snapshot requires its conversion receipt and source inventory")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_raw = inventory_path.read_bytes()
    inventory = json.loads(inventory_raw.decode("utf-8"))
    if _sha256_bytes(inventory_raw) != receipt.get("source_inventory_sha256"):
        raise ValueError("derived model source inventory hash mismatch")
    revision = receipt.get("revision")
    rows = inventory.get("files") if isinstance(inventory, dict) else None
    by_name = {row.get("file"): row for row in rows if isinstance(row, dict)} if isinstance(rows, list) else {}
    receipt_rows = receipt.get("files")
    if (receipt.get("model_id") != MODEL_ID or receipt.get("external_revision_verified") is not False
            or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or not isinstance(receipt_rows, list) or len(receipt_rows) != len(_MODEL_FILES)
            or set(by_name) != _MODEL_FILES):
        raise ValueError("derived model receipt metadata is invalid")
    receipt_by_name = {row.get("file"): row for row in receipt_rows if isinstance(row, dict)}
    if set(receipt_by_name) != _MODEL_FILES:
        raise ValueError("derived model receipt file set is invalid")
    for name in sorted(_MODEL_FILES):
        row, receipt_row = by_name[name], receipt_by_name[name]
        if (row.get("revision") != revision or row.get("sha256") != receipt_row.get("sha256")
                or row.get("etag") != receipt_row.get("etag") or row.get("metadata_matches_bytes") is not True):
            raise ValueError(f"derived model inventory and receipt differ for {name}")
        path = root / name
        if not path.is_file():
            raise ValueError(f"derived model file is missing: {name}")
        actual = _sha256(path)
        expected = receipt.get("derived_config_sha256") if name == "config.json" else row.get("sha256")
        if actual != expected:
            raise ValueError(f"derived model file hash mismatch: {name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("_commit_hash") != revision:
        raise ValueError("derived model config revision does not match the verified receipt")
    if receipt.get("source_config_sha256") != by_name["config.json"].get("sha256"):
        raise ValueError("source config hash in the conversion receipt is inconsistent")
    return receipt


def _feature_payload(result: FeatureResult) -> dict[str, Any]:
    payload = result.to_dict()
    return {"result": payload, "result_sha256": _sha256_bytes(_canonical_bytes(payload))}


def _cache_binding(
    protein: ProteinRecord,
    bundle: Mapping[str, Any],
    provenance: Mapping[str, str],
    pooling_policy: str,
) -> dict[str, Any]:
    key = build_feature_cache_key(
        protein, model_id=MODEL_ID, model_revision=provenance["revision"],
        model_artifact_sha256=provenance["model_artifact_sha256"],
        pooling_policy=pooling_policy,
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
    )
    return {
        "accession": protein.accession,
        "taxonomy_id": protein.taxonomy_id,
        "sequence_length": protein.sequence_length,
        "sequence_sha256": protein.sequence_sha256,
        "entity_version": bundle["entity_version"],
        "entity_bundle_hash": bundle["content_sha256"],
        "source_kind": protein.source_kind,
        "model_id": MODEL_ID,
        "model_revision": provenance["revision"],
        "model_artifact_sha256": provenance["model_artifact_sha256"],
        "pooling_policy": pooling_policy,
        "cache_key": key,
    }


def _load_source_cache_manifest(
    cache_from: os.PathLike[str] | str,
    *,
    bundle: Mapping[str, Any],
    bundle_file_sha256: str,
    provenance: Mapping[str, str],
    max_tokens: int,
    runtime_versions: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    root = Path(os.path.expanduser(os.fspath(cache_from)))
    if not root.is_absolute():
        raise ValueError("cache-from must be an absolute snapshot directory")
    _assert_no_reparse_components(root)
    root = root.resolve(strict=True)
    manifest_path = root / "feature_manifest.json"
    _assert_no_reparse_components(manifest_path)
    if not manifest_path.is_file():
        raise ValueError("cache-from feature_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest.get("model", {})
    inference = manifest.get("inference", {})
    if (manifest.get("bundle_file_sha256") != bundle_file_sha256
            or manifest.get("entity_bundle_hash") != bundle.get("content_sha256")
            or manifest.get("entity_version") != bundle.get("entity_version")
            or model.get("model_id") != MODEL_ID
            or model.get("revision") != provenance["revision"]
            or model.get("model_artifact_sha256") != provenance["model_artifact_sha256"]
            or model.get("config_sha256") != provenance["config_sha256"]
            or model.get("tokenizer_sha256") != provenance["tokenizer_sha256"]
            or model.get("weights_sha256") != provenance["weights_sha256"]
            or inference.get("pooling") != POOLING_VERSION
            or inference.get("max_tokens") != max_tokens
            or inference.get("runtime_versions") != dict(runtime_versions)):
        raise ValueError("cache-from manifest does not match the current bundle/model/pooling/runtime request")
    rows = manifest.get("entities")
    if not isinstance(rows, list):
        raise ValueError("cache-from manifest has no per-entity records")
    return root, manifest, _sha256(manifest_path)


def _load_cache_entry(
    path: Path,
    *,
    expected: Mapping[str, Any],
    expected_dimension: int,
    source_manifest_row: Mapping[str, Any],
) -> FeatureResult:
    source_file_sha256 = source_manifest_row.get("cache_file_sha256")
    if not isinstance(source_file_sha256, str) or _sha256(path) != source_file_sha256:
        raise ValueError("feature cache source-file SHA-256 does not match its source manifest")
    entry = json.loads(path.read_text(encoding="utf-8"))
    payload = entry.get("result") if isinstance(entry, dict) else None
    if not isinstance(payload, dict) or entry.get("result_sha256") != _sha256_bytes(_canonical_bytes(payload)):
        raise ValueError("feature cache payload integrity check failed")
    result = FeatureResult(**payload)
    if result.status != "ready" or result.test_provenance is not None:
        raise ValueError("test-only or non-ready features cannot enter the real feature cache")
    result_data = result.to_dict()
    for field, value in expected.items():
        if result_data.get(field) != value or source_manifest_row.get(field) != value:
            raise ValueError(f"feature cache identity/model/pooling binding mismatch: {field}")
    embedding = result.features.get("esm_embedding")
    source_embedding = source_manifest_row.get("features", {}).get("esm_embedding")
    if (not isinstance(embedding, list) or len(embedding) != expected_dimension
            or source_manifest_row.get("status") != "ready"
            or source_manifest_row.get("finite") is not True
            or embedding != source_embedding
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                   for value in embedding)):
        raise ValueError("feature cache embedding differs from its source manifest or model dimension")
    return result


def _save_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _runtime_versions(device: str) -> dict[str, Any]:
    packages = {}
    for distribution in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = None
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        device_name = torch.cuda.get_device_name(0) if device.startswith("cuda") and cuda_available else None
    except Exception:
        cuda_available = False
        cuda_version = None
        device_name = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "device": device,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_name": device_name,
    }


def run_snapshot(
    *,
    entities_path: os.PathLike[str] | str,
    model_dir: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    max_tokens: int = 1024,
    cache_from: os.PathLike[str] | str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    output = resolve_new_output(output_dir)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
        raise ValueError("max_tokens must be between 1 and the registered 1024-token limit")
    unresolved_model_path = Path(os.path.abspath(os.fspath(model_dir)))
    _assert_no_reparse_components(unresolved_model_path)
    model_path = unresolved_model_path.resolve(strict=False)
    if model_path.name != MODEL_DIR_NAME:
        raise ValueError(f"model-dir must be the local {MODEL_DIR_NAME} snapshot directory")
    conversion_receipt = verify_derived_snapshot(model_path) if model_path.is_dir() else None
    entities = Path(entities_path).resolve(strict=True)
    bundle = load_entity_bundle(str(entities), verify_sources=True)
    bundle_sha = _sha256(entities)
    model_parent = model_path.parent
    extractor = ProteinFeatureExtractor(
        output_dir=None, esm_model_name=MODEL_ID, device=device,
        max_tokens=max_tokens, cache_dir=str(model_parent),
    )
    if extractor.esm_model is None:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
    loaded = extractor.load_esm_model()
    provenance = extractor._runtime_model_provenance if loaded else None
    runtime_versions = _runtime_versions(extractor.device if loaded else device)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8")) if model_path.joinpath("config.json").is_file() else {}
    embedding_dim = getattr(getattr(extractor.esm_model, "config", None), "hidden_size", None) if loaded else config.get("hidden_size")
    model_files = []
    if model_path.is_dir():
        for filename in sorted(_MODEL_FILES):
            file_path = model_path / filename
            if file_path.is_file():
                model_files.append({"file": filename, "bytes": file_path.stat().st_size, "sha256": _sha256(file_path)})
    cache_snapshot = None
    cache_manifest = None
    cache_manifest_sha256 = None
    cache_root = None
    if cache_from:
        cache_snapshot, cache_manifest, cache_manifest_sha256 = _load_source_cache_manifest(
            cache_from, bundle=bundle, bundle_file_sha256=bundle_sha,
            provenance=provenance, max_tokens=max_tokens,
            runtime_versions=runtime_versions,
        )
        cache_root = cache_snapshot / "cache"
        if not cache_root.is_dir():
            raise ValueError("cache-from cache directory is missing")
    rows: list[dict[str, Any]] = []
    cache_events: dict[str, str] = {}
    for protein_data in bundle.get("proteins", []):
        protein = ProteinRecord(**protein_data)
        sequence = protein.full_sequence or ""
        base = {
            "accession": protein.accession, "taxonomy_id": protein.taxonomy_id,
            "sequence_length": protein.sequence_length,
            "sequence_sha256": protein.sequence_sha256,
            "entity_version": bundle.get("entity_version"),
            "entity_bundle_hash": bundle.get("content_sha256"),
            "bundle_file_sha256": bundle_sha,
            "model_id": MODEL_ID,
            "model_revision": provenance.get("revision") if provenance else config.get("_commit_hash"),
            "model_artifact_sha256": provenance.get("model_artifact_sha256") if provenance else None,
            "config_sha256": provenance.get("config_sha256") if provenance else None,
            "tokenizer_sha256": provenance.get("tokenizer_sha256") if provenance else None,
            "weights_sha256": provenance.get("weights_sha256") if provenance else None,
            "pooling_policy": f"{POOLING_VERSION}:max_tokens={max_tokens}",
            "max_tokens": max_tokens, "embedding_dim": embedding_dim,
        }
        if protein.sequence_status != "verified" or protein.identity_status != "verified":
            row = {**base, "status": "unverified_source", "reason": "source_sequence_or_identity_not_verified", "finite": False, "features": {}}
        elif len(sequence) > MAX_SUPPORTED_RESIDUES or len(sequence) + 2 > max_tokens:
            row = {**base, "status": "unsupported_length", "reason": "full_sequence_exceeds_registered_token_domain", "finite": False, "features": {}}
        elif not loaded:
            row = {**base, "status": "model_missing", "reason": "revision_verified_local_esm_unavailable", "finite": False, "features": {}}
        else:
            expected_binding = _cache_binding(protein, bundle, provenance, base["pooling_policy"])
            cache_key = expected_binding["cache_key"]
            cached_result = None
            if cache_root is not None:
                cache_path = cache_root / f"{cache_key}.json"
                try:
                    _assert_no_reparse_components(cache_path)
                    matching_rows = [
                        row for row in cache_manifest["entities"]
                        if row.get("accession") == protein.accession
                        and row.get("taxonomy_id") == protein.taxonomy_id
                        and row.get("sequence_sha256") == protein.sequence_sha256
                    ]
                    if len(matching_rows) != 1:
                        raise ValueError("source cache manifest lacks a unique matching protein row")
                    if not cache_path.is_file():
                        raise ValueError("source cache file is missing")
                    cached_result = _load_cache_entry(
                        cache_path, expected=expected_binding,
                        expected_dimension=int(embedding_dim),
                        source_manifest_row=matching_rows[0],
                    )
                    cache_events[protein.accession or ""] = "disk_cache_hit"
                except Exception as exc:
                    row = {**base, "status": "error", "reason": f"cache_integrity_rejected:{type(exc).__name__}", "finite": False, "features": {}, "cache_key": cache_key}
                    rows.append(row)
                    continue
            if cached_result is None:
                result = extractor._extract_validated_record(protein, bundle, test_only=False)
            else:
                result = cached_result
            result_data = result.to_dict()
            embedding = result_data.get("features", {}).get("esm_embedding", [])
            finite = bool(embedding) and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
                for value in embedding
            )
            row = {**base, **result_data, "finite": finite, "embedding_dim": len(embedding) if embedding else embedding_dim,
                   "cache_event": cache_events.get(protein.accession or "", "first_run_inference")}
            if result.status == "ready" and finite:
                cache_destination = output / "cache" / f"{cache_key}.json"
                _save_json_new(cache_destination, _feature_payload(result))
                row["cache_file_sha256"] = _sha256(cache_destination)
            elif result.status != "ready":
                row["reason"] = result.reason or "feature_extraction_not_ready"
        rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    ready = bool(loaded and rows and counts.get("ready", 0) == len(rows))
    manifest = {
        "schema_version": 1, "mode": "local_only_feature_snapshot",
        "status": "ready" if ready else "blocked",
        "blocked_reasons": [] if ready else [
            "whole_cohort_has_non_ready_or_unsupported_entities",
            *( ["local_model_unavailable"] if not loaded else [] ),
        ],
        "entities_path": str(entities), "bundle_file_sha256": bundle_sha,
        "entity_bundle_hash": bundle.get("content_sha256"),
        "entity_version": bundle.get("entity_version"),
        "model": {
            "model_id": MODEL_ID, "revision": provenance.get("revision") if provenance else config.get("_commit_hash"),
            "external_revision_verified": False,
            "model_artifact_sha256": provenance.get("model_artifact_sha256") if provenance else None,
            "config_sha256": provenance.get("config_sha256") if provenance else _sha256(model_path / "config.json") if model_path.joinpath("config.json").is_file() else None,
            "tokenizer_sha256": provenance.get("tokenizer_sha256") if provenance else None,
            "weights_sha256": provenance.get("weights_sha256") if provenance else None,
            "files": model_files, "embedding_dim": embedding_dim,
            "loading_diagnostics": extractor._loading_diagnostics if loaded else None,
            "source_conversion_receipt": str(model_path / "conversion_receipt.json") if model_path.joinpath("conversion_receipt.json").is_file() else None,
            "source_conversion_receipt_sha256": _sha256(model_path / "conversion_receipt.json") if model_path.joinpath("conversion_receipt.json").is_file() else None,
            "source_revision_verified_by_local_inventory": conversion_receipt is not None,
        },
        "inference": {"device": extractor.device if loaded else device, "eval": bool(loaded and not extractor.esm_model.training),
                      "inference_mode": True, "pooling": POOLING_VERSION,
                      "max_tokens": max_tokens, "max_supported_residues": MAX_SUPPORTED_RESIDUES,
                      "threads": 1, "cache_source": str(cache_snapshot) if cache_snapshot else None,
                      "cache_source_manifest_sha256": cache_manifest_sha256,
                      "runtime_versions": runtime_versions,
                      "cache_events": cache_events, "allclose_atol": 1e-6, "allclose_rtol": 1e-6},
        "entity_count": len(rows), "status_counts": counts,
        "support_gate": {"status": "blocked", "reason": "formal_candidate_support_shortfall",
                         "train": {"active": 55, "inactive": 5},
                         "val": {"active": 0, "inactive": 0},
                         "test": {"active": 0, "inactive": 0}},
        "supervised_models": {"cnn": "skipped_untrained", "transformer": "skipped_untrained", "fusion": "empty"},
        "entities": rows,
    }
    _save_json_new(output / "feature_manifest.json", manifest)
    for row in rows:
        name = f"{row.get('taxonomy_id')}-{row.get('accession') or 'unknown'}.json"
        _save_json_new(output / "entities" / name, row)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a local-only source-bound ESM feature snapshot.")
    parser.add_argument("--entities", required=True, help="Source-verified entities.json")
    parser.add_argument("--model-dir", required=True, help=f"Local revision-bearing {MODEL_DIR_NAME} directory")
    parser.add_argument("--out", required=True, help="New absolute GUID output directory")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--cache-from", help="Read-only feature cache from a prior snapshot")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        manifest = run_snapshot(
            entities_path=args.entities, model_dir=args.model_dir, output_dir=args.out,
            max_tokens=args.max_tokens, cache_from=args.cache_from, device=args.device,
        )
    except Exception as exc:
        print(json.dumps({"status": "blocked", "reason": f"{type(exc).__name__}:{exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": manifest["status"], "out": args.out,
                      "entity_count": manifest["entity_count"], "status_counts": manifest["status_counts"]}, ensure_ascii=False))
    return 0 if manifest["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
