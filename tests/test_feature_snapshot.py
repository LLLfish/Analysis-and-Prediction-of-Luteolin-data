"""Contracts for local-only, identity-bound ESM feature snapshots."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

import data_contracts
import evidence_builder
import feature_snapshot
import protein_features


FIXTURE = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _bundle(tmp_path: Path, sequence: str = "ACDEFG") -> tuple[Path, dict]:
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manifest["uniprot"]["P12345"]["sequence"]["value"] = sequence
    manifest["uniprot"]["P12345"]["sequence"]["length"] = len(sequence)
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    root = tmp_path / "bundle"
    evidence_builder.ingest_local_manifest(manifest_path, root)
    path = root / "entities.json"
    return path, evidence_builder.load_entity_bundle(path, verify_sources=True)


def _guid_path(parent: Path) -> Path:
    return parent / uuid.uuid4().hex


def _model_destination(parent: Path) -> Path:
    return parent / f"model-{uuid.uuid4().hex}" / feature_snapshot.MODEL_DIR_NAME


def _ready_result(bundle: dict) -> data_contracts.FeatureResult:
    protein = data_contracts.ProteinRecord(**bundle["proteins"][0])
    pooling = f"{feature_snapshot.POOLING_VERSION}:max_tokens=1024"
    artifact = hashlib.sha256(b"bound local model snapshot").hexdigest()
    key = protein_features.build_feature_cache_key(
        protein, model_id=feature_snapshot.MODEL_ID,
        model_revision="a" * 40, model_artifact_sha256=artifact,
        pooling_policy=pooling, entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
    )
    return data_contracts.FeatureResult(
        status="ready", features={"esm_embedding": [0.25, -1.5]},
        accession=protein.accession, taxonomy_id=protein.taxonomy_id,
        sequence_length=protein.sequence_length,
        sequence_sha256=protein.sequence_sha256,
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
        source_kind="offline_snapshot", model_id=feature_snapshot.MODEL_ID,
        model_revision="a" * 40, model_artifact_sha256=artifact,
        pooling_policy=pooling, cache_key=key,
    )


def _provenance(result: data_contracts.FeatureResult) -> dict[str, str]:
    return {
        "revision": result.model_revision,
        "model_artifact_sha256": result.model_artifact_sha256,
        "config_sha256": hashlib.sha256(b"config").hexdigest(),
        "tokenizer_sha256": hashlib.sha256(b"tokenizer").hexdigest(),
        "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
    }


def _source_row(result: data_contracts.FeatureResult, cache_path: Path) -> dict:
    return {**result.to_dict(), "finite": True, "cache_file_sha256": feature_snapshot._sha256(cache_path)}


def test_output_path_requires_new_guid_and_rejects_protected_roots(tmp_path):
    output = _guid_path(tmp_path)
    assert feature_snapshot.resolve_new_output(output) == output.resolve()
    with pytest.raises(ValueError, match="GUID"):
        feature_snapshot.resolve_new_output(tmp_path / "plain-name")
    with pytest.raises(ValueError, match="absolute"):
        feature_snapshot.resolve_new_output("relative-guid")
    protected = Path(protein_features.__file__).resolve().parent / ".model_cache" / uuid.uuid4().hex
    with pytest.raises(ValueError, match="protected"):
        feature_snapshot.resolve_new_output(protected)
    output.mkdir()
    with pytest.raises(FileExistsError):
        feature_snapshot.resolve_new_output(output)


def test_missing_model_reports_blocked_with_per_entity_source_identity(tmp_path):
    entities, bundle = _bundle(tmp_path)
    manifest = feature_snapshot.run_snapshot(
        entities_path=entities,
        model_dir=tmp_path / feature_snapshot.MODEL_DIR_NAME,
        output_dir=_guid_path(tmp_path),
    )
    assert manifest["status"] == "blocked"
    assert manifest["status_counts"] == {"model_missing": 1}
    result = manifest["entities"][0]
    assert result["accession"] == bundle["proteins"][0]["accession"]
    assert result["taxonomy_id"] == bundle["proteins"][0]["taxonomy_id"]
    assert result["sequence_sha256"] == bundle["proteins"][0]["sequence_sha256"]
    assert result["entity_bundle_hash"] == bundle["content_sha256"]
    assert manifest["supervised_models"] == {
        "cnn": "skipped_untrained", "transformer": "skipped_untrained", "fusion": "empty",
    }


def test_sequence_outside_registered_domain_is_retained_and_unsupported(tmp_path):
    entities, bundle = _bundle(tmp_path, sequence="A" * 1023)
    manifest = feature_snapshot.run_snapshot(
        entities_path=entities,
        model_dir=tmp_path / feature_snapshot.MODEL_DIR_NAME,
        output_dir=_guid_path(tmp_path),
    )
    result = manifest["entities"][0]
    assert result["status"] == "unsupported_length"
    assert result["sequence_length"] == 1023
    assert result["sequence_sha256"] == hashlib.sha256(b"A" * 1023).hexdigest()
    assert bundle["proteins"][0]["full_sequence"] == "A" * 1023


def test_disk_cache_recomputes_binding_and_checks_trusted_manifest_vector(tmp_path):
    _, bundle = _bundle(tmp_path)
    result = _ready_result(bundle)
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(feature_snapshot._feature_payload(result)), encoding="utf-8")
    protein = replace(
        data_contracts.ProteinRecord(**bundle["proteins"][0]),
        source_kind=result.source_kind,
    )
    binding = feature_snapshot._cache_binding(
        protein, bundle, _provenance(result), result.pooling_policy,
    )
    source_row = _source_row(result, path)
    assert feature_snapshot._load_cache_entry(
        path, expected=binding, expected_dimension=2, source_manifest_row=source_row,
    ) == result

    wrong_identity = replace(result, sequence_sha256="e" * 64)
    path.write_text(json.dumps(feature_snapshot._feature_payload(wrong_identity)), encoding="utf-8")
    with pytest.raises(ValueError, match="sequence_sha256"):
        feature_snapshot._load_cache_entry(
            path, expected=binding, expected_dimension=2,
            source_manifest_row=_source_row(wrong_identity, path),
        )

    wrong_model = replace(result, model_revision="c" * 40)
    path.write_text(json.dumps(feature_snapshot._feature_payload(wrong_model)), encoding="utf-8")
    with pytest.raises(ValueError, match="model_revision"):
        feature_snapshot._load_cache_entry(
            path, expected=binding, expected_dimension=2,
            source_manifest_row=_source_row(wrong_model, path),
        )

    test_only = replace(
        result, status="test_only", reason="explicit mock result",
        test_provenance="mock fixture",
    )
    path.write_text(json.dumps(feature_snapshot._feature_payload(test_only)), encoding="utf-8")
    with pytest.raises(ValueError, match="test-only"):
        feature_snapshot._load_cache_entry(
            path, expected=binding, expected_dimension=2,
            source_manifest_row=_source_row(test_only, path),
        )

    path.write_text(json.dumps(feature_snapshot._feature_payload(result)), encoding="utf-8")
    source_row = _source_row(result, path)
    path.write_text(json.dumps(feature_snapshot._feature_payload(result)), encoding="utf-8")
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["result"]["features"]["esm_embedding"][0] = 99.0
    entry["result_sha256"] = feature_snapshot._sha256_bytes(feature_snapshot._canonical_bytes(entry["result"]))
    path.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(ValueError, match="source-file SHA-256"):
        feature_snapshot._load_cache_entry(
            path, expected=binding, expected_dimension=2, source_manifest_row=source_row,
        )


def test_cache_source_manifest_binds_bundle_model_and_pooling(tmp_path):
    _, bundle = _bundle(tmp_path)
    result = _ready_result(bundle)
    provenance = _provenance(result)
    manifest = {
        "bundle_file_sha256": "f" * 64,
        "entity_bundle_hash": bundle["content_sha256"],
        "entity_version": bundle["entity_version"],
        "model": {"model_id": feature_snapshot.MODEL_ID,
                  "revision": provenance["revision"],
                  "model_artifact_sha256": provenance["model_artifact_sha256"],
                  "config_sha256": provenance["config_sha256"],
                  "tokenizer_sha256": provenance["tokenizer_sha256"],
                  "weights_sha256": provenance["weights_sha256"]},
        "inference": {"pooling": feature_snapshot.POOLING_VERSION, "max_tokens": 1024,
                      "runtime_versions": feature_snapshot._runtime_versions("cpu")},
        "entities": [],
    }
    root = tmp_path / "source-snapshot"
    root.mkdir()
    path = root / "feature_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded_root, loaded, _ = feature_snapshot._load_source_cache_manifest(
        root, bundle=bundle, bundle_file_sha256="f" * 64,
        provenance=provenance, max_tokens=1024,
        runtime_versions=feature_snapshot._runtime_versions("cpu"),
    )
    assert loaded_root == root.resolve()
    assert loaded == manifest
    manifest["model"]["revision"] = "d" * 40
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle/model/pooling"):
        feature_snapshot._load_source_cache_manifest(
            root, bundle=bundle, bundle_file_sha256="f" * 64,
            provenance=provenance, max_tokens=1024,
            runtime_versions=feature_snapshot._runtime_versions("cpu"),
        )
    manifest["model"]["revision"] = provenance["revision"]
    manifest["inference"]["runtime_versions"]["python"] = "0.0"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime request"):
        feature_snapshot._load_source_cache_manifest(
            root, bundle=bundle, bundle_file_sha256="f" * 64,
            provenance=provenance, max_tokens=1024,
            runtime_versions=feature_snapshot._runtime_versions("cpu"),
        )


def test_model_derivation_checks_each_source_hash_and_revision_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    metadata = source / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    revision = "b" * 40
    rows = []
    for name in sorted(feature_snapshot._MODEL_FILES):
        content = b'{"model_type":"esm"}' if name == "config.json" else name.encode("utf-8")
        (source / name).write_bytes(content)
        etag = hashlib.sha256(content).hexdigest()
        (metadata / f"{name}.metadata").write_text(f"{revision}\n{etag}\n1.0\n", encoding="utf-8")
        rows.append({
            "file": name, "bytes": len(content), "sha256": etag,
            "revision": revision, "etag": etag,
            "etag_kind": "sha256", "metadata_matches_bytes": True,
        })
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"files": rows}), encoding="utf-8")
    destination = _model_destination(tmp_path / "derived")
    receipt = feature_snapshot.derive_revision_snapshot(source, inventory, destination)
    assert receipt["revision"] == revision
    assert receipt["external_revision_verified"] is False
    assert receipt["source_config_sha256"] != receipt["derived_config_sha256"]
    assert json.loads((destination / "config.json").read_text(encoding="utf-8"))["_commit_hash"] == revision
    assert feature_snapshot.verify_derived_snapshot(destination)["revision"] == revision
    (destination / "model.safetensors").write_bytes(b"tampered derived weights")
    with pytest.raises(ValueError, match="file hash mismatch"):
        feature_snapshot.verify_derived_snapshot(destination)

    wrong_revision = "c" * 40
    (metadata / "config.json.metadata").write_text(
        f"{wrong_revision}\n{rows[[row['file'] for row in rows].index('config.json')]['etag']}\n1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        feature_snapshot.derive_revision_snapshot(
            source, inventory, _model_destination(tmp_path / "wrong"),
        )

    (metadata / "config.json.metadata").write_text(
        f"{revision}\n{rows[[row['file'] for row in rows].index('config.json')]['etag']}\n1.0\n",
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"tampered weights")
    with pytest.raises(ValueError, match="bytes or local revision/ETag"):
        feature_snapshot.derive_revision_snapshot(
            source, inventory, _model_destination(tmp_path / "bad-weights"),
        )


def test_model_derivation_refuses_protected_source_overlap_and_reparse_destinations(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    inventory = tmp_path / "missing-inventory.json"
    protected = Path(protein_features.__file__).resolve().parent / ".model_cache" / f"model-{uuid.uuid4().hex}" / feature_snapshot.MODEL_DIR_NAME
    with pytest.raises(ValueError, match="protected"):
        feature_snapshot.derive_revision_snapshot(source, inventory, protected)
    assert not protected.exists()

    overlap = source / f"model-{uuid.uuid4().hex}" / feature_snapshot.MODEL_DIR_NAME
    with pytest.raises(ValueError, match="source snapshot"):
        feature_snapshot.derive_revision_snapshot(source, inventory, overlap)
    assert not overlap.exists()

    with pytest.raises(ValueError, match="absolute"):
        feature_snapshot.derive_revision_snapshot(source, inventory, Path("relative") / feature_snapshot.MODEL_DIR_NAME)

    target = tmp_path / "reparse-target"
    target.mkdir()
    link = tmp_path / "reparse-parent"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable on this host")
    redirected = link / f"model-{uuid.uuid4().hex}" / feature_snapshot.MODEL_DIR_NAME
    with pytest.raises(ValueError, match="reparse"):
        feature_snapshot.derive_revision_snapshot(source, inventory, redirected)
    assert not (target / redirected.parent.name / feature_snapshot.MODEL_DIR_NAME).exists()


def test_test_only_results_cannot_be_reused_as_real_disk_cache(tmp_path):
    cache = tmp_path / "test-cache.json"
    cache.write_text(json.dumps({"result": {"status": "test_only"}, "result_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        feature_snapshot._load_cache_entry(
            cache, expected={}, expected_dimension=1,
            source_manifest_row={"cache_file_sha256": feature_snapshot._sha256(cache)},
        )
