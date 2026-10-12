"""Unit contracts for admission decisions; no production data enters fixtures."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

import readiness_gate


def _support_fixture():
    assignments = [
        {"split": "train", "molecule_cluster_id": "m1", "protein_cluster_id": "p1"},
        {"split": "train", "molecule_cluster_id": "m2", "protein_cluster_id": "p2"},
    ]
    manifest = {"split": {"pair_assignments": assignments}}
    policy = {
        "minimum_labeled_examples_per_class": {"train": 100, "val": 20, "test": 20},
        "minimum_distinct_connected_molecule_clusters": {"train": None, "val": 5, "test": 5},
        "minimum_distinct_connected_protein_clusters": {"train": None, "val": 5, "test": 5},
    }
    counts = {"train": {"active": 55, "inactive": 5}, "val": {"active": 0, "inactive": 0},
              "test": {"active": 0, "inactive": 0}}
    return manifest, policy, {"class_support": counts}


def test_independently_counted_support_does_not_trust_readiness_boolean():
    manifest, policy, audit = _support_fixture()
    manifest["training_ready"] = True
    support, ready = readiness_gate._support(audit, manifest, policy)
    assert not ready
    assert support["counts"]["train"]["active"] == 55
    assert support["counts"]["val"]["molecule_clusters"] == 0
    assert support["thresholds"]["examples_per_class"]["val"] == 20


@pytest.mark.parametrize("gates", [(True, False, True), (True, True, False), (False, True, True)])
def test_any_failed_gate_keeps_admission_blocked(gates):
    status, reasons, code = readiness_gate._admission_status(*gates)
    assert status == "blocked"
    assert code == 4
    assert len(reasons) == gates.count(False)


def test_all_three_verified_gates_are_the_only_ready_path():
    assert readiness_gate._admission_status(True, True, True) == ("ready", [], 0)


def test_resource_budget_block_is_exit_four_and_keeps_support_unverified(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    report, code = readiness_gate._budget_blocked(
        {"manifest_sha256": "a" * 64}, {"version_id": "formal-test"}, receipt,
        {"feature_manifest_sha256": "b" * 64, "status": "blocked"},
        readiness_gate.GLOBAL_CELLS_CAP, "test_resource_limit",
    )
    assert code == 4
    assert report["status"] == "blocked"
    assert report["tool_gate"]["status"] == "blocked"
    assert report["support_gate"]["status"] == "unverified"
    assert report["unverified"][0] == "independent_audit_not_run_within_registered_budget"


def test_feature_manifest_wrong_bundle_binding_is_rejected_before_snapshot_io(tmp_path, monkeypatch):
    payload = {"entity_bundle_hash": "wrong", "entity_version": "v", "entities": []}
    raw = json.dumps(payload).encode("utf-8")
    path = tmp_path / "feature_manifest.json"
    path.write_bytes(raw)
    monkeypatch.setattr(readiness_gate, "_json", lambda _path: (payload, raw))
    monkeypatch.setattr(readiness_gate, "_sha", lambda _raw: "46d13b05db30c15e39cc4b455a5ab3ec1a0faf6265017b32301b08212da0c1c5")
    with pytest.raises(readiness_gate.ContractError, match="different corpus"):
        readiness_gate._feature_gate(path, {"entities": {"entity_version": "v"}}, {"content_sha256": "right"})


def _feature_fixture(tmp_path, monkeypatch):
    revision, artifact = "a" * 40, "b" * 64
    config_sha, tokenizer_sha, weights_sha = "c" * 64, "d" * 64, "e" * 64
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    conversion = model_dir / "conversion_receipt.json"
    conversion.write_text("{}", encoding="utf-8")
    proteins, rows = [], []
    for index in range(44):
        accession = f"P{index:05d}"
        sequence_sha = (str(index % 10) * 64)
        protein = {"accession": accession, "taxonomy_id": 9606, "sequence_length": 100,
                   "sequence_sha256": sequence_sha}
        proteins.append(protein)
        key = f"cache-{accession}"
        embedding = [0.25] * 320
        result = {"features": {"esm_embedding": embedding}, "sequence_sha256": sequence_sha,
                  "model_artifact_sha256": artifact, "cache_key": key, "status": "ready"}
        entry = {"result": result, "result_sha256": readiness_gate._canonical_hash(result)}
        cache_path = tmp_path / "snapshot" / "cache" / f"{key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(entry, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        rows.append({**protein, "entity_bundle_hash": "bundle", "model_revision": revision,
                     "model_artifact_sha256": artifact, "status": "ready", "finite": True,
                     "embedding_dim": 320, "features": {"esm_embedding": embedding},
                     "pooling_policy": "residue-mask-mean-v1:max_tokens=1024", "cache_key": key,
                     "cache_file_sha256": readiness_gate._file_sha(cache_path)})
    model = {"model_id": "facebook/esm2_t6_8M_UR50D", "revision": revision,
             "external_revision_verified": False, "model_artifact_sha256": artifact,
             "config_sha256": config_sha, "tokenizer_sha256": tokenizer_sha,
             "weights_sha256": weights_sha, "embedding_dim": 320,
             "source_conversion_receipt": str(conversion),
             "source_conversion_receipt_sha256": readiness_gate._file_sha(conversion)}
    feature = {"entity_bundle_hash": "bundle", "entity_version": "entities-v2-fixture",
               "model": model, "entities": rows, "status_counts": {"ready": 44}}
    path = tmp_path / "snapshot" / "feature_manifest.json"
    path.write_text(json.dumps(feature), encoding="utf-8")
    monkeypatch.setattr(readiness_gate, "EXPECTED_FEATURE_MANIFEST_SHA256", readiness_gate._file_sha(path))
    monkeypatch.setattr(readiness_gate, "EXPECTED_MODEL_REVISION", revision)
    monkeypatch.setattr(readiness_gate, "EXPECTED_MODEL_ARTIFACT_SHA256", artifact)
    monkeypatch.setattr(readiness_gate, "EXPECTED_FEATURE_STATUS_COUNTS", {"ready": 44})
    monkeypatch.setattr(readiness_gate, "EXPECTED_UNSUPPORTED_ACCESSIONS", set())
    monkeypatch.setattr(readiness_gate, "verify_derived_snapshot", lambda _path: {})
    monkeypatch.setattr(readiness_gate.ProteinFeatureExtractor, "_local_model_provenance", lambda *_a, **_k: {
        "revision": revision, "model_artifact_sha256": artifact, "config_sha256": config_sha,
        "tokenizer_sha256": tokenizer_sha, "weights_sha256": weights_sha,
    })
    monkeypatch.setattr(readiness_gate, "ProteinRecord", lambda **kwargs: kwargs)
    monkeypatch.setattr(readiness_gate, "build_feature_cache_key", lambda protein, **_kwargs: f"cache-{protein['accession']}")
    bundle = {"content_sha256": "bundle", "entity_version": "entities-v2-fixture", "proteins": proteins}
    return path, feature, bundle


def test_feature_gate_recomputes_model_provenance_and_each_cache_binding(tmp_path, monkeypatch):
    path, feature, bundle = _feature_fixture(tmp_path, monkeypatch)
    observed, ready = readiness_gate._feature_gate(path, {"entities": {"entity_version": "entities-v2-fixture"}}, bundle)
    assert ready is True
    assert observed["status_counts"] == {"ready": 44}
    cache = path.parent / "cache" / "cache-P00000.json"
    cache.write_text("tampered", encoding="utf-8")
    with pytest.raises(readiness_gate.ContractError, match="cache bytes hash mismatch"):
        readiness_gate._feature_gate(path, {"entities": {"entity_version": "entities-v2-fixture"}}, bundle)
    cache.unlink()
    row = feature["entities"][0]
    entry = {"result": {"features": row["features"], "sequence_sha256": row["sequence_sha256"],
                         "model_artifact_sha256": row["model_artifact_sha256"], "cache_key": row["cache_key"]}}
    cache.write_text(json.dumps(entry), encoding="utf-8")
    monkeypatch.setattr(readiness_gate.ProteinFeatureExtractor, "_local_model_provenance", lambda *_a, **_k: {
        "revision": "f" * 40, "model_artifact_sha256": "f" * 64, "config_sha256": "f" * 64,
        "tokenizer_sha256": "f" * 64, "weights_sha256": "f" * 64,
    })
    with pytest.raises(readiness_gate.ContractError, match="derived local model file inventory"):
        readiness_gate._feature_gate(path, {"entities": {"entity_version": "entities-v2-fixture"}}, bundle)


def test_frozen_policy_tamper_is_rejected_by_registered_hash(tmp_path):
    policy = {"minimum_labeled_examples_per_class": {"train": 1}}
    path = tmp_path / "support.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(readiness_gate.ContractError, match="registered frozen policy"):
        readiness_gate._verify_support_policy(path)


def test_synthetic_receipt_cannot_claim_formal_protocol(tmp_path):
    fake_config = tmp_path / "corpus_formal_v1.json"
    fake_config.write_text("{}", encoding="utf-8")
    fake_receipt = tmp_path / "formal_expansion_receipt.json"
    fake_receipt.write_text(json.dumps({"status": "test_only"}), encoding="utf-8")
    with pytest.raises(readiness_gate.ContractError, match="immutable 02-03 acceptance receipt"):
        readiness_gate._verify_protocol(fake_config, fake_receipt, tmp_path / "manifest.json", {})


def test_new_output_requires_absolute_unused_guid_and_rejects_relative(tmp_path):
    output = tmp_path / uuid.uuid4().hex
    assert readiness_gate.resolve_new_output(output) == output
    with pytest.raises(readiness_gate.ContractError, match="absolute"):
        readiness_gate.resolve_new_output(uuid.uuid4().hex)
    with pytest.raises(readiness_gate.ContractError, match="GUID"):
        readiness_gate.resolve_new_output(tmp_path / "plain")
    output.mkdir()
    with pytest.raises(readiness_gate.ContractError, match="overwrite"):
        readiness_gate.resolve_new_output(output)


def test_output_guard_rejects_protected_path(tmp_path, monkeypatch):
    protected_root = tmp_path / "protected"
    monkeypatch.setattr(readiness_gate, "ROOT", protected_root)
    protected_root.mkdir()
    with pytest.raises(readiness_gate.ContractError, match="protected"):
        readiness_gate.resolve_new_output(protected_root / "output_dashboard.html" / uuid.uuid4().hex)


def test_cli_exit_contract_ready_blocked_and_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness_gate, "evaluate", lambda *_args: ({"status": "ready"}, 0))
    ready = tmp_path / uuid.uuid4().hex
    assert readiness_gate.main(["--manifest", "m", "--features", "f", "--support-policy", "s",
                                "--formal-receipt", "r", "--out", str(ready)]) == 0
    assert json.loads((ready / "training_admission.json").read_text(encoding="utf-8"))["status"] == "ready"
    monkeypatch.setattr(readiness_gate, "evaluate", lambda *_args: ({"status": "blocked"}, 4))
    blocked = tmp_path / uuid.uuid4().hex
    assert readiness_gate.main(["--manifest", "m", "--features", "f", "--support-policy", "s",
                                "--formal-receipt", "r", "--out", str(blocked)]) == 4
    monkeypatch.setattr(readiness_gate, "evaluate", lambda *_args: (_ for _ in ()).throw(readiness_gate.ContractError("tampered")))
    rejected = tmp_path / uuid.uuid4().hex
    assert readiness_gate.main(["--manifest", "m", "--features", "f", "--support-policy", "s",
                                "--formal-receipt", "r", "--out", str(rejected)]) == 3
