"""Offline contracts for identity-bound protein features."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import data_contracts
import evidence_builder as builder
import protein_features


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _manifest(sequence: str | None = None) -> dict:
    manifest = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if sequence is not None:
        manifest["uniprot"]["P12345"]["sequence"]["value"] = sequence
        manifest["uniprot"]["P12345"]["sequence"]["length"] = len(sequence)
    return manifest


def _bundle(tmp_path: Path, name: str = "bundle", sequence: str | None = None) -> tuple[Path, dict]:
    source = tmp_path / f"{name}-manifest.json"
    source.write_text(json.dumps(_manifest(sequence), ensure_ascii=False), encoding="utf-8")
    output = tmp_path / name
    builder.ingest_local_manifest(source, output)
    return output / "entities.json", builder.load_entity_bundle(output / "entities.json")


class _MockTokenizer:
    model_max_length = 8

    def __call__(self, sequence: str, **kwargs):
        import torch

        residue_ids = [ord(residue) for residue in sequence]
        ids = [997, *residue_ids, 991, 989]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * (len(sequence) + 2) + [0]], dtype=torch.long),
            "special_tokens_mask": torch.tensor([[1] + [0] * len(sequence) + [1, 1]], dtype=torch.long),
        }


class _MockModel:
    config = SimpleNamespace(max_position_embeddings=8)

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        import torch

        self.calls.append(input_ids.detach().cpu().tolist())
        values = input_ids.to(dtype=torch.float32)
        hidden = torch.stack((values, values * 2), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def _extractor(tmp_path: Path, *, max_tokens: int = 8):
    return protein_features.ProteinFeatureExtractor(
        output_dir=str(tmp_path / "features"),
        esm_model_name="mock/esm",
        model_revision="mock-r1",
        max_tokens=max_tokens,
        model=_MockModel(),
        tokenizer=_MockTokenizer(),
        cache_dir=str(tmp_path / "model-cache"),
        test_only=True,
    )


def _synthetic_feature_result(bundle: dict) -> data_contracts.FeatureResult:
    protein = data_contracts.ProteinRecord(**bundle["proteins"][0])
    model_id = "mock/esm"
    model_revision = "mock-r1"
    artifact_hash = hashlib.sha256(b"synthetic test-only model artifact").hexdigest()
    pooling = "residue-mask-mean-v1:max_tokens=8"
    cache_key = protein_features.build_feature_cache_key(
        protein,
        model_id=model_id,
        model_revision=model_revision,
        model_artifact_sha256=artifact_hash,
        pooling_policy=pooling,
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
    )
    return data_contracts.FeatureResult(
        status="test_only",
        reason="synthetic fixture only",
        features={"esm_embedding": [1.0, 2.0]},
        accession=protein.accession,
        taxonomy_id=protein.taxonomy_id,
        sequence_length=protein.sequence_length,
        sequence_sha256=protein.sequence_sha256,
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
        source_kind="synthetic_test",
        model_id=model_id,
        model_revision=model_revision,
        model_artifact_sha256=artifact_hash,
        pooling_policy=pooling,
        cache_key=cache_key,
        test_provenance="synthetic entity bundle and deterministic mock embedding",
    )


def test_identity_bound_feature_contract_is_exposed():
    assert "test_only" in inspect.signature(protein_features.ProteinFeatureExtractor).parameters
    assert callable(getattr(protein_features, "validate_sequence_identity", None))
    assert callable(getattr(protein_features.ProteinFeatureExtractor, "extract_from_bundle", None))
    assert callable(getattr(protein_features.ProteinFeatureExtractor, "extract_for_protein", None))
    assert getattr(data_contracts, "StageStatus", None) is not None
    assert getattr(data_contracts, "FeatureResult", None) is not None


def test_feature_result_requires_verified_bundle_model_and_finite_embedding(tmp_path):
    params = inspect.signature(data_contracts.FeatureResult).parameters
    assert "model_artifact_sha256" in params
    assert "model_id" in params
    assert "test_provenance" in params

    _, bundle = _bundle(tmp_path, sequence="ACDE")
    record = data_contracts.ProteinRecord(**bundle["proteins"][0])
    artifact_hash = hashlib.sha256(b"offline test snapshot").hexdigest()
    pooling = "residue-mask-mean-v1:max_tokens=8"
    cache_key = protein_features.build_feature_cache_key(
        record,
        model_id="mock/esm",
        model_revision="mock-r1",
        model_artifact_sha256=artifact_hash,
        pooling_policy=pooling,
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
    )
    base = {
        "reason": None,
        "features": {"esm_embedding": [1.0, 2.0]},
        "accession": record.accession,
        "taxonomy_id": record.taxonomy_id,
        "sequence_length": record.sequence_length,
        "sequence_sha256": record.sequence_sha256,
        "entity_version": bundle["entity_version"],
        "entity_bundle_hash": bundle["content_sha256"],
        "model_id": "mock/esm",
        "model_revision": "mock-r1",
        "model_artifact_sha256": artifact_hash,
        "pooling_policy": pooling,
        "cache_key": cache_key,
    }

    with pytest.raises(ValueError, match="synthetic|online"):
        data_contracts.FeatureResult(
            status="ready", source_kind="synthetic_test", **base,
        )
    with pytest.raises(ValueError, match="embedding"):
        data_contracts.FeatureResult(
            status="ready",
            source_kind="online_snapshot",
            **{**base, "features": {"esm_embedding": []}},
        )
    with pytest.raises(ValueError, match="embedding"):
        data_contracts.FeatureResult(
            status="ready",
            source_kind="online_snapshot",
            **{**base, "features": {"esm_embedding": [[1.0, 2.0]]}},
        )
    with pytest.raises(ValueError, match="embedding"):
        data_contracts.FeatureResult(
            status="ready",
            source_kind="online_snapshot",
            **{**base, "features": {"esm_embedding": [float("nan")]}},
        )
    with pytest.raises(ValueError, match="sequence_sha256"):
        data_contracts.FeatureResult(
            status="ready",
            source_kind="online_snapshot",
            **{**base, "sequence_sha256": "not-a-hash"},
        )

    mock_result = data_contracts.FeatureResult(
        status="test_only",
        reason="synthetic fixture/model mock",
        features={"esm_embedding": [1.0, 2.0]},
        accession=record.accession,
        taxonomy_id=record.taxonomy_id,
        sequence_length=record.sequence_length,
        sequence_sha256=record.sequence_sha256,
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
        source_kind="synthetic_test",
        model_id="mock/esm",
        model_revision="mock-r1",
        model_artifact_sha256=artifact_hash,
        pooling_policy=pooling,
        cache_key=cache_key,
        test_provenance="synthetic fixture plus deterministic mock model",
    )
    assert mock_result.status == "test_only"
    assert mock_result.source_kind == "synthetic_test"


def test_injected_model_and_arbitrary_revision_are_not_runtime_provenance(tmp_path):
    pytest.importorskip("torch")
    extractor = protein_features.ProteinFeatureExtractor(
        output_dir=str(tmp_path / "features"),
        esm_model_name="facebook/esm2_t6_8M_UR50D",
        model_revision="arbitrary-user-revision",
        model=_MockModel(),
        tokenizer=_MockTokenizer(),
        cache_dir=str(tmp_path / "model-cache"),
    )

    assert extractor.load_esm_model() is False
    assert extractor.model_revision is None


def test_local_snapshot_provenance_hashes_config_tokenizer_and_weights(tmp_path):
    provenance_builder = getattr(
        protein_features.ProteinFeatureExtractor, "_local_model_provenance", None,
    )
    assert callable(provenance_builder)
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    revision = "a" * 40
    (model_dir / "config.json").write_text(
        json.dumps({"_commit_hash": revision, "model_type": "esm"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text('{"vocab":{"A":1}}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights-v1")

    first = provenance_builder(model_dir, model_name="facebook/esm2_t6_8M_UR50D")
    (model_dir / "model.safetensors").write_bytes(b"weights-v2")
    second = provenance_builder(model_dir, model_name="facebook/esm2_t6_8M_UR50D")

    assert first["revision"] == second["revision"] == revision
    assert first["config_sha256"] == second["config_sha256"]
    assert first["tokenizer_sha256"] == second["tokenizer_sha256"]
    assert first["weights_sha256"] != second["weights_sha256"]
    assert first["model_artifact_sha256"] != second["model_artifact_sha256"]


def test_partial_esm_checkpoint_loading_cannot_be_reported_as_pretrained():
    validator = getattr(
        protein_features.ProteinFeatureExtractor, "_assert_complete_esm_weights", None,
    )
    assert callable(validator)
    validator({
        "missing_keys": [], "mismatched_keys": [], "unexpected_keys": ["lm_head.decoder.weight"],
        "error_msgs": [],
    })
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        validator({
            "missing_keys": ["encoder.layer.0.weight"], "mismatched_keys": [],
            "unexpected_keys": [], "error_msgs": [],
        })


def test_feature_save_requires_explicit_unprotected_output_and_refuses_conflicts(tmp_path):
    default_extractor = protein_features.ProteinFeatureExtractor()
    assert default_extractor.output_dir is None
    with pytest.raises(ValueError, match="output_dir"):
        default_extractor.save_features()

    protected_root = Path(protein_features.__file__).resolve().parent / "features" / "protein"
    protected_extractor = protein_features.ProteinFeatureExtractor(output_dir=str(protected_root))
    with pytest.raises(ValueError, match="protected"):
        protected_extractor.save_features()

    safe_root = tmp_path / "safe-output"
    embeddings = safe_root / "embeddings"
    embeddings.mkdir(parents=True)
    destination = embeddings / "existing.json"
    destination.write_bytes(b"keep-existing-feature")
    _, bundle = _bundle(tmp_path)
    result = _synthetic_feature_result(bundle)
    safe_extractor = protein_features.ProteinFeatureExtractor(output_dir=str(safe_root))
    with pytest.raises(FileExistsError, match="existing|conflict"):
        safe_extractor.save_features("existing.json", result=result)
    assert destination.read_bytes() == b"keep-existing-feature"


def test_feature_save_requires_explicit_result_and_writes_full_provenance(tmp_path):
    _, bundle = _bundle(tmp_path)
    result = _synthetic_feature_result(bundle)
    extractor = protein_features.ProteinFeatureExtractor(output_dir=str(tmp_path / "safe-output"))
    extractor.features = {"esm_embedding": [999.0]}

    with pytest.raises(ValueError, match="explicit FeatureResult"):
        extractor.save_features("unbound.json")
    assert not (tmp_path / "safe-output" / "embeddings" / "unbound.json").exists()

    path = Path(extractor.save_features("provenance.json", result=result))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == result.to_dict()
    assert saved["status"] == "test_only"
    assert saved["test_provenance"] == result.test_provenance
    assert saved["sequence_sha256"] == result.sequence_sha256
    assert saved["entity_bundle_hash"] == result.entity_bundle_hash
    assert saved["model_artifact_sha256"] == result.model_artifact_sha256
    assert saved["pooling_policy"] == result.pooling_policy
    assert saved["cache_key"] == result.cache_key


def test_extractor_tracks_latest_result_across_extraction_cache_and_failure(tmp_path, monkeypatch):
    bundle_path, _ = _bundle(tmp_path)
    extractor = protein_features.ProteinFeatureExtractor(
        output_dir=str(tmp_path / "feature-output"),
        esm_model_name="mock/esm",
        model_revision="mock-r1",
        model=object(),
        tokenizer=object(),
        test_only=True,
    )
    monkeypatch.setattr(
        extractor,
        "_extract_residue_embedding",
        lambda sequence: (protein_features.np.array([1.0, 2.0]), "ready", None),
    )

    first = extractor.extract_from_bundle(str(bundle_path), test_only=True)[0]
    assert extractor.last_result == first
    assert first.status == "test_only"

    monkeypatch.setattr(
        extractor,
        "_extract_residue_embedding",
        lambda sequence: pytest.fail("cache hit must not rerun the model"),
    )
    cached = extractor.extract_from_bundle(str(bundle_path), test_only=True)[0]
    assert cached == first
    assert extractor.last_result == cached

    failed = extractor.extract_from_bundle(str(bundle_path), test_only=False)[0]
    assert failed.status == "unverified_source"
    assert extractor.last_result == failed


def test_unbound_sequence_api_cannot_emit_protein_features(tmp_path):
    extractor = _extractor(tmp_path)

    result = extractor.extract_all_features("ACDEFGHIKLMNPQRSTVWY", "sample_protein")

    assert result["status"] == "unverified_source"
    assert result["reason"] == "entity_bundle_required_for_protein_features"
    assert result["success"] is False
    assert result["features"] == {}
    assert extractor.esm_model.calls == []


def test_residue_pooling_excludes_cls_eos_and_padding_and_keeps_bundle_provenance(tmp_path):
    pytest.importorskip("torch")
    bundle_path, bundle = _bundle(tmp_path, sequence="ACD")
    extractor = _extractor(tmp_path)

    results = extractor.extract_from_bundle(bundle_path, test_only=True)

    assert len(results) == 1
    result = results[0]
    assert result.status == "test_only"
    assert result.sequence_sha256 == bundle["proteins"][0]["sequence_sha256"]
    assert result.entity_version == bundle["entity_version"]
    assert result.entity_bundle_hash == bundle["content_sha256"]
    assert result.source_kind == "synthetic_test"
    expected_mean = sum(ord(residue) for residue in "ACD") / 3
    assert result.features["esm_embedding"] == pytest.approx([expected_mean, expected_mean * 2])
    assert extractor.esm_model.calls == [[[997, *[ord(residue) for residue in "ACD"], 991, 989]]]


def test_real_extraction_rejects_synthetic_source_unless_test_only_opt_in(tmp_path):
    pytest.importorskip("torch")
    bundle_path, _ = _bundle(tmp_path, sequence="ACDE")
    extractor = _extractor(tmp_path)

    blocked = extractor.extract_from_bundle(bundle_path)
    allowed_for_mock_test = extractor.extract_from_bundle(bundle_path, test_only=True)

    assert blocked[0].status == "unverified_source"
    assert blocked[0].features == {}
    assert allowed_for_mock_test[0].status == "test_only"
    assert allowed_for_mock_test[0].source_kind == "synthetic_test"


def test_nonstandard_residue_is_preserved_and_rejected_not_filtered(tmp_path):
    pytest.importorskip("torch")
    bundle_path, bundle = _bundle(tmp_path, sequence="ACDX")
    extractor = _extractor(tmp_path)

    result = extractor.extract_from_bundle(bundle_path, test_only=True)[0]

    assert bundle["proteins"][0]["raw_sequence"] == "ACDX"
    assert bundle["proteins"][0]["full_sequence"] == "ACDX"
    assert result.status == "invalid_sequence"
    assert result.features == {}
    assert result.reason
    assert extractor.esm_model.calls == []


def test_long_sequence_returns_unsupported_without_truncating_bundle_record(tmp_path):
    pytest.importorskip("torch")
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 2
    bundle_path, bundle = _bundle(tmp_path, sequence=sequence)
    extractor = _extractor(tmp_path, max_tokens=8)

    result = extractor.extract_from_bundle(bundle_path, test_only=True)[0]

    assert result.status == "unsupported_length"
    assert result.sequence_length == len(sequence)
    assert result.sequence_sha256 == hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    assert result.features == {}
    assert extractor.esm_model.calls == []


def test_cache_key_separates_identity_sequence_model_revision_and_pooling_policy(tmp_path):
    _, bundle = _bundle(tmp_path, sequence="ACDE")
    record_type = data_contracts.ProteinRecord
    record = record_type(**bundle["proteins"][0])
    cache_key = getattr(protein_features, "build_feature_cache_key", None)
    assert callable(cache_key)

    bundle_args = {
        "entity_version": bundle["entity_version"],
        "entity_bundle_hash": bundle["content_sha256"],
        "model_id": "mock/esm",
        "model_artifact_sha256": "f" * 64,
    }
    base = cache_key(
        record, model_revision="mock-r1", pooling_policy="residue-mask-mean-v1", **bundle_args,
    )
    other_model = cache_key(
        record, model_revision="mock-r2", pooling_policy="residue-mask-mean-v1", **bundle_args,
    )
    other_policy = cache_key(
        record, model_revision="mock-r1", pooling_policy="residue-only-max-v1", **bundle_args,
    )
    other_identity = cache_key(
        replace(record, accession="Q54321", returned_accession="Q54321"),
        model_revision="mock-r1",
        pooling_policy="residue-mask-mean-v1",
        **bundle_args,
    )
    other_bundle = cache_key(
        record,
        model_revision="mock-r1",
        pooling_policy="residue-mask-mean-v1",
        model_id="mock/esm",
        model_artifact_sha256="f" * 64,
        entity_version="entities-v2-aaaaaaaaaaaaaaaa",
        entity_bundle_hash="a" * 64,
    )
    other_artifact = cache_key(
        record,
        model_revision="mock-r1",
        pooling_policy="residue-mask-mean-v1",
        model_id="mock/esm",
        model_artifact_sha256="e" * 64,
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
    )

    assert len({base, other_model, other_policy, other_identity, other_bundle, other_artifact}) == 6


def test_cached_or_concurrent_extraction_does_not_leak_between_proteins(tmp_path):
    pytest.importorskip("torch")
    from concurrent.futures import ThreadPoolExecutor

    first_path, first_bundle = _bundle(tmp_path, "first", "ACD")
    second_path, second_bundle = _bundle(tmp_path, "second", "WYK")
    extractor = _extractor(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(extractor.extract_from_bundle, first_path, test_only=True)
        second_future = pool.submit(extractor.extract_from_bundle, second_path, test_only=True)
        first = first_future.result()[0]
        second = second_future.result()[0]

    assert first.sequence_sha256 == first_bundle["proteins"][0]["sequence_sha256"]
    assert second.sequence_sha256 == second_bundle["proteins"][0]["sequence_sha256"]
    assert first.features["esm_embedding"] != second.features["esm_embedding"]
    assert first.cache_key != second.cache_key
    assert extractor.sequence in {"ACD", "WYK"}
