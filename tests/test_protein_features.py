"""Offline contracts for identity-bound protein features."""

from __future__ import annotations

from dataclasses import replace
import hashlib
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
    )


def test_identity_bound_feature_contract_is_exposed():
    assert callable(getattr(protein_features, "validate_sequence_identity", None))
    assert callable(getattr(protein_features.ProteinFeatureExtractor, "extract_from_bundle", None))
    assert callable(getattr(protein_features.ProteinFeatureExtractor, "extract_for_protein", None))
    assert getattr(data_contracts, "StageStatus", None) is not None
    assert getattr(data_contracts, "FeatureResult", None) is not None


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

    results = extractor.extract_from_bundle(bundle_path, allow_synthetic=True)

    assert len(results) == 1
    result = results[0]
    assert result.status == "ready"
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
    allowed_for_mock_test = extractor.extract_from_bundle(bundle_path, allow_synthetic=True)

    assert blocked[0].status == "unverified_source"
    assert blocked[0].features == {}
    assert allowed_for_mock_test[0].status == "ready"
    assert allowed_for_mock_test[0].source_kind == "synthetic_test"


def test_nonstandard_residue_is_preserved_and_rejected_not_filtered(tmp_path):
    pytest.importorskip("torch")
    bundle_path, bundle = _bundle(tmp_path, sequence="ACDX")
    extractor = _extractor(tmp_path)

    result = extractor.extract_from_bundle(bundle_path, allow_synthetic=True)[0]

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

    result = extractor.extract_from_bundle(bundle_path, allow_synthetic=True)[0]

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

    base = cache_key(record, model_revision="mock-r1", pooling_policy="residue-mask-mean-v1")
    other_model = cache_key(record, model_revision="mock-r2", pooling_policy="residue-mask-mean-v1")
    other_policy = cache_key(record, model_revision="mock-r1", pooling_policy="residue-only-max-v1")
    other_identity = cache_key(
        replace(record, accession="Q54321", returned_accession="Q54321"),
        model_revision="mock-r1",
        pooling_policy="residue-mask-mean-v1",
    )

    assert len({base, other_model, other_policy, other_identity}) == 4


def test_cached_or_concurrent_extraction_does_not_leak_between_proteins(tmp_path):
    pytest.importorskip("torch")
    from concurrent.futures import ThreadPoolExecutor

    first_path, first_bundle = _bundle(tmp_path, "first", "ACD")
    second_path, second_bundle = _bundle(tmp_path, "second", "WYK")
    extractor = _extractor(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(extractor.extract_from_bundle, first_path, allow_synthetic=True)
        second_future = pool.submit(extractor.extract_from_bundle, second_path, allow_synthetic=True)
        first = first_future.result()[0]
        second = second_future.result()[0]

    assert first.sequence_sha256 == first_bundle["proteins"][0]["sequence_sha256"]
    assert second.sequence_sha256 == second_bundle["proteins"][0]["sequence_sha256"]
    assert first.features["esm_embedding"] != second.features["esm_embedding"]
    assert first.cache_key != second.cache_key
    assert extractor.sequence in {"ACD", "WYK"}
