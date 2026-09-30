"""Offline tests for learned-weight guards and pipeline readiness reporting."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import cnn_module
import evidence_builder as builder
import main as main_module
import output_module
import transformer_module
from data_contracts import FeatureResult


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _bundle(tmp_path: Path) -> tuple[Path, dict]:
    output = tmp_path / "synthetic-bundle"
    builder.ingest_local_manifest(FIXTURE_PATH, output)
    path = output / "entities.json"
    return path, builder.load_entity_bundle(path)


def _readiness() -> dict:
    return {
        "overall_status": "partial",
        "stages": {
            "protein": {"status": "ready", "reason": None},
            "cnn": {"status": "skipped_untrained", "reason": "verified_trained_weights_not_loaded"},
            "transformer": {"status": "skipped_untrained", "reason": "verified_trained_weights_not_loaded"},
        },
    }


class _ForwardSpy:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        import torch

        self.calls += 1
        return torch.ones((1, 64))


def test_cnn_does_not_run_when_only_weights_status_string_is_supplied(tmp_path):
    assert "weights_status" in inspect.signature(cnn_module.MolecularCNN).parameters
    pytest.importorskip("torch")
    spy = _ForwardSpy()
    wrapper = cnn_module.MolecularCNN(
        output_dir=str(tmp_path / "cnn"),
        weights_status="verified",
        model=spy,
    )

    result = wrapper.extract_features(np.ones((1, 4, 4, 4), dtype=np.float32))

    assert result is None
    assert spy.calls == 0
    assert wrapper.status.status == "skipped_untrained"


def test_transformer_has_no_random_or_concat_fallback_without_verified_weights(tmp_path):
    assert "weights_status" in inspect.signature(transformer_module.FeatureFusion).parameters
    pytest.importorskip("torch")
    spy = _ForwardSpy()
    wrapper = transformer_module.FeatureFusion(
        output_dir=str(tmp_path / "fusion"),
        weights_status="verified",
        model=spy,
    )

    result = wrapper.fuse_features(
        np.ones(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )

    assert result is None
    assert spy.calls == 0
    assert wrapper.status.status == "skipped_untrained"


class _ProteinExtractorSpy:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.legacy_calls = []
        self.bundle_paths = []

    def load_esm_model(self, *args, **kwargs):
        return False

    def extract_all_features(self, sequence, protein_name):
        self.legacy_calls.append((sequence, protein_name))
        return {"success": True, "features": {"esm_embedding": [999.0]}}

    def extract_from_bundle(self, bundle_path):
        self.bundle_paths.append(Path(bundle_path))
        protein = self.bundle["proteins"][0]
        return [FeatureResult(
            status="ready",
            features={"esm_embedding": [1.0, 2.0]},
            accession=protein["accession"],
            taxonomy_id=protein["taxonomy_id"],
            sequence_length=protein["sequence_length"],
            sequence_sha256=protein["sequence_sha256"],
            entity_version=self.bundle["entity_version"],
            entity_bundle_hash=self.bundle["content_sha256"],
            source_kind="synthetic_test",
            model_revision="mock-r1",
            pooling_policy="residue-mask-mean-v1",
            cache_key="offline-test-only",
        )]


def _framework_for_test(bundle: dict, tmp_path: Path):
    framework = main_module.NetworkPharmacologyFramework.__new__(main_module.NetworkPharmacologyFramework)
    framework.output_dir = str(tmp_path / "framework-output")
    framework.results = {}
    framework.ligand_extractor = None
    framework.protein_extractor = _ProteinExtractorSpy(bundle)
    framework.gnn_analyzer = None
    framework.cnn_module = None
    framework.transformer_module = None
    return framework


def test_main_feature_pipeline_consumes_entity_bundle_and_reports_partial(tmp_path):
    bundle_path, bundle = _bundle(tmp_path)
    framework = _framework_for_test(bundle, tmp_path)

    features = framework.run_feature_extraction({
        "entity_bundle_path": str(bundle_path),
        "interactions": [],
        "ppi_source_kind": "synthetic_test",
    })

    assert framework.protein_extractor.bundle_paths == [bundle_path]
    assert framework.protein_extractor.legacy_calls == []
    assert features["protein"]["results"][0]["source_kind"] == "synthetic_test"
    assert features["cnn"]["status"] == "skipped_untrained"
    assert features["feature_readiness"]["overall_status"] == "partial"


def test_feature_fusion_never_fills_missing_modalities_with_random_arrays(monkeypatch, tmp_path):
    framework = main_module.NetworkPharmacologyFramework.__new__(main_module.NetworkPharmacologyFramework)
    framework.results = {}
    framework.transformer_module = None
    random_calls = []
    monkeypatch.setattr(np.random, "randn", lambda *shape: random_calls.append(shape))

    result = framework.run_feature_fusion({"protein": {}, "cnn": {}, "gnn": {}, "ligand": {}})

    assert random_calls == []
    assert result["status"] == "skipped_untrained"
    assert result["fused_features"] is None


def test_report_includes_machine_readable_stage_readiness(tmp_path):
    generator = output_module.OutputGenerator(output_dir=str(tmp_path / "report"))
    readiness = _readiness()

    report = generator.generate_summary_report({"feature_readiness": readiness})

    assert report["feature_readiness"] == readiness
    assert report["data_quality"]["overall_status"] == "partial"
    assert report["data_quality"]["feature_stages"]["cnn"]["status"] == "skipped_untrained"


def test_framework_json_summary_uses_statuses_instead_of_truthy_feature_dicts(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    framework = main_module.NetworkPharmacologyFramework.__new__(main_module.NetworkPharmacologyFramework)
    framework.output_dir = str(output_dir)
    readiness = _readiness()
    framework.results = {
        "data_collection": {},
        "feature_extraction": {
            "protein": {"status": "ready", "features": {"esm_embedding": [1.0]}},
            "cnn": {"status": "skipped_untrained", "reason": "verified_trained_weights_not_loaded"},
            "feature_readiness": readiness,
        },
        "feature_fusion": {"status": "skipped_untrained", "fused_features": None},
    }

    saved_path = framework.save_results()
    saved = json.loads(Path(saved_path).read_text(encoding="utf-8"))

    assert saved["summary"]["feature_extraction"]["status"] == "partial"
    assert saved["summary"]["feature_extraction"]["stage_statuses"]["cnn"] == "skipped_untrained"
    assert saved["summary"]["feature_fusion"]["status"] == "skipped_untrained"


def test_new_local_entity_feature_pipeline_entry_point_is_available():
    assert callable(getattr(main_module, "run_feature_pipeline", None))
