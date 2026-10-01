"""Offline tests for learned-weight guards and pipeline readiness reporting."""

from __future__ import annotations

import builtins
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

import cnn_module
import data_contracts
import evidence_builder as builder
import main as main_module
import output_module
import protein_features
import transformer_module
from data_contracts import FeatureResult, StageStatus


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _bundle(tmp_path: Path) -> tuple[Path, dict]:
    output = tmp_path / "synthetic-bundle"
    builder.ingest_local_manifest(FIXTURE_PATH, output)
    path = output / "entities.json"
    return path, builder.load_entity_bundle(path)


def _issue_test_feature_result(output_dir: Path, bundle_path: Path, monkeypatch):
    extractor = protein_features.ProteinFeatureExtractor(
        output_dir=str(output_dir),
        esm_model_name="mock/esm",
        model_revision="mock-r1",
        model=object(),
        tokenizer=object(),
        test_only=True,
    )
    monkeypatch.setattr(
        extractor,
        "_extract_residue_embedding",
        lambda sequence: (np.array([1.0, 2.0], dtype=np.float32), "ready", None),
    )
    result = extractor.extract_from_bundle(str(bundle_path), test_only=True)[0]
    return extractor, result


def _fake_full_pipeline_framework(output_root: Path):
    framework = main_module.NetworkPharmacologyFramework(
        output_dir=str(output_root), allow_online=True,
    )
    calls = {"collection": 0, "reports": 0}
    dataset = {
        "compound": {"name": "synthetic"},
        "targets": [],
        "proteins": [],
        "interactions": [],
        "collection_status": "complete",
        "status": "complete",
    }

    class FakeCollector:
        def run_pipeline(self):
            calls["collection"] += 1
            return dataset

    framework.data_collector = FakeCollector()

    def run_feature_extraction(current_dataset):
        result = {
            "feature_readiness": {"overall_status": "ready", "stages": {}},
            "gnn": {"centrality": {"synthetic": 1.0}},
        }
        framework.results["feature_extraction"] = result
        return result

    def run_feature_fusion(features):
        result = {"status": "ready"}
        framework.results["feature_fusion"] = result
        return result

    def run_docking(current_dataset):
        result = {"binding_affinity": None}
        framework.results["docking"] = result
        return result

    def run_knowledge_graph(current_dataset):
        result = {"graph": {"stats": {"num_nodes": 0, "num_edges": 0}}}
        framework.results["knowledge_graph"] = result
        return result

    def run_output_generation():
        calls["reports"] += 1
        report_path = output_root / "output" / "reports" / "summary.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"report_number": calls["reports"]}), encoding="utf-8")
        framework.results["output"] = {"report_path": str(report_path)}

    framework.run_feature_extraction = run_feature_extraction
    framework.run_feature_fusion = run_feature_fusion
    framework.run_docking = run_docking
    framework.run_knowledge_graph = run_knowledge_graph
    framework.run_output_generation = run_output_generation
    return framework, calls


def _readiness() -> dict:
    return {
        "overall_status": "partial",
        "stages": {
            "protein": {"status": "ready", "reason": None},
            "cnn": {"status": "skipped_untrained", "reason": "verified_trained_weights_not_loaded"},
            "transformer": {"status": "skipped_untrained", "reason": "verified_trained_weights_not_loaded"},
        },
    }


def _install_write_guard(monkeypatch):
    apl_root = Path(main_module.__file__).resolve().parent
    forbidden = [
        apl_root / "data",
        apl_root / "framework_output",
        apl_root / "offline_data.json",
        apl_root / ".model_cache",
        apl_root / "features",
        apl_root / "output_dashboard.html",
    ]

    def is_forbidden(path):
        try:
            candidate = os.path.realpath(os.path.abspath(os.fspath(path)))
        except (TypeError, ValueError):
            return False
        for root in forbidden:
            protected = os.path.realpath(os.path.abspath(os.fspath(root)))
            try:
                common = os.path.commonpath((candidate, protected))
            except ValueError:
                continue
            if os.path.normcase(common) in {
                os.path.normcase(candidate),
                os.path.normcase(protected),
            }:
                return True
        return False

    original_open = builtins.open
    original_makedirs = os.makedirs
    original_mkdir = os.mkdir
    original_path_open = Path.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")) and is_forbidden(file):
            raise AssertionError(f"write escaped into protected path: {file}")
        return original_open(file, mode, *args, **kwargs)

    def guarded_makedirs(name, *args, **kwargs):
        if is_forbidden(name):
            raise AssertionError(f"directory creation escaped into protected path: {name}")
        return original_makedirs(name, *args, **kwargs)

    def guarded_mkdir(name, *args, **kwargs):
        if is_forbidden(name):
            raise AssertionError(f"directory creation escaped into protected path: {name}")
        return original_mkdir(name, *args, **kwargs)

    def guarded_path_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")) and is_forbidden(path):
            raise AssertionError(f"path write escaped into protected path: {path}")
        return original_path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "makedirs", guarded_makedirs)
    monkeypatch.setattr(os, "mkdir", guarded_mkdir)
    monkeypatch.setattr(Path, "open", guarded_path_open)


class _ForwardSpy:
    def __init__(self):
        self.calls = 0

    def parameters(self):
        import torch

        return iter([torch.nn.Parameter(torch.zeros(1))])

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

    wrapper._verified_weights = True
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
        protein = data_contracts.ProteinRecord(**self.bundle["proteins"][0])
        artifact_hash = "d" * 64
        pooling = "residue-mask-mean-v1:max_tokens=8"
        cache_key = protein_features.build_feature_cache_key(
            protein,
            model_id="mock/esm",
            model_revision="mock-r1",
            model_artifact_sha256=artifact_hash,
            pooling_policy=pooling,
            entity_version=self.bundle["entity_version"],
            entity_bundle_hash=self.bundle["content_sha256"],
        )
        return [FeatureResult(
            status="test_only",
            reason="synthetic_test_result_not_for_runtime",
            features={"esm_embedding": [1.0, 2.0]},
            accession=protein.accession,
            taxonomy_id=protein.taxonomy_id,
            sequence_length=protein.sequence_length,
            sequence_sha256=protein.sequence_sha256,
            entity_version=self.bundle["entity_version"],
            entity_bundle_hash=self.bundle["content_sha256"],
            source_kind="synthetic_test",
            model_id="mock/esm",
            model_revision="mock-r1",
            model_artifact_sha256=artifact_hash,
            pooling_policy=pooling,
            cache_key=cache_key,
            test_provenance="mock tokenizer/model contract only",
        )]

def _framework_for_test(bundle: dict, tmp_path: Path):
    framework = main_module.NetworkPharmacologyFramework.__new__(main_module.NetworkPharmacologyFramework)
    framework.output_dir = str(tmp_path / "framework-output")
    framework.results = {}
    framework.ligand_extractor = None
    framework.protein_extractor = _ProteinExtractorSpy(bundle)
    framework.gnn_analyzer = None
    framework.cnn_module = type("UntrainedCNN", (), {
        "status": StageStatus("skipped_untrained", "verified_trained_weights_not_loaded"),
    })()
    framework.transformer_module = type("UntrainedFusion", (), {
        "status": StageStatus("skipped_untrained", "verified_trained_weights_not_loaded"),
        "_has_verified_weights": lambda self: False,
    })()
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
    assert features["protein"]["results"][0]["status"] == "unverified_source"
    assert features["cnn"]["status"] == "skipped_untrained"
    assert features["feature_readiness"]["overall_status"] == "unavailable"


def test_feature_fusion_never_fills_missing_modalities_with_random_arrays(monkeypatch, tmp_path):
    framework = main_module.NetworkPharmacologyFramework.__new__(main_module.NetworkPharmacologyFramework)
    framework.results = {}
    framework.transformer_module = type("UntrainedFusion", (), {
        "_has_verified_weights": lambda self: False,
    })()
    random_calls = []
    monkeypatch.setattr(np.random, "randn", lambda *shape: random_calls.append(shape))

    result = framework.run_feature_fusion({"protein": {}, "cnn": {}, "gnn": {}, "ligand": {}})

    assert random_calls == []
    assert result["status"] == "skipped_untrained"
    assert result["fused_features"] is None


def test_report_includes_machine_readable_stage_readiness(tmp_path, monkeypatch):
    _install_write_guard(monkeypatch)
    generator = output_module.OutputGenerator(output_dir=str(tmp_path / "report"))
    readiness = _readiness()

    report = generator.generate_summary_report({"feature_readiness": readiness})
    dashboard_path = Path(generator.generate_dashboard_html(report))
    dashboard = dashboard_path.read_text(encoding="utf-8")

    assert report["feature_readiness"] == readiness
    assert report["data_quality"]["overall_status"] == "partial"
    assert report["data_quality"]["feature_stages"]["cnn"]["status"] == "skipped_untrained"
    assert "特征与模型状态" in dashboard
    assert "skipped_untrained" in dashboard
    assert "framework_output/" not in dashboard

def test_framework_json_summary_uses_statuses_instead_of_truthy_feature_dicts(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    _install_write_guard(monkeypatch)
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


def test_dashboard_html_uses_requested_output_and_preserves_legacy_path(tmp_path, monkeypatch):
    _install_write_guard(monkeypatch)
    legacy_base = tmp_path / "legacy"
    legacy_base.mkdir()
    legacy_dashboard = legacy_base / "output_dashboard.html"
    legacy_dashboard.write_bytes(b"preexisting dashboard bytes")
    monkeypatch.setattr(output_module, "_BASE_DIR", str(legacy_base))
    target = tmp_path / "requested-output"
    generator = output_module.OutputGenerator(output_dir=str(target))
    report = generator.generate_summary_report({"feature_readiness": _readiness()})

    dashboard_path = Path(generator.generate_dashboard_html(report))

    assert dashboard_path == target / "dashboard.html"
    assert dashboard_path.is_file()
    dashboard = dashboard_path.read_text(encoding="utf-8")
    assert legacy_dashboard.read_bytes() == b"preexisting dashboard bytes"
    assert "framework_output/" not in dashboard
    assert "本次运行未生成可用数据文件" in dashboard


def test_dashboard_links_only_existing_files_inside_this_run_root(tmp_path, monkeypatch):
    _install_write_guard(monkeypatch)
    run_root = tmp_path / "current-run"
    run_root.mkdir()
    (run_root / "framework_results.json").write_text("{}", encoding="utf-8")
    generator = output_module.OutputGenerator(
        output_dir=str(run_root / "output"),
        run_root=str(run_root),
    )
    report = generator.generate_summary_report({
        "feature_readiness": _readiness(),
        "data_files": {"完整结果 (JSON)": "framework_results.json"},
    })

    page = Path(generator.generate_dashboard_html(report)).read_text(encoding="utf-8")

    assert 'href="../framework_results.json"' in page
    assert "framework_output/" not in page
    assert "本次运行未生成可用数据文件" not in page
def test_feature_readiness_reducer_reports_partial_for_a_mixed_real_stage_contract():
    readiness = main_module.NetworkPharmacologyFramework._feature_readiness({
        "protein": StageStatus("ready"),
        "cnn": StageStatus("skipped_untrained", "verified_trained_weights_not_loaded"),
        "transformer": StageStatus("skipped_untrained", "verified_trained_weights_not_loaded"),
    })
    assert readiness["overall_status"] == "partial"


@pytest.mark.parametrize(
    ("ppi_status", "expected_status"),
    [("partial", "partial"), ("invalid", "unavailable"), ("dependency_unavailable", "unavailable")],
)
def test_network_statistics_status_preserves_ppi_failure_state(tmp_path, ppi_status, expected_status):
    class PpiResult:
        status = ppi_status
        node_count = 1
        edge_count = 0
        edges = []

        def to_dict(self):
            return {"status": self.status, "edges": []}

    framework = main_module.NetworkPharmacologyFramework.__new__(main_module.NetworkPharmacologyFramework)
    framework.results = {}
    framework.ligand_extractor = None
    framework.protein_extractor = None
    framework.gnn_analyzer = None
    framework.cnn_module = type("UntrainedCNN", (), {
        "status": StageStatus("skipped_untrained", "weights_missing"),
    })()
    framework.transformer_module = type("UntrainedFusion", (), {
        "status": StageStatus("skipped_untrained", "weights_missing"),
        "_has_verified_weights": lambda self: False,
    })()
    framework._canonicalize_dataset_ppi = lambda dataset: PpiResult()

    features = framework.run_feature_extraction({"interactions": []})

    assert features["gnn"]["model_type"] == "network_statistics_only"
    assert features["feature_readiness"]["stages"]["network_statistics"]["status"] == expected_status


def test_default_framework_is_read_only_and_returns_partial_without_online_opt_in(monkeypatch):
    calls = []
    from data_collection import LuteolinDataCollector
    monkeypatch.setattr("data_collection.requests.get", lambda *args, **kwargs: calls.append(args[0]))
    monkeypatch.setattr("data_collection.requests.post", lambda *args, **kwargs: calls.append(args[0]))

    framework = main_module.NetworkPharmacologyFramework()

    assert framework.output_dir is None
    dataset = framework.run_data_collection()
    assert dataset["status"] == "partial"
    assert dataset["collection_status"] == "not_requested"
    pipeline = framework.run_full_pipeline()
    assert pipeline["data_collection"]["status"] == "partial"
    assert pipeline["pipeline_status"] == "partial"
    assert calls == []


def test_framework_output_root_must_be_new_and_outside_protected_paths(tmp_path):
    import inspect

    constructor = main_module.NetworkPharmacologyFramework
    assert "allow_online" in inspect.signature(constructor).parameters
    with pytest.raises(ValueError, match="output|protected"):
        constructor(allow_online=True)

    apl_root = Path(main_module.__file__).resolve().parent
    with pytest.raises(ValueError, match="protected"):
        constructor(output_dir=str(apl_root / "data"))

    existing = tmp_path / "existing-run"
    existing.mkdir()
    sentinel = existing / "keep.json"
    sentinel.write_bytes(b"keep-existing-output")
    with pytest.raises(FileExistsError, match="existing|new"):
        constructor(output_dir=str(existing))
    assert sentinel.read_bytes() == b"keep-existing-output"


def test_run_output_directory_resolution_rejects_a_junction_into_protected_data(tmp_path, monkeypatch):
    resolver = getattr(main_module, "_resolve_new_output_dir", None)
    assert callable(resolver)
    protected = Path(main_module.__file__).resolve().parent / "framework_output"
    alias = tmp_path / "junction"
    candidate = alias / "new-run"
    original_realpath = os.path.realpath

    def resolve_junction(path, *args, **kwargs):
        candidate_path = os.path.abspath(os.fspath(path))
        alias_path = os.path.abspath(os.fspath(alias))
        if os.path.normcase(candidate_path).startswith(os.path.normcase(alias_path)):
            suffix = os.path.relpath(candidate_path, alias_path)
            return os.path.join(os.fspath(protected), suffix)
        return original_realpath(path, *args, **kwargs)

    monkeypatch.setattr(main_module.os.path, "realpath", resolve_junction)

    with pytest.raises(ValueError, match="protected|intersect"):
        resolver(candidate)
    assert not candidate.exists()


def test_optional_torch_absence_keeps_model_modules_and_main_importable(tmp_path):
    import subprocess
    import sys

    source = r'''
import builtins
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("torch intentionally hidden for the dependency test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
import cnn_module
import transformer_module
import main
assert cnn_module.TORCH_AVAILABLE is False
assert transformer_module.TORCH_AVAILABLE is False
assert callable(main.run_feature_pipeline)
cnn = cnn_module.MolecularCNN()
fusion = transformer_module.FeatureFusion()
assert cnn.status.status == "unavailable"
assert fusion.status.status == "unavailable"
assert cnn.extract_features(None) is None
assert fusion.fuse_features(None, None, None) is None
'''
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", source],
        cwd=str(Path(main_module.__file__).resolve().parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_escapes_html_text_and_script_json(tmp_path, monkeypatch):
    _install_write_guard(monkeypatch)
    payload = "</script><script>alert(7)</script>"
    generator = output_module.OutputGenerator(output_dir=str(tmp_path / "report"))
    monkeypatch.setattr(output_module, "rank_targets", lambda *args, **kwargs: [
        (payload, 0.5, {
            "gene_symbol": payload, "organism": payload,
            "activity_type": payload, "activity_value": payload, "activity_unit": payload,
        })
    ])
    monkeypatch.setattr(output_module, "organism_distribution", lambda *args, **kwargs: [(payload, 1)])
    monkeypatch.setattr(generator, "_build_molecule_viewer_html", lambda: "")
    report = generator.generate_summary_report({
        "compound": {
            "name": payload, "name_cn": payload, "molecular_formula": payload,
            "molecular_weight": payload, "pubchem_cid": payload, "chembl_id": payload,
            "inchikey": payload, "smiles": payload,
        },
        "targets": [{
            "gene_symbol": payload, "organism": payload,
            "activity_type": payload, "activity_value": payload, "activity_unit": payload,
        }],
        "knowledge_graph": {
            "graph": {
                "nodes": [payload],
                "edges": [[payload, payload]],
                "stats": {"num_nodes": 1, "num_edges": 1, "num_relations": 1},
            },
        },
        "feature_readiness": _readiness(),
    })

    page = Path(generator.generate_dashboard_html(report)).read_text(encoding="utf-8")

    assert payload not in page
    assert "<script>alert(7)</script>" not in page
    assert r"\u003c/script\u003e" in page


def test_save_methods_create_configured_parents_and_reject_path_traversal(tmp_path, monkeypatch):
    _install_write_guard(monkeypatch)
    protein_root = tmp_path / "protein-output"
    bundle_path, _ = _bundle(tmp_path)
    protein_extractor, protein_result = _issue_test_feature_result(protein_root, bundle_path, monkeypatch)
    protein_path = Path(protein_extractor.save_features(result=protein_result))
    assert protein_path.parent == protein_root / "embeddings"
    assert protein_path.is_file()
    with pytest.raises(ValueError, match="filename|path"):
        protein_extractor.save_features("../escape.json")

    cnn_root = tmp_path / "cnn-output"
    cnn = cnn_module.MolecularCNN(output_dir=str(cnn_root))
    cnn.save_features(np.zeros((1, 2), dtype=np.float32))
    assert (cnn_root / "cnn_features.npy").is_file()
    with pytest.raises(ValueError, match="filename|path"):
        cnn.save_features(np.zeros((1, 2), dtype=np.float32), "../escape.npy")

    fusion_root = tmp_path / "fusion-output"
    fusion = transformer_module.FeatureFusion(output_dir=str(fusion_root))
    fusion.save_fused_features(np.zeros((1, 2), dtype=np.float32))
    assert (fusion_root / "fused_features.npy").is_file()
    with pytest.raises(ValueError, match="filename|path"):
        fusion.save_fused_features(np.zeros((1, 2), dtype=np.float32), "../escape.npy")

    report_root = tmp_path / "report-output"
    generator = output_module.OutputGenerator(output_dir=str(report_root))
    report_path = Path(generator.save_report({"status": "partial"}, "result.json"))
    assert report_path == report_root / "reports" / "result.json"
    assert report_path.is_file()
    with pytest.raises(ValueError, match="filename|path"):
        generator.save_report({}, "../escape.json")
    assert not (report_root / "escape.json").exists()


def test_feature_result_from_synthetic_source_never_counts_as_ready(tmp_path):
    bundle_path, bundle = _bundle(tmp_path)
    protein = data_contracts.ProteinRecord(**bundle["proteins"][0])
    artifact_hash = "d" * 64
    pooling = "residue-mask-mean-v1:max_tokens=8"
    cache_key = protein_features.build_feature_cache_key(
        protein,
        model_id="mock/esm",
        model_revision="mock-r1",
        model_artifact_sha256=artifact_hash,
        pooling_policy=pooling,
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
    )
    synthetic_result = FeatureResult(
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
        model_id="mock/esm",
        model_revision="mock-r1",
        model_artifact_sha256=artifact_hash,
        pooling_policy=pooling,
        cache_key=cache_key,
        test_provenance="offline fixture and deterministic mock",
    )
    framework = main_module.NetworkPharmacologyFramework.__new__(main_module.NetworkPharmacologyFramework)
    framework.results = {}
    framework.ligand_extractor = None
    framework.protein_extractor = type(
        "TestOnlyExtractor",
        (),
        {"extract_from_bundle": lambda self, path: [synthetic_result]},
    )()
    framework.gnn_analyzer = None
    framework.cnn_module = type("UntrainedCNN", (), {
        "status": StageStatus("skipped_untrained", "weights_missing"),
    })()
    framework.transformer_module = type("UntrainedFusion", (), {
        "status": StageStatus("skipped_untrained", "weights_missing"),
        "_has_verified_weights": lambda self: False,
    })()

    result = framework.run_feature_extraction({
        "entity_bundle_path": str(bundle_path),
        "interactions": [],
        "ppi_source_kind": "synthetic_test",
    })

    assert result["protein"]["results"][0]["status"] == "unverified_source"
    assert result["protein"]["results"][0]["features"] == {}
    assert result["feature_readiness"]["overall_status"] == "unavailable"


def test_main_reports_partial_only_when_a_verified_stage_is_ready():
    stages = {
        "protein": StageStatus("ready"),
        "cnn": StageStatus("skipped_untrained", "weights_missing"),
        "transformer": StageStatus("skipped_untrained", "weights_missing"),
    }
    assert main_module.NetworkPharmacologyFramework._feature_readiness(stages)["overall_status"] == "partial"


class _PpiStatusFixture:
    def __init__(self, status):
        self.status = status
        self.node_count = 1
        self.edge_count = 0
        self.edges = []

    def to_dict(self):
        return {"status": self.status, "edges": []}


@pytest.mark.parametrize(
    ("ppi_status", "expected_status"),
    [
        ("partial", "partial"),
        ("invalid", "unavailable"),
        ("dependency_unavailable", "unavailable"),
    ],
)
def test_network_statistics_stage_does_not_erase_ppi_failure(ppi_status, expected_status):
    framework = main_module.NetworkPharmacologyFramework.__new__(main_module.NetworkPharmacologyFramework)
    framework.results = {}
    framework.ligand_extractor = None
    framework.protein_extractor = None
    framework.gnn_analyzer = None
    framework.cnn_module = type("UntrainedCNN", (), {
        "status": StageStatus("skipped_untrained", "weights_missing"),
    })()
    framework.transformer_module = type("UntrainedFusion", (), {
        "status": StageStatus("skipped_untrained", "weights_missing"),
        "_has_verified_weights": lambda self: False,
    })()
    ppi_result = _PpiStatusFixture(ppi_status)
    framework._canonicalize_dataset_ppi = lambda dataset: ppi_result

    result = framework.run_feature_extraction({"interactions": []})

    assert result["feature_readiness"]["stages"]["network_statistics"]["status"] == expected_status


def test_dashboard_html_escapes_untrusted_html_and_script_data(tmp_path, monkeypatch):
    _install_write_guard(monkeypatch)
    payload = "</script><script>alert(7)</script>"
    generator = output_module.OutputGenerator(output_dir=str(tmp_path / "report"))
    monkeypatch.setattr(output_module, "rank_targets", lambda *args, **kwargs: [
        (payload, 0.5, {
            "gene_symbol": payload, "organism": payload,
            "activity_type": payload, "activity_value": payload, "activity_unit": payload,
        })
    ])
    monkeypatch.setattr(output_module, "organism_distribution", lambda *args, **kwargs: [(payload, 1)])
    monkeypatch.setattr(generator, "_build_molecule_viewer_html", lambda: "")
    report = generator.generate_summary_report({
        "compound": {
            "name": payload, "name_cn": payload, "molecular_formula": payload,
            "molecular_weight": payload, "pubchem_cid": payload, "chembl_id": payload,
            "inchikey": payload, "smiles": payload,
        },
        "targets": [{
            "gene_symbol": payload, "organism": payload,
            "activity_type": payload, "activity_value": payload, "activity_unit": payload,
        }],
        "knowledge_graph": {
            "graph": {
                "nodes": [payload],
                "edges": [[payload, payload]],
                "stats": {"num_nodes": 1, "num_edges": 1, "num_relations": 1},
            },
        },
        "feature_readiness": _readiness(),
    })

    dashboard_path = Path(generator.generate_dashboard_html(report))
    dashboard = dashboard_path.read_text(encoding="utf-8")

    assert payload not in dashboard
    assert "<script>alert(7)</script>" not in dashboard
    assert r"\u003c/script\u003e" in dashboard
    assert "framework_output/" not in dashboard


def test_cnn_and_fusion_savers_do_not_default_to_protected_feature_roots():
    cnn = cnn_module.MolecularCNN()
    fusion = transformer_module.FeatureFusion()
    assert cnn.output_dir is None
    assert fusion.output_dir is None
    with pytest.raises(ValueError, match="output_dir"):
        cnn.save_features(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="output_dir"):
        fusion.save_fused_features(np.zeros((1, 2), dtype=np.float32))


def test_configured_save_methods_create_parents_and_reject_path_traversal(tmp_path, monkeypatch):
    _install_write_guard(monkeypatch)
    protein_root = tmp_path / "protein"
    bundle_path, _ = _bundle(tmp_path)
    extractor, protein_result = _issue_test_feature_result(protein_root, bundle_path, monkeypatch)
    protein_path = Path(extractor.save_features(result=protein_result))
    assert protein_path.is_file()
    assert protein_path.parent == protein_root / "embeddings"
    with pytest.raises(ValueError, match="filename|path"):
        extractor.save_features("../escape.json")

    cnn_root = tmp_path / "cnn"
    cnn = cnn_module.MolecularCNN(output_dir=str(cnn_root))
    cnn.save_features(np.zeros((1, 2), dtype=np.float32))
    assert (cnn_root / "cnn_features.npy").is_file()
    with pytest.raises(ValueError, match="filename|path"):
        cnn.save_features(np.zeros((1, 2), dtype=np.float32), "../escape.npy")

    fusion_root = tmp_path / "fusion"
    fusion = transformer_module.FeatureFusion(output_dir=str(fusion_root))
    fusion.save_fused_features(np.zeros((1, 2), dtype=np.float32))
    assert (fusion_root / "fused_features.npy").is_file()
    with pytest.raises(ValueError, match="filename|path"):
        fusion.save_fused_features(np.zeros((1, 2), dtype=np.float32), "../escape.npy")

    report_root = tmp_path / "report"
    generator = output_module.OutputGenerator(output_dir=str(report_root))
    report_path = Path(generator.save_report({"status": "partial"}, "result.json"))
    assert report_path == report_root / "reports" / "result.json"
    assert report_path.is_file()
    with pytest.raises(ValueError, match="filename|path"):
        generator.save_report({}, "../escape.json")
    assert not (report_root / "escape.json").exists()


def test_run_output_directory_resolution_rejects_symlinked_protected_parent(tmp_path, monkeypatch):
    resolver = getattr(main_module, "_resolve_new_output_dir", None)
    assert callable(resolver)
    protected = Path(main_module.__file__).resolve().parent / "framework_output"
    alias = tmp_path / "junction"
    candidate = alias / "new-run"
    original_realpath = os.path.realpath

    def resolve_junction(path, *args, **kwargs):
        candidate_path = os.path.abspath(os.fspath(path))
        alias_path = os.path.abspath(os.fspath(alias))
        try:
            common = os.path.commonpath((candidate_path, alias_path))
        except ValueError:
            common = None
        if common and os.path.normcase(common) == os.path.normcase(alias_path):
            suffix = os.path.relpath(candidate_path, alias_path)
            return os.path.join(os.fspath(protected), suffix)
        return original_realpath(path, *args, **kwargs)

    monkeypatch.setattr(main_module.os.path, "realpath", resolve_junction)
    with pytest.raises(ValueError, match="protected|intersect"):
        resolver(candidate)
    assert not candidate.exists()


def test_local_entity_feature_pipeline_creates_status_report_in_new_output_root(tmp_path, monkeypatch):
    assert callable(getattr(main_module, "run_feature_pipeline", None))
    bundle_path, _ = _bundle(tmp_path)
    output_dir = tmp_path / "p0-output"
    _install_write_guard(monkeypatch)

    results = main_module.main([
        "--entity-bundle", str(bundle_path),
        "--output-dir", str(output_dir),
    ])

    saved_path = output_dir / "framework_results.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    readiness = results["feature_extraction"]["feature_readiness"]
    assert readiness["stages"]["protein"]["status"] == "unverified_source"
    assert readiness["stages"]["cnn"]["status"] == "skipped_untrained"
    assert readiness["stages"]["transformer"]["status"] == "skipped_untrained"
    assert saved["summary"]["feature_extraction"]["status"] == "unavailable"
    dashboard_path = output_dir / "output" / "dashboard.html"
    dashboard = dashboard_path.read_text(encoding="utf-8")
    assert dashboard_path.is_file()
    assert "特征与模型状态" in dashboard
    assert "skipped_untrained" in dashboard
    assert results["output"]["feature_readiness"] == readiness
    assert results["output"]["gnn_summary"]["model_type"] == "network_statistics_only"
    assert output_dir.is_dir()


def test_full_pipeline_is_single_use_and_new_root_runs_independently(tmp_path):
    output_root = tmp_path / "first-run"
    framework, calls = _fake_full_pipeline_framework(output_root)

    result = framework.run_full_pipeline()
    assert result["pipeline_status"] == "complete"
    results_path = output_root / "framework_results.json"
    report_path = output_root / "output" / "reports" / "summary.json"
    results_before = results_path.read_bytes()
    report_before = report_path.read_bytes()

    with pytest.raises(FileExistsError, match="single|fresh|new"):
        framework.run_full_pipeline()

    assert calls == {"collection": 1, "reports": 1}
    assert results_path.read_bytes() == results_before
    assert report_path.read_bytes() == report_before
    assert json.loads(results_before)["results"]["pipeline_status"] == "complete"

    second_root = tmp_path / "second-run"
    second_framework, second_calls = _fake_full_pipeline_framework(second_root)
    second_result = second_framework.run_full_pipeline()
    assert second_result["pipeline_status"] == "complete"
    assert second_calls == {"collection": 1, "reports": 1}
    second_saved = json.loads((second_root / "framework_results.json").read_text(encoding="utf-8"))
    assert second_saved["results"]["pipeline_status"] == "complete"


def test_full_pipeline_raises_when_result_persistence_fails(tmp_path):
    output_root = tmp_path / "save-fails"
    framework, calls = _fake_full_pipeline_framework(output_root)
    framework.save_results = lambda: ""

    with pytest.raises(RuntimeError, match="persist|save"):
        framework.run_full_pipeline()

    assert framework.results["pipeline_status"] == "failed"
    assert framework.results["pipeline_status_reason"] == "framework_results_persistence_failed"
    assert calls["reports"] == 1
    assert not (output_root / "framework_results.json").exists()


def test_feature_only_public_pipeline_raises_when_result_persistence_fails(tmp_path, monkeypatch):
    bundle_path, _ = _bundle(tmp_path)
    output_root = tmp_path / "feature-only-save-fails"

    class FailedSaveFramework:
        def __init__(self, output_dir):
            self.output_dir = output_dir
            self.results = {}

        def run_feature_extraction(self, dataset):
            return {"feature_readiness": {"overall_status": "unavailable"}}

        def run_feature_fusion(self, features):
            return {"status": "unavailable"}

        def run_output_generation(self):
            return {}

        def save_results(self):
            return ""

    monkeypatch.setattr(main_module, "NetworkPharmacologyFramework", FailedSaveFramework)
    with pytest.raises(RuntimeError, match="could not be saved"):
        main_module.run_feature_pipeline(str(bundle_path), str(output_root))
