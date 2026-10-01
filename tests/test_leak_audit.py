"""Adversarial contracts for the independent, source-bound split auditor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import dataset_builder
import evidence_builder


APL_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _auditor():
    spec = importlib.util.find_spec("leak_audit")
    assert spec is not None, "planned independent leak_audit module should be present"
    return importlib.import_module("leak_audit")


def _fixture() -> dict:
    value = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    value["source_kind"] = "synthetic_test"
    return value


def _add_molecule(fixture: dict, molecule_id: str, smiles: str, activity_id: int) -> None:
    chembl = fixture["chembl"]
    molecule = deepcopy(chembl["molecules"][1])
    molecule["molecule_chembl_id"] = molecule_id
    molecule["pref_name"] = f"synthetic {molecule_id}"
    molecule["molecule_structures"]["canonical_smiles"] = smiles
    chembl["molecules"].append(molecule)

    page = deepcopy(chembl["activities_by_molecule"]["CHEMBL123456"]["pages"][0])
    activity = deepcopy(page["activities"][0])
    activity.update(
        activity_id=activity_id,
        molecule_chembl_id=molecule_id,
        target_chembl_id="CHEMBL910201",
        assay_chembl_id="CHEMBL910001",
        document_chembl_id="CHEMBL910101",
        standard_type="Ki",
        standard_relation="=",
        standard_value="100",
        standard_units="nM",
    )
    page["activities"] = [activity]
    page["molecule_chembl_id"] = molecule_id
    page["page_meta"].update(limit=1, next=None, offset=0, previous=None, total_count=1)
    chembl["activities_by_molecule"][molecule_id] = {"pages": [page]}


def _write_bundle(tmp_path: Path, fixture: dict | None = None) -> tuple[Path, dict]:
    payload = deepcopy(fixture) if fixture is not None else _fixture()
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    staging = tmp_path / "bundle"
    result = evidence_builder.ingest_local_manifest(source_manifest, staging)
    assert result["status"] == "complete"
    bundle_path = staging / "entities.json"
    return bundle_path, evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)


def _memberships(clusters, assignment: dict) -> dict[str, str]:
    rows = {}
    for cluster in clusters:
        partition = assignment.get(cluster.cluster_id)
        if partition in {"train", "val", "test"}:
            rows.update({identifier: partition for identifier in cluster.member_ids})
    return rows


def _candidate_manifest(tmp_path: Path, bundle_path: Path, bundle: dict) -> Path:
    """Build an untrusted candidate claim; audit tests never use it as evidence."""
    molecules = dataset_builder.build_molecule_clusters(bundle_path)
    proteins = dataset_builder.build_protein_clusters(bundle_path)
    split = dataset_builder.strict_double_cold_split(bundle_path, allow_test_only=True)
    claimed_split = {
        "molecule_memberships": _memberships(molecules.clusters, split.molecule_cluster_assignment),
        "protein_memberships": _memberships(proteins.clusters, split.protein_cluster_assignment),
        "pair_assignments": [asdict(row) for row in split.pair_assignments],
        "crosspair_quarantine": [asdict(row) for row in split.crosspair_quarantine],
        "unassigned_pairs": [asdict(row) for row in split.unassigned_pairs],
        "test_only_pair_assignments": [asdict(row) for row in split.test_only_pair_assignments],
        "ambiguous_label_quarantine": [asdict(row) for row in split.ambiguous_label_quarantine],
        "class_support": {key: dict(value) for key, value in split.class_support.items()},
        "molecule_cluster_ids": {row["entity_id"]: "forged-" + row["entity_id"] for row in molecules.entities},
        "protein_cluster_ids": {row["entity_id"]: "forged-" + row["entity_id"] for row in proteins.entities},
        "training_ready": True,
    }
    entity_bytes = bundle_path.read_bytes()
    manifest = {
        "schema_version": 1,
        "entities": {
            "path": "bundle/entities.json",
            "schema_version": bundle["schema_version"],
            "entity_version": bundle["entity_version"],
            "content_sha256": bundle["content_sha256"],
            "file_sha256": hashlib.sha256(entity_bytes).hexdigest(),
        },
        "source_provenance": bundle["source_provenance"],
        "split": claimed_split,
        "version_control": {"status": "not_initialized", "commit": None},
    }
    path = tmp_path / "candidate-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _finding_codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


def test_clean_small_bundle_is_auditable_but_never_implies_training_ready(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path)
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)

    report = _auditor().audit_split(manifest_path)

    assert report.exit_code == 0
    assert report.status == "clean"
    assert report.training_ready is False
    assert report.class_support != {"train": {"active": 100, "inactive": 100}}


def test_independent_neighbors_detect_forged_cluster_ids_even_if_builder_is_disabled(tmp_path, monkeypatch):
    fixture = _fixture()
    original = next(row for row in fixture["chembl"]["molecules"] if row["molecule_chembl_id"] == "CHEMBL151")
    original_smiles = original["molecule_structures"]["canonical_smiles"]
    _add_molecule(fixture, "CHEMBL700010", original_smiles, 990010)
    bundle_path, bundle = _write_bundle(tmp_path, fixture)
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    molecule_ids = sorted(row["molecule_chembl_id"] for row in bundle["molecules"] if row["structure_status"] == "verified")
    assert "CHEMBL151" in molecule_ids and "CHEMBL700010" in molecule_ids
    manifest["split"]["molecule_memberships"].update({"CHEMBL151": "train", "CHEMBL700010": "test"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("independent audit called dataset_builder clustering/split output")

    monkeypatch.setattr(dataset_builder, "build_molecule_clusters", forbidden)
    monkeypatch.setattr(dataset_builder, "build_protein_clusters", forbidden)
    monkeypatch.setattr(dataset_builder, "strict_double_cold_split", forbidden)
    auditor = _auditor()
    neighbors = auditor.independently_recompute_neighbors(bundle_path)
    report = auditor.audit_split(manifest_path)

    assert neighbors.status == "ok"
    assert any(set(edge.entity_ids) == {"CHEMBL151", "CHEMBL700010"} for edge in neighbors.molecule_neighbors)
    assert report.exit_code == 3
    finding = next(row for row in report.findings if row.code == "molecule_neighbor_cross_split")
    assert set(finding.entity_ids) == {"CHEMBL151", "CHEMBL700010"}
    assert finding.metric is not None and finding.threshold is not None
    assert finding.source_evidence


def test_raw_source_hash_mismatch_and_unknown_entity_mapping_are_concrete_violations(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path)
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["molecule_memberships"]["CHEMBL999999"] = "train"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    raw = next(row for row in bundle["source_provenance"] if row.get("source_path"))
    raw_path = bundle_path.parent / raw["source_path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"tampered")

    report = _auditor().audit_split(manifest_path)

    assert report.exit_code == 3
    assert {"source_hash_mismatch", "unknown_entity_mapping"} <= _finding_codes(report)
    assert any(row.source_evidence for row in report.findings if row.code == "source_hash_mismatch")


def test_duplicate_ledger_references_and_caller_support_counts_are_not_trusted(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path)
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest["split"]
    rows = (split["pair_assignments"] + split["crosspair_quarantine"] + split["unassigned_pairs"]
            + split["test_only_pair_assignments"] + split["ambiguous_label_quarantine"])
    assert rows
    duplicate = deepcopy(rows[0])
    bucket = next(name for name in ("pair_assignments", "crosspair_quarantine", "unassigned_pairs",
                                    "test_only_pair_assignments", "ambiguous_label_quarantine")
                  if rows[0] in split[name])
    split[bucket].append(duplicate)
    split["class_support"] = {name: {"active": 500, "inactive": 500} for name in ("train", "val", "test")}
    split["training_ready"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    report = _auditor().audit_split(manifest_path)

    assert report.exit_code == 3
    assert "duplicate_pair_assignment" in _finding_codes(report)
    assert "class_support_mismatch" in _finding_codes(report)
    assert report.training_ready is False


def test_cli_malicious_fixture_returns_audit_violation_exit_code(tmp_path):
    fixture = _fixture()
    original = next(row for row in fixture["chembl"]["molecules"] if row["molecule_chembl_id"] == "CHEMBL151")
    _add_molecule(fixture, "CHEMBL700011", original["molecule_structures"]["canonical_smiles"], 990011)
    bundle_path, bundle = _write_bundle(tmp_path, fixture)
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["molecule_memberships"].update({"CHEMBL151": "train", "CHEMBL700011": "val"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "leak_audit.py"), "--manifest", str(manifest_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, check=False,
    )

    assert completed.returncode == 3, (completed.stdout or "") + (completed.stderr or "")
    assert "molecule_neighbor_cross_split" in completed.stdout
