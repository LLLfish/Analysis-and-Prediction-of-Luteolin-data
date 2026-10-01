"""Adversarial contracts for the independent, source-bound split auditor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import builtins
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import data_contracts
import dataset_builder
import evidence_builder
import source_audit


APL_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _auditor():
    spec = importlib.util.find_spec("leak_audit")
    assert spec is not None, "planned independent leak_audit module should be present"
    return importlib.import_module("leak_audit")


def _fixture() -> dict:
    value = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    value["source_kind"] = "synthetic_test"
    # The baseline audit manifest must contain only pair-resolvable evidence;
    # non-human and PROTEIN COMPLEX rejection remains covered by entity tests.
    eligible_targets = {
        row["target_chembl_id"]
        for row in value["chembl"]["targets"]
        if row.get("target_type") == "SINGLE PROTEIN" and row.get("organism") == "Homo sapiens"
    }
    for molecule_pages in value["chembl"]["activities_by_molecule"].values():
        pages = molecule_pages["pages"]
        retained_rows = [
            row
            for page in pages
            for row in page.get("activities", [])
            if row.get("target_chembl_id") in eligible_targets
        ]
        page = deepcopy(pages[0])
        page["activities"] = retained_rows
        page["page_meta"].update(
            offset=0, limit=len(retained_rows), total_count=len(retained_rows),
            next=None, previous=None,
        )
        molecule_pages["pages"] = [page]
    return value


def _two_protein_fixture() -> dict:
    fixture = _fixture()
    target = next(row for row in fixture["chembl"]["targets"] if row["target_chembl_id"] == "CHEMBL910201")
    target["target_components"][0]["accession"] = "P23456"
    protein = deepcopy(fixture["uniprot"]["P12345"])
    protein["primaryAccession"] = "P23456"
    fixture["uniprot"]["P23456"] = protein
    return fixture


def _two_measurement_fixture(second_value: str) -> dict:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["source_kind"] = "online_snapshot"
    fixture["purpose"] = "synthetic_test_only_official_shape_mock"
    chembl = fixture["chembl"]

    def measurement(source_rows: list[dict], activity_id: int, molecule_id: str, endpoint: str, value: str) -> dict:
        row = deepcopy(source_rows[0])
        row.update(
            activity_id=activity_id,
            molecule_chembl_id=molecule_id,
            standard_type=endpoint,
            standard_relation="=",
            standard_value=value,
            standard_units="nM",
        )
        return row

    def replace_pages(molecule_id: str, rows: list[dict]) -> None:
        source_page = chembl["activities_by_molecule"][molecule_id]["pages"][0]
        page = deepcopy(source_page)
        page["page_meta"].update(limit=len(rows), next=None, offset=0, previous=None, total_count=len(rows))
        page["molecule_chembl_id"] = molecule_id
        page["activities"] = rows
        chembl["activities_by_molecule"][molecule_id] = {"pages": [page]}

    luteolin_source = chembl["activities_by_molecule"]["CHEMBL151"]["pages"][0]["activities"]
    replace_pages("CHEMBL151", [
        measurement(luteolin_source, 990101, "CHEMBL151", "Ki", "100"),
        measurement(luteolin_source, 990102, "CHEMBL151", "Kd", second_value),
    ])
    caffeine_source = chembl["activities_by_molecule"]["CHEMBL123456"]["pages"][0]["activities"]
    replace_pages("CHEMBL123456", [measurement(caffeine_source, 990103, "CHEMBL123456", "Ki", "100")])
    return fixture


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
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(fixture) if fixture is not None else _fixture()
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    staging = tmp_path / "bundle"
    result = evidence_builder.ingest_local_manifest(source_manifest, staging)
    assert result["status"] == "complete", result
    bundle_path = staging / "entities.json"
    return bundle_path, evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)


def _write_empty_bundle(tmp_path: Path) -> tuple[Path, dict]:
    staging = tmp_path / "empty-bundle"
    staging.mkdir()
    evidence_builder.write_entity_bundle(
        staging, activities=[], source_provenance=[], molecules=[], proteins=[],
    )
    bundle_path = staging / "entities.json"
    return bundle_path, evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)


def _memberships(clusters, assignment: dict) -> dict[str, str]:
    rows = {}
    for cluster in clusters:
        partition = assignment.get(cluster.cluster_id)
        if partition in {"train", "val", "test"}:
            rows.update({identifier: partition for identifier in cluster.member_ids})
    return rows


def _frozen_policy_refs(root: Path) -> tuple[dict, dict]:
    references = {}
    claims = {}
    for name, filename in (("split", "split_policy.json"), ("label", "label_policy.json")):
        raw = (APL_ROOT / "policies" / filename).read_bytes()
        parsed = json.loads(raw.decode("utf-8-sig"))
        semantic_hash = evidence_builder.canonical_json_sha256(parsed)
        references[name] = {
            "path": f"policies/{filename}",
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "semantic_sha256": semantic_hash,
            "schema_version": parsed["schema_version"],
            "policy_version": parsed["policy_version"],
        }
        claims[f"{name}_policy_file_sha256"] = hashlib.sha256(raw).hexdigest()
        claims[f"{name}_policy_semantic_sha256"] = semantic_hash
        claims[f"{name}_policy_schema_version"] = parsed["schema_version"]
        claims[f"{name}_policy_version"] = parsed["policy_version"]
        target = root / references[name]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    claims["label_policy_sha256"] = references["label"]["file_sha256"]
    return references, claims


def _candidate_manifest(tmp_path: Path, bundle_path: Path, bundle: dict) -> Path:
    """Build an untrusted candidate claim; audit tests never use it as evidence."""
    molecules = dataset_builder.build_molecule_clusters(bundle_path)
    proteins = dataset_builder.build_protein_clusters(bundle_path)
    split = dataset_builder.strict_double_cold_split(bundle_path, allow_test_only=True)
    policy_refs, policy_claims = _frozen_policy_refs(tmp_path)
    claimed_split = {
        "strict_split_status": split.strict_split_status,
        "split_policy_sha256": split.policy_sha256,
        **policy_claims,
        "entity_version": bundle["entity_version"],
        "entity_bundle_hash": bundle["content_sha256"],
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
        "training_ready": split.training_ready,
    }
    entity_bytes = bundle_path.read_bytes()
    pair_payload = {key: claimed_split[key] for key in (
        "pair_assignments", "crosspair_quarantine", "unassigned_pairs",
        "test_only_pair_assignments", "ambiguous_label_quarantine",
    )}
    manifest = {
        "schema_version": 2,
        "entities": {
            "path": bundle_path.relative_to(tmp_path).as_posix(),
            "schema_version": bundle["schema_version"],
            "entity_version": bundle["entity_version"],
            "content_sha256": bundle["content_sha256"],
            "file_sha256": hashlib.sha256(entity_bytes).hexdigest(),
        },
        "source_provenance": bundle["source_provenance"],
        "policies": policy_refs,
        "activities": {
            "entity_version": bundle["entity_version"],
            "entity_bundle_hash": bundle["content_sha256"],
            "record_count": len(bundle["activities"]),
            "semantic_sha256": evidence_builder.canonical_json_sha256(bundle["activities"]),
        },
        "pairs": {
            "entity_version": bundle["entity_version"],
            "entity_bundle_hash": bundle["content_sha256"],
            "semantic_sha256": evidence_builder.canonical_json_sha256(pair_payload),
            "ledger_counts": {key: len(value) for key, value in pair_payload.items()},
        },
        "split": claimed_split,
        "version_control": {"status": "not_initialized", "commit": None},
    }
    candidate_path = tmp_path / ".manifest.audit.tmp.testfixture"

    def write_candidate() -> None:
        candidate_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    manifest["semantic_sha256"] = data_contracts.semantic_manifest_hash(manifest)
    write_candidate()
    first_audit = _auditor()._audit_candidate_split(candidate_path)
    assert first_audit.exit_code == 0, first_audit.to_dict()
    manifest["training_ready"] = first_audit.training_ready
    claimed_split["training_ready"] = first_audit.training_ready
    manifest["audit"] = {"status": "clean", "exit_code": 0, "finding_count": 0}
    manifest["semantic_sha256"] = data_contracts.semantic_manifest_hash(manifest)
    write_candidate()
    receipt_audit = _auditor()._audit_candidate_split(candidate_path)
    assert receipt_audit.exit_code == 0, receipt_audit.to_dict()
    receipt = source_audit.build_frozen_corpus_readiness_receipt(manifest, receipt_audit)
    receipt_bytes = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    receipt_path = tmp_path / "source_readiness_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    manifest["source_readiness_contract_version"] = 1
    manifest["source_readiness_receipt_ref"] = {
        "path": "source_readiness_receipt.json",
        "file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "semantic_sha256": data_contracts.canonical_json_sha256(receipt),
    }
    manifest["semantic_sha256"] = data_contracts.semantic_manifest_hash(manifest)
    candidate_path.unlink()
    path = tmp_path / "candidate-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _finding_codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


def _rebind_entity_manifest(manifest_path: Path, bundle_path: Path, bundle: dict) -> None:
    digest = evidence_builder.canonical_json_sha256(evidence_builder._bundle_hash_payload(bundle))
    version = f"entities-v2-{digest[:16]}"
    bundle["entity_version"] = version
    bundle["content_sha256"] = digest
    for activity in bundle["activities"]:
        activity["entity_bundle_version"] = version
        activity["entity_bundle_hash"] = digest
    raw = json.dumps(bundle, ensure_ascii=False, sort_keys=True).encode("utf-8")
    bundle_path.write_bytes(raw)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entities"].update(
        entity_version=version,
        content_sha256=digest,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    manifest["activities"].update(
        entity_version=version,
        entity_bundle_hash=digest,
        record_count=len(bundle["activities"]),
        semantic_sha256=evidence_builder.canonical_json_sha256(bundle["activities"]),
    )
    pair_payload = {key: manifest["split"][key] for key in (
        "pair_assignments", "crosspair_quarantine", "unassigned_pairs",
        "test_only_pair_assignments", "ambiguous_label_quarantine",
    )}
    manifest["pairs"].update(
        entity_version=version,
        entity_bundle_hash=digest,
        semantic_sha256=evidence_builder.canonical_json_sha256(pair_payload),
        ledger_counts={key: len(value) for key, value in pair_payload.items()},
    )
    manifest["split"].update(entity_version=version, entity_bundle_hash=digest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def test_clean_small_bundle_is_auditable_but_never_implies_training_ready(tmp_path):
    bundle_path, bundle = _write_empty_bundle(tmp_path)
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


def test_protein_neighbor_audit_recomputes_full_sequence_identity_and_both_coverages(tmp_path, monkeypatch):
    bundle_path, bundle = _write_bundle(tmp_path, _two_protein_fixture())
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["protein_memberships"].update({"9606|P12345": "train", "9606|P23456": "val"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("independent audit called dataset_builder clustering/split output")

    monkeypatch.setattr(dataset_builder, "build_protein_clusters", forbidden)
    monkeypatch.setattr(dataset_builder, "strict_double_cold_split", forbidden)
    auditor = _auditor()
    neighbors = auditor.independently_recompute_neighbors(bundle_path)
    report = auditor.audit_split(manifest_path)

    hit = next(row for row in neighbors.protein_neighbors if set(row.entity_ids) == {"9606|P12345", "9606|P23456"})
    assert hit.metric == 1.0
    assert hit.details["coverage_a"] == hit.details["coverage_b"] == 1.0
    assert hit.details["matrix"] == "BLOSUM62"
    assert hit.details["open_gap_score"] == -10.0
    assert hit.details["extend_gap_score"] == -0.5
    assert hit.details["epsilon"] == 0.0
    assert report.exit_code == 3
    assert any(row.code == "protein_neighbor_cross_split" for row in report.findings)


@pytest.mark.parametrize(
    ("coordinates", "expected_identity", "expected_coverage"),
    [
        (((0, 10), (0, 10)), 0.3, 1.0),
        (((0, 8, 10, 10), (0, 8, 8, 10)), 0.375, 0.8),
    ],
)
def test_protein_identity_and_bidirectional_coverage_thresholds_are_inclusive(
    tmp_path, monkeypatch, coordinates, expected_identity, expected_coverage,
):
    import Bio.Align

    fixture = _two_protein_fixture()
    for accession, sequence in (("P12345", "AAAAAAAAAA"), ("P23456", "AAACCCCCCC")):
        fixture["uniprot"][accession]["sequence"]["value"] = sequence
        fixture["uniprot"][accession]["sequence"]["length"] = len(sequence)
    bundle_path, _bundle = _write_bundle(tmp_path, fixture)

    class CoordinateMatrix:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (2, len(rows[0]))

        def __getitem__(self, index):
            row, column = index
            return self.rows[row][column]

    class Alignment:
        def __init__(self):
            self.coordinates = CoordinateMatrix(coordinates)

    class FakeAligner:
        def align(self, _sequence_a, _sequence_b):
            return [Alignment()]

    aligner = FakeAligner()
    monkeypatch.setattr(Bio.Align, "PairwiseAligner", lambda: aligner)
    result = _auditor().independently_recompute_neighbors(bundle_path)

    assert result.status == "ok"
    hit = next(row for row in result.protein_neighbors if set(row.entity_ids) == {"9606|P12345", "9606|P23456"})
    assert hit.metric == pytest.approx(expected_identity)
    assert hit.details["coverage_a"] == pytest.approx(expected_coverage)
    assert hit.details["coverage_b"] == pytest.approx(expected_coverage)


def test_alignment_budget_exhaustion_is_not_a_clean_audit(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path, _two_protein_fixture())
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    auditor = _auditor()

    neighbors = auditor.independently_recompute_neighbors(bundle_path, budget_overrides={"alignment_cells": 1})
    report = auditor.audit_split(manifest_path, budget_overrides={"alignment_cells": 1})

    assert neighbors.status == "size_limit"
    assert neighbors.exit_code == 4
    assert report.status == "size_limit"
    assert report.exit_code == 4
    assert report.training_ready is False


def test_missing_bio_alignment_dependency_returns_exit4_not_clean(tmp_path, monkeypatch):
    bundle_path, _bundle = _write_bundle(tmp_path, _two_protein_fixture())
    auditor = _auditor()
    original_import = builtins.__import__

    def without_bio(name, *args, **kwargs):
        if name == "Bio" or name.startswith("Bio."):
            raise ImportError("Bio.Align intentionally unavailable for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_bio)
    result = auditor.independently_recompute_neighbors(bundle_path)

    assert result.status == "dependency_missing"
    assert result.exit_code == 4


def test_pair_support_counts_multiple_endpoints_once_and_quarantines_ambiguous_labels(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path, _two_measurement_fixture("200"))
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    report = _auditor().audit_split(manifest_path)

    assert report.class_support["train"]["active"] + report.class_support["val"]["active"] + report.class_support["test"]["active"] == 2
    assert report.training_ready is False

    conflict_path, conflict_bundle = _write_bundle(tmp_path / "conflict", _two_measurement_fixture("20000"))
    conflict_manifest_path = _candidate_manifest(tmp_path / "conflict", conflict_path, conflict_bundle)
    conflict_manifest = json.loads(conflict_manifest_path.read_text(encoding="utf-8"))
    ambiguous_rows = conflict_manifest["split"]["ambiguous_label_quarantine"]
    assert any("ambiguous_multi_label" in row["reasons"] for row in ambiguous_rows)
    conflict = _auditor().audit_split(conflict_manifest_path)
    assert conflict.training_ready is False
    assert all(row.code != "ambiguous_multi_label_marker_missing" for row in conflict.findings)


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


def test_rehashed_structure_or_full_sequence_changes_still_fail_source_semantics(tmp_path):
    for variant in ("molecule", "protein"):
        case_root = tmp_path / variant
        bundle_path, bundle = _write_bundle(case_root)
        manifest_path = _candidate_manifest(case_root, bundle_path, bundle)
        if variant == "molecule":
            molecule = next(row for row in bundle["molecules"] if row.get("structure_status") == "verified")
            molecule["canonical_smiles"] = "CCO"
            molecule["structure_sha256"] = hashlib.sha256(b"CCO").hexdigest()
        else:
            protein = next(row for row in bundle["proteins"] if row.get("sequence_status") == "verified")
            protein["raw_sequence"] = "ACDE"
            protein["full_sequence"] = "ACDE"
            protein["sequence_length"] = 4
            protein["sequence_sha256"] = hashlib.sha256(b"ACDE").hexdigest()
        _rebind_entity_manifest(manifest_path, bundle_path, bundle)

        report = _auditor().audit_split(manifest_path)

        assert report.exit_code == 3
        assert any("derived fields differ from raw source" in row.message for row in report.findings)


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


def test_cli_alignment_budget_failure_returns_exit4_not_clean(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path, _two_protein_fixture())
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "leak_audit.py"),
         "--manifest", str(manifest_path), "--max-alignment-cells", "1"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, check=False,
    )

    assert completed.returncode == 4, (completed.stdout or "") + (completed.stderr or "")
    assert '"status": "size_limit"' in completed.stdout
    audit_report = json.loads(completed.stdout)
    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert audit_report["split_policy_sha256"] == frozen_manifest["policies"]["split"]["semantic_sha256"]
    assert audit_report["split_policy_file_sha256"] == frozen_manifest["policies"]["split"]["file_sha256"]
    assert "split_policy_hash_mismatch" not in {row["code"] for row in audit_report["findings"]}
    assert "split_policy_hash_mismatch" not in {row["code"] for row in audit_report["findings"]}


def test_cli_missing_bio_dependency_returns_exit4(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path, _two_protein_fixture())
    manifest_path = _candidate_manifest(tmp_path, bundle_path, bundle)
    code = (
        "import builtins,sys;\n"
        f"sys.path.insert(0,{str(APL_ROOT)!r});\n"
        "original=builtins.__import__;\n"
        "def blocked(name,*args,**kwargs):\n"
        "    if name=='Bio' or name.startswith('Bio.'): raise ImportError('Bio.Align unavailable');\n"
        "    return original(name,*args,**kwargs)\n"
        "builtins.__import__=blocked;\n"
        "import leak_audit;\n"
        "raise SystemExit(leak_audit.main(sys.argv[1:]))\n"
    )
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-c", code, "--manifest", str(manifest_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, check=False,
    )

    assert completed.returncode == 4, (completed.stdout or "") + (completed.stderr or "")
    assert '"status": "dependency_missing"' in completed.stdout
