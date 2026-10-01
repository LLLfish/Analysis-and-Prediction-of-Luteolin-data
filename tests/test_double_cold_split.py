"""Behavior tests for whole-cluster double-cold splitting and pair quarantine."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import dataset_builder
import evidence_builder
import label_policy


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"
POLICY_PATH = Path(__file__).parents[1] / "policies" / "split_policy.json"


def _add_molecule(fixture, molecule_id, smiles, measurements):
    chembl = fixture["chembl"]
    molecule = deepcopy(chembl["molecules"][1])
    molecule["molecule_chembl_id"] = molecule_id
    molecule["pref_name"] = f"synthetic {molecule_id}"
    molecule["molecule_structures"]["canonical_smiles"] = smiles
    chembl["molecules"].append(molecule)

    template_page = chembl["activities_by_molecule"]["CHEMBL123456"]["pages"][0]
    template_activity = template_page["activities"][0]
    activities = []
    for offset, (activity_id, endpoint, value) in enumerate(measurements):
        activity = deepcopy(template_activity)
        activity.update(
            activity_id=activity_id,
            standard_type=endpoint,
            standard_relation="=",
            standard_value=value,
            standard_units="nM",
            molecule_chembl_id=molecule_id,
            target_chembl_id="CHEMBL910201",
            assay_chembl_id="CHEMBL910001",
            document_chembl_id="CHEMBL910101",
        )
        activities.append(activity)
    page = deepcopy(template_page)
    page["page_meta"].update(
        limit=len(activities), next=None, offset=0, previous=None, total_count=len(activities),
    )
    page["activities"] = activities
    page["molecule_chembl_id"] = molecule_id
    chembl["activities_by_molecule"][molecule_id] = {"pages": [page]}


def _expanded_conflict_and_unknown_fixture():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["source_kind"] = "synthetic_test"
    _add_molecule(fixture, "CHEMBL700001", "c1ccccc1", [
        (880001, "Ki", "100"),
        (880002, "Ki", "20000"),
        (880003, "Kd", "100"),
        (880004, "IC50", "100"),
    ])
    _add_molecule(fixture, "CHEMBL700002", "CC(=O)O", [(880005, "Ki", "100")])
    _add_molecule(fixture, "CHEMBL700003", "C1CCCCC1", [(880006, "Ki", None)])
    return fixture


def _write_bundle(tmp_path: Path, fixture):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "bundle"
    result = evidence_builder.ingest_local_manifest(manifest_path, output_dir)
    assert result["status"] == "complete"
    bundle_path = output_dir / "entities.json"
    bundle = evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)
    return bundle_path, bundle


def _empty_bundle(tmp_path: Path):
    output_dir = tmp_path / "empty-bundle"
    output_dir.mkdir()
    return evidence_builder.write_entity_bundle(
        output_dir,
        activities=[],
        source_provenance=[],
        molecules=[],
        proteins=[],
    ), output_dir / "entities.json"


def _all_pair_rows(result):
    return (
        list(result.pair_assignments)
        + list(result.crosspair_quarantine)
        + list(result.unassigned_pairs)
        + list(result.test_only_pair_assignments)
    )


def test_split_uses_grouped_source_verified_labels_and_never_promotes_test_only_or_conflicts(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path, _expanded_conflict_and_unknown_fixture())
    groups, ledger = label_policy.group_evidence(bundle["activities"])

    result = dataset_builder.strict_double_cold_split(bundle_path, allow_test_only=True)

    assert result.strict_split_status == "insufficient"
    assert result.training_ready is False
    assert result.test_only is True
    assert result.entity_version == bundle["entity_version"]
    assert result.entity_bundle_hash == bundle["content_sha256"]
    assert result.source_provenance
    assert result.test_only_pair_assignments
    assert result.pair_assignments == ()
    assert result.class_support == {
        "train": {"active": 0, "inactive": 0},
        "val": {"active": 0, "inactive": 0},
        "test": {"active": 0, "inactive": 0},
    }

    conflict_key = "CHEMBL700001|9606|P12345"
    conflict_group = next(group for group in groups if group.pair_key == conflict_key)
    records = [json.loads(raw) for raw in conflict_group.records_json]
    assert {row["standard_type"] for row in records} == {"Ki", "Kd", "IC50"}
    assert len([group for group in groups if group.pair_key == conflict_key]) == 1
    conflict_assignments = [row for row in _all_pair_rows(result) if row.pair_key == conflict_key]
    assert len(conflict_assignments) == 1
    assert conflict_assignments[0].split_group_id == conflict_group.split_group_id
    assert not (set(conflict_assignments[0].labels) & {"active", "inactive"})

    unknown_key = "CHEMBL700003|9606|P12345"
    unknown_assignments = [row for row in _all_pair_rows(result) if row.pair_key == unknown_key]
    assert len(unknown_assignments) == 1
    assert not (set(unknown_assignments[0].labels) & {"active", "inactive"})
    assert ledger.input_count == len(bundle["activities"])


def test_whole_cluster_assignments_quarantine_crosspairs_and_reorder_stably(tmp_path, monkeypatch):
    bundle_path, bundle = _write_bundle(tmp_path, _expanded_conflict_and_unknown_fixture())

    first = dataset_builder.strict_double_cold_split(bundle_path, allow_test_only=True)

    assert len(first.molecule_clusters) == 5
    assert len(first.protein_clusters) == 1
    assert set(first.molecule_cluster_assignment.values()) == {"train", "val"}
    assert set(first.protein_cluster_assignment.values()) == {"train"}
    assert first.crosspair_quarantine
    assert all(row.split is None and "crosspair" in " ".join(row.reasons)
               for row in first.crosspair_quarantine)
    assert isinstance(first.split_sha256, str) and len(first.split_sha256) == 64
    for cluster in first.molecule_clusters:
        assert cluster.cluster_id in first.molecule_cluster_assignment
    for cluster in first.protein_clusters:
        assert cluster.cluster_id in first.protein_cluster_assignment

    original_group_evidence = label_policy.group_evidence
    monkeypatch.setattr(
        dataset_builder,
        "group_evidence",
        lambda rows: original_group_evidence(list(rows)[::-1]),
    )
    reordered = dataset_builder.strict_double_cold_split(bundle_path, allow_test_only=True)

    assert reordered.entity_bundle_hash == bundle["content_sha256"]
    assert reordered.split_sha256 == first.split_sha256
    assert reordered.molecule_cluster_assignment == first.molecule_cluster_assignment
    assert reordered.protein_cluster_assignment == first.protein_cluster_assignment
    assert reordered.pair_assignments == first.pair_assignments
    assert reordered.crosspair_quarantine == first.crosspair_quarantine


def test_zero_one_and_single_cluster_inputs_remain_insufficient(tmp_path):
    _, empty_path = _empty_bundle(tmp_path)
    empty = dataset_builder.strict_double_cold_split(empty_path)
    assert empty.strict_split_status == "insufficient"
    assert empty.training_ready is False
    assert not empty.molecule_clusters and not empty.protein_clusters

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["source_kind"] = "synthetic_test"
    fixture["chembl"]["molecules"] = [fixture["chembl"]["molecules"][0]]
    fixture["chembl"]["activities_by_molecule"] = {
        "CHEMBL151": fixture["chembl"]["activities_by_molecule"]["CHEMBL151"]
    }
    single_path, _ = _write_bundle(tmp_path / "single", fixture)

    single = dataset_builder.strict_double_cold_split(single_path, allow_test_only=True)

    assert single.strict_split_status == "insufficient"
    assert single.training_ready is False
    assert len(single.molecule_clusters) == 1
    assert len(single.protein_clusters) == 1
    assert set(single.molecule_cluster_assignment.values()) == {"train"}
    assert set(single.protein_cluster_assignment.values()) == {"train"}


def test_split_budget_limit_is_not_reported_as_ready(tmp_path):
    bundle_path, _ = _write_bundle(tmp_path, _expanded_conflict_and_unknown_fixture())
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["budgets"]["molecule_pairs"] = 0
    policy_path = tmp_path / "limited-split-policy.json"
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    result = dataset_builder.strict_double_cold_split(bundle_path, policy_path=policy_path)

    assert result.strict_split_status == "size_limit"
    assert result.training_ready is False
    assert not result.molecule_clusters
    assert not result.pair_assignments


def test_missing_alignment_dependency_is_reported_without_fallback(tmp_path, monkeypatch):
    bundle_path, bundle = _write_bundle(tmp_path, _expanded_conflict_and_unknown_fixture())
    monkeypatch.setattr(dataset_builder, "load_entity_bundle", lambda *_args, **_kwargs: bundle)
    original_import = __import__

    def without_bio(name, *args, **kwargs):
        if name == "Bio" or name.startswith("Bio."):
            raise ModuleNotFoundError("simulated missing Bio.Align")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_bio)
    result = dataset_builder.strict_double_cold_split(bundle_path)

    assert result.strict_split_status == "dependency_missing"
    assert result.training_ready is False
    assert not result.pair_assignments
