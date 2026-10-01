"""Contract tests for producer-bound molecular and protein clustering."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

import data_contracts
import dataset_builder
import evidence_builder


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"
POLICY_PATH = Path(__file__).parents[1] / "policies" / "split_policy.json"


def _write_bundle(tmp_path: Path, fixture=None):
    payload = deepcopy(fixture) if fixture is not None else json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["source_kind"] = "synthetic_test"
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "bundle"
    result = evidence_builder.ingest_local_manifest(manifest_path, output_dir)
    assert result["status"] == "complete"
    bundle_path = output_dir / "entities.json"
    bundle = evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)
    return bundle_path, bundle


def _protein(accession: str, sequence: str, *, taxonomy_id: int = 9606):
    source = {
        "source_id": f"uniprot:{accession}",
        "source_path": f"raw/uniprot/{accession}.json",
        "raw_sha256": "a" * 64,
    }
    return asdict(data_contracts.ProteinRecord(
        accession=accession,
        taxonomy_id=taxonomy_id,
        raw_sequence=sequence,
        full_sequence=sequence,
        sequence_length=len(sequence),
        sequence_sha256=hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        sequence_status="verified",
        sequence_status_reason=None,
        identity_status="verified",
        source_kind="synthetic_test",
        source_provenance=[source],
        component_evidence={"references": []},
        returned_accession=accession,
        returned_taxonomy_id=taxonomy_id,
    ))


def _molecule(molecule_id: str, smiles: str):
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "structure_status": "verified",
        "source_kind": "synthetic_test",
        "source_provenance": [{
            "source_id": f"chembl_molecule:{molecule_id}",
            "source_path": f"raw/chembl/molecule-{molecule_id}.json",
            "raw_sha256": "b" * 64,
        }],
    }


def _rehash_bundle(bundle):
    digest = evidence_builder.canonical_json_sha256(evidence_builder._bundle_hash_payload(bundle))
    version = f"entities-v2-{digest[:16]}"
    bundle["entity_version"] = version
    bundle["content_sha256"] = digest
    for activity in bundle["activities"]:
        activity["entity_bundle_version"] = version
        activity["entity_bundle_hash"] = digest


def _policy_with_budget(tmp_path: Path, **overrides):
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["budgets"].update(overrides)
    path = tmp_path / "split-policy.json"
    path.write_text(json.dumps(policy, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def test_producer_bundle_is_the_only_cluster_input_and_hashes_propagate(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path)

    molecules = dataset_builder.build_molecule_clusters(bundle_path)
    proteins = dataset_builder.build_protein_clusters(bundle_path)

    assert molecules.entity_version == proteins.entity_version == bundle["entity_version"]
    assert molecules.entity_bundle_hash == proteins.entity_bundle_hash == bundle["content_sha256"]
    assert molecules.policy_sha256 == proteins.policy_sha256
    assert molecules.source_provenance and proteins.source_provenance
    assert {row["entity_id"] for row in molecules.entities} == {
        row["molecule_chembl_id"] for row in bundle["molecules"]
        if row["structure_status"] == "verified"
    }
    expected_proteins = {
        f"{row['taxonomy_id']}|{row['accession']}"
        for row in bundle["proteins"]
        if row["identity_status"] == "verified" and row["sequence_status"] == "verified"
    }
    assert {row["entity_id"] for row in proteins.entities} == expected_proteins
    assert molecules.test_only is True
    assert proteins.training_ready is False


def test_molecule_neighbor_uses_identity_scaffold_and_inclusive_morgan_threshold(monkeypatch):
    identity = dataset_builder.molecule_neighbor(
        _molecule("CHEMBL1", "CCO"), _molecule("CHEMBL1", "CCN"),
    )
    assert [(row.reason, row.score) for row in identity] == [("identity", 1.0)]

    scaffold = dataset_builder.molecule_neighbor(
        _molecule("CHEMBL2", "Oc1ccccc1"), _molecule("CHEMBL3", "Cc1ccccc1"),
    )
    assert any(row.reason == "murcko_scaffold" for row in scaffold)

    from rdkit import DataStructs
    monkeypatch.setattr(DataStructs, "TanimotoSimilarity", lambda *_: 0.8)
    inclusive = dataset_builder.molecule_neighbor(
        _molecule("CHEMBL4", "CCO"), _molecule("CHEMBL5", "CCN"),
    )
    assert any(row.reason == "morgan_tanimoto" and row.score == 0.8 for row in inclusive)

    monkeypatch.setattr(DataStructs, "TanimotoSimilarity", lambda *_: 0.799999)
    below = dataset_builder.molecule_neighbor(
        _molecule("CHEMBL6", "CCO"), _molecule("CHEMBL7", "CCN"),
    )
    assert not any(row.reason == "morgan_tanimoto" for row in below)


def test_empty_scaffolds_do_not_connect_acyclic_molecules():
    neighbors = dataset_builder.molecule_neighbor(
        _molecule("CHEMBL8", "CCO"), _molecule("CHEMBL9", "CCN"),
    )
    assert not any(row.reason == "murcko_scaffold" for row in neighbors)


def test_protein_neighbor_uses_non_gap_identity_and_inclusive_bidirectional_coverage():
    identity_boundary = dataset_builder.protein_neighbor(
        _protein("P12345", "A" * 10), _protein("Q12345", "AAA" + "C" * 7),
    )
    assert any(row.reason == "sequence_similarity" for row in identity_boundary)
    match = next(row for row in identity_boundary if row.reason == "sequence_similarity")
    assert match.details["identity"] == pytest.approx(0.3)
    assert match.details["coverage_a"] == pytest.approx(1.0)
    assert match.details["coverage_b"] == pytest.approx(1.0)

    coverage_boundary = dataset_builder.protein_neighbor(
        _protein("P23456", "ACDEFGHIKL"), _protein("Q23456", "ACDEFGHI"),
    )
    assert any(row.reason == "sequence_similarity" for row in coverage_boundary)
    coverage_match = next(row for row in coverage_boundary if row.reason == "sequence_similarity")
    assert coverage_match.details["coverage_a"] == pytest.approx(0.8)
    assert coverage_match.details["coverage_b"] == pytest.approx(1.0)

    below = dataset_builder.protein_neighbor(
        _protein("P34567", "ACDEFGHIKL"), _protein("Q34567", "ACDEFGH"),
    )
    assert not any(row.reason == "sequence_similarity" for row in below)


def test_connected_clusters_keep_transitive_chain_and_stable_minimum_member_id():
    edges = (
        dataset_builder.NeighborEvidence("A", "B", "morgan_tanimoto", 0.9),
        dataset_builder.NeighborEvidence("B", "C", "morgan_tanimoto", 0.9),
    )

    clusters = dataset_builder.connected_clusters(["C", "A", "B"], edges, prefix="molecule-cluster-v1")

    assert len(clusters) == 1
    assert clusters[0].member_ids == ("A", "B", "C")
    assert clusters[0].cluster_id == "molecule-cluster-v1:A"
    assert dataset_builder.connected_clusters([], ()) == ()
    singleton = dataset_builder.connected_clusters(["one"], ())
    assert len(singleton) == 1 and singleton[0].member_ids == ("one",)


@pytest.mark.parametrize("tamper", ["structure", "sequence"])
def test_source_semantics_reject_rehashed_structure_or_sequence_tampering(tmp_path, tamper):
    bundle_path, original = _write_bundle(tmp_path)
    bundle = deepcopy(original)
    if tamper == "structure":
        molecule = next(row for row in bundle["molecules"] if row["structure_status"] == "verified")
        molecule["canonical_smiles"] = "CCO"
        molecule["raw_structure"]["canonical_smiles"] = "CCO"
        molecule["structure_sha256"] = hashlib.sha256(b"CCO").hexdigest()
    else:
        protein = next(row for row in bundle["proteins"] if row["sequence_status"] == "verified")
        changed = "A" * protein["sequence_length"]
        protein["raw_sequence"] = changed
        protein["full_sequence"] = changed
        protein["sequence_sha256"] = hashlib.sha256(changed.encode("utf-8")).hexdigest()
    _rehash_bundle(bundle)
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(dataset_builder.DatasetBuildError, match="source|molecule|protein|sequence"):
        if tamper == "structure":
            dataset_builder.build_molecule_clusters(bundle_path)
        else:
            dataset_builder.build_protein_clusters(bundle_path)


def test_unknown_bundle_version_and_missing_raw_source_cannot_produce_clusters(tmp_path):
    bundle_path, bundle = _write_bundle(tmp_path)
    unknown_path = tmp_path / "unknown-entities.json"
    unknown = deepcopy(bundle)
    unknown["schema_version"] = 999
    unknown_path.write_text(json.dumps(unknown, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(dataset_builder.DatasetBuildError, match="schema|version"):
        dataset_builder.build_molecule_clusters(unknown_path)

    source = next(row for row in bundle["source_provenance"] if row["source_id"].startswith("chembl_molecule:"))
    (bundle_path.parent / source["source_path"]).unlink()
    with pytest.raises(dataset_builder.DatasetBuildError, match="source|missing"):
        dataset_builder.build_molecule_clusters(bundle_path)


def test_full_long_sequence_hash_matches_feature_contract(tmp_path):
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 75
    fixture["uniprot"]["P12345"]["sequence"]["value"] = sequence
    fixture["uniprot"]["P12345"]["sequence"]["length"] = len(sequence)
    bundle_path, bundle = _write_bundle(tmp_path, fixture)

    result = dataset_builder.build_protein_clusters(bundle_path)
    assert any(row["entity_id"] == "9606|P12345" for row in result.entities)
    entity = next(row for row in result.entities if row["entity_id"] == "9606|P12345")
    source_protein = next(row for row in bundle["proteins"] if row["accession"] == "P12345")
    feature = data_contracts.FeatureResult(
        status="skipped_untrained",
        reason="fixture has no verified feature model",
        accession="P12345",
        taxonomy_id=9606,
        sequence_length=len(sequence),
        sequence_sha256=source_protein["sequence_sha256"],
        entity_version=bundle["entity_version"],
        entity_bundle_hash=bundle["content_sha256"],
    )

    assert entity["sequence_length"] == len(sequence)
    assert entity["sequence_sha256"] == feature.sequence_sha256
    assert feature.sequence_length > 1022


def test_pair_budget_and_alignment_cell_budget_fail_closed(tmp_path):
    bundle_path, _ = _write_bundle(tmp_path)
    limited_policy = _policy_with_budget(tmp_path, molecule_pairs=0)

    limited = dataset_builder.build_molecule_clusters(bundle_path, policy_path=limited_policy)

    assert limited.status == "size_limit"
    assert limited.clusters == ()
    assert limited.training_ready is False
    with pytest.raises(dataset_builder.DatasetBuildError) as exc_info:
        dataset_builder.protein_neighbor(
            _protein("P45678", "AA"), _protein("Q45678", "CC"),
            policy_path=_policy_with_budget(tmp_path, alignment_cells=3),
        )
    assert exc_info.value.code == "size_limit"


def test_missing_clustering_dependency_is_reported_without_fallback(tmp_path, monkeypatch):
    bundle_path, bundle = _write_bundle(tmp_path)
    monkeypatch.setattr(dataset_builder, "load_entity_bundle", lambda *_args, **_kwargs: bundle)
    original_import = __import__

    def without_rdkit(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ModuleNotFoundError("simulated missing RDKit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_rdkit)
    result = dataset_builder.build_molecule_clusters(bundle_path)

    assert result.status == "dependency_missing"
    assert result.clusters == ()
    assert result.training_ready is False
