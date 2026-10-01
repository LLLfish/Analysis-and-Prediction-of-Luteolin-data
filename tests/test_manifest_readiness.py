"""TDD contracts for immutable frozen artifacts and truthful source readiness."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import pytest

import data_contracts
import dataset_builder
import evidence_builder
import leak_audit


APL_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"
READY_MOCK_SMILES = (
    "C1CCCCC1", "C1CCOC1", "C1CCNC1", "C1CCSC1", "C1COCCO1",
    "C1COCCN1", "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
    "c1ncc[nH]1", "c1ccc2ccccc2c1", "c1ccc2ncccc2c1", "C1C2CC3CC1CC(C2)C3",
    "c1cncnc1", "c1cnccn1", "c1ccnnc1", "c1ncncn1", "c1cn[nH]c1",
    "c1cc[nH]c1",
)


def _store():
    spec = importlib.util.find_spec("artifact_store")
    assert spec is not None, "planned artifact_store module should be present"
    return importlib.import_module("artifact_store")


def _source_audit():
    spec = importlib.util.find_spec("source_audit")
    assert spec is not None, "planned source_audit module should be present"
    return importlib.import_module("source_audit")


def _policy_bytes_and_hashes(root: Path | None = None) -> tuple[dict, dict]:
    policy_root = APL_ROOT / "policies"
    references = {}
    claims = {}
    for name, filename in (("split", "split_policy.json"), ("label", "label_policy.json")):
        raw = (policy_root / filename).read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
        semantic_sha256 = data_contracts.canonical_json_sha256(value)
        references[name] = {
            "path": f"policies/{filename}",
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "semantic_sha256": semantic_sha256,
            "schema_version": value["schema_version"],
            "policy_version": value["policy_version"],
        }
        claims[f"{name}_policy_file_sha256"] = hashlib.sha256(raw).hexdigest()
        claims[f"{name}_policy_semantic_sha256"] = semantic_sha256
        claims[f"{name}_policy_schema_version"] = value["schema_version"]
        claims[f"{name}_policy_version"] = value["policy_version"]
        if root is not None:
            destination = root / references[name]["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
    claims["label_policy_sha256"] = references["label"]["file_sha256"]
    return references, claims


def _empty_verified_bundle(tmp_path: Path) -> tuple[Path, dict]:
    bundle_root = tmp_path / "source-bundle"
    raw_root = bundle_root / "raw"
    raw_root.mkdir(parents=True)
    raw_bytes = b'{"fixture":"synthetic_test","metadata":"source-receipt"}'
    (raw_root / "metadata.json").write_bytes(raw_bytes)
    provenance = [{
        "source_id": "fixture:metadata",
        "source_path": "raw/metadata.json",
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_kind": "synthetic_test",
    }]
    evidence_builder.write_entity_bundle(
        bundle_root,
        activities=[],
        source_provenance=provenance,
        molecules=[],
        proteins=[],
        source_kind="synthetic_test",
    )
    entity_path = bundle_root / "entities.json"
    return entity_path, evidence_builder.load_entity_bundle(entity_path, verify_sources=True)


def _split_payload(bundle_path: Path, bundle: dict) -> dict:
    result = dataset_builder.strict_double_cold_split(bundle_path, allow_test_only=True)
    _references, policy_claims = _policy_bytes_and_hashes()
    return {
        "strict_split_status": result.strict_split_status,
        "split_policy_sha256": result.policy_sha256,
        **policy_claims,
        "entity_version": bundle["entity_version"],
        "entity_bundle_hash": bundle["content_sha256"],
        "molecule_memberships": {},
        "protein_memberships": {},
        "pair_assignments": [],
        "crosspair_quarantine": [],
        "unassigned_pairs": [],
        "test_only_pair_assignments": [],
        "ambiguous_label_quarantine": [],
        "class_support": {name: {"active": 0, "inactive": 0} for name in ("train", "val", "test")},
        "training_ready": False,
    }


def _source_verified_ready_mock_bundle(tmp_path: Path) -> Path:
    """Build an online-shaped local producer fixture for tool-only readiness tests."""
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["source_kind"] = "online_snapshot"
    payload["purpose"] = "synthetic_test_only_tool_capability_fixture"
    chembl = payload["chembl"]
    base_molecule = deepcopy(chembl["molecules"][1])
    base_activity = deepcopy(chembl["activities_by_molecule"]["CHEMBL123456"]["pages"][0]["activities"][0])
    base_page = deepcopy(chembl["activities_by_molecule"]["CHEMBL123456"]["pages"][0])
    base_target = deepcopy(chembl["targets"][0])
    base_component = deepcopy(payload["uniprot"]["P12345"])
    base_assay = deepcopy(next(row for row in chembl["assays"] if row["assay_chembl_id"] == "CHEMBL910001"))
    base_document = deepcopy(next(row for row in chembl["documents"] if row["document_chembl_id"] == "CHEMBL910101"))
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    molecules = []
    targets = []
    assays = []
    documents = []
    uniprot = {}
    activity_pages = {}
    for molecule_index, smiles in enumerate(READY_MOCK_SMILES):
        molecule_id = f"CHEMBL920{molecule_index + 1:03d}"
        molecule = deepcopy(base_molecule)
        molecule.update(molecule_chembl_id=molecule_id, pref_name=f"tool fixture molecule {molecule_index + 1}")
        molecule["molecule_structures"]["canonical_smiles"] = smiles
        molecules.append(molecule)

        activity_rows = []
        for protein_index in range(len(READY_MOCK_SMILES)):
            accession = f"P{protein_index + 1:05d}"
            target_id = f"CHEMBL930{protein_index + 1:03d}"
            assay_id = f"CHEMBL940{protein_index + 1:03d}"
            document_id = f"CHEMBL950{protein_index + 1:03d}"
            if molecule_index == 0:
                target = deepcopy(base_target)
                target.update(target_chembl_id=target_id, pref_name=f"tool fixture target {protein_index + 1}")
                target["target_components"] = [{
                    "component_id": protein_index + 1,
                    "accession": accession,
                    "component_type": "PROTEIN",
                    "organism": "Homo sapiens",
                }]
                targets.append(target)

                protein = deepcopy(base_component)
                protein["primaryAccession"] = accession
                sequence = "".join(random.Random(7000 + protein_index).choices(amino_acids, k=60))
                protein["sequence"]["value"] = sequence
                protein["sequence"]["length"] = len(sequence)
                uniprot[accession] = protein

                assay = deepcopy(base_assay)
                assay.update(assay_chembl_id=assay_id, target_chembl_id=target_id, confidence_score=9)
                assays.append(assay)
                document = deepcopy(base_document)
                document.update(document_chembl_id=document_id, doi=f"10.0000/tool-fixture.{protein_index + 1:03d}")
                documents.append(document)

            row = deepcopy(base_activity)
            row.update(
                activity_id=97000000 + molecule_index * len(READY_MOCK_SMILES) + protein_index,
                molecule_chembl_id=molecule_id,
                target_chembl_id=target_id,
                assay_chembl_id=assay_id,
                document_chembl_id=document_id,
                standard_type="Ki",
                standard_relation="=",
                standard_value="100" if (molecule_index + protein_index) % 2 == 0 else "100000",
                standard_units="nM",
            )
            activity_rows.append(row)

        if molecule_index == 0:
            # One measured but independent-only endpoint remains quarantined;
            # train/val/test qualified support remains supplied by other pairs.
            activity_rows[0]["standard_type"] = "IC50"

        page = deepcopy(base_page)
        page["molecule_chembl_id"] = molecule_id
        page["activities"] = activity_rows
        page["page_meta"].update(
            limit=len(activity_rows), offset=0, next=None, previous=None,
            total_count=len(activity_rows),
        )
        activity_pages[molecule_id] = {"pages": [page]}

    unverified_target_id = "CHEMBL930999"
    unverified_assay_id = "CHEMBL940999"
    unverified_document_id = "CHEMBL950999"
    unverified_target = deepcopy(base_target)
    unverified_target.update(
        target_chembl_id=unverified_target_id,
        pref_name="tool fixture target with no sequence receipt",
        target_components=[{
            "component_id": 999,
            "accession": "P99999",
            "component_type": "PROTEIN",
            "organism": "Homo sapiens",
        }],
    )
    targets.append(unverified_target)
    unverified_assay = deepcopy(base_assay)
    unverified_assay.update(assay_chembl_id=unverified_assay_id, target_chembl_id=unverified_target_id, confidence_score=9)
    assays.append(unverified_assay)
    unverified_document = deepcopy(base_document)
    unverified_document.update(document_chembl_id=unverified_document_id, doi="10.0000/tool-fixture.unverified")
    documents.append(unverified_document)
    unverified_row = deepcopy(base_activity)
    unverified_row.update(
        activity_id=97999999,
        molecule_chembl_id="CHEMBL920001",
        target_chembl_id=unverified_target_id,
        assay_chembl_id=unverified_assay_id,
        document_chembl_id=unverified_document_id,
        standard_type="Ki",
        standard_relation="=",
        standard_value="100",
        standard_units="nM",
    )
    unverified_page = activity_pages["CHEMBL920001"]["pages"][0]
    unverified_page["activities"].append(unverified_row)
    unverified_page["page_meta"]["limit"] += 1
    unverified_page["page_meta"]["total_count"] += 1

    chembl["molecules"] = molecules
    chembl["targets"] = targets
    chembl["assays"] = assays
    chembl["documents"] = documents
    chembl["activities_by_molecule"] = activity_pages
    payload["uniprot"] = uniprot

    source_manifest = tmp_path / "ready-tool-fixture.json"
    source_manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    staging = tmp_path / "ready-tool-producer"
    result = evidence_builder.ingest_local_manifest(source_manifest, staging)
    assert result["status"] == "complete", result
    bundle_path = staging / "entities.json"
    assert evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)["source_kind"] == "online_snapshot"
    molecule_clusters = dataset_builder.build_molecule_clusters(bundle_path)
    protein_clusters = dataset_builder.build_protein_clusters(bundle_path)
    assert len(molecule_clusters.clusters) == len(READY_MOCK_SMILES)
    assert len(protein_clusters.clusters) == len(READY_MOCK_SMILES)
    return bundle_path


def _fake_legacy_sources(root: Path) -> None:
    data_raw = root / "data" / "raw"
    framework_raw = root / "framework_output" / "data" / "raw"
    data_raw.mkdir(parents=True)
    framework_raw.mkdir(parents=True)
    (data_raw / "chembl_luteolin_targets.json").write_text(json.dumps({
        "targets": [{"target_chembl_id": "CHEMBL900201", "activity_value": "100", "activity_unit": "nM"}],
        "source": "legacy",
    }), encoding="utf-8")
    (data_raw / "uniprot_proteins.json").write_text(json.dumps({
        "proteins": [{"accession": "P12345", "length": 18, "organism": "Homo sapiens"}],
    }), encoding="utf-8")
    (data_raw / "tcmsp_luteolin.json").write_text(json.dumps({
        "source": "online", "targets": [{"name": "target"}],
    }), encoding="utf-8")
    (framework_raw / "chembl_luteolin_targets.json").write_text(json.dumps({
        "targets": [{"target_chembl_id": "CHEMBL900201", "pchembl_value": "7.0"}],
    }), encoding="utf-8")
    (root / "offline_data.json").write_text(json.dumps({
        "compound": {"name": "Luteolin"},
        "targets": [{"gene_symbol": "EGFR", "target_chembl_id": "CHEMBL1"}],
        "proteins": {"EGFR": {"uniprot_id": "P00533", "length": 1210}},
        "interactions": [{"protein_a": "EGFR", "protein_b": "GRB2"}],
        "tcmsp": {"source": "online", "targets": [
            {"target_chembl_id": "CHEMBL2"}, {"target_chembl_id": "CHEMBL3"},
        ]},
    }), encoding="utf-8")


def test_semantic_manifest_hash_ignores_timestamps_but_keeps_source_bytes_and_policy():
    first = {
        "schema_version": 1,
        "generated_at": "2026-10-01T00:00:00Z",
        "entities": {"file_sha256": "a" * 64, "content_sha256": "b" * 64},
        "source_provenance": [{"raw_sha256": "c" * 64, "retrieved_at": "2026-09-30T00:00:00Z"}],
        "policy_sha256": "d" * 64,
    }
    second = json.loads(json.dumps(first))
    second["generated_at"] = "2026-10-02T00:00:00Z"
    second["source_provenance"][0]["retrieved_at"] = "2026-10-02T00:00:00Z"

    digest = _store().semantic_manifest_hash

    assert digest(first) == digest(second)
    second["entities"]["file_sha256"] = "e" * 64
    assert digest(first) != digest(second)


def test_frozen_writer_copies_exact_entities_and_all_raw_bytes_and_audits_the_published_directory(tmp_path):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    output_root = tmp_path / "frozen"
    output_root.mkdir()
    writer = _store().FrozenManifestWriter(output_root, version_id="v1", code_root=APL_ROOT)

    result = writer.publish(bundle_path, split=split)

    assert result.status == "complete"
    manifest_path = Path(result.manifest_path)
    frozen_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (frozen_root / manifest["entities"]["path"]).read_bytes() == bundle_path.read_bytes()
    code_state = manifest["code_version_control"]
    head = subprocess.run(["git", "-C", str(APL_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert code_state["status"] == "git"
    assert code_state["branch"] == "codex/p0-data-foundation"
    assert code_state["commit"] == head
    assert code_state["dirty"] is True
    assert "output_dashboard.html" in code_state["dirty_paths"]
    for source in bundle["source_provenance"]:
        relative = Path(source["source_path"])
        assert (frozen_root / relative).read_bytes() == (bundle_path.parent / relative).read_bytes()
        assert hashlib.sha256((frozen_root / relative).read_bytes()).hexdigest() == source["raw_sha256"]
    assert manifest["activities"]["entity_bundle_hash"] == bundle["content_sha256"]
    assert manifest["pairs"]["entity_bundle_hash"] == bundle["content_sha256"]
    assert manifest["split"]["entity_bundle_hash"] == bundle["content_sha256"]
    report = leak_audit.audit_split(manifest_path)
    assert report.exit_code == 0
    assert report.training_ready is False


def test_competing_writers_have_one_success_and_never_publish_a_partial_manifest(tmp_path, monkeypatch):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    output_root = tmp_path / "race"
    output_root.mkdir()
    store = _store()
    writer_type = store.FrozenManifestWriter
    writers = [writer_type(output_root, version_id="same-version") for _ in range(2)]
    audit_started = threading.Event()
    release_audit = threading.Event()
    original_audit = store.audit_split

    def hold_first_writer(manifest_path):
        audit_started.set()
        if not release_audit.wait(timeout=10):
            raise TimeoutError("test did not release first writer")
        return original_audit(manifest_path)

    monkeypatch.setattr(store, "audit_split", hold_first_writer)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(writers[0].publish, bundle_path, split=split)
        assert audit_started.wait(timeout=10)
        second = writers[1].publish(bundle_path, split=split)
        release_audit.set()
        first = first_future.result(timeout=30)
        results = [first, second]

    statuses = [result.status for result in results]
    assert statuses.count("complete") == 1
    assert sum(result.exit_code == 0 for result in results) == 1
    assert all(result.exit_code in {0, 2} for result in results)
    assert all(status in {"complete", "busy", "conflict"} for status in statuses)
    final_dir = output_root / "same-version"
    if "complete" in statuses:
        manifest_path = final_dir / "manifest.json"
        assert manifest_path.is_file()
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["semantic_sha256"]


def test_interruption_before_manifest_commit_leaves_no_completed_version_and_cleans_only_own_temps(tmp_path, monkeypatch):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    output_root = tmp_path / "interrupted"
    output_root.mkdir()
    store = _store()
    writer = store.FrozenManifestWriter(output_root, version_id="interrupted-v1")
    original_replace = store.os.replace

    def interrupt_manifest(source, destination):
        if Path(destination).name == "manifest.json":
            raise OSError("injected interruption before final manifest replace")
        return original_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", interrupt_manifest)
    result = writer.publish(bundle_path, split=split)

    version_dir = output_root / "interrupted-v1"
    assert result.status == "incomplete"
    assert not (version_dir / "manifest.json").exists()
    assert (version_dir / "entities.json").is_file()
    assert not list(version_dir.glob("*.tmp.*"))


def test_interruption_during_nested_snapshot_copy_cleans_its_temp_without_publishing(tmp_path, monkeypatch):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    output_root = tmp_path / "nested-interruption"
    output_root.mkdir()
    store = _store()
    writer = store.FrozenManifestWriter(output_root, version_id="nested-interruption-v1")
    original_replace = store.os.replace

    def interrupt_raw_copy(source, destination):
        if Path(destination).name == "metadata.json":
            raise OSError("injected interruption during nested raw snapshot replace")
        return original_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", interrupt_raw_copy)
    result = writer.publish(bundle_path, split=split)

    version_dir = output_root / "nested-interruption-v1"
    assert result.status == "incomplete"
    assert not (version_dir / "manifest.json").exists()
    assert (version_dir / "entities.json").is_file()
    assert not list(version_dir.rglob("*.tmp.*"))
    assert not (version_dir / "raw" / "metadata.json").exists()


def test_input_change_after_independent_audit_blocks_manifest_commit(tmp_path, monkeypatch):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    output_root = tmp_path / "source-change"
    output_root.mkdir()
    store = _store()
    writer = store.FrozenManifestWriter(output_root, version_id="changed-input-v1")
    original_audit = store.audit_split
    source_path = bundle_path.parent / "raw" / "metadata.json"

    def change_source_after_audit(manifest_path):
        report = original_audit(manifest_path)
        source_path.write_bytes(source_path.read_bytes() + b"changed-during-publication")
        return report

    monkeypatch.setattr(store, "audit_split", change_source_after_audit)
    result = writer.publish(bundle_path, split=split)

    assert result.status == "incomplete"
    assert not (output_root / "changed-input-v1" / "manifest.json").exists()


def test_writer_rejects_output_root_inside_protected_history_without_creating_it(tmp_path):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    protected_target = APL_ROOT / "data" / f"p0_01_06_forbidden_{tmp_path.name}"
    result = _store().FrozenManifestWriter(protected_target, version_id="v1").publish(bundle_path, split=split)

    assert result.exit_code == 2
    assert result.status == "incomplete"
    assert not protected_target.exists()


def test_existing_version_is_immutable_when_input_claim_changes(tmp_path):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    output_root = tmp_path / "immutable"
    output_root.mkdir()
    writer_type = _store().FrozenManifestWriter
    writer = writer_type(output_root, version_id="v1")
    first = writer.publish(bundle_path, split=split)
    manifest_path = Path(first.manifest_path)
    before = manifest_path.read_bytes()
    changed = dict(split)
    changed["strict_split_status"] = "complete"

    second = writer.publish(bundle_path, split=changed)

    assert second.status == "conflict"
    assert second.exit_code == 2
    assert manifest_path.read_bytes() == before


def test_corrupt_existing_manifest_is_conflict_not_success_or_repair(tmp_path):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    output_root = tmp_path / "corrupt-existing"
    writer = _store().FrozenManifestWriter(output_root, version_id="v1")
    first = writer.publish(bundle_path, split=split)
    manifest_path = Path(first.manifest_path)
    manifest_path.write_bytes(b"{}")
    corrupt_bytes = manifest_path.read_bytes()

    assert leak_audit.audit_split(manifest_path).exit_code != 0
    second = writer.publish(bundle_path, split=split)

    assert second.status == "conflict"
    assert second.exit_code == 2
    assert manifest_path.read_bytes() == corrupt_bytes


def test_no_git_input_records_not_initialized_and_null_commit(tmp_path):
    bundle_path, bundle = _empty_verified_bundle(tmp_path)
    split = _split_payload(bundle_path, bundle)
    output_root = tmp_path / "no-git-output"
    writer = _store().FrozenManifestWriter(output_root, version_id="no-git-v1", code_root=tmp_path / "not-a-repository")

    result = writer.publish(bundle_path, split=split)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

    assert result.status == "complete"
    assert manifest["code_version_control"]["status"] == "not_initialized"
    assert manifest["code_version_control"]["commit"] is None


def test_three_legacy_sources_are_reported_separately_and_tcms_is_static_reference(tmp_path):
    apl_root = tmp_path / "fake-apl"
    _fake_legacy_sources(apl_root)
    out_root = tmp_path / "source-reports"

    collection = _source_audit().audit_legacy_sources(apl_root, output_dir=out_root)

    by_id = {row["source_id"]: row for row in collection["sources"]}
    assert set(by_id) == {"apl_data", "framework_output_data", "offline_data"}
    assert all(row["training_ready"] is False for row in by_id.values())
    assert by_id["offline_data"]["observed_counts"]["activity_records"] == 0
    assert by_id["offline_data"]["observed_counts"]["unverified_activity_records"] == 0
    assert by_id["offline_data"]["label_statistics_status"] == "not_evaluated_no_activity_records"
    assert by_id["offline_data"]["observed_counts"]["activity_records_status"] == "no_source_activity_rows_observed"
    assert by_id["offline_data"]["observed_counts"]["confirmed_active"] == 0
    assert by_id["offline_data"]["observed_counts"]["confirmed_inactive"] == 0
    assert by_id["offline_data"]["observed_counts"]["compound_records"] == 1
    assert by_id["offline_data"]["observed_counts"]["protein_records"] == 1
    assert by_id["offline_data"]["observed_counts"]["ppi_edges"] == 1
    assert by_id["offline_data"]["observed_counts"]["distinct_target_ids"] == 1
    for source_id in ("apl_data", "framework_output_data"):
        source = by_id[source_id]
        assert source["observed_counts"]["unverified_activity_records"] == 1
        assert source["observed_counts"]["activity_records_status"] == "unverified_legacy_source_rows"
        assert source["label_statistics_status"] == "not_evaluated"
        assert source["observed_counts"]["confirmed_active"] is None
        assert source["observed_counts"]["confirmed_inactive"] is None
        assert source["observed_counts"]["unknown"] is None
        assert source["observed_counts"]["quarantine"] is None
        assert "source_label_statistics_not_evaluated" in source["reasons"]
        assert "no_source_verified_confirmed_active_measurements" not in source["reasons"]
    offline_file = by_id["offline_data"]["files"][0]
    assert offline_file["embedded_static_reference"]["source_kind"] == "static_reference"
    assert offline_file["embedded_static_reference"]["collection_counts"]["$.targets"] == 2
    assert offline_file["embedded_static_reference"]["label_statistics_status"] == "not_evaluated_static_reference"
    tcmsp = next(file for file in by_id["apl_data"]["files"] if file["relative_path"].endswith("tcmsp_luteolin.json"))
    assert tcmsp["source_kind"] == "static_reference"
    assert by_id["apl_data"]["source_sha256"] != by_id["framework_output_data"]["source_sha256"]
    assert len(collection["report_paths"]) == 3
    assert all(Path(path).is_file() for path in collection["report_paths"].values())
    assert collection["code_version_control"]["status"] == "not_initialized"
    for path in collection["report_paths"].values():
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        assert report["code_version_control"]["status"] == "not_initialized"
        assert report["code_version_control"]["commit"] is None


def test_source_audit_cli_returns_zero_for_successful_not_ready_reports_and_two_for_bad_input(tmp_path):
    apl_root = tmp_path / "fake-apl"
    _fake_legacy_sources(apl_root)
    output_root = tmp_path / "cli-reports"
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "source_audit.py"),
         "--apl-root", str(apl_root), "--out", str(output_root)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, check=False,
    )
    bad = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "source_audit.py"),
         "--apl-root", str(tmp_path / "missing-root"), "--out", str(tmp_path / "bad-output")],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, check=False,
    )

    assert completed.returncode == 0, (completed.stdout or "") + (completed.stderr or "")
    result = json.loads(completed.stdout)
    assert len(result["sources"]) == 3
    assert all(row["training_ready"] is False for row in result["sources"])
    assert bad.returncode == 2


def test_dataset_builder_publishes_only_through_the_frozen_writer(tmp_path, monkeypatch):
    bundle_path, _bundle = _empty_verified_bundle(tmp_path)
    output_root = tmp_path / "builder-output"
    output_root.mkdir()
    called = []
    store = _store()

    class FakeWriter:
        def __init__(self, root, version_id, **kwargs):
            called.append((Path(root), version_id))

        def publish(self, entity_path, *, split, policy_paths=None):
            called.append((Path(entity_path), split["entity_bundle_hash"]))
            called.append(policy_paths)
            return type("Result", (), {"status": "complete", "exit_code": 0})()

    monkeypatch.setattr(store, "FrozenManifestWriter", FakeWriter)
    result = dataset_builder.publish_frozen_dataset(bundle_path, output_root, version_id="builder-v1")

    assert result.status == "complete"
    assert called[0] == (output_root, "builder-v1")
    assert called[1][0] == bundle_path


def test_dataset_builder_cli_returns_zero_for_tool_success_with_training_not_ready(tmp_path):
    bundle_path, _bundle = _empty_verified_bundle(tmp_path)
    output_root = tmp_path / "cli-frozen"
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "dataset_builder.py"),
         "--entities", str(bundle_path), "--out", str(output_root), "--version", "cli-v1"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=45, check=False,
    )

    assert completed.returncode == 0, (completed.stdout or "") + (completed.stderr or "")
    result = json.loads(completed.stdout)
    assert result["status"] == "complete"
    assert result["audit_report"]["training_ready"] is False

    repeated = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "dataset_builder.py"),
         "--entities", str(bundle_path), "--out", str(output_root), "--version", "cli-v1"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=45, check=False,
    )
    assert repeated.returncode == 2, (repeated.stdout or "") + (repeated.stderr or "")
    assert json.loads(repeated.stdout)["status"] == "conflict"


def _custom_policy_sources(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    default_split_path = APL_ROOT / "policies" / "split_policy.json"
    default_label_path = APL_ROOT / "policies" / "label_policy.json"
    split_policy = json.loads(default_split_path.read_text(encoding="utf-8"))
    split_policy["budgets"]["alignment_cells"] = 1_000_001
    label_policy = json.loads(default_label_path.read_text(encoding="utf-8"))
    split_path = tmp_path / "custom-split-policy.json"
    label_path = tmp_path / "custom-label-policy.json"
    split_path.write_text(json.dumps(split_policy, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    label_path.write_text(json.dumps(label_policy, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    assert label_path.read_bytes() != default_label_path.read_bytes()
    assert data_contracts.canonical_json_sha256(json.loads(label_path.read_text(encoding="utf-8"))) == data_contracts.canonical_json_sha256(label_policy)
    return split_path, label_path, split_policy, label_policy


def test_custom_policies_are_frozen_and_default_audit_uses_only_snapshot_bytes(tmp_path):
    bundle_path, _bundle = _empty_verified_bundle(tmp_path)
    custom_split, custom_label, split_value, label_value = _custom_policy_sources(tmp_path)
    output_root = tmp_path / "custom-policy-frozen"
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})

    built = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "dataset_builder.py"),
         "--entities", str(bundle_path), "--out", str(output_root), "--version", "custom-v1",
         "--policy", str(custom_split), "--label-policy", str(custom_label)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60, check=False,
    )

    assert built.returncode == 0, (built.stdout or "") + (built.stderr or "")
    build_result = json.loads(built.stdout)
    assert build_result["status"] == "complete"
    version_dir = output_root / "custom-v1"
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    refs = manifest["policies"]
    assert refs["split"]["path"] == "policies/split_policy.json"
    assert refs["label"]["path"] == "policies/label_policy.json"
    assert (version_dir / refs["split"]["path"]).read_bytes() == custom_split.read_bytes()
    assert (version_dir / refs["label"]["path"]).read_bytes() == custom_label.read_bytes()
    assert refs["split"]["file_sha256"] == hashlib.sha256(custom_split.read_bytes()).hexdigest()
    assert refs["split"]["semantic_sha256"] == data_contracts.canonical_json_sha256(split_value)
    assert refs["split"]["policy_version"] == split_value["policy_version"]
    assert refs["label"]["file_sha256"] == hashlib.sha256(custom_label.read_bytes()).hexdigest()
    assert refs["label"]["semantic_sha256"] == data_contracts.canonical_json_sha256(label_value)
    assert manifest["split"]["split_policy_file_sha256"] == refs["split"]["file_sha256"]
    assert manifest["split"]["split_policy_sha256"] == refs["split"]["semantic_sha256"]
    assert manifest["split"]["label_policy_sha256"] == refs["label"]["file_sha256"]
    assert manifest["split"]["label_policy_semantic_sha256"] == refs["label"]["semantic_sha256"]

    audit_command = [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "leak_audit.py"), "--manifest", str(manifest_path)]
    audited = subprocess.run(
        audit_command, cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60, check=False,
    )
    assert audited.returncode == 0, (audited.stdout or "") + (audited.stderr or "")
    audit_report = json.loads(audited.stdout)
    assert audit_report["split_policy_sha256"] == refs["split"]["semantic_sha256"]
    assert audit_report["split_policy_file_sha256"] == refs["split"]["file_sha256"]
    assert audit_report["label_policy_sha256"] == refs["label"]["file_sha256"]
    assert audit_report["label_policy_semantic_sha256"] == refs["label"]["semantic_sha256"]

    custom_split.write_bytes(b"changed after publication")
    custom_label.unlink()
    after_input_change = subprocess.run(
        audit_command, cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60, check=False,
    )
    assert after_input_change.returncode == 0, (after_input_change.stdout or "") + (after_input_change.stderr or "")
    assert json.loads(after_input_change.stdout)["split_policy_file_sha256"] == refs["split"]["file_sha256"]

    wrong_split = tmp_path / "wrong-split-policy.json"
    wrong_label = tmp_path / "wrong-label-policy.json"
    wrong_split.write_bytes((APL_ROOT / "policies" / "split_policy.json").read_bytes())
    wrong_label.write_bytes((APL_ROOT / "policies" / "label_policy.json").read_bytes())
    for option, wrong_path, finding_code in (
        ("--policy", wrong_split, "split_policy_override_mismatch"),
        ("--label-policy", wrong_label, "label_policy_override_mismatch"),
    ):
        override = subprocess.run(
            [*audit_command, option, str(wrong_path)], cwd=tmp_path, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False,
        )
        assert override.returncode == 3, (override.stdout or "") + (override.stderr or "")
        assert finding_code in {row["code"] for row in json.loads(override.stdout)["findings"]}

    tamper_cases = (
        ("split", "path", "policies/other.json", "split_policy_path_mismatch"),
        ("split", "file_sha256", "0" * 64, "split_policy_snapshot_hash_mismatch"),
        ("split", "semantic_sha256", "1" * 64, "split_policy_semantic_mismatch"),
        ("split", "policy_version", "forged-split-v9", "split_policy_version_mismatch"),
        ("label", "path", "policies/other-label.json", "label_policy_path_mismatch"),
        ("label", "file_sha256", "f" * 64, "label_policy_snapshot_hash_mismatch"),
        ("label", "semantic_sha256", "2" * 64, "label_policy_semantic_mismatch"),
        ("label", "policy_version", "forged-label-v9", "label_policy_version_mismatch"),
    )
    for index, (kind, field, value, finding_code) in enumerate(tamper_cases):
        tampered = json.loads(json.dumps(manifest))
        tampered["policies"][kind][field] = value
        tampered["semantic_sha256"] = _store().semantic_manifest_hash(tampered)
        tampered_path = version_dir / f"tampered-policy-ref-{index}.json"
        tampered_path.write_text(json.dumps(tampered, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        report = leak_audit.audit_split(tampered_path)
        assert report.exit_code == 3
        assert finding_code in {row.code for row in report.findings}

    for index, (field, value, finding_code) in enumerate((
        ("split_policy_file_sha256", "0" * 64, "split_policy_claim_mismatch"),
        ("label_policy_sha256", "f" * 64, "label_policy_claim_mismatch"),
    )):
        tampered = json.loads(json.dumps(manifest))
        tampered["split"][field] = value
        tampered["semantic_sha256"] = _store().semantic_manifest_hash(tampered)
        tampered_path = version_dir / f"tampered-policy-claim-{index}.json"
        tampered_path.write_text(json.dumps(tampered, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        report = leak_audit.audit_split(tampered_path)
        assert report.exit_code == 3
        assert finding_code in {row.code for row in report.findings}

    frozen_label = version_dir / refs["label"]["path"]
    frozen_label.write_bytes(frozen_label.read_bytes() + b"\n")
    label_tamper = leak_audit.audit_split(manifest_path)
    assert label_tamper.exit_code == 3
    assert "label_policy_snapshot_hash_mismatch" in {row.code for row in label_tamper.findings}

    frozen_split = version_dir / refs["split"]["path"]
    frozen_split.write_bytes(frozen_split.read_bytes() + b"\n")
    frozen_tamper = leak_audit.audit_split(manifest_path)
    assert frozen_tamper.exit_code == 3
    assert "split_policy_snapshot_hash_mismatch" in {row.code for row in frozen_tamper.findings}


def test_custom_policy_input_change_during_publication_leaves_no_manifest(tmp_path, monkeypatch):
    bundle_path, _bundle = _empty_verified_bundle(tmp_path)
    custom_split, custom_label, _split_value, _label_value = _custom_policy_sources(tmp_path)
    output_root = tmp_path / "changing-policy-output"
    store = _store()
    original_audit = store.audit_split
    changed = False

    def change_policy_after_provisional_audit(manifest_path, **kwargs):
        nonlocal changed
        report = original_audit(manifest_path, **kwargs)
        if not changed:
            custom_split.write_bytes(custom_split.read_bytes() + b"\n")
            changed = True
        return report

    monkeypatch.setattr(store, "audit_split", change_policy_after_provisional_audit)
    result = dataset_builder.publish_frozen_dataset(
        bundle_path, output_root, version_id="policy-change-v1",
        policy_path=custom_split, label_policy_path=custom_label,
    )

    assert changed is True
    assert result.exit_code == 2
    assert result.status == "incomplete"
    assert not (output_root / "policy-change-v1" / "manifest.json").exists()


def test_public_cli_promotes_builder_provisional_claim_only_after_independent_ready_audit(tmp_path):
    bundle_path = _source_verified_ready_mock_bundle(tmp_path)
    output_root = tmp_path / "ready-frozen"
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "dataset_builder.py"),
         "--entities", str(bundle_path), "--out", str(output_root), "--version", "ready-v1"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=90, check=False,
    )

    assert completed.returncode == 0, (completed.stdout or "") + (completed.stderr or "")
    result = json.loads(completed.stdout)
    assert result["status"] == "complete"
    assert result["audit_report"]["training_ready"] is True
    manifest_path = output_root / "ready-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split"]["training_ready"] is True
    assert manifest["audit"]["status"] == "clean"
    assert manifest["split"]["crosspair_quarantine"]
    assert manifest["split"]["unassigned_pairs"]
    independent = leak_audit.audit_split(manifest_path)
    assert independent.exit_code == 0
    assert independent.training_ready is True

    tampered = json.loads(json.dumps(manifest))
    tampered["training_ready"] = False
    tampered["split"]["training_ready"] = False
    tampered["semantic_sha256"] = _store().semantic_manifest_hash(tampered)
    tampered_path = manifest_path.parent / "tampered-ready-state.json"
    tampered_path.write_text(json.dumps(tampered, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    rejected = leak_audit.audit_split(tampered_path)
    assert rejected.exit_code == 3
    assert any(row.code == "training_ready_mismatch" for row in rejected.findings)


def test_dataset_builder_cli_incomplete_version_argument_returns_two_without_output(tmp_path):
    bundle_path, _bundle = _empty_verified_bundle(tmp_path)
    output_root = tmp_path / "invalid-version-output"
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(APL_ROOT / "dataset_builder.py"),
         "--entities", str(bundle_path), "--out", str(output_root), "--version", "../escape"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=45, check=False,
    )

    assert completed.returncode == 2
    assert not output_root.exists()
