"""Offline entity identity, sequence, provenance, and bundle contracts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path

import pytest

import data_contracts
import evidence_builder as builder


APL_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _fixture():
    with FIXTURE_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _plain(record):
    return asdict(record) if is_dataclass(record) else dict(record)


def _single_component():
    fixture = _fixture()
    return builder.resolve_target_components(
        fixture["chembl"]["targets"][0],
        source_kind="synthetic_test",
        source_path="raw/source_manifest.json",
        source_sha256="a" * 64,
    )[0]


def _protein_record_type():
    record_type = getattr(data_contracts, "ProteinRecord", None)
    assert record_type is not None, "shared ProteinRecord contract is required"
    return record_type


def test_shared_protein_record_is_strictly_identity_and_sequence_bound():
    record_type = _protein_record_type()
    validate = getattr(data_contracts, "validate_protein_record", None)
    assert callable(validate), "data_contracts must expose ProteinRecord validation"
    sequence = "ACDE"
    record = record_type(
        accession="P12345",
        taxonomy_id=9606,
        raw_sequence="ac de",
        full_sequence=sequence,
        sequence_length=4,
        sequence_sha256=hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        sequence_status="verified",
        sequence_status_reason=None,
        identity_status="verified",
        source_kind="synthetic_test",
        source_provenance=[{
            "source_path": "raw/uniprot/P12345.json",
            "raw_sha256": "b" * 64,
        }],
        component_evidence={"target_chembl_id": "CHEMBL900201"},
    )

    assert validate(record) is None
    with pytest.raises((TypeError, ValueError)):
        validate(replace(record, sequence_length=3))
    with pytest.raises((TypeError, ValueError)):
        validate(replace(record, taxonomy_id="9606"))
    with pytest.raises((TypeError, ValueError)):
        validate(replace(record, sequence_sha256="0" * 64))


def test_rdkit_molecule_normalization_keeps_raw_structure_and_native_identity():
    fixture = _fixture()
    make_record = getattr(builder, "molecule_record_from_source", None)
    assert callable(make_record), "MoleculeRecord normalization API is required"
    source_hash = "c" * 64
    first = make_record(
        fixture["chembl"]["molecules"][0],
        source_kind="synthetic_test",
        source_path="raw/source_manifest.json",
        source_sha256=source_hash,
    )
    second = make_record(
        fixture["chembl"]["molecules"][1],
        source_kind="synthetic_test",
        source_path="raw/source_manifest.json",
        source_sha256=source_hash,
    )
    first_data = _plain(first)
    second_data = _plain(second)

    assert first_data["molecule_chembl_id"] == "CHEMBL151"
    assert first_data["raw_structure"]["canonical_smiles"] == fixture["chembl"]["molecules"][0]["molecule_structures"]["canonical_smiles"]
    assert first_data["canonical_smiles"] == "O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
    assert second_data["molecule_chembl_id"] == "CHEMBL123456"
    assert first_data["canonical_smiles"] != second_data["canonical_smiles"]
    assert first_data["structure_status"] == second_data["structure_status"] == "verified"

    missing_id = make_record(
        {"pref_name": "No identifier", "molecule_structures": {"canonical_smiles": "CCO"}},
        source_kind="synthetic_test",
        source_path="raw/source_manifest.json",
        source_sha256=source_hash,
    )
    assert _plain(missing_id)["molecule_chembl_id"] is None
    assert _plain(missing_id)["structure_status"] == "unverified"
    assert _plain(missing_id)["canonical_smiles"] is None

    salt = make_record(
        {"molecule_chembl_id": "CHEMBL999999", "molecule_structures": {"canonical_smiles": "CCO.Cl"}},
        source_kind="synthetic_test",
        source_path="raw/source_manifest.json",
        source_sha256=source_hash,
    )
    assert _plain(salt)["structure_status"] == "excluded"
    assert _plain(salt)["structure_status_reason"] == "multiple_fragments"
    assert _plain(salt)["canonical_smiles"] is None


def test_target_component_resolution_requires_human_single_protein_and_accession():
    fixture = _fixture()
    resolver = getattr(builder, "resolve_target_components", None)
    assert callable(resolver), "component-to-accession resolver is required"
    source_hash = "d" * 64
    single = resolver(
        fixture["chembl"]["targets"][0], source_kind="synthetic_test",
        source_path="raw/source_manifest.json", source_sha256=source_hash,
    )
    assert len(single) == 1
    assert single[0]["resolution_status"] == "verified"
    assert single[0]["accession"] == "P12345"
    assert single[0]["taxonomy_id"] == 9606
    assert single[0]["taxonomy_evidence_path"] in {"target.tax_id", "target_components[0].component_tax_id"}
    assert single[0]["identity_key"] == [9606, "P12345"]

    complex_rows = resolver(
        fixture["chembl"]["targets"][1], source_kind="synthetic_test",
        source_path="raw/source_manifest.json", source_sha256=source_hash,
    )
    assert len(complex_rows) == 2
    assert all(row["resolution_status"] == "excluded" for row in complex_rows)
    assert all(row["reason_code"] == "target_type_not_single_protein" for row in complex_rows)

    rat_rows = resolver(
        fixture["chembl"]["targets"][2], source_kind="synthetic_test",
        source_path="raw/source_manifest.json", source_sha256=source_hash,
    )
    assert rat_rows[0]["resolution_status"] == "excluded"
    assert rat_rows[0]["reason_code"] == "non_human_taxonomy"

    symbol_only = resolver({
        "target_chembl_id": "CHEMBL999998",
        "target_type": "SINGLE PROTEIN",
        "organism": "Homo sapiens",
        "tax_id": 9606,
        "target_components": [{"component_type": "PROTEIN", "component_synonyms": ["AKT1"]}],
    }, source_kind="synthetic_test", source_path="raw/source_manifest.json", source_sha256=source_hash)
    assert symbol_only[0]["accession"] is None
    assert symbol_only[0]["resolution_status"] == "unverified"
    assert symbol_only[0]["reason_code"] == "missing_component_accession"


def test_uniprot_json_and_fasta_plus_taxonomy_metadata_hash_the_complete_sequence(tmp_path):
    fixture = _fixture()
    hydrate = getattr(builder, "hydrate_protein_record", None)
    assert callable(hydrate), "source-bound sequence hydration API is required"
    component = _single_component()
    payload = fixture["uniprot"]["P12345"]
    json_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_path = tmp_path / "P12345.json"
    json_path.write_bytes(json_bytes)
    from_json = _plain(hydrate(
        component,
        source_kind="synthetic_test",
        source_path="raw/uniprot/P12345.json",
        uniprot_json=payload,
        raw_response_bytes=json_bytes,
    ))

    assert from_json["accession"] == "P12345"
    assert from_json["taxonomy_id"] == 9606
    assert from_json["raw_sequence"] == "ACDEFGHIKLMNPQRSTVWY"
    assert from_json["full_sequence"] == "ACDEFGHIKLMNPQRSTVWY"
    assert from_json["sequence_length"] == 20
    assert from_json["sequence_sha256"] == hashlib.sha256(from_json["full_sequence"].encode("utf-8")).hexdigest()
    assert from_json["sequence_status"] == "verified"
    assert from_json["source_provenance"][0]["raw_sha256"] == hashlib.sha256(json_bytes).hexdigest()

    fasta_path = tmp_path / "P12345.fasta"
    fasta_path.write_text(">sp|P12345|SYNTHETIC_HUMAN\nACDEFGHIKL\nMNPQRSTVWY\n", encoding="utf-8")
    metadata_path = tmp_path / "P12345-metadata.json"
    metadata_path.write_text(json.dumps({
        "primaryAccession": "P12345",
        "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
        "sequence": {"length": 20},
    }), encoding="utf-8")
    from_fasta = _plain(hydrate(
        component,
        source_kind="synthetic_test",
        fasta_path=fasta_path,
        taxonomy_metadata_path=metadata_path,
    ))
    assert from_fasta["sequence_status"] == "verified"
    assert from_fasta["full_sequence"] == from_json["full_sequence"]
    assert from_fasta["sequence_sha256"] == from_json["sequence_sha256"]
    assert len(from_fasta["source_provenance"]) == 2


def test_wrong_accession_taxonomy_length_and_nonstandard_residues_stay_unverified_or_invalid():
    fixture = _fixture()
    hydrate = getattr(builder, "hydrate_protein_record", None)
    assert callable(hydrate), "source-bound sequence hydration API is required"
    component = _single_component()

    cases = [
        ("P99999", 9606, "ACDE", 4, "accession_mismatch"),
        ("P12345", 10090, "ACDE", 4, "taxonomy_mismatch"),
        ("P12345", 9606, "ACDE", 5, "declared_length_mismatch"),
        ("P12345", 9606, "ACDX", 4, "nonstandard_residue"),
    ]
    for accession, taxonomy, sequence, length, reason in cases:
        raw = {
            "primaryAccession": accession,
            "organism": {"scientificName": "Homo sapiens", "taxonId": taxonomy},
            "sequence": {"value": sequence, "length": length},
        }
        raw_bytes = json.dumps(raw, separators=(",", ":")).encode("utf-8")
        result = _plain(hydrate(
            component,
            source_kind="synthetic_test",
            source_path="raw/uniprot/returned.json",
            uniprot_json=raw,
            raw_response_bytes=raw_bytes,
        ))
        assert result["sequence_status"] == "invalid"
        assert result["sequence_status_reason"] == reason
        assert result["identity_status"] != "verified" or reason == "declared_length_mismatch"

    missing = _plain(hydrate(
        component,
        source_kind="synthetic_test",
        source_path="raw/uniprot/missing-sequence.json",
        uniprot_json={
            "primaryAccession": "P12345",
            "organism": {"taxonId": 9606},
            "sequence": {"length": 20},
        },
        raw_response_bytes=b'{"missing":"sequence"}',
    ))
    assert missing["sequence_status"] == "missing"
    assert missing["sequence_status_reason"] == "sequence_value_missing"


def test_local_bundle_contains_canonical_molecules_proteins_exclusions_and_verifiable_hashes(tmp_path, capsys):
    ingest = getattr(builder, "ingest_local_manifest", None)
    assert callable(ingest), "local manifest builder is required"
    verify_cli = getattr(builder, "main", None)
    assert callable(verify_cli), "entity bundle verification CLI is required"
    output_dir = tmp_path / "bundle-one"
    result = ingest(FIXTURE_PATH, output_dir)
    bundle_path = output_dir / "entities.json"
    bundle = builder.load_entity_bundle(bundle_path)

    assert result["status"] == "complete"
    assert bundle["schema_version"] == 2
    assert bundle["entity_version"] == result["entity_version"]
    assert bundle["content_sha256"] == result["content_sha256"]
    assert {row["molecule_chembl_id"] for row in bundle["molecules"]} == {
        "CHEMBL151", "CHEMBL123456",
    }
    assert all(row["structure_status"] == "verified" for row in bundle["molecules"])
    assert {row["accession"] for row in bundle["proteins"]} == {"P12345"}
    protein = bundle["proteins"][0]
    assert protein["taxonomy_id"] == 9606
    assert protein["sequence_sha256"] == hashlib.sha256(protein["full_sequence"].encode("utf-8")).hexdigest()
    assert len(bundle["activities"]) == 4
    assert all(row["entity_bundle_hash"] == bundle["content_sha256"] for row in bundle["activities"])
    assert all(row["entity_bundle_version"] == bundle["entity_version"] for row in bundle["activities"])
    assert any(row["reason_code"] == "target_type_not_single_protein" for row in bundle["exclusions"])
    assert "targets" not in bundle

    second_output = tmp_path / "bundle-two"
    second_result = ingest(FIXTURE_PATH, second_output)
    assert second_result["content_sha256"] == result["content_sha256"]

    exit_code = verify_cli(["verify-bundle", str(bundle_path)])
    cli_result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert cli_result["status"] == "verified"
    assert cli_result["content_sha256"] == bundle["content_sha256"]


def test_bundle_loader_rejects_content_and_raw_source_tampering(tmp_path):
    ingest = getattr(builder, "ingest_local_manifest", None)
    assert callable(ingest), "local manifest builder is required"
    output_dir = tmp_path / "tamper-content"
    ingest(FIXTURE_PATH, output_dir)
    bundle_path = output_dir / "entities.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle.get("molecules"), "versioned entity bundle must include normalized molecules"
    bundle["molecules"][0]["canonical_smiles"] = "CCO"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        builder.load_entity_bundle(bundle_path)

    second_output = tmp_path / "tamper-source"
    ingest(FIXTURE_PATH, second_output)
    second_bundle = builder.load_entity_bundle(second_output / "entities.json")
    source = next(item for item in second_bundle["source_provenance"] if item["source_path"].endswith("page-0001.json"))
    with (second_output / source["source_path"]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="raw source hash mismatch"):
        builder.load_entity_bundle(second_output / "entities.json")


def test_default_sequence_hydration_is_offline_and_online_get_is_bounded(tmp_path, monkeypatch):
    hydrate = getattr(builder, "hydrate_protein_record", None)
    assert callable(hydrate), "source-bound sequence hydration API is required"
    component = _single_component()
    calls = []

    class Response:
        def __init__(self, payload, url):
            self.url = url
            self.status_code = 200
            self.body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.headers = {"Content-Length": str(len(self.body)), "Content-Type": "application/json"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield self.body

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response(_fixture()["uniprot"]["P12345"], url)

    monkeypatch.setattr("evidence_builder.requests.get", get)
    offline = _plain(hydrate(component))
    assert calls == []
    assert offline["sequence_status"] in {"missing", "unverified"}

    online = _plain(hydrate(
        component,
        online=True,
        output_root=tmp_path / "online-snapshot",
        timeout=7,
        max_requests=1,
        max_response_bytes=100_000,
    ))
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://rest.uniprot.org/uniprotkb/P12345.json"
    assert kwargs["allow_redirects"] is False
    assert kwargs["stream"] is True
    assert kwargs["timeout"] <= 7
    assert online["sequence_status"] == "verified"
    assert (tmp_path / "online-snapshot" / "raw" / "uniprot" / "P12345.json").is_file()
