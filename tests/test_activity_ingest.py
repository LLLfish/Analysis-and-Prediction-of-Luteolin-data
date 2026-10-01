"""Offline contracts for complete ChEMBL evidence ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from data_collection import LuteolinDataCollector


APL_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _fixture():
    with FIXTURE_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


class _Response:
    def __init__(self, payload, url, status_code=200):
        self.payload = payload
        self.url = url
        self.status_code = status_code
        self.content = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(self.content)),
        }
        self.text = self.content.decode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size=8192):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]


def _chembl_get_from_fixture(fixture, calls):
    target_by_id = {
        row["target_chembl_id"]: row for row in fixture["chembl"]["targets"]
    }
    assay_by_id = {
        row["assay_chembl_id"]: row for row in fixture["chembl"]["assays"]
    }
    document_by_id = {
        row["document_chembl_id"]: row for row in fixture["chembl"]["documents"]
    }
    molecule_by_id = {
        row["molecule_chembl_id"]: row for row in fixture["chembl"]["molecules"]
    }

    def fake_get(url, *args, **kwargs):
        calls.append((url, kwargs))
        parsed = urlparse(url)
        if parsed.hostname not in {"www.ebi.ac.uk", "tcmsp-e.com"}:
            raise AssertionError(f"unexpected request host: {parsed.hostname}")
        path = parsed.path
        if path.endswith("/molecule/search.json"):
            payload = {"molecules": [molecule_by_id["CHEMBL151"]]}
        elif "/molecule/" in path and path.endswith(".json"):
            molecule_id = path.rsplit("/", 1)[-1].removesuffix(".json")
            payload = molecule_by_id[molecule_id]
        elif path.endswith("/activity.json"):
            query = parse_qs(parsed.query)
            molecule_id = query.get("molecule_chembl_id", ["CHEMBL151"])[0]
            offset = int(query.get("offset", ["0"])[0])
            pages = fixture["chembl"]["activities_by_molecule"][molecule_id]["pages"]
            payload = next(
                page for page in pages if page["page_meta"]["offset"] == offset
            )
        elif "/target/" in path and path.endswith(".json"):
            target_id = path.rsplit("/", 1)[-1].removesuffix(".json")
            payload = target_by_id[target_id]
        elif "/assay/" in path and path.endswith(".json"):
            assay_id = path.rsplit("/", 1)[-1].removesuffix(".json")
            payload = assay_by_id[assay_id]
        elif "/document/" in path and path.endswith(".json"):
            document_id = path.rsplit("/", 1)[-1].removesuffix(".json")
            payload = document_by_id[document_id]
        else:
            raise AssertionError(f"unexpected request path: {path}")
        return _Response(payload, url)

    return fake_get


def test_default_collector_is_side_effect_free_and_run_pipeline_requires_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr("data_collection.requests.get", lambda *args, **kwargs: calls.append(args[0]))
    monkeypatch.setattr("data_collection.requests.post", lambda *args, **kwargs: calls.append(args[0]))

    collector = LuteolinDataCollector()

    assert collector.output_dir is None
    with pytest.raises(ValueError, match="output|online"):
        collector.run_pipeline()
    assert calls == []


def test_public_online_collection_requires_explicit_opt_in(tmp_path, monkeypatch):
    calls = []

    def blocked_request(url, *args, **kwargs):
        calls.append(url)
        raise RuntimeError("mocked offline request")

    monkeypatch.setattr("data_collection.requests.get", blocked_request)
    monkeypatch.setattr("data_collection.requests.post", blocked_request)
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "no-online-opt-in"))
    monkeypatch.setattr(collector, "_fetch_pubchem_via_pubchempy", lambda: (_ for _ in ()).throw(RuntimeError("use REST fallback")))

    for fetch in (
        collector.fetch_pubchem_data,
        collector.fetch_chembl_data,
        collector.fetch_tcmsp_data,
        lambda: collector.fetch_string_data(["AKT1"]),
    ):
        with pytest.raises(PermissionError, match="allow_online"):
            fetch()
    assert calls == []


def test_collector_output_root_resolution_rejects_junction_into_protected_data(tmp_path, monkeypatch):
    import os
    import data_collection

    alias = tmp_path / "junction"
    candidate = alias / "new-run"
    protected = APL_ROOT / "data"
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

    monkeypatch.setattr(data_collection.os.path, "realpath", resolve_junction)
    with pytest.raises(ValueError, match="protected"):
        LuteolinDataCollector(output_dir=str(candidate))
    assert not candidate.exists()


def test_collector_output_root_rejects_protected_and_existing_paths(tmp_path):
    protected = APL_ROOT / "data"
    with pytest.raises(ValueError, match="protected"):
        LuteolinDataCollector(output_dir=str(protected))

    existing = tmp_path / "existing-output"
    existing.mkdir()
    marker = existing / "keep.json"
    marker.write_bytes(b"keep-existing-data")
    with pytest.raises(FileExistsError, match="existing"):
        LuteolinDataCollector(output_dir=str(existing))
    assert marker.read_bytes() == b"keep-existing-data"


def test_string_snapshots_are_content_addressed_and_conflicts_never_overwrite(tmp_path, monkeypatch):
    raw_bytes = b"[]"

    class EmptyResponse:
        status_code = 200
        content = raw_bytes

        def raise_for_status(self):
            return None

        def json(self):
            return []

    collector = LuteolinDataCollector(
        output_dir=str(tmp_path / "immutable-string"), allow_online=True,
    )
    monkeypatch.setattr(
        "data_collection.requests.post", lambda *args, **kwargs: EmptyResponse(),
    )
    first = collector.fetch_string_data(["AKT1"])
    first_path = Path(collector.output_dir) / first["snapshot_path"]
    assert first_path.read_bytes() == raw_bytes
    second = collector.fetch_string_data(["AKT1"])
    assert second["snapshot_path"] == first["snapshot_path"]
    assert first_path.read_bytes() == raw_bytes

    first_path.write_bytes(b"keep-conflicting-bytes")
    failed = collector.fetch_string_data(["AKT1"])
    assert failed["request_status"] == "failed"
    assert first_path.read_bytes() == b"keep-conflicting-bytes"


def test_collector_fetches_and_hands_off_complete_activity_pages(tmp_path, monkeypatch):
    fixture = _fixture()
    calls = []
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "collector"), allow_online=True)
    collector.source_kind = fixture["fixture_kind"]
    monkeypatch.setattr(
        "data_collection.requests.get", _chembl_get_from_fixture(fixture, calls)
    )

    result = collector.fetch_chembl_data()

    assert result.get("pagination_status") == "complete"
    assert result.get("metadata_status") == "complete"
    pages = result.get("activity_pages")
    assert isinstance(pages, list) and len(pages) == 2
    activity_rows = [row for page in pages for row in page["activities"]]
    assert [row["activity_id"] for row in activity_rows] == [700001, 700002, 700003]
    assert activity_rows[0]["assay_chembl_id"] == "CHEMBL900001"
    assert activity_rows[0]["document_chembl_id"] == "CHEMBL900101"
    assert activity_rows[0]["standard_relation"] == "="
    assert activity_rows[0]["recorded_extra_field"] == "preserve-me-1"
    assert all(len(page["response_sha256"]) == 64 for page in pages)
    assert all(page["source_kind"] == "synthetic_test" for page in pages)
    assert collector.raw_data["chembl"]["activity_pages"] == pages
    assert any("offset=2" in url for url, _ in calls)
    cleaned = collector.clean_and_standardize()
    evidence_by_id = {row["activity_id"]: row for row in cleaned["activity_evidence"]}
    assert set(evidence_by_id) == {700001, 700002, 700003}
    assert evidence_by_id[700001]["raw_activity_record"]["recorded_extra_field"] == "preserve-me-1"
    assert evidence_by_id[700001]["assay_chembl_id"] == "CHEMBL900001"
    assert evidence_by_id[700001]["document_chembl_id"] == "CHEMBL900101"
    assert evidence_by_id[700001]["confidence_score"] == 9
    assert evidence_by_id[700001]["confidence_source"] == "assay"
    assert "confidence_score" not in evidence_by_id[700001]["raw_activity_record"]
    assert evidence_by_id[700001]["assay_record"]["assay_chembl_id"] == "CHEMBL900001"
    assert evidence_by_id[700001]["assay_record"]["assay_cell_type"] is None
    assert evidence_by_id[700001]["document_record"]["document_chembl_id"] == "CHEMBL900101"
    assert evidence_by_id[700001]["document_record"]["doi"] == "10.0000/synthetic.001"
    assert evidence_by_id[700001]["assay_record_source_sha256"]
    assert evidence_by_id[700001]["document_record_source_sha256"]
    assert evidence_by_id[700001]["source_url"].startswith("https://www.ebi.ac.uk/chembl/api/data/activity")
    assert evidence_by_id[700001]["query"]["molecule_chembl_id"] == "CHEMBL151"
    assert evidence_by_id[700001]["page_offset"] == 0
    assert evidence_by_id[700001]["pagination_next"].endswith("offset=2")
    assert evidence_by_id[700001]["response_sha256"] == evidence_by_id[700001]["source_sha256"]


def test_legacy_rows_remain_unverified_and_missing_ids_stay_null(tmp_path):
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "legacy"))
    collector.raw_data["chembl"] = {
        "chembl_id": "CHEMBL151",
        "targets": [{
            "target_chembl_id": "CHEMBL900201",
            "target_name": "Legacy display target",
            "organism": "Homo sapiens",
            "activity_type": "Ki",
            "activity_value": "120",
            "activity_unit": "nM",
        }],
    }

    cleaned = collector.clean_and_standardize()

    evidence = cleaned.get("activity_evidence")
    assert isinstance(evidence, list) and len(evidence) == 1
    assert evidence[0]["activity_id"] is None
    assert evidence[0]["assay_chembl_id"] is None
    assert evidence[0]["document_chembl_id"] is None
    assert evidence[0]["source_kind"] == "legacy_unverified"
    assert evidence[0]["evidence_status"] == "unverified"
    assert "activity_evidence" not in cleaned.get("training_tables", {})


def test_static_tcmsp_payload_stays_static_after_http_success(tmp_path, monkeypatch):
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "tcmsp"), allow_online=True)
    calls = []

    def successful_static_probe(url, **kwargs):
        calls.append((url, kwargs))
        return _Response({"ok": True}, url)

    monkeypatch.setattr(
        "data_collection.requests.get",
        successful_static_probe,
    )

    result = collector.fetch_tcmsp_data()

    assert result["source_kind"] == "static_reference"
    assert result["source"] != "online"
    assert result["evidence_status"] == "unverified"
    assert calls[0][1]["stream"] is True
    assert calls[0][1]["allow_redirects"] is False


def test_local_builder_exports_page_status_and_full_activity_records(tmp_path, monkeypatch):
    builder = pytest.importorskip("evidence_builder")
    fixture = _fixture()
    calls = []
    monkeypatch.setattr(
        "evidence_builder.requests.get",
        lambda *args, **kwargs: calls.append(args[0]),
    )
    output_dir = tmp_path / "local-output"

    result = builder.ingest_local_manifest(FIXTURE_PATH, output_dir)

    assert calls == []
    assert result["pagination_status"] == "complete"
    assert result["activity_count"] == 4
    assert result["source_kind"] == "synthetic_test"
    bundle = builder.load_entity_bundle(output_dir / "entities.json")
    assert bundle["schema_version"] == 2
    assert bundle["content_sha256"] == result["content_sha256"]
    assert len(bundle["activities"]) == 4
    activity = next(row for row in bundle["activities"] if row["activity_id"] == 700001)
    assert activity["entity_bundle_hash"] == bundle["content_sha256"]
    assert activity["raw_activity_record"]["assay_chembl_id"]
    assert activity["raw_activity_record"]["recorded_extra_field"] == "preserve-me-1"
    assert activity["target_type"] == "SINGLE PROTEIN"
    assert activity["target_taxonomy_id"] == 9606
    assert activity["target_components"] == fixture["chembl"]["targets"][0]["target_components"]
    assert activity["confidence_score"] == 9
    assert activity["confidence_source"] == "assay"
    assert "confidence_score" not in activity["raw_activity_record"]
    assert activity["assay_record"]["assay_chembl_id"] == "CHEMBL900001"
    assert activity["document_record"]["document_chembl_id"] == "CHEMBL900101"
    assert activity["standard_relation"] == "="
    assert activity["standard_units"] == "nM"
    complex_activity = next(row for row in bundle["activities"] if row["activity_id"] == 700002)
    assert len(complex_activity["target_components"]) == 2
    assert bundle["source_provenance"][0]["raw_sha256"]


def test_record_budget_writes_raw_page_but_no_complete_bundle(tmp_path, monkeypatch):
    builder = pytest.importorskip("evidence_builder")
    fixture = _fixture()
    calls = []
    monkeypatch.setattr(
        "evidence_builder.requests.get", _chembl_get_from_fixture(fixture, calls)
    )
    output_dir = tmp_path / "bounded-output"

    result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], output_dir, max_pages=10, max_records=1,
        max_requests=32, max_response_bytes=100_000,
    )

    assert result["pagination_status"] == "incomplete"
    assert result["activity_count"] == 1
    assert result["status"] == "incomplete"
    assert not (output_dir / "entities.json").exists()
    assert (output_dir / "partial_ingest.json").is_file()
    assert list((output_dir / "raw" / "chembl" / "activities").rglob("*.json"))


def test_invalid_next_url_is_not_requested_and_cannot_mark_complete(tmp_path, monkeypatch):
    builder = pytest.importorskip("evidence_builder")
    fixture = _fixture()
    fixture["chembl"]["activities_by_molecule"]["CHEMBL151"]["pages"][0]["page_meta"]["next"] = (
        "https://evil.invalid/chembl/api/data/activity.json?offset=2"
    )
    calls = []
    monkeypatch.setattr(
        "evidence_builder.requests.get", _chembl_get_from_fixture(fixture, calls)
    )
    output_dir = tmp_path / "invalid-next"

    result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], output_dir, max_pages=10, max_records=100,
        max_requests=32, max_response_bytes=100_000,
    )

    assert result["pagination_status"] == "incomplete"
    assert any("invalid next URL" in reason for reason in result["incomplete_reasons"])
    assert all(urlparse(url).hostname == "www.ebi.ac.uk" for url, _ in calls)
    assert not (output_dir / "entities.json").exists()


def test_interrupted_fetch_is_incomplete_without_a_request(tmp_path, monkeypatch):
    builder = pytest.importorskip("evidence_builder")
    calls = []
    monkeypatch.setattr(
        "evidence_builder.requests.get",
        lambda *args, **kwargs: calls.append(args[0]),
    )
    output_dir = tmp_path / "interrupted"

    result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], output_dir, interruption=lambda: True,
    )

    assert result["pagination_status"] == "incomplete"
    assert result["request_count"] == 0
    assert calls == []
    assert not (output_dir / "entities.json").exists()


def test_fetch_cli_collects_multiple_molecules_and_writes_verified_bundle(tmp_path, monkeypatch, capsys):
    builder = pytest.importorskip("evidence_builder")
    fixture = _fixture()
    calls = []
    monkeypatch.setattr(
        "evidence_builder.requests.get", _chembl_get_from_fixture(fixture, calls)
    )
    output_dir = tmp_path / "cli-fetch"

    exit_code = builder.main([
        "fetch-chembl",
        "--molecule-id", "CHEMBL151",
        "--molecule-id", "CHEMBL123456",
        "--out", str(output_dir),
        "--max-pages", "10",
        "--max-records", "20",
        "--max-requests", "32",
        "--max-response-bytes", "100000",
    ])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["pagination_status"] == "complete"
    bundle = builder.load_entity_bundle(output_dir / "entities.json")
    assert {row["molecule_chembl_id"] for row in bundle["activities"]} == {
        "CHEMBL151", "CHEMBL123456",
    }
    assert len(bundle["activities"]) == 4
    assert {row["molecule_chembl_id"] for row in bundle["molecules"]} == {
        "CHEMBL151", "CHEMBL123456",
    }
    assert all(row["structure_status"] == "verified" for row in bundle["molecules"])
    assert {row["accession"] for row in bundle["proteins"]} == {"P12345"}
    assert all(row["sequence_status"] == "missing" for row in bundle["proteins"])
    assert all(row["sequence_status_reason"] == "local_sequence_source_missing" for row in bundle["proteins"])
    first_activity = next(row for row in bundle["activities"] if row["activity_id"] == 700001)
    assert first_activity["confidence_score"] == 9
    assert first_activity["confidence_source"] == "assay"
    assert "confidence_score" not in first_activity["raw_activity_record"]
    assert first_activity["assay_record"]["assay_chembl_id"] == "CHEMBL900001"
    assert first_activity["document_record"]["document_chembl_id"] == "CHEMBL900101"
    assert all(Path(row["source_path"]).name.endswith(".json") for row in bundle["activities"])
    assert all((output_dir / source["source_path"]).is_file() for source in bundle["source_provenance"])
    assert {source["source_id"] for source in bundle["source_provenance"]}.issuperset({
        "chembl_assay:CHEMBL900001", "chembl_document:CHEMBL900101",
        "chembl_target:CHEMBL900201", "chembl_molecule:CHEMBL151",
    })
    assert {"CHEMBL151", "CHEMBL123456"}.issubset({
        parse_qs(urlparse(url).query).get("molecule_chembl_id", [None])[0]
        for url, _ in calls if "/activity.json" in url
    })


def test_page_limit_and_response_size_limits_never_create_complete_bundle(tmp_path, monkeypatch):
    builder = pytest.importorskip("evidence_builder")
    fixture = _fixture()
    monkeypatch.setattr(
        "evidence_builder.requests.get", _chembl_get_from_fixture(fixture, [])
    )
    page_output = tmp_path / "page-limit"
    page_result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], page_output, max_pages=1, max_records=20,
        max_requests=32, max_response_bytes=100_000,
    )
    assert page_result["pagination_status"] == "incomplete"
    assert page_result["page_count"] == 1
    assert page_result["activity_count"] == 2
    assert not (page_output / "entities.json").exists()

    byte_output = tmp_path / "byte-limit"
    calls = []

    def too_large(url, **kwargs):
        calls.append(url)
        response = _Response({"molecule_chembl_id": "CHEMBL151"}, url)
        response.headers["Content-Length"] = "1000"
        return response

    monkeypatch.setattr("evidence_builder.requests.get", too_large)
    byte_result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], byte_output, max_response_bytes=64,
    )
    assert byte_result["pagination_status"] == "incomplete"
    assert byte_result["request_count"] == 1
    assert calls == ["https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL151.json"]
    assert not (byte_output / "entities.json").exists()


def test_truncated_local_manifest_preserves_raw_and_stays_incomplete(tmp_path):
    builder = pytest.importorskip("evidence_builder")
    manifest = _fixture()
    manifest["chembl"]["activities_by_molecule"]["CHEMBL151"]["pages"] = [
        manifest["chembl"]["activities_by_molecule"]["CHEMBL151"]["pages"][0]
    ]
    manifest_path = tmp_path / "本地活动清单.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "incomplete-local"

    result = builder.ingest_local_manifest(manifest_path, output_dir)

    assert result["status"] == "incomplete"
    assert result["pagination_status"] == "incomplete"
    assert not (output_dir / "entities.json").exists()
    assert (output_dir / "partial_ingest.json").is_file()
    assert (output_dir / "raw" / "source_manifest.json").is_file()
    assert list((output_dir / "raw" / "chembl" / "activities").rglob("*.json"))


def test_request_budget_includes_molecule_pages_and_linked_metadata(tmp_path, monkeypatch):
    builder = pytest.importorskip("evidence_builder")
    fixture = _fixture()
    calls = []
    monkeypatch.setattr(
        "evidence_builder.requests.get", _chembl_get_from_fixture(fixture, calls)
    )
    output_dir = tmp_path / "request-budget"

    result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], output_dir, max_pages=10, max_records=20,
        max_requests=3, max_response_bytes=100_000,
    )

    assert result["request_count"] == 3
    assert len(calls) == 3
    assert result["pagination_status"] == "complete"
    assert result["metadata_status"] == "incomplete"
    assert result["status"] == "incomplete"
    assert any("max_requests reached" in reason for reason in result["incomplete_reasons"])
    assert not (output_dir / "entities.json").exists()
    assert (output_dir / "partial_ingest.json").is_file()


def test_http_redirect_is_not_followed_to_another_host(tmp_path, monkeypatch):
    builder = pytest.importorskip("evidence_builder")
    calls = []

    def redirect(url, **kwargs):
        calls.append((url, kwargs))
        response = _Response({"redirect": True}, url, status_code=302)
        response.headers["Location"] = "https://evil.invalid/chembl/api/data/molecule/CHEMBL151.json"
        return response

    monkeypatch.setattr("evidence_builder.requests.get", redirect)
    output_dir = tmp_path / "redirected-response"

    result = builder.fetch_chembl_bounded(["CHEMBL151"], output_dir)

    assert result["status"] == "incomplete"
    assert result["request_count"] == 1
    assert len(calls) == 1
    assert calls[0][1]["allow_redirects"] is False
    assert not (output_dir / "entities.json").exists()


def test_symbol_only_uniprot_lookup_is_not_resolved_by_first_search_hit(tmp_path, monkeypatch):
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "symbol-only"))
    calls = []

    def search_response(url, **kwargs):
        calls.append(url)
        return _Response({
            "results": [{
                "primaryAccession": "P12345",
                "sequence": {"value": "ACDE", "length": 4},
            }],
        }, url)

    monkeypatch.setattr("data_collection.requests.get", search_response)

    result = collector.fetch_uniprot_data(["SYNTHETICGENE"])

    assert calls == []
    assert result["proteins"] == []
    assert result["resolution_status"] == "unverified"
    assert result["unresolved_requests"][0]["reason"] == "symbol_search_disabled"


def test_missing_verified_gene_symbols_do_not_trigger_string_request(tmp_path, monkeypatch):
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "empty-string"))
    calls = []
    monkeypatch.setattr(
        "data_collection.requests.post",
        lambda url, **kwargs: calls.append(url),
    )

    result = collector.fetch_string_data([])

    assert calls == []
    assert result["source_kind"] == "legacy_unverified"
    assert result["interactions"] == []
    assert result["resolution_status"] == "empty"


def test_ppi_request_status_survives_cleaning_for_empty_success_failure_and_no_request(tmp_path, monkeypatch):
    from main import NetworkPharmacologyFramework

    def network_result(dataset):
        framework = object.__new__(NetworkPharmacologyFramework)
        framework.results = {}
        framework.ligand_extractor = None
        framework.protein_extractor = None
        framework.gnn_analyzer = None
        framework.cnn_module = None
        framework.transformer_module = None
        features = framework.run_feature_extraction(dataset)
        return features["ppi"], features["feature_readiness"]["stages"]["network_statistics"]

    no_request = LuteolinDataCollector(output_dir=str(tmp_path / "ppi-not-requested"))
    unrequested = no_request.fetch_string_data([])
    unrequested_clean = no_request.clean_and_standardize()
    assert unrequested["request_status"] == "not_requested"
    assert unrequested_clean["ppi_trace"]["request_status"] == "not_requested"
    unrequested_contract, unrequested_network = network_result(unrequested_clean)
    assert unrequested_contract["status"] == "unavailable"
    assert unrequested_network["status"] == "unavailable"
    assert unrequested_network["reason"] == "ppi_collection_not_requested"

    class EmptyResponse:
        status_code = 200
        content = b"[]"

        def raise_for_status(self):
            return None

        def json(self):
            return []

    successful_empty = LuteolinDataCollector(output_dir=str(tmp_path / "ppi-empty-success"), allow_online=True)
    monkeypatch.setattr(
        "data_collection.requests.post", lambda *args, **kwargs: EmptyResponse(),
    )
    empty_result = successful_empty.fetch_string_data(["AKT1"])
    empty_clean = successful_empty.clean_and_standardize()
    assert empty_result["request_status"] == "complete"
    assert empty_result["source_kind"] == "online_snapshot"
    assert empty_result["interactions"] == []
    assert empty_clean["ppi_trace"]["request_status"] == "complete"
    assert empty_clean["ppi_trace"]["snapshot_sha256"] == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    empty_contract, empty_network = network_result(empty_clean)
    assert empty_contract["status"] == "empty"
    assert empty_network["status"] == "empty"

    failed = LuteolinDataCollector(output_dir=str(tmp_path / "ppi-failed"), allow_online=True)
    monkeypatch.setattr(
        "data_collection.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline fixture")),
    )
    failed_result = failed.fetch_string_data(["AKT1"])
    failed_clean = failed.clean_and_standardize()
    assert failed_result["request_status"] == "failed"
    assert failed_result["source_kind"] == "legacy_unverified"
    assert failed_clean["ppi_trace"]["request_status"] == "failed"
    assert failed_clean["interactions"] == []
    failed_contract, failed_network = network_result(failed_clean)
    assert failed_contract["status"] == "unavailable"
    assert failed_network["status"] == "unavailable"
    assert failed_network["reason"] == "ppi_collection_failed"


def test_run_pipeline_hands_complete_raw_evidence_to_cleaning(tmp_path):
    fixture = _fixture()
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "pipeline"), allow_online=True)
    collector.source_kind = "synthetic_test"
    collector.gene_mapping = {"AKT1": "AKT1"}
    collector.fetch_pubchem_data = lambda: collector.raw_data.setdefault("pubchem", {
        "cid": 5280445,
        "properties": {"Canonical_SMILES": "CCO"},
    })
    collector.fetch_chembl_data = lambda: collector.raw_data.setdefault("chembl", {
        "chembl_id": "CHEMBL151",
        "source_kind": "synthetic_test",
        "targets": [{
            "target_chembl_id": "CHEMBL900201",
            "target_name": "AKT1 kinase",
            "organism": "Homo sapiens",
        }],
        "activity_pages": fixture["chembl"]["activities_by_molecule"]["CHEMBL151"]["pages"],
        "target_records": {
            row["target_chembl_id"]: row for row in fixture["chembl"]["targets"]
        },
            "assay_records": {
                row["assay_chembl_id"]: row for row in fixture["chembl"]["assays"]
            },
            "document_records": {
                row["document_chembl_id"]: row for row in fixture["chembl"]["documents"]
            },
    })
    collector.fetch_tcmsp_data = lambda: collector.raw_data.setdefault("tcmsp", {
        "source_kind": "static_reference", "evidence_status": "unverified",
    })
    observed_string_seeds = []

    def capture_string_seeds(symbols):
        observed_string_seeds.extend(symbols)
        collector.raw_data["string"] = {"interactions": [], "source_kind": "legacy_unverified"}
        return collector.raw_data["string"]

    collector.fetch_string_data = capture_string_seeds
    collector.save_processed_data = lambda: "not-written-in-test"

    result = collector.run_pipeline()

    assert observed_string_seeds == []
    assert {row["activity_id"] for row in result["activity_evidence"]} == {
        700001, 700002, 700003,
    }
    assert all(row["evidence_status"] == "observed" for row in result["activity_evidence"])
    assert all(row["is_valid_target"] is False for row in result["targets"])
    display_target = next(row for row in result["targets"] if row["target_chembl_id"] == "CHEMBL900201")
    assert display_target["target_name"] == fixture["chembl"]["targets"][0]["pref_name"]
    assert display_target["gene_symbol"] is None


def test_collector_groups_multiple_molecule_page_chains_without_discarding_capture(tmp_path, monkeypatch):
    fixture = _fixture()
    calls = []
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "multi-collector"), allow_online=True)
    collector.source_kind = "synthetic_test"
    collector.chembl_molecule_ids = ["CHEMBL151", "CHEMBL123456"]
    monkeypatch.setattr(
        "data_collection.requests.get", _chembl_get_from_fixture(fixture, calls)
    )

    result = collector.fetch_chembl_data()
    cleaned = collector.clean_and_standardize()

    assert result["chembl_id"] is None
    assert result["pagination_status"] == "complete"
    assert result["metadata_status"] == "complete"
    assert result["collection_status"] == "complete"
    assert len(result["activity_pages"]) == 3
    assert len(result["activities"]) == 4
    assert {row["activity_id"] for row in cleaned["activity_evidence"]} == {
        700001, 700002, 700003, 800001,
    }
    assert {row["molecule_chembl_id"] for row in cleaned["activity_evidence"]} == {
        "CHEMBL151", "CHEMBL123456",
    }


def test_projection_error_does_not_replace_successfully_captured_raw_pages(tmp_path, monkeypatch):
    fixture = _fixture()
    calls = []
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "projection-error"), allow_online=True)
    collector.source_kind = "synthetic_test"
    monkeypatch.setattr(
        "data_collection.requests.get", _chembl_get_from_fixture(fixture, calls)
    )
    monkeypatch.setattr(
        "evidence_builder.activity_evidence_from_collector_payload",
        lambda payload: (_ for _ in ()).throw(RuntimeError("projection failure fixture")),
    )

    result = collector.fetch_chembl_data()

    assert result["projection_status"] == "error"
    assert result["pagination_status"] == "complete"
    assert result["collection_status"] == "incomplete"
    assert len(result["activity_pages"]) == 2
    assert [row["activity_id"] for row in result["activities"]] == [700001, 700002, 700003]
    assert "projection failure fixture" in result["projection_errors"][0]
    assert all((Path(collector.output_dir) / page["source_path"]).is_file() for page in result["activity_pages"])


def test_failed_required_assay_metadata_keeps_pagination_complete_but_bundle_incomplete(tmp_path, monkeypatch):
    builder = pytest.importorskip("evidence_builder")
    fixture = _fixture()
    calls = []
    base_get = _chembl_get_from_fixture(fixture, calls)

    def fail_one_assay(url, *args, **kwargs):
        if "/assay/CHEMBL900001.json" in url:
            calls.append((url, kwargs))
            raise TimeoutError("synthetic assay metadata timeout")
        return base_get(url, *args, **kwargs)

    monkeypatch.setattr("evidence_builder.requests.get", fail_one_assay)
    output_dir = tmp_path / "missing-assay-metadata"

    result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], output_dir, max_pages=10, max_records=20,
        max_requests=32, max_response_bytes=100_000,
    )

    assert result["pagination_status"] == "complete"
    assert result["metadata_status"] == "incomplete"
    assert result["status"] == "incomplete"
    assert any("assay:CHEMBL900001" in reason for reason in result["incomplete_reasons"])
    assert (output_dir / "partial_ingest.json").is_file()
    assert not (output_dir / "entities.json").exists()
    assert list((output_dir / "raw" / "chembl" / "activities").rglob("*.json"))
