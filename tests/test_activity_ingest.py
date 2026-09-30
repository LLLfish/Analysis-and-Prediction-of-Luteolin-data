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
            payload = {"assay_chembl_id": assay_id, "confidence_score": 9}
        elif "/document/" in path and path.endswith(".json"):
            document_id = path.rsplit("/", 1)[-1].removesuffix(".json")
            payload = {"document_chembl_id": document_id}
        else:
            raise AssertionError(f"unexpected request path: {path}")
        return _Response(payload, url)

    return fake_get


def test_collector_fetches_and_hands_off_complete_activity_pages(tmp_path, monkeypatch):
    fixture = _fixture()
    calls = []
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "collector"))
    collector.source_kind = fixture["fixture_kind"]
    monkeypatch.setattr(
        "data_collection.requests.get", _chembl_get_from_fixture(fixture, calls)
    )

    result = collector.fetch_chembl_data()

    assert result.get("pagination_status") == "complete"
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
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "tcmsp"))
    monkeypatch.setattr(
        "data_collection.requests.get",
        lambda url, **kwargs: _Response({"ok": True}, url),
    )

    result = collector.fetch_tcmsp_data()

    assert result["source_kind"] == "static_reference"
    assert result["source"] != "online"
    assert result["evidence_status"] == "unverified"


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
    assert all(Path(row["source_path"]).name.endswith(".json") for row in bundle["activities"])
    assert all((output_dir / source["source_path"]).is_file() for source in bundle["source_provenance"])
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


def test_run_pipeline_hands_complete_raw_evidence_to_cleaning(tmp_path):
    fixture = _fixture()
    collector = LuteolinDataCollector(output_dir=str(tmp_path / "pipeline"))
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

