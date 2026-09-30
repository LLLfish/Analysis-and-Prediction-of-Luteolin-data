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
    assert bundle["activities"][0]["entity_bundle_hash"] == bundle["content_sha256"]
    assert bundle["activities"][0]["raw_activity_record"]["assay_chembl_id"]
    assert bundle["activities"][0]["raw_activity_record"]["recorded_extra_field"] == "preserve-me-1"
    assert bundle["source_provenance"][0]["raw_sha256"]

