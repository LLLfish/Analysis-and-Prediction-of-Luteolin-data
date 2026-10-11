"""Focused contracts for the bounded real-corpus pilot bridge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import parse_qs, urlparse

import corpus_pipeline
import dataset_builder
import evidence_builder as builder
import label_policy
import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "activity_pages.json"


class _Response:
    def __init__(self, payload, url):
        self.body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.url = url
        self.status_code = 200
        self.headers = {"Content-Length": str(len(self.body))}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset + chunk_size]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source(root, source_id, relative, value):
    raw = builder._json_bytes(value)
    saved, digest = builder._write_raw_bytes(root, relative, raw)
    return {"source_id": source_id, "source_kind": "online_snapshot", "source_path": saved, "raw_sha256": digest}


def test_ki_only_policy_hash_is_bound_and_default_policy_stays_ki_kd():
    apl = Path(builder.__file__).resolve().parent
    corpus, corpus_bytes = corpus_pipeline._load_base_contract()
    split_path, ki_path = corpus_pipeline._verify_bound_policies(corpus)
    ki_policy, ki_hash, _ = label_policy._load_policy(ki_path)
    default_policy, _, _ = label_policy._load_policy(apl / "policies" / "label_policy.json")

    assert ki_policy["endpoints"] == {"labeled": ["Ki"], "independent_only": ["Kd", "IC50"]}
    assert _sha(ki_path.read_bytes()) == corpus["label_policy"]["sha256"]
    assert default_policy["endpoints"] == {"labeled": ["Ki", "Kd"], "independent_only": ["IC50"]}
    assert len(corpus_bytes) > 0 and len(ki_hash) == 64 and split_path.is_file()


def test_empty_or_duplicate_run_ids_fail_before_source_access(tmp_path):
    corpus, raw = corpus_pipeline._load_base_contract()
    split_path, label_path = corpus_pipeline._verify_bound_policies(corpus)
    run = {
        "base_corpus_sha256": _sha(raw), "base_corpus_path": "configs/corpus_v1.json",
        "split_policy_path": str(split_path.relative_to(Path(builder.__file__).resolve().parent)).replace("\\", "/"),
        "split_policy_sha256": _sha(split_path.read_bytes()),
        "label_policy_path": str(label_path.relative_to(Path(builder.__file__).resolve().parent)).replace("\\", "/"),
        "label_policy_sha256": _sha(label_path.read_bytes()),
        "selected_molecule_chembl_ids": [],
    }
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(corpus_pipeline.CorpusPipelineError, match="non-empty unique real ChEMBL ID"):
        corpus_pipeline._load_run_config(path)


def test_uniprot_budget_is_cumulative_and_all_component_references_are_retained(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chembl = fixture["chembl"]
    root = tmp_path / "bundle"
    root.mkdir()
    molecule = chembl["molecules"][0]
    target_one = json.loads(json.dumps(chembl["targets"][0]))
    target_two = json.loads(json.dumps(target_one))
    target_two.update(target_chembl_id="CHEMBL900202", pref_name="Second verified target")
    target_two["target_components"][0]["accession"] = "Q12345"
    target_three = json.loads(json.dumps(target_one))
    target_three.update(target_chembl_id="CHEMBL900203", pref_name="Third verified target")
    provenance = [
        _source(root, "chembl_molecule:CHEMBL151", "raw/chembl/molecules/CHEMBL151.json", molecule),
        _source(root, "chembl_target:CHEMBL900201", "raw/chembl/targets/CHEMBL900201.json", target_one),
        _source(root, "chembl_target:CHEMBL900202", "raw/chembl/targets/CHEMBL900202.json", target_two),
        _source(root, "chembl_target:CHEMBL900203", "raw/chembl/targets/CHEMBL900203.json", target_three),
    ]
    rows = [
        {"molecule_chembl_id": "CHEMBL151", "target_chembl_id": target_one["target_chembl_id"]},
        {"molecule_chembl_id": "CHEMBL151", "target_chembl_id": target_two["target_chembl_id"]},
        {"molecule_chembl_id": "CHEMBL151", "target_chembl_id": target_three["target_chembl_id"]},
    ]
    result = {"molecule_records": {"CHEMBL151": molecule},
              "target_records": {target["target_chembl_id"]: target for target in (target_one, target_two, target_three)},
              "activity_pages": [{"activities": rows}]}
    calls = []

    def uniprot_get(url, **kwargs):
        calls.append(url)
        accession = urlparse(url).path.rsplit("/", 1)[-1].removesuffix(".json")
        return _Response(fixture["uniprot"][accession], url)

    budget = {"max_requests": 1, "request_count": 0, "max_response_bytes": 1_000_000,
              "max_total_bytes": 1_000_000, "response_bytes": 0, "timeout": 30}
    molecules, proteins, exclusions = builder._entities_from_chembl_capture(
        result, provenance, uniprot_root=root / "uniprot-fetch", uniprot_budget=budget,
        uniprot_http_get=uniprot_get,
    )

    assert len(calls) == budget["request_count"] == 1
    assert budget["response_bytes"] > 0
    assert any(item["source_id"] == "uniprot:P12345" for item in provenance)
    assert len(proteins) == 2
    first = next(item for item in proteins if item.accession == "P12345")
    refs = first.component_evidence["references"]
    assert len(refs) == 2
    assert refs[0]["target_type"] == "SINGLE PROTEIN"
    assert refs[0]["organism"] == "Homo sapiens"
    assert refs[0]["taxonomy_id"] == 9606
    assert refs[0]["component_index"] == 0
    assert refs[0]["target_record_sha256"]
    assert refs[0]["target_record_source"]["source_path"] == "raw/chembl/targets/CHEMBL900201.json"
    assert any(item.sequence_status == "verified" for item in proteins)
    assert any(item.sequence_status != "verified" for item in proteins)
    assert exclusions == []

    bundle = builder.write_entity_bundle(
        root, activities=[], source_provenance=provenance, molecules=molecules,
        proteins=proteins, exclusions=exclusions, source_kind="online_snapshot",
    )
    checked = builder.load_entity_bundle(root / "entities.json", verify_sources=True)
    assert checked["content_sha256"] == bundle["content_sha256"]
    assert len(checked["source_provenance"]) == len(provenance)
    assert _sha((root / first.source_provenance[0]["source_path"]).read_bytes()) == first.source_provenance[0]["raw_sha256"]


def test_trace_parser_defaults_to_offline_and_has_explicit_online_gate():
    parser = corpus_pipeline.build_argument_parser()
    args = parser.parse_args(["trace", "--config", "run.json", "--out", "C:/tmp/" + "a" * 32])
    assert args.allow_online is False
    assert args.command == "trace"


def _scoped_http_get(fixture, *, target_organism="Homo sapiens", next_url=None, total_count=1):
    chembl = fixture["chembl"]
    row = json.loads(json.dumps(chembl["activities_by_molecule"]["CHEMBL151"]["pages"][0]["activities"][0]))
    row["target_organism"] = target_organism
    molecule = chembl["molecules"][0]
    target = chembl["targets"][0]
    assay = chembl["assays"][0]
    document = chembl["documents"][0]
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        path = urlparse(url).path
        if path.endswith("/molecule/CHEMBL151.json"):
            return _Response(molecule, url)
        if path.endswith("/activity.json"):
            query = parse_qs(urlparse(url).query)
            assert query.get("standard_type") == ["Ki"]
            assert query.get("target_organism") == ["Homo sapiens"]
            page = {"page_meta": {"limit": 100, "next": next_url, "offset": 0,
                                  "previous": None, "total_count": total_count},
                    "activities": [row]}
            return _Response(page, url)
        if path.endswith("/target/CHEMBL900201.json"):
            return _Response(target, url)
        if path.endswith("/assay/CHEMBL900001.json"):
            return _Response(assay, url)
        if path.endswith("/document/CHEMBL900101.json"):
            return _Response(document, url)
        raise AssertionError(f"unexpected source request: {url}")

    return fake_get, calls


def test_scoped_ki_human_fetch_binds_filters_and_verifies_raw_pages(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fake_get, calls = _scoped_http_get(fixture)
    output = tmp_path / "scoped-ki-human"

    result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], output, max_pages=2, max_records=10, max_requests=16,
        max_response_bytes=100_000, timeout=2, http_get=fake_get,
        activity_filters={"standard_type": "Ki", "target_organism": "Homo sapiens"},
    )

    assert result["status"] == "complete"
    bundle = builder.load_entity_bundle(output / "entities.json", verify_sources=True)
    assert len(bundle["activities"]) == 1
    activity = bundle["activities"][0]
    assert activity["standard_type"] == "Ki"
    assert activity["target_organism"] == "Homo sapiens"
    assert activity["query"]["filters"] == {"standard_type": "Ki", "target_organism": "Homo sapiens"}
    assert parse_qs(urlparse(activity["source_url"]).query)["standard_type"] == ["Ki"]
    assert parse_qs(urlparse(activity["source_url"]).query)["target_organism"] == ["Homo sapiens"]
    page_receipt = next(item for item in bundle["source_provenance"] if item["source_id"].startswith("chembl_activity:"))
    assert hashlib.sha256((output / page_receipt["source_path"]).read_bytes()).hexdigest() == page_receipt["raw_sha256"]
    assert any("/activity.json?" in url for url in calls)

    tampered_root = tmp_path / "scope-marker-stripped"
    tampered_root.mkdir()
    shutil.copytree(output / "raw", tampered_root / "raw")
    tampered_activities = json.loads(json.dumps(bundle["activities"]))
    tampered_activities[0]["query"].pop("filters")
    with pytest.raises(builder.EvidenceInputError, match="scope marker is missing"):
        builder.write_entity_bundle(
            tampered_root, activities=tampered_activities,
            source_provenance=bundle["source_provenance"], molecules=bundle["molecules"],
            proteins=bundle["proteins"], exclusions=bundle.get("exclusions", []),
            source_kind="online_snapshot",
        )


def test_scoped_ki_human_fetch_rejects_out_of_scope_rows_and_next_pages(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    filters = {"standard_type": "Ki", "target_organism": "Homo sapiens"}
    wrong_get, _ = _scoped_http_get(fixture, target_organism="Rattus norvegicus")
    wrong_output = tmp_path / "out-of-scope-row"
    wrong = builder.fetch_chembl_bounded(
        ["CHEMBL151"], wrong_output, max_pages=2, max_records=10, max_requests=16,
        max_response_bytes=100_000, timeout=2, http_get=wrong_get, activity_filters=filters,
    )
    assert wrong["status"] == "incomplete"
    assert any("activity_filter_mismatch:target_organism" in reason for reason in wrong["incomplete_reasons"])
    assert not (wrong_output / "entities.json").exists()

    drift_next = "/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL151&limit=100&offset=1"
    next_get, next_calls = _scoped_http_get(fixture, next_url=drift_next, total_count=2)
    next_output = tmp_path / "scope-drift-next"
    drift = builder.fetch_chembl_bounded(
        ["CHEMBL151"], next_output, max_pages=2, max_records=10, max_requests=16,
        max_response_bytes=100_000, timeout=2, http_get=next_get, activity_filters=filters,
    )
    assert drift["pagination_status"] == "incomplete"
    assert any("invalid next URL" in reason for reason in drift["incomplete_reasons"])
    assert sum("/activity.json" in url for url in next_calls) == 1
    assert not (next_output / "entities.json").exists()


def test_target_cohort_scope_is_sorted_bound_to_each_page_row_and_next_url(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    filters = {"standard_type": "Ki", "target_organism": "Homo sapiens",
               "target_chembl_id__in": ["CHEMBL900201"]}
    fake_get, calls = _scoped_http_get(fixture)
    output = tmp_path / "target-cohort"
    result = builder.fetch_chembl_bounded(
        ["CHEMBL151"], output, max_pages=2, max_records=10, max_requests=16,
        max_response_bytes=100_000, timeout=2, http_get=fake_get, activity_filters=filters,
    )
    assert result["status"] == "complete"
    bundle = builder.load_entity_bundle(output / "entities.json", verify_sources=True)
    activity = bundle["activities"][0]
    assert activity["query"]["filters"] == filters
    activity_query = parse_qs(urlparse(activity["source_url"]).query)
    assert activity_query["target_chembl_id__in"] == ["CHEMBL900201"]
    assert all("target_chembl_id__in=CHEMBL900201" in call for call in calls if "/activity.json?" in call)

    bad_filters = {**filters, "target_chembl_id__in": ["CHEMBL900202"]}
    bad_output = tmp_path / "target-out-of-scope"
    bad = builder.fetch_chembl_bounded(
        ["CHEMBL151"], bad_output, max_pages=2, max_records=10, max_requests=16,
        max_response_bytes=100_000, timeout=2, http_get=fake_get, activity_filters=bad_filters,
    )
    assert bad["status"] == "incomplete"
    assert any("activity_filter_mismatch:target_chembl_id__in" in reason for reason in bad["incomplete_reasons"])
    assert not (bad_output / "entities.json").exists()


def test_target_cohort_scope_stripping_is_rejected_and_legacy_unscoped_remains_compatible():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    row = fixture["chembl"]["activities_by_molecule"]["CHEMBL151"]["pages"][0]["activities"][0]
    pages = [{"page_meta": {"limit": 100, "next": None, "offset": 0, "previous": None, "total_count": 1},
              "activities": [row], "source_url": "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL151&limit=100&offset=0&standard_type=Ki&target_organism=Homo+sapiens&target_chembl_id__in=CHEMBL900201",
              "query_scope": {}}]
    result = builder.activity_evidence_from_pages("CHEMBL151", pages, source_kind="online_snapshot")
    assert result[0].pagination_status == "incomplete"
    with pytest.raises(builder.EvidenceInputError, match="target_chembl_id__in"):
        builder._normalize_activity_filters({"standard_type": "Ki", "target_organism": "Homo sapiens",
                                             "target_chembl_id__in": ["CHEMBL900202", "CHEMBL900201"]})


def test_v1_2_selection_projection_never_reads_activity_values_or_classes():
    row = {"molecule_chembl_id": "CHEMBL1", "target_chembl_id": "CHEMBL4302",
           "assay_chembl_id": "CHEMBL952395", "standard_relation": "=",
           "standard_value": object(), "pchembl_value": object(), "activity_comment": object()}
    assert corpus_pipeline._activity_selection_fields(row) == ("CHEMBL1", "CHEMBL4302", "CHEMBL952395", "=")


def test_v1_2_trace_entry_smoke_binds_target_scope_and_blocks_test_only_source(tmp_path, monkeypatch):
    corpus, corpus_raw = corpus_pipeline._load_base_contract()
    split_path, label_path = corpus_pipeline._verify_bound_policies(corpus)
    decision, decision_raw = corpus_pipeline._read_json(corpus_pipeline._pilot_budget_v1_2_path())
    filters = {**decision["activity_filters"], "target_chembl_id__in": ["CHEMBL900201"]}
    config = {
        "purpose": "bounded_development_pilot_v1.2", "pilot_budget": decision["cumulative_caps"],
        "prior_usage": {"chembl_requests": 256, "activity_pages": 14, "activity_rows": 1169,
                        "uniprot_requests": 0, "uniprot_response_bytes": 0},
        "activity_filters": filters, "budget_decision_sha256": _sha(decision_raw),
        "selected_molecule_chembl_ids": ["CHEMBL151", "CHEMBL123456"],
        "base_corpus_sha256": _sha(corpus_raw), "selection_receipt_sha256": "a" * 64,
        "source_snapshot": "ChEMBL_37", "capture_cutoff_utc": "2026-10-11T00:00:00Z",
        "target_cohort_receipt_path": "target_cohort_receipt.json",
    }
    selection = {"selection_request_count": 0, "selection_activity_pages": 0, "selection_activity_rows": 0}
    config_file = tmp_path / "v1_2_config.json"
    config_file.write_text("{}", encoding="utf-8")
    (tmp_path / "target_cohort_receipt.json").write_text(json.dumps({"selection_elapsed_seconds": 0}), encoding="utf-8")
    monkeypatch.setattr(corpus_pipeline, "_load_run_config", lambda _path: (config, corpus, selection, split_path, label_path))
    status_payload = {"status": "UP", "chembl_db_version": "ChEMBL_37"}
    status_bytes = corpus_pipeline._json_bytes(status_payload)
    monkeypatch.setattr(corpus_pipeline, "_bounded_get", lambda *_args, **_kwargs: (status_payload, status_bytes, _sha(status_bytes)))
    seen = {}

    def fixture_capture(molecule_ids, output_dir, **kwargs):
        seen.update(kwargs)
        assert molecule_ids == ["CHEMBL151", "CHEMBL123456"]
        return builder.ingest_local_manifest(FIXTURE, output_dir)

    monkeypatch.setattr(builder, "fetch_chembl_bounded", fixture_capture)
    output = tmp_path / ("d" * 32)
    report = corpus_pipeline.trace(config_file, output, allow_online=True)
    assert seen["activity_filters"] == filters
    assert seen["timeout"] == decision["cumulative_caps"]["per_request_timeout_seconds"]
    assert report["status"] == "blocked"
    assert report["reason"] == "no_verified_real_human_single_protein_ki_row"
    assert report["total_budget_usage"]["chembl_requests"] == 257
    assert not (output / "frozen").exists()


def test_mixed_scoped_and_unscoped_pages_are_rejected():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    row = json.loads(json.dumps(fixture["chembl"]["activities_by_molecule"]["CHEMBL151"]["pages"][0]["activities"][0]))
    filters = {"standard_type": "Ki", "target_organism": "Homo sapiens"}
    pages = [
        {"page_meta": {"limit": 100, "next": None, "offset": 0, "previous": None, "total_count": 2},
         "activities": [row], "source_url": "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL151&limit=100&offset=0",
         "query_scope": {}},
        {"page_meta": {"limit": 100, "next": None, "offset": 1, "previous": None, "total_count": 2},
         "activities": [row], "source_url": "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL151&standard_type=Ki&target_organism=Homo+sapiens&limit=100&offset=1",
         "query_scope": filters},
    ]
    with pytest.raises(builder.EvidenceInputError, match="do not share one exact query filter scope"):
        builder.activity_evidence_from_pages("CHEMBL151", pages, source_kind="online_snapshot")

def test_per_request_timeout_option_bounds_each_call_without_changing_default(tmp_path):
    molecule = {"molecule_chembl_id": "CHEMBL151", "molecule_structures": {}}
    page = {"page_meta": {"limit": 100, "next": None, "offset": 0,
                          "previous": None, "total_count": 0}, "activities": []}

    def slow_get(url, **kwargs):
        time.sleep(0.04)
        payload = molecule if "/molecule/" in url else page
        return _Response(payload, url)

    default = builder.fetch_chembl_bounded(
        ["CHEMBL151"], tmp_path / "batch-timeout", max_pages=1, max_records=1,
        max_requests=2, max_response_bytes=100_000, timeout=0.06, http_get=slow_get,
    )
    per_request = builder.fetch_chembl_bounded(
        ["CHEMBL151"], tmp_path / "per-request-timeout", max_pages=1, max_records=1,
        max_requests=2, max_response_bytes=100_000, timeout=0.06, http_get=slow_get,
        per_request_timeout=True,
    )
    wall_bounded = builder.fetch_chembl_bounded(
        ["CHEMBL151"], tmp_path / "wall-bounded", max_pages=1, max_records=1,
        max_requests=2, max_response_bytes=100_000, timeout=0.06, http_get=slow_get,
        per_request_timeout=True, total_timeout_seconds=0.06,
    )

    assert default["status"] == "incomplete"
    assert per_request["status"] == "complete"
    assert per_request["budgets"]["per_request_timeout"] is True
    assert wall_bounded["status"] == "incomplete"
    assert wall_bounded["budgets"]["total_timeout_seconds"] == 0.06


def test_v1_1_runtime_split_copy_preflights_and_audits_test_only_without_readiness(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chembl = fixture["chembl"]
    ki_row = next(row for page in chembl["activities_by_molecule"]["CHEMBL151"]["pages"]
                  for row in page["activities"] if row["activity_id"] == 700001)
    first_page = json.loads(json.dumps(chembl["activities_by_molecule"]["CHEMBL151"]["pages"][0]))
    first_page["page_meta"].update({"next": None, "offset": 0, "total_count": 1})
    first_page["activities"] = [ki_row]
    if isinstance(first_page.get("raw_response"), dict):
        first_page["raw_response"]["page_meta"] = json.loads(json.dumps(first_page["page_meta"]))
        first_page["raw_response"]["activities"] = [ki_row]
    chembl["molecules"] = chembl["molecules"][:1]
    chembl["targets"] = [chembl["targets"][0]]
    chembl["assays"] = [chembl["assays"][0]]
    chembl["documents"] = [chembl["documents"][0]]
    chembl["activities_by_molecule"] = {"CHEMBL151": {"pages": [first_page]}}
    manifest_path = tmp_path / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    bundle_root = tmp_path / "fixture-bundle"
    ingest = builder.ingest_local_manifest(manifest_path, bundle_root)
    assert ingest["status"] == "complete"
    bundle_path = bundle_root / "entities.json"

    decision_path = corpus_pipeline._pilot_budget_v1_1_path()
    decision, decision_raw = corpus_pipeline._read_json(decision_path)
    runtime_path, runtime = corpus_pipeline._make_runtime_split_policy(
        tmp_path / "runtime-copy", Path(dataset_builder._POLICY_PATH), decision,
    )
    preflight = corpus_pipeline._preflight_workload(bundle_path, runtime["policy"])
    assert preflight["status"] == "ready"
    assert runtime["runtime_file_sha256"] == _sha(runtime_path.read_bytes())
    assert runtime["base_file_sha256"] == _sha(Path(dataset_builder._POLICY_PATH).read_bytes())
    assert runtime["policy"]["molecule"] == json.loads(Path(dataset_builder._POLICY_PATH).read_text(encoding="utf-8"))["molecule"]
    assert runtime["policy"]["split"] == json.loads(Path(dataset_builder._POLICY_PATH).read_text(encoding="utf-8"))["split"]
    assert runtime["policy"]["budgets"]["alignment_cells"] == decision["pilot_split_runtime_caps"]["alignment_cells"]

    frozen = dataset_builder.publish_frozen_dataset(
        bundle_path, tmp_path / "test-only-freeze", version_id="pilot-smoke",
        policy_path=runtime_path,
        label_policy_path=Path(label_policy.__file__).resolve().parent / "configs" / "label_policy_ki_v1.json",
        allow_test_only=True,
    )
    if frozen.status != "complete":
        print(json.dumps(frozen.to_dict(), ensure_ascii=False, indent=2))
    assert frozen.status == "complete", frozen.to_dict()
    assert frozen.audit_report["status"] == "clean"
    assert frozen.audit_report["training_ready"] is False
    assert _sha(decision_raw) == _sha(decision_path.read_bytes())


@pytest.mark.parametrize("prior_uniprot_requests,prior_uniprot_bytes", [(0, 0), (2, 5000)])
def test_trace_persists_release_request_attempt_before_failed_get(tmp_path, monkeypatch, prior_uniprot_requests, prior_uniprot_bytes):
    corpus, corpus_raw = corpus_pipeline._load_base_contract()
    split_path, label_path = corpus_pipeline._verify_bound_policies(corpus)
    decision, _ = corpus_pipeline._read_json(corpus_pipeline._pilot_budget_v1_1_path())
    config = {
        "purpose": "bounded_development_pilot_v1.1", "pilot_budget": decision["cumulative_caps"],
        "prior_usage": {"chembl_requests": 27, "activity_pages": 10, "activity_rows": 911,
                        "uniprot_requests": prior_uniprot_requests, "uniprot_response_bytes": prior_uniprot_bytes},
        "activity_filters": decision["activity_filters"], "budget_decision_sha256": _sha(corpus_pipeline._pilot_budget_v1_1_path().read_bytes()),
        "selected_molecule_chembl_ids": ["CHEMBL1", "CHEMBL10"],
        "base_corpus_sha256": _sha(corpus_raw), "selection_receipt_sha256": "a" * 64,
        "source_snapshot": "ChEMBL_37", "capture_cutoff_utc": "2026-10-11T00:00:00Z",
    }
    selection = {"selection_request_count": 9, "selection_activity_pages": 2, "selection_activity_rows": 200}
    config_path = tmp_path / "failed-boundary-config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(corpus_pipeline, "_load_run_config", lambda _path: (config, corpus, selection, split_path, label_path))

    def fail_status(*_args, **_kwargs):
        raise TimeoutError("bounded status request timeout")

    monkeypatch.setattr(corpus_pipeline, "_bounded_get", fail_status)
    output = tmp_path / ("c" * 32)
    result = corpus_pipeline.trace(config_path, output, allow_online=True)

    ledger_path = output / "source-boundary" / "status-request-attempt.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["reason"] == "chembl_release_boundary_request_failed"
    assert ledger["request_count"] == result["source_boundary"]["request_count"] == 1
    assert ledger["status"] == "failed"
    assert result["total_budget_usage"]["chembl_requests"] == 28
    assert result["total_budget_usage"]["uniprot_requests"] == prior_uniprot_requests
    assert result["total_budget_usage"]["uniprot_response_bytes"] == prior_uniprot_bytes
    assert result["total_budget_usage"]["caps"]["uniprot_requests"] == 64
    assert not (output / "bundle").exists()


def test_trace_preflights_release_plus_capture_budget_before_status_request(tmp_path, monkeypatch):
    corpus, corpus_raw = corpus_pipeline._load_base_contract()
    split_path, label_path = corpus_pipeline._verify_bound_policies(corpus)
    decision, _ = corpus_pipeline._read_json(corpus_pipeline._pilot_budget_v1_1_path())
    config = {
        "purpose": "bounded_development_pilot_v1.1",
        "pilot_budget": {**decision["cumulative_caps"], "chembl_requests": 28},
        "prior_usage": {"chembl_requests": 27, "activity_pages": 10, "activity_rows": 911,
                        "uniprot_requests": 2, "uniprot_response_bytes": 5000},
        "activity_filters": decision["activity_filters"],
        "budget_decision_sha256": _sha(corpus_pipeline._pilot_budget_v1_1_path().read_bytes()),
        "selected_molecule_chembl_ids": ["CHEMBL1", "CHEMBL10"],
        "base_corpus_sha256": _sha(corpus_raw), "selection_receipt_sha256": "b" * 64,
        "source_snapshot": "ChEMBL_37", "capture_cutoff_utc": "2026-10-11T00:00:00Z",
    }
    selection = {"selection_request_count": 9, "selection_activity_pages": 2,
                 "selection_activity_rows": 200}
    config_path = tmp_path / "no-budget-config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(corpus_pipeline, "_load_run_config", lambda _path: (config, corpus, selection, split_path, label_path))
    calls = []
    monkeypatch.setattr(corpus_pipeline, "_bounded_get", lambda *_args, **_kwargs: calls.append("GET"))

    output = tmp_path / ("e" * 32)
    result = corpus_pipeline.trace(config_path, output, allow_online=True)
    assert result["reason"] == "insufficient_budget_for_release_check_and_capture"
    assert result["source_boundary"]["request_count"] == 0
    assert result["total_budget_usage"]["chembl_requests"] == 27
    assert result["total_budget_usage"]["uniprot_requests"] == 2
    assert result["total_budget_usage"]["uniprot_response_bytes"] == 5000
    assert calls == []


def test_v1_2_selection_request_is_counted_before_failure_and_receipted_finally(tmp_path):
    ledger_path = tmp_path / "target-selection" / "attempt-ledger.json"
    ledger = {"decision_sha256": "f" * 64, "request_count": 0, "attempts": [], "status": "in_progress"}

    def fail_after_observing_pending_state():
        on_disk = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert on_disk["request_count"] == 1
        assert on_disk["attempts"][0]["status"] == "pending"
        raise TimeoutError("bounded metadata request timed out")

    with pytest.raises(TimeoutError):
        corpus_pipeline._run_persisted_selection_attempt(
            ledger_path, ledger, {"kind": "assay", "identifier": "CHEMBL123"},
            fail_after_observing_pending_state,
        )
    closed = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert closed["request_count"] == 1
    assert closed["status"] == "failed"
    assert closed["attempts"][0]["status"] == "failed"

def test_v1_1_trace_cli_smoke_keeps_synthetic_fixture_blocked(tmp_path, monkeypatch, capsys):
    fixture_path = FIXTURE
    corpus, corpus_raw = corpus_pipeline._load_base_contract()
    split_path, label_path = corpus_pipeline._verify_bound_policies(corpus)
    decision, _ = corpus_pipeline._read_json(corpus_pipeline._pilot_budget_v1_1_path())
    ids = ["CHEMBL151", "CHEMBL123456"]
    config = {
        "purpose": "bounded_development_pilot_v1.1", "pilot_budget": decision["cumulative_caps"],
        "prior_usage": {"chembl_requests": 0, "activity_pages": 0, "activity_rows": 0,
                        "uniprot_requests": 0, "uniprot_response_bytes": 0},
        "activity_filters": decision["activity_filters"],
        "budget_decision_sha256": _sha(corpus_pipeline._pilot_budget_v1_1_path().read_bytes()),
        "selected_molecule_chembl_ids": ids,
        "base_corpus_sha256": _sha(corpus_raw), "selection_receipt_sha256": "a" * 64,
        "source_snapshot": "ChEMBL_37", "capture_cutoff_utc": "2026-10-11T00:00:00Z",
    }
    selection = {"selection_request_count": 0, "selection_activity_pages": 0,
                 "selection_activity_rows": 0}
    config_file = tmp_path / "test-run-config.json"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(corpus_pipeline, "_load_run_config", lambda _path: (config, corpus, selection, split_path, label_path))
    status_payload = {"status": "UP", "chembl_db_version": "ChEMBL_37"}
    status_bytes = corpus_pipeline._json_bytes(status_payload)
    monkeypatch.setattr(corpus_pipeline, "_bounded_get", lambda *_args, **_kwargs: (status_payload, status_bytes, _sha(status_bytes)))
    fetch_kwargs = {}

    def fixture_capture(molecule_ids, output_dir, **kwargs):
        fetch_kwargs.update(kwargs)
        assert molecule_ids == ids
        return builder.ingest_local_manifest(fixture_path, output_dir)

    monkeypatch.setattr(builder, "fetch_chembl_bounded", fixture_capture)
    output = tmp_path / ("b" * 32)
    exit_code = corpus_pipeline.main([
        "trace", "--config", str(config_file), "--out", str(output), "--allow-online",
    ])
    cli_report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert cli_report["status"] == "blocked"
    assert cli_report["reason"] == "no_verified_real_human_single_protein_ki_row"
    assert cli_report["ingest"]["source_kind"] == "synthetic_test"
    assert fetch_kwargs["activity_filters"] == {"standard_type": "Ki", "target_organism": "Homo sapiens"}
    assert fetch_kwargs["per_request_timeout"] is True
    assert 0 < fetch_kwargs["total_timeout_seconds"] <= decision["cumulative_caps"]["new_attempt_wall_seconds"]
    assert not (output / "frozen").exists()
    assert (output / "trace_report.json").is_file()


@pytest.mark.parametrize("choices", [
    [{"molecule_chembl_id": "CHEMBL1", "target_chembl_id": "CHEMBL4302"}],
    [{"molecule_chembl_id": "CHEMBL1", "target_chembl_id": "CHEMBL4302"},
     {"molecule_chembl_id": "CHEMBL1", "target_chembl_id": "CHEMBL1955"}],
])
def test_target_cohort_rejects_missing_or_duplicate_fixed_molecule(choices):
    with pytest.raises(corpus_pipeline.CorpusPipelineError, match="every fixed molecule"):
        corpus_pipeline._cohort_target_ids(choices, ["CHEMBL1", "CHEMBL10"], 2)


def test_target_cohort_allows_two_fixed_molecules_to_share_one_target():
    choices = [{"molecule_chembl_id": "CHEMBL1", "target_chembl_id": "CHEMBL4302"},
               {"molecule_chembl_id": "CHEMBL10", "target_chembl_id": "CHEMBL4302"}]
    assert corpus_pipeline._cohort_target_ids(choices, ["CHEMBL1", "CHEMBL10"], 2) == ["CHEMBL4302"]


def test_offline_trace_reports_fixture_without_unbound_budget_or_network(tmp_path, monkeypatch):
    corpus, corpus_raw = corpus_pipeline._load_base_contract()
    split_path, label_path = corpus_pipeline._verify_bound_policies(corpus)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ids = sorted(row["molecule_chembl_id"] for row in fixture["chembl"]["molecules"])
    config = {"verified_manifest_path": str(FIXTURE), "base_corpus_sha256": _sha(corpus_raw),
              "selection_receipt_sha256": "a" * 64, "source_snapshot": "synthetic-test-only",
              "capture_cutoff_utc": "2026-10-11T00:00:00Z", "selected_molecule_chembl_ids": ids}
    selection = {"selection_request_count": 0, "selection_activity_pages": 0, "selection_activity_rows": 0}
    config_file = tmp_path / "offline-run.json"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(corpus_pipeline, "_load_run_config", lambda _path: (config, corpus, selection, split_path, label_path))
    calls = []
    monkeypatch.setattr(corpus_pipeline, "_bounded_get", lambda *_args, **_kwargs: calls.append("GET"))
    report = corpus_pipeline.trace(config_file, tmp_path / ("f" * 32))
    assert report["status"] == "blocked"
    assert report["reason"] == "no_verified_real_human_single_protein_ki_row"
    assert report["prior_usage"] == {"chembl_requests": 0, "activity_pages": 0, "activity_rows": 0,
                                      "uniprot_requests": 0, "uniprot_response_bytes": 0}
    assert calls == []


@pytest.mark.parametrize("states,expected", [
    (["pending"], 35.0),
    (["succeeded"], 5.0),
    (["failed"], 5.0),
    (["pending", "pending"], 65.0),
])
def test_interrupted_selection_calls_consume_their_timeout_on_retry(states, expected):
    ledger = {"request_count": len(states), "elapsed_seconds": 5.0,
              "attempts": [{"status": status} for status in states]}
    before = json.dumps(ledger, sort_keys=True)
    elapsed, charge = corpus_pipeline._selection_attempt_elapsed(ledger, 30)
    assert elapsed == expected
    assert charge == expected - 5
    assert json.dumps(ledger, sort_keys=True) == before


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf"), "0", None])
def test_selection_recovery_rejects_invalid_elapsed_ledger(elapsed):
    with pytest.raises(corpus_pipeline.CorpusPipelineError, match="invalid elapsed"):
        corpus_pipeline._selection_attempt_elapsed(
            {"request_count": 1, "elapsed_seconds": elapsed, "attempts": [{"status": "pending"}]}, 30,
        )


@pytest.mark.parametrize("requests,elapsed,error", [
    (-1, 0.0, "ledger counter"),
    (1, -1, "invalid elapsed"),
    (1, "Infinity", "invalid elapsed"),
])
def test_successful_prior_cohort_is_validated_before_preparing_retry(tmp_path, monkeypatch, requests, elapsed, error):
    corpus, _ = corpus_pipeline._load_base_contract()
    decision, _ = corpus_pipeline._read_json(corpus_pipeline._pilot_budget_v1_2_path())
    old_config = {"purpose": "bounded_development_pilot_v1.1",
                  "base_corpus_sha256": decision["base_config_sha256"],
                  "selected_molecule_chembl_ids": decision["selected_molecule_chembl_ids"]}
    old_selection = {"source_receipts": []}
    config = tmp_path / "old-config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(corpus_pipeline, "_load_run_config", lambda _path: (old_config, corpus, old_selection, None, None))
    prior = tmp_path / "old-trace"
    (prior / "bundle").mkdir(parents=True)
    (prior / "trace_report.json").write_text(json.dumps({"total_budget_usage": decision["cumulative_prior_usage"]}), encoding="utf-8")
    (prior / "bundle" / "partial_ingest.json").write_text(json.dumps({"status": "incomplete", "source_provenance": []}), encoding="utf-8")
    carried = tmp_path / "successful-selection"
    carried.mkdir()
    (carried / "target_cohort_receipt.json").write_text(json.dumps({
        "decision_version": "pilot-v1.2", "metadata_request_count": requests,
        "selection_elapsed_seconds": elapsed, "source_receipts": [],
    }), encoding="utf-8")
    calls = []
    monkeypatch.setattr(corpus_pipeline, "_bounded_get", lambda *_args, **_kwargs: calls.append("GET"))
    output = tmp_path / ("a" * 32)
    with pytest.raises(corpus_pipeline.CorpusPipelineError, match=error):
        corpus_pipeline.prepare_pilot_v1_2(config, prior, output, prior_selection_attempts=[carried])
    assert calls == []
    assert not output.exists()
