import hashlib
import json
from pathlib import Path
import shutil
import formal_corpus
import evidence_builder
import dataset_builder
import pytest
from urllib.parse import parse_qs, urlparse


def _caps():
    return formal_corpus._verify_config(formal_corpus.CONFIG_PATH)[1]["caps"]


def test_formal_config_binds_new_budget_and_immutable_source_policies():
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    assert config["protocol_version"] == "corpus-formal-v1.0"
    assert config["base_corpus_sha256"] == hashlib.sha256((formal_corpus.APL_ROOT / "configs/corpus_v1.json").read_bytes()).hexdigest()
    assert decision["caps"]["chembl_requests"] == 2048
    assert decision["caps"]["uniprot_requests"] == 128
    assert config["luteolin_case"]["chembl_id"] is None
    assert config["development_exclusion"]["molecule_ids"] == ["CHEMBL1", "CHEMBL10"]


def test_workload_uses_global_unique_pairs_and_full_sequence_cells():
    caps = _caps()
    workload = formal_corpus._workload(4, [100, 200, 300], caps, 1_000_000)
    assert workload["status"] == "within_budget"
    assert workload["molecule_pairs"] == 6
    assert workload["protein_pairs"] == 3
    assert workload["alignment_cells"] == 100 * 200 + 100 * 300 + 200 * 300
    assert workload["max_single_alignment_cells"] == 60_000
    assert workload["builder_estimated_seconds"] == workload["auditor_estimated_seconds"]


def test_workload_counts_long_verified_sequences_even_when_esm_unsupported():
    caps = _caps()
    workload = formal_corpus._workload(2, [1280, 1363], caps, 4_764_854.39208085)
    assert workload["protein_pairs"] == 1
    assert workload["alignment_cells"] == 1_744_640
    assert workload["status"] == "within_budget"


def test_six_full_pass_worst_case_scope_fails_global_resource_budget_without_relaxation():
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    measurement = decision["measurement"]
    molecule = measurement["molecule_neighbor"]
    workload = formal_corpus._workload(
        128, [1022] * 64, decision["caps"], measurement["alignment_cells_per_second"],
        [1280, 1363], 3, molecule["warm_median_seconds"], molecule["cold_first_seconds"],
    )
    assert workload["status"] == "size_limit"
    assert workload["publisher_audit_passes"] == 3
    assert workload["full_neighbor_pass_count"] == 6
    assert workload["alignment_cells"] <= decision["caps"]["builder_alignment_cells"]
    assert workload["pipeline_total_alignment_cells"] > decision["caps"]["pipeline_total_alignment_cells"]
    assert decision["caps"]["pipeline_total_alignment_cells"] == 6_856_000_000
    assert decision["caps"]["pipeline_total_wall_seconds"] == 2820


def test_formal_source_bundle_rejects_mixed_fixture_or_legacy_entity():
    bundle = {
        "source_kind": "online_snapshot",
        "molecules": [{"source_kind": "online_snapshot"}],
        "proteins": [{"source_kind": "online_snapshot"}],
        "activities": [{"source_kind": "synthetic_test"}],
        "source_provenance": [{"source_kind": "online_snapshot"}],
    }
    try:
        formal_corpus._require_formal_online_snapshot(bundle)
    except formal_corpus.FormalCorpusError as exc:
        assert "activities[0]" in str(exc)
    else:
        raise AssertionError("mixed-source formal bundle accepted")


def test_pair_cap_stops_before_expensive_cell_enumeration():
    class HugeLengths:
        def __len__(self):
            return 100_000

        def __iter__(self):
            raise AssertionError("alignment cells must not be enumerated after pair cap")

    caps = _caps()
    workload = formal_corpus._workload(2, HugeLengths(), caps, 1.0)
    assert workload["status"] == "size_limit"
    assert workload["protein_pairs"] == 4_999_950_000
    assert workload["alignment_cells"] is None


def test_long_sequences_remain_excluded_with_explicit_reason_and_hash():
    long_sequence = "A" * 1280
    short_sequence = "C" * 500
    bundle = {"proteins": [
        {"accession": "P00001", "taxonomy_id": 9606, "full_sequence": long_sequence,
         "sequence_sha256": hashlib.sha256(long_sequence.encode()).hexdigest(),
         "sequence_status": "verified", "identity_status": "verified", "source_kind": "online_snapshot"},
        {"accession": "P00002", "taxonomy_id": 9606, "full_sequence": short_sequence,
         "sequence_sha256": hashlib.sha256(short_sequence.encode()).hexdigest(),
         "sequence_status": "verified", "identity_status": "verified", "source_kind": "online_snapshot"},
    ]}
    supported, excluded = formal_corpus._sequence_applicability(bundle)
    assert [item["accession"] for item in supported] == ["P00002"]
    assert excluded[0]["accession"] == "P00001"
    assert excluded[0]["sequence_length"] == 1280
    assert excluded[0]["sequence_sha256"] == hashlib.sha256(long_sequence.encode()).hexdigest()
    assert excluded[0]["reason"] == "unsupported_length_full_sequence_retained"


def test_preflight_cli_is_offline_and_expand_requires_explicit_online_flag():
    parser = formal_corpus.build_argument_parser()
    preflight = parser.parse_args(["preflight", "--config", "c.json", "--entities", "e.json", "--out", "C:/tmp/" + "a" * 32])
    expand = parser.parse_args(["expand", "--config", "c.json", "--out", "C:/tmp/" + "b" * 32])
    assert preflight.command == "preflight"
    assert not hasattr(preflight, "allow_online")
    assert expand.allow_online is False


def test_preflight_rejects_existing_and_non_guid_roots(tmp_path):
    existing = tmp_path / ("a" * 32)
    existing.mkdir()
    try:
        formal_corpus._safe_new_root(existing)
    except formal_corpus.FormalCorpusError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("pre-existing output root accepted")
    try:
        formal_corpus._safe_new_root(tmp_path / "not-guid")
    except formal_corpus.FormalCorpusError as exc:
        assert "GUID" in str(exc)
    else:
        raise AssertionError("non-GUID output root accepted")


class _FakeResponse:
    def __init__(self, body, url, *, content_length=None):
        self.body = body
        self.url = url
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.reads = 0
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        self.reads += 1
        yield self.body

    def close(self):
        self.closed = True


def _make_ledger(tmp_path, *, registry=None):
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    root = tmp_path / "batch"
    root.mkdir(exist_ok=True)
    raw_sha = hashlib.sha256(formal_corpus.CONFIG_PATH.read_bytes()).hexdigest()
    ledger = formal_corpus.RequestLedger(root, config, decision, registry=registry, config_sha256=raw_sha)
    return root, config, decision, ledger, raw_sha


def test_request_attempt_is_durable_before_http_call_and_raw_bytes_are_bound(tmp_path, monkeypatch):
    root, _, _, ledger, _ = _make_ledger(tmp_path)
    url = "https://www.ebi.ac.uk/chembl/api/data/status.json"
    expected = b'{"database_version":"ChEMBL-test"}'

    def fake_get(request_url, **kwargs):
        current = json.loads((root / "request_ledger.json").read_text(encoding="utf-8"))
        assert current["attempts"][-1]["status"] == "pending"
        assert current["attempts"][-1]["url"] == request_url == url
        return _FakeResponse(expected, request_url, content_length=len(expected))

    monkeypatch.setattr(evidence_builder.requests, "get", fake_get)
    response = ledger.http_get(url, timeout=30, stream=True)
    assert b"".join(response.iter_content()) == expected
    attempt = ledger.attempts[-1]
    assert attempt["status"] == "succeeded"
    raw = (root / attempt["raw_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == attempt["raw_sha256"]
    assert ledger.summary()["chembl_requests"] == 1


def test_declared_oversize_response_is_not_read_and_charged_conservatively(tmp_path, monkeypatch):
    _, _, _, ledger, _ = _make_ledger(tmp_path)
    url = "https://www.ebi.ac.uk/chembl/api/data/status.json"
    fake = _FakeResponse(b"should-not-be-read", url, content_length=8_388_609)
    monkeypatch.setattr(evidence_builder.requests, "get", lambda *_a, **_k: fake)
    response = ledger.http_get(url, timeout=30)
    with pytest.raises(formal_corpus.FormalCorpusError, match="already refused"):
        list(response.iter_content())
    attempt = ledger.attempts[-1]
    assert fake.reads == 0 and fake.closed
    assert attempt["status"] == "failed"
    assert attempt["response_bytes"] == 0
    assert attempt["charged_bytes"] == 8_388_608
    assert ledger.summary()["chembl_response_bytes_charged"] == 8_388_608


def test_stream_oversize_actual_bytes_are_charged_once_even_if_failure_repeats(tmp_path, monkeypatch):
    _, _, _, ledger, _ = _make_ledger(tmp_path)
    url = "https://www.ebi.ac.uk/chembl/api/data/status.json"
    fake = _FakeResponse(b"x" * 8_388_609, url)
    monkeypatch.setattr(evidence_builder.requests, "get", lambda *_a, **_k: fake)
    response = ledger.http_get(url, timeout=30)
    with pytest.raises(formal_corpus.FormalCorpusError, match="exceeded 8 MiB"):
        list(response.iter_content())
    attempt = ledger.attempts[-1]
    charged = attempt["charged_bytes"]
    assert attempt["status"] == "failed"
    assert attempt["response_bytes"] == 8_388_609
    assert charged == 8_388_609
    response._fail("second failure must not double-account")
    assert ledger.attempts[-1]["charged_bytes"] == charged


def test_pending_attempt_recovery_charges_request_timeout_and_max_bytes(tmp_path):
    old_root, config, decision, ledger, raw_sha = _make_ledger(tmp_path)
    ledger.before("https://www.ebi.ac.uk/chembl/api/data/activity.json?limit=100&offset=0")
    new_root = tmp_path / "resumed"
    new_root.mkdir()
    resumed = formal_corpus.RequestLedger(new_root, config, decision, prior_root=old_root, config_sha256=raw_sha)
    usage = resumed._used()
    assert resumed.attempts[0]["status"] == "recovered_unknown"
    assert usage["requests"] == 1 and usage["pages"] == 1 and usage["rows"] == 100
    assert usage["bytes"] == 8_388_608 and usage["wall"] == 30


def test_batch_registry_requires_latest_ledger_and_rejects_tampered_counters_before_new_root(tmp_path):
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    config_sha = hashlib.sha256(formal_corpus.CONFIG_PATH.read_bytes()).hexdigest()
    registry_path = tmp_path / "registry.json"
    first_root = tmp_path / "first"
    first_root.mkdir()
    registry = formal_corpus.BatchRegistry(config, config_sha, first_root, None, registry_path=registry_path)
    ledger = formal_corpus.RequestLedger(first_root, config, decision, registry=registry, config_sha256=config_sha)
    ledger.before("https://www.ebi.ac.uk/chembl/api/data/status.json")
    assert formal_corpus.BatchRegistry.validate_start(config, config_sha, first_root, registry_path=registry_path) == first_root.resolve()
    with pytest.raises(formal_corpus.FormalCorpusError, match="requires --resume-from"):
        formal_corpus.BatchRegistry.validate_start(config, config_sha, None, registry_path=registry_path)
    data = json.loads((first_root / "request_ledger.json").read_text(encoding="utf-8"))
    data["attempts"][0]["charged_bytes"] = -1
    (first_root / "request_ledger.json").write_bytes(formal_corpus._json_bytes(data))
    with pytest.raises(formal_corpus.FormalCorpusError, match="SHA mismatch"):
        formal_corpus.BatchRegistry.validate_start(config, config_sha, first_root, registry_path=registry_path)
    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_data["latest_ledger_sha256"] = hashlib.sha256((first_root / "request_ledger.json").read_bytes()).hexdigest()
    registry_path.write_bytes(formal_corpus._json_bytes(registry_data))
    with pytest.raises(formal_corpus.FormalCorpusError, match="non-negative strict integer"):
        formal_corpus.BatchRegistry.validate_start(config, config_sha, first_root, registry_path=registry_path)
    assert not (tmp_path / "second").exists()


def test_discovery_byte_cap_preserves_capture_reserve(tmp_path):
    _, _, _, ledger, _ = _make_ledger(tmp_path)
    for index in range(16):
        ledger.attempts.append({"attempt_id": index + 1, "phase": "discovery", "service": "chembl",
                                "status": "succeeded", "activity_page": False,
                                "request_timeout_seconds": 0.01, "elapsed_seconds": 0.01,
                                "charged_wall_seconds": 0.01, "response_bytes": 8_388_608,
                                "charged_bytes": 8_388_608, "url": f"https://www.ebi.ac.uk/chembl/api/data/status.json?i={index}"})
    with pytest.raises(formal_corpus.FormalCorpusError, match="discovery response-byte reserve"):
        ledger.before("https://www.ebi.ac.uk/chembl/api/data/status.json?last=1")


def _projection_page(*, offset, total, next_url, standard_type="Ki", organism="Homo sapiens"):
    return {"activities": [{"activity_id": 1, "standard_type": standard_type,
                            "target_organism": organism}],
            "page_meta": {"offset": offset, "limit": 100, "total_count": total, "next": next_url}}


def test_discovery_fails_on_row_scope_drift():
    with pytest.raises(formal_corpus.FormalCorpusError, match="changed the registered Ki/human query scope"):
        formal_corpus._validate_activity_projection_page(
            _projection_page(offset=0, total=1, next_url=None, standard_type="Kd"),
            expected_offset=0, projection=["activity_id", "standard_type", "target_organism"],
        )
    with pytest.raises(formal_corpus.FormalCorpusError, match="changed the registered Ki/human query scope"):
        formal_corpus._validate_activity_projection_page(
            _projection_page(offset=0, total=1, next_url=None, organism="Mus musculus"),
            expected_offset=0, projection=["activity_id", "standard_type", "target_organism"],
        )


def test_discovery_fails_on_total_count_drift_and_early_end_but_allows_fixed_window_truncation():
    projection = ["activity_id", "standard_type", "target_organism"]
    formal_corpus._validate_activity_projection_page(
        _projection_page(offset=0, total=200, next_url="/next"), expected_offset=0, projection=projection,
    )
    with pytest.raises(formal_corpus.FormalCorpusError, match="total_count changed"):
        formal_corpus._validate_activity_projection_page(
            _projection_page(offset=100, total=201, next_url=None), expected_offset=100,
            projection=projection, expected_total=200,
        )
    with pytest.raises(formal_corpus.FormalCorpusError, match="ended before total_count"):
        formal_corpus._validate_activity_window_end(200, 100, None)
    formal_corpus._validate_activity_window_end(2000, 1000, "/chembl/api/data/activity.json?offset=1000")


def test_luteolin_lookup_uses_complete_exact_name_window_and_lowest_stable_id(tmp_path, monkeypatch):
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    root = tmp_path / "lookup"
    root.mkdir()
    ledger = formal_corpus.RequestLedger(root, config, decision,
                                         config_sha256=hashlib.sha256(formal_corpus.CONFIG_PATH.read_bytes()).hexdigest())
    payload = {"molecules": [
        {"molecule_chembl_id": "CHEMBL90", "pref_name": "Luteolin",
         "molecule_structures": {"canonical_smiles": "CC"}},
        {"molecule_chembl_id": "CHEMBL12", "pref_name": "luteolin",
         "molecule_structures": {"canonical_smiles": "CO"}},
    ], "page_meta": {"offset": 0, "limit": 100, "total_count": 2, "next": None}}
    detail = {"molecule_chembl_id": "CHEMBL12", "pref_name": "luteolin",
              "molecule_structures": {"canonical_smiles": "CO"}}

    def fake_get(url, **_kwargs):
        body = formal_corpus._json_bytes(detail if url.endswith("/molecule/CHEMBL12.json") else payload)
        return _FakeResponse(body, url, content_length=len(body))

    monkeypatch.setattr(evidence_builder.requests, "get", fake_get)
    molecule, receipt = formal_corpus._lookup_luteolin(ledger, config)
    assert receipt["lookup_complete"] is True
    assert receipt["selected_chembl_id"] == "CHEMBL12"
    assert receipt["exact_name_candidate_ids"] == ["CHEMBL12", "CHEMBL90"]


def test_luteolin_lookup_seal_and_reader_bind_complete_pages_plus_native_id_detail(tmp_path, monkeypatch):
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    config_sha = hashlib.sha256(formal_corpus.CONFIG_PATH.read_bytes()).hexdigest()
    root = tmp_path / "lookup-seal"
    root.mkdir()
    ledger = formal_corpus.RequestLedger(root, config, decision, config_sha256=config_sha)
    search_payload = {"molecules": [
        {"molecule_chembl_id": "CHEMBL90", "pref_name": "Luteolin",
         "molecule_structures": {"canonical_smiles": "CC"}},
        {"molecule_chembl_id": "CHEMBL12", "pref_name": "luteolin",
         "molecule_structures": {"canonical_smiles": "CO"}},
    ], "page_meta": {"offset": 0, "limit": 100, "total_count": 2, "next": None}}
    detail_payload = {"molecule_chembl_id": "CHEMBL12", "pref_name": "Luteolin",
                      "molecule_structures": {"canonical_smiles": "CO"}}

    def fake_get(url, **_kwargs):
        payload = detail_payload if url.endswith("/molecule/CHEMBL12.json") else search_payload
        body = formal_corpus._json_bytes(payload)
        return _FakeResponse(body, url, content_length=len(body))

    monkeypatch.setattr(evidence_builder.requests, "get", fake_get)
    _molecule, case_receipt = formal_corpus._lookup_luteolin(ledger, config)
    selection = {
        "schema_version": 1, "purpose": "formal_bounded_candidate_selection",
        "protocol_version": config["protocol_version"], "config_semantic_sha256": formal_corpus._sha(formal_corpus._json_bytes(config)),
        "base_corpus_sha256": config["base_corpus_sha256"], "decision_sha256": config["budget"]["decision_sha256"],
        "selection_status": "sealed", "global_population_claim": False,
        "selected_molecule_ids": ["CHEMBL500"], "selected_target_chembl_ids": ["CHEMBL700"],
        "selected_relation_evidence": [{"activity_id": "1", "molecule_chembl_id": "CHEMBL500",
                                         "target_chembl_id": "CHEMBL700", "standard_type": "Ki",
                                         "standard_relation": "="}],
        "forbidden_selection_fields_read": [], "luteolin_case": case_receipt,
    }
    window = {"source_snapshot": "ChEMBL_37", "query_window": {"coverage": "bounded_window"}}
    sealed, _ = formal_corpus._seal_selection(
        root, ledger, config, config_sha, selection, window, case_receipt,
        "2026-10-11T00:00:00Z",
    )
    resolved = formal_corpus._lookup_luteolin_from_selection(sealed)
    assert resolved["molecule_chembl_id"] == "CHEMBL12"
    assert case_receipt["lookup_complete"] is True
    assert case_receipt["detail_receipt"]["native_molecule_chembl_id"] == "CHEMBL12"


def test_luteolin_lookup_rejects_early_end_instead_of_claiming_complete(tmp_path, monkeypatch):
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    root = tmp_path / "lookup"
    root.mkdir()
    ledger = formal_corpus.RequestLedger(root, config, decision,
                                         config_sha256=hashlib.sha256(formal_corpus.CONFIG_PATH.read_bytes()).hexdigest())
    payload = {"molecules": [{"molecule_chembl_id": "CHEMBL12", "pref_name": "luteolin",
                              "molecule_structures": {"canonical_smiles": "CO"}}],
               "page_meta": {"offset": 0, "limit": 100, "total_count": 2, "next": None}}
    body = formal_corpus._json_bytes(payload)
    monkeypatch.setattr(evidence_builder.requests, "get",
                        lambda url, **_kwargs: _FakeResponse(body, url, content_length=len(body)))
    with pytest.raises(formal_corpus.FormalCorpusError, match="ended before its total_count"):
        formal_corpus._lookup_luteolin(ledger, config)


def test_development_neighbor_exclusion_is_transitive_across_formal_cluster_components():
    touched, components = formal_corpus._development_touched_cluster_ids(
        ["formal-cluster:A", "formal-cluster:B"], ["development:protein:P"],
        [{"entity_a": "formal-cluster:A", "entity_b": "formal-cluster:B", "reason": "formal_neighbor", "score": 0.9},
         {"entity_a": "formal-cluster:A", "entity_b": "development:protein:P", "reason": "development_neighbor", "score": 0.4}],
    )
    assert touched == {"formal-cluster:A", "formal-cluster:B"}
    assert components[0]["development_members"] == ["development:protein:P"]


def _capture_binding_fixture(tmp_path, capture_accession_by_target=None):
    selected_accession_by_target = {"CHEMBL700": "P00001", "CHEMBL701": "P00002"}
    capture_accession_by_target = capture_accession_by_target or selected_accession_by_target
    bundle_root = tmp_path / "captured"
    selection_root = tmp_path / "selection"
    (bundle_root / "raw" / "target").mkdir(parents=True)
    (selection_root / "raw" / "target").mkdir(parents=True)
    target_sources = []
    discovery_receipts = []
    qualities = []
    proteins = {}
    for index, (target_id, selected_accession) in enumerate(selected_accession_by_target.items()):
        capture_accession = capture_accession_by_target[target_id]
        discovery_target = {"target_chembl_id": target_id, "target_type": "SINGLE PROTEIN", "organism": "Homo sapiens",
                  "tax_id": 9606, "target_components": [{"component_type": "PROTEIN", "accession": selected_accession,
                  "component_tax_id": 9606, "organism": "Homo sapiens"}]}
        discovery_raw = formal_corpus._json_bytes(discovery_target)
        discovery_sha = hashlib.sha256(discovery_raw).hexdigest()
        capture_target = {**discovery_target, "target_components": [{"component_type": "PROTEIN", "accession": capture_accession,
                       "component_tax_id": 9606, "organism": "Homo sapiens"}]}
        raw = formal_corpus._json_bytes(capture_target)
        raw_sha = hashlib.sha256(raw).hexdigest()
        relative = f"raw/target/{target_id}.json"
        captured_path = bundle_root / relative
        captured_path.write_bytes(raw)
        source_url = f"{evidence_builder.CHEMBL_API_ROOT}/target/{target_id}.json"
        target_sources.append({"source_id": f"chembl_target:{target_id}", "source_kind": "online_snapshot",
                               "source_path": relative, "raw_sha256": raw_sha, "source_url": source_url})
        resolved = evidence_builder.resolve_target_components(capture_target, source_kind="online_snapshot",
                                                              source_path=relative, source_sha256=raw_sha,
                                                              source_url=source_url)[0]
        sequence = "ACDEFG"
        proteins[capture_accession] = {"accession": capture_accession, "taxonomy_id": 9606, "full_sequence": sequence,
                               "sequence_length": len(sequence), "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                               "sequence_status": "verified", "identity_status": "verified",
                               "component_evidence": {"references": [resolved]}}
        discovery_relative = f"raw/target/{target_id}.json"
        discovery_path = selection_root / discovery_relative
        discovery_path.write_bytes(discovery_raw)
        discovery_receipts.append({"attempt_id": index + 1, "service": "chembl", "phase": "discovery",
                                   "status": "succeeded", "url": source_url, "batch_root": str(selection_root),
                                   "raw_path": discovery_relative, "raw_sha256": discovery_sha,
                                   "source_kind": "online_snapshot"})
        qualities.append({"target_chembl_id": target_id, "accession": selected_accession, "taxonomy_id": 9606,
                          "target_raw_sha256": discovery_sha,
                          "target_receipt": {"url": source_url, "raw_path": discovery_relative, "raw_sha256": discovery_sha}})
    activities = []
    sealed_rows = []
    for index, target_id in enumerate(selected_accession_by_target):
        row = {"activity_id": str(1001 + index), "molecule_chembl_id": "CHEMBL500", "target_chembl_id": target_id,
               "assay_chembl_id": f"CHEMBL8{index:04d}", "document_chembl_id": f"CHEMBL9{index:04d}",
               "standard_type": "Ki", "standard_relation": "=", "target_organism": "Homo sapiens"}
        activities.append(row)
        sealed_rows.append(dict(row))
    selection = {"selected_molecule_ids": ["CHEMBL500"], "selected_target_chembl_ids": list(selected_accession_by_target),
                 "target_quality_evidence": qualities, "raw_source_receipts": discovery_receipts,
                 "selected_relation_evidence": sealed_rows}
    bundle = {"source_kind": "online_snapshot", "molecules": [{"molecule_chembl_id": "CHEMBL500"}],
              "activities": activities, "proteins": list(proteins.values()), "source_provenance": target_sources}
    return bundle_root, bundle, selection, {"pagination_status": "complete", "metadata_status": "complete"}


def test_complete_capture_checks_each_target_component_and_sealed_activity_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(formal_corpus, "_require_formal_online_snapshot", lambda _bundle: None)
    bundle_root, bundle, selection, result = _capture_binding_fixture(tmp_path)
    verified = formal_corpus._validate_complete_capture(bundle, result, selection, bundle_root)
    assert verified["sealed_activity_rows_verified"] == 2
    assert verified["selected_target_component_bindings_verified"] == 2
    bundle["activities"] = [dict(bundle["activities"][1]), {**bundle["activities"][0], "activity_id": "1003"}]
    with pytest.raises(formal_corpus.FormalCorpusError, match="omitted or replaced"):
        formal_corpus._validate_complete_capture(bundle, result, selection, bundle_root)


def test_capture_rejects_cross_target_component_swap_even_when_other_target_keeps_old_accession(monkeypatch, tmp_path):
    monkeypatch.setattr(formal_corpus, "_require_formal_online_snapshot", lambda _bundle: None)
    bundle_root, bundle, selection, result = _capture_binding_fixture(
        tmp_path, {"CHEMBL700": "P00002", "CHEMBL701": "P00001"},
    )
    with pytest.raises(formal_corpus.FormalCorpusError, match="selected target CHEMBL700"):
        formal_corpus._validate_complete_capture(bundle, result, selection, bundle_root)


def test_runtime_compute_budget_uses_actual_clock_and_stops_after_over_cap_pass():
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    caps = dict(decision["caps"])
    caps["pipeline_total_wall_seconds"] = 10
    now = [0.0]
    budget = formal_corpus.RuntimeComputeBudget(caps, clock=lambda: now[0])

    def slow_pass():
        now[0] = 3.0
        return "finished"

    with pytest.raises(formal_corpus.FormalSizeLimit, match="actual_strict_builder_wall_limit_exceeded"):
        budget.run_pass("strict_builder", slow_pass, 2.0)
    assert budget.records == [{"pass": "strict_builder", "wall_seconds": 3.0,
                               "cap_seconds": 2.0, "status": "size_limit"}]
    assert budget.to_dict()["actual_elapsed_seconds"] == 3.0


def test_runtime_compute_budget_accounts_failed_pass_duration_and_preserves_normal_errors():
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    caps = dict(decision["caps"])
    caps["pipeline_total_wall_seconds"] = 20
    now = [0.0]
    budget = formal_corpus.RuntimeComputeBudget(caps, clock=lambda: now[0])

    def slow_failure():
        now[0] = 3.0
        raise RuntimeError("slow audit failed")

    with pytest.raises(formal_corpus.FormalSizeLimit, match="after_failure"):
        budget.run_pass("independent_audit", slow_failure, 2.0)
    assert budget.records[-1]["status"] == "size_limit"
    assert budget.records[-1]["wall_seconds"] == 3.0

    now[0] = 4.0
    normal_failure_budget = formal_corpus.RuntimeComputeBudget(caps, clock=lambda: now[0])

    def quick_failure():
        now[0] = 5.0
        raise RuntimeError("ordinary failure")

    with pytest.raises(RuntimeError, match="ordinary failure"):
        normal_failure_budget.run_pass("strict_builder", quick_failure, 2.0)
    assert normal_failure_budget.records[-1]["status"] == "failed"
    assert normal_failure_budget.records[-1]["wall_seconds"] == 1.0


def test_offline_mocked_expand_runs_selection_capture_derive_and_all_real_audits(monkeypatch, tmp_path):
    config, _decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    molecule_id, target_id, assay_id, document_id = "CHEMBL500", "CHEMBL700", "CHEMBL800", "CHEMBL900"
    Chem, _DataStructs, _fingerprints, _MurckoScaffold = dataset_builder._require_rdkit()
    luteolin_smiles = Chem.MolToSmiles(
        Chem.MolFromSmiles("C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O"),
        canonical=True, isomericSmiles=True,
    )
    luteolin = {"molecule_chembl_id": "CHEMBL151", "pref_name": "Luteolin", "molecule_type": "Small molecule",
                "molecule_structures": {"canonical_smiles": luteolin_smiles}}
    molecule = {"molecule_chembl_id": molecule_id, "pref_name": "Nitrile probe", "molecule_type": "Small molecule",
                "molecule_structures": {"canonical_smiles": "C#N"}}
    target = {"target_chembl_id": target_id, "target_type": "SINGLE PROTEIN", "organism": "Homo sapiens",
              "tax_id": 9606, "target_components": [{"component_type": "PROTEIN", "accession": "P12345",
              "component_tax_id": 9606, "organism": "Homo sapiens"}]}
    assay = {"assay_chembl_id": assay_id, "target_chembl_id": target_id, "confidence_score": 9}
    document = {"document_chembl_id": document_id, "title": "Offline fixture document"}
    activity = {"activity_id": 700001, "molecule_chembl_id": molecule_id, "target_chembl_id": target_id,
                "assay_chembl_id": assay_id, "document_chembl_id": document_id, "standard_type": "Ki",
                "standard_relation": "=", "standard_value": "100", "standard_units": "nM",
                "pchembl_value": "7.00", "target_organism": "Homo sapiens", "target_tax_id": 9606}
    projection = config["selection_window"]["projection"]

    def payload_for(url):
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if parsed.path.endswith("/status.json"):
            return {"chembl_db_version": "ChEMBL_37"}
        if parsed.path.endswith("/molecule.json"):
            return {"molecules": [luteolin], "page_meta": {"offset": 0, "limit": 100,
                    "total_count": 1, "next": None, "previous": None}}
        if parsed.path.endswith("/molecule/CHEMBL151.json"):
            return luteolin
        if parsed.path.endswith(f"/molecule/{molecule_id}.json"):
            return molecule
        if parsed.path.endswith("/activity.json"):
            row = activity if "molecule_chembl_id" in params else {key: activity[key] for key in projection}
            return {"activities": [row], "page_meta": {"offset": 0, "limit": 100, "total_count": 1,
                    "next": None, "previous": None}}
        if parsed.path.endswith(f"/target/{target_id}.json"):
            return target
        if parsed.path.endswith(f"/assay/{assay_id}.json"):
            return assay
        if parsed.path.endswith(f"/document/{document_id}.json"):
            return document
        if parsed.path.endswith("/uniprotkb/P12345.json"):
            sequence = "ACDEFG"
            return {"primaryAccession": "P12345", "organism": {"taxonId": 9606},
                    "sequence": {"value": sequence, "length": len(sequence)}}
        raise AssertionError(f"unexpected mocked HTTP endpoint: {url}")

    def fake_get(url, **_kwargs):
        body = formal_corpus._json_bytes(payload_for(url))
        return _FakeResponse(body, url, content_length=len(body))

    monkeypatch.setattr(evidence_builder.requests, "get", fake_get)
    registry_path = tmp_path / "isolated-formal-batch-ledger.json"
    original_validate = formal_corpus.BatchRegistry.validate_start.__func__
    original_init = formal_corpus.BatchRegistry.__init__

    def isolated_validate(cls, cfg, config_sha, resume, *, registry_path=None):
        return original_validate(cls, cfg, config_sha, resume, registry_path=registry_path or registry_path_for_test)

    registry_path_for_test = registry_path

    def isolated_init(self, cfg, config_sha, output_root, resume, *, registry_path=None):
        return original_init(self, cfg, config_sha, output_root, resume, registry_path=registry_path_for_test)

    monkeypatch.setattr(formal_corpus.BatchRegistry, "validate_start", classmethod(isolated_validate))
    monkeypatch.setattr(formal_corpus.BatchRegistry, "__init__", isolated_init)
    root = formal_corpus.APL_ROOT.parent / ".task-tmp" / __import__("uuid").uuid4().hex
    result = formal_corpus.expand(formal_corpus.CONFIG_PATH, root, allow_online=True)
    assert result["status"] == "candidate_audited_not_ready"
    assert result["training_ready"] is False
    assert result["independent_audit"]["status"] == "clean"
    assert result["source_window_complete_global"] is False
    passes = result["runtime_compute_budget"]["passes"]
    assert [row["pass"] for row in passes] == [
        "eligibility_cluster_and_development_neighbors", "strict_builder",
        "publisher_candidate_audit_1", "publisher_candidate_audit_2",
        "publisher_manifest_audit", "independent_audit",
    ]
    assert len(passes) == 6 and all(row["status"] == "within_budget" for row in passes)
    capture_summary = json.loads((root / "capture" / "capture_result.json").read_text(encoding="utf-8"))
    assert capture_summary["capture_integrity"]["selected_target_component_bindings_verified"] == 1
    manifest_path = Path(result["frozen_manifest_path"])
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    derived_bundle = evidence_builder.load_entity_bundle(root / "formal_bundle" / "entities.json", verify_sources=True)
    assert formal_corpus._sha(manifest_raw) == result["frozen_manifest_sha256"]
    assert manifest["entities"]["content_sha256"] == result["derived_bundle_sha256"] == derived_bundle["content_sha256"]
    assert manifest["source_provenance"] == derived_bundle["source_provenance"]
    assert manifest["audit"] == {"status": "clean", "exit_code": 0, "finding_count": 0}
    shutil.rmtree(root)
