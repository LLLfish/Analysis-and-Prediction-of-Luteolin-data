import hashlib
import json
from pathlib import Path

import formal_corpus


def test_formal_config_binds_new_budget_and_immutable_source_policies():
    config, decision = formal_corpus._verify_config(formal_corpus.CONFIG_PATH)
    assert config["protocol_version"] == "corpus-formal-v1.0"
    assert config["base_corpus_sha256"] == hashlib.sha256((formal_corpus.APL_ROOT / "configs/corpus_v1.json").read_bytes()).hexdigest()
    assert decision["caps"]["chembl_requests"] == 2048
    assert decision["caps"]["uniprot_requests"] == 128
    assert config["luteolin_case"]["chembl_id"] is None
    assert config["development_exclusion"]["molecule_ids"] == ["CHEMBL1", "CHEMBL10"]


def test_workload_uses_global_unique_pairs_and_full_sequence_cells():
    caps = {
        "builder_molecule_pairs": 8128, "builder_protein_pairs": 2016,
        "builder_alignment_cells": 2_200_000_000, "auditor_alignment_cells": 2_200_000_000,
        "builder_wall_seconds": 900, "auditor_wall_seconds": 900,
    }
    workload = formal_corpus._workload(4, [100, 200, 300], caps, 1_000_000)
    assert workload["status"] == "within_budget"
    assert workload["molecule_pairs"] == 6
    assert workload["protein_pairs"] == 3
    assert workload["alignment_cells"] == 100 * 200 + 100 * 300 + 200 * 300
    assert workload["max_single_alignment_cells"] == 60_000
    assert workload["builder_estimated_seconds"] == workload["auditor_estimated_seconds"]


def test_pair_cap_stops_before_expensive_cell_enumeration():
    class HugeLengths:
        def __len__(self):
            return 100_000

        def __iter__(self):
            raise AssertionError("alignment cells must not be enumerated after pair cap")

    caps = {"builder_molecule_pairs": 8128, "builder_protein_pairs": 2016}
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
