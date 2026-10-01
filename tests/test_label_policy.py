"""Contracts for deterministic grouping of complete ChEMBL activity evidence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path

import evidence_builder as builder
from label_policy import group_evidence


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "activity_pages.json"


def _producer_evidence(*, duplicate_first: bool = False):
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    chembl = fixture["chembl"]
    activities = [
        copy.deepcopy(row)
        for page in chembl["activities_by_molecule"]["CHEMBL151"]["pages"]
        for row in page["activities"]
    ]
    measured = next(row for row in activities if row["activity_id"] == 700001)
    measured["conformation_id"] = "conf-A"
    measured["sequence_chunk_id"] = "chunk-1"

    second = copy.deepcopy(measured)
    second.update(
        activity_id=700004,
        assay_chembl_id="CHEMBL900004",
        document_chembl_id="CHEMBL900104",
        standard_type="Kd",
        standard_value="100",
    )
    third = copy.deepcopy(measured)
    third.update(
        activity_id=700005,
        assay_chembl_id="CHEMBL900005",
        document_chembl_id="CHEMBL900105",
        standard_type="IC50",
        standard_value="250",
    )
    activities.extend((second, third))
    if duplicate_first:
        activities.append(copy.deepcopy(measured))

    target_records = {
        item["target_chembl_id"]: copy.deepcopy(item)
        for item in chembl["targets"]
    }
    assays = {item["assay_chembl_id"]: copy.deepcopy(item) for item in chembl["assays"]}
    documents = {item["document_chembl_id"]: copy.deepcopy(item) for item in chembl["documents"]}
    for assay_id in ("CHEMBL900004", "CHEMBL900005"):
        assay = copy.deepcopy(assays["CHEMBL900001"])
        assay["assay_chembl_id"] = assay_id
        assays[assay_id] = assay
    for document_id in ("CHEMBL900104", "CHEMBL900105"):
        document = copy.deepcopy(documents["CHEMBL900101"])
        document["document_chembl_id"] = document_id
        documents[document_id] = document

    page = {
        "page_meta": {
            "limit": len(activities),
            "next": None,
            "offset": 0,
            "previous": None,
            "total_count": len(activities),
        },
        "activities": activities,
        "molecule_chembl_id": "CHEMBL151",
    }
    payload = {
        "chembl_id": "CHEMBL151",
        "source_kind": "synthetic_test",
        "activity_pages": [page],
        "target_records": target_records,
        "assay_records": assays,
        "document_records": documents,
    }
    result = builder.activity_evidence_from_collector_payload(payload)
    assert result and all(isinstance(row, builder.ActivityEvidence) for row in result)
    return result


def _digest(result):
    groups, ledger = result
    return tuple(group.group_sha256 for group in groups), ledger.sha256


def test_group_activity_evidence_from_real_producer_keeps_all_pair_endpoints_together():
    records = _producer_evidence()

    groups, ledger = group_evidence(records)

    assert len(groups) == 1
    group = groups[0]
    assert group.molecule_chembl_id == "CHEMBL151"
    assert group.protein_accession == "P12345"
    assert group.taxonomy_id == 9606
    grouped_records = [json.loads(item) for item in group.records_json]
    assert {str(item["activity_id"]) for item in grouped_records} == {"700001", "700004", "700005"}
    assert {item["standard_type"] for item in grouped_records} == {"Ki", "Kd", "IC50"}
    assert {item["assay_chembl_id"] for item in grouped_records} == {
        "CHEMBL900001", "CHEMBL900004", "CHEMBL900005",
    }
    first_record = next(item for item in grouped_records if item["activity_id"] == 700001)
    assert first_record["raw_activity_record"]["conformation_id"] == "conf-A"
    assert first_record["raw_activity_record"]["sequence_chunk_id"] == "chunk-1"
    assert group.split_group_id.startswith("pair-v1-")
    assert ledger.input_count == len(records)


def test_identical_activity_source_is_deduplicated_and_every_input_gets_one_ledger_row():
    records = _producer_evidence(duplicate_first=True)

    groups, ledger = group_evidence(records)

    assert len(groups) == 1
    assert len(groups[0].evidence_ids) == 3
    expected_categories = {"retained", "duplicate", "excluded", "unverified", "conflict_candidate"}
    assert set(ledger.counts) == expected_categories
    assert ledger.counts["duplicate"] == 1
    assert ledger.counts["excluded"] == 2
    assert sum(ledger.counts.values()) == len(records)


def test_empty_single_reordered_and_concurrent_grouping_are_deterministic():
    records = _producer_evidence(duplicate_first=True)

    empty_groups, empty_ledger = group_evidence([])
    assert empty_groups == ()
    assert empty_ledger.input_count == 0

    one_groups, one_ledger = group_evidence(records[:1])
    assert len(one_groups) == 1
    assert one_ledger.input_count == 1

    expected = _digest(group_evidence(records))
    assert _digest(group_evidence(list(reversed(records)))) == expected
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(lambda _: _digest(group_evidence(records)), range(8)))
    assert parallel == [expected] * len(parallel)
