"""Contracts for deterministic grouping of complete ChEMBL activity evidence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import json
from pathlib import Path

import evidence_builder as builder
import label_policy
from label_policy import apply_label_policy, group_evidence
import pytest


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


def test_different_linked_source_receipts_are_not_collapsed_as_activity_duplicates():
    base = next(row for row in _producer_evidence() if row.activity_id == 700001)
    alternate = replace(
        base,
        assay_record_source_path="raw/assay-revision.json",
        assay_record_source_sha256="a" * 64,
    )

    groups, ledger = group_evidence([base, alternate])

    assert len(groups) == 1
    assert len(groups[0].evidence_ids) == 2
    assert ledger.counts["duplicate"] == 0
    assert {
        json.loads(record)["assay_record_source_path"]
        for record in groups[0].records_json
    } == {None, "raw/assay-revision.json"}


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


def _test_policy(tmp_path: Path) -> Path:
    policy = {
        "schema_version": 1,
        "policy_version": "label-policy-test-v1",
        "entity_bundle_schema_version": 2,
        "confirmed_source_kinds": ["online_snapshot"],
        "test_only_source_kinds": ["synthetic_test"],
        "entity_eligibility": {
            "target_type": "SINGLE PROTEIN",
            "organism": "Homo sapiens",
            "taxonomy_id": 9606,
            "component_type": "PROTEIN",
            "component_count": 1,
            "accession_pattern": "^[A-Z0-9]{6}(?:[A-Z0-9]{4})?$",
        },
        "endpoints": {"labeled": ["Ki", "Kd"], "independent_only": ["IC50"]},
        "exact_relation": "=",
        "minimum_confidence": 8,
        "units_to_nanomolar": {
            "pM": "0.001", "nM": "1", "uM": "1000", "µM": "1000",
            "μM": "1000", "mM": "1000000", "M": "1000000000",
        },
        "thresholds": {"active_min_p": 6, "inactive_max_p": 5},
        "assay_compatibility_fields": [
            "assay_type", "assay_category", "assay_test_type", "assay_organism",
            "assay_tax_id", "assay_cell_type", "assay_subcellular_fraction",
            "bao_format", "bao_label",
        ],
    }
    path = tmp_path / "label_policy.json"
    path.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
    return path


def _with_measurement(record, activity_id, endpoint, value, unit, *, relation="=", pchembl="5.0", confidence=None):
    raw = copy.deepcopy(record.raw_activity_record)
    raw.update(
        activity_id=activity_id,
        standard_type=endpoint,
        standard_relation=relation,
        standard_value=value,
        standard_units=unit,
        pchembl_value=pchembl,
    )
    changes = {
        "activity_id": activity_id,
        "standard_type": endpoint,
        "standard_relation": relation,
        "standard_value": value,
        "standard_units": unit,
        "pchembl_value": pchembl,
        "raw_activity_record": raw,
    }
    if confidence is not None:
        assay = copy.deepcopy(record.assay_record)
        assay["confidence_score"] = confidence
        changes.update(
            confidence_score=confidence,
            assay_confidence_score=confidence,
            assay_record=assay,
        )
    return replace(record, **changes)


def _with_assay_variant(record, activity_id, endpoint, value, unit, assay_id, assay_category):
    result = _with_measurement(record, activity_id, endpoint, value, unit)
    raw = copy.deepcopy(result.raw_activity_record)
    raw["assay_chembl_id"] = assay_id
    assay = copy.deepcopy(result.assay_record)
    assay["assay_chembl_id"] = assay_id
    assay["assay_category"] = assay_category
    return replace(
        result,
        assay_chembl_id=assay_id,
        raw_activity_record=raw,
        assay_record=assay,
    )


def _with_molecule(record, molecule_id):
    raw = copy.deepcopy(record.raw_activity_record)
    raw["molecule_chembl_id"] = molecule_id
    return replace(record, molecule_chembl_id=molecule_id, raw_activity_record=raw)


def _apply_test_records(records, tmp_path, *, allow_test_only=True, entity_bundle_path=None, policy_path=None):
    groups, ledger = group_evidence(records)
    return apply_label_policy(
        groups,
        ledger=ledger,
        policy_path=policy_path or _test_policy(tmp_path),
        entity_bundle_path=entity_bundle_path,
        allow_test_only=allow_test_only,
    )


def test_versioned_policy_declares_only_exact_human_ki_kd_label_rules():
    policy_path = Path(label_policy.__file__).resolve().parent / "policies" / "label_policy.json"

    assert policy_path.is_file()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["policy_version"]
    assert policy["endpoints"] == {"labeled": ["Ki", "Kd"], "independent_only": ["IC50"]}
    assert policy["exact_relation"] == "="
    assert policy["minimum_confidence"] == 8
    assert policy["thresholds"] == {"active_min_p": 6, "inactive_max_p": 5}
    assert policy["entity_eligibility"]["taxonomy_id"] == 9606


def test_measured_value_boundaries_are_inclusive_units_convert_and_pchembl_is_ignored(tmp_path):
    base = next(row for row in _producer_evidence() if row.activity_id == 700001)
    active = _apply_test_records([_with_measurement(base, 710001, "Ki", "1", "uM", pchembl="1.0")], tmp_path)
    inactive = _apply_test_records([_with_measurement(base, 710002, "Ki", "10", "uM", pchembl="9.9")], tmp_path)
    gray = _apply_test_records([_with_measurement(base, 710003, "Ki", "5", "uM", pchembl="10.0")], tmp_path)

    assert [(item.label, item.endpoint) for item in active.test_only_labeled_pairs] == [("active", "Ki")]
    assert [(item.label, item.endpoint) for item in inactive.test_only_labeled_pairs] == [("inactive", "Ki")]
    assert len(gray.gray) == 1 and gray.gray[0].label == "gray"
    assert active.counts["confirmed_active"] == inactive.counts["confirmed_inactive"] == 0
    assert active.counts["test_only_active"] == 1
    assert active.training_ready is inactive.training_ready is gray.training_ready is False
    assert active.test_only_labeled_pairs[0].evidence_ids


def test_synthetic_labels_require_explicit_test_only_opt_in(tmp_path):
    base = next(row for row in _producer_evidence() if row.activity_id == 700001)
    record = _with_measurement(base, 715001, "Ki", "1", "uM")

    result = _apply_test_records([record], tmp_path, allow_test_only=False)

    assert result.confirmed_labeled_pairs == ()
    assert result.test_only_labeled_pairs == ()
    assert len(result.unverified) == 1
    assert "synthetic_test_only" in result.unverified[0].reasons
    assert result.counts["confirmed_active"] == 0


def test_compatible_assay_conflict_is_quarantined_and_ki_kd_remain_separate(tmp_path):
    base = next(row for row in _producer_evidence() if row.activity_id == 700001)
    records = [
        _with_measurement(base, 720001, "Ki", "100", "nM"),
        _with_measurement(base, 720002, "Ki", "20", "uM"),
        _with_measurement(base, 720003, "Kd", "200", "nM"),
        _with_assay_variant(base, 720004, "Ki", "100", "nM", "CHEMBL900004", "functional"),
        _with_molecule(_with_measurement(base, 720005, "Ki", "100", "nM"), "CHEMBL123456"),
    ]

    result = _apply_test_records(records, tmp_path)

    groups, _ = group_evidence(records)
    activity_by_evidence = {
        evidence_id: json.loads(record_json)["activity_id"]
        for group in groups
        for evidence_id, record_json in zip(group.evidence_ids, group.records_json)
    }
    pair_conflicts = [row for row in result.conflicts if row.pair_key == "CHEMBL151|9606|P12345"]
    assert len(pair_conflicts) == 3
    assert all(row.label == "conflict" for row in pair_conflicts)
    ki_conflict = next(
        row for row in pair_conflicts
        if row.endpoint == "Ki" and "active_inactive_same_endpoint_assay_condition" in row.reasons
    )
    assert {activity_by_evidence[evidence_id] for evidence_id in ki_conflict.evidence_ids} == {720001, 720002}
    assert sum("pair_quarantined_by_conflicting_endpoint" in row.reasons for row in pair_conflicts) == 2
    assert [row.pair_key for row in result.test_only_labeled_pairs] == ["CHEMBL123456|9606|P12345"]
    assert {group.pair_key for group in groups} == {"CHEMBL151|9606|P12345", "CHEMBL123456|9606|P12345"}


def test_apply_rejects_caller_pair_key_tampering(tmp_path):
    record = next(row for row in _producer_evidence() if row.activity_id == 700001)
    groups, ledger = group_evidence([record])
    tampered = replace(groups[0], pair_key="WRONG_PAIR")

    with pytest.raises(label_policy.LabelPolicyInputError, match="pair_key"):
        apply_label_policy(
            [tampered], ledger=ledger, policy_path=_test_policy(tmp_path), allow_test_only=True,
        )


def test_apply_rejects_forked_groups_for_one_canonical_pair(tmp_path):
    base = next(row for row in _producer_evidence() if row.activity_id == 700001)
    records = [
        _with_measurement(base, 725001, "Ki", "100", "nM"),
        _with_measurement(base, 725002, "Kd", "200", "nM"),
    ]
    combined_groups, ledger = group_evidence(records)
    left_groups, _ = group_evidence([records[0]])
    right_groups, _ = group_evidence([records[1]])

    assert len(combined_groups) == len(left_groups) == len(right_groups) == 1
    assert left_groups[0].pair_key == right_groups[0].pair_key
    with pytest.raises(label_policy.LabelPolicyInputError, match="duplicate or forked"):
        apply_label_policy(
            [left_groups[0], right_groups[0]], ledger=ledger,
            policy_path=_test_policy(tmp_path), allow_test_only=True,
        )


def test_source_verified_official_shape_mock_admits_confirmed_labels_without_readiness(tmp_path):
    # A local, official-shape mock exercises the real producer and bundle verifier only.
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["source_kind"] = "online_snapshot"
    fixture["purpose"] = "synthetic_test_only_official_shape_mock"
    chembl = fixture["chembl"]

    luteolin_pages = chembl["activities_by_molecule"]["CHEMBL151"]["pages"]
    luteolin_rows = [copy.deepcopy(row) for page in luteolin_pages for row in page["activities"]]
    luteolin_active = next(row for row in luteolin_rows if row["activity_id"] == 700001)
    luteolin_inactive = copy.deepcopy(luteolin_active)
    luteolin_inactive.update(activity_id=740001, standard_value="20000", standard_units="nM")
    luteolin_kd = copy.deepcopy(luteolin_active)
    luteolin_kd.update(activity_id=740002, standard_type="Kd", standard_value="200", standard_units="nM")
    luteolin_rows.extend((luteolin_inactive, luteolin_kd))
    luteolin_pages[:] = [{
        "page_meta": {"limit": len(luteolin_rows), "next": None, "offset": 0,
                       "previous": None, "total_count": len(luteolin_rows)},
        "activities": luteolin_rows,
    }]

    caffeine_pages = chembl["activities_by_molecule"]["CHEMBL123456"]["pages"]
    caffeine_rows = [copy.deepcopy(row) for page in caffeine_pages for row in page["activities"]]
    caffeine_active = copy.deepcopy(caffeine_rows[0])
    caffeine_active.update(activity_id=840001, standard_relation="=", standard_value="500")
    caffeine_rows.append(caffeine_active)
    caffeine_pages[:] = [{
        "page_meta": {"limit": len(caffeine_rows), "next": None, "offset": 0,
                       "previous": None, "total_count": len(caffeine_rows)},
        "activities": caffeine_rows,
    }]

    manifest_path = tmp_path / "official_shape_mock_test_only.json"
    manifest_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    bundle_root = tmp_path / "verified_mock_bundle"
    ingest = builder.ingest_local_manifest(manifest_path, bundle_root)
    assert ingest["status"] == "complete"
    bundle_path = bundle_root / "entities.json"
    bundle = builder.load_entity_bundle(bundle_path, verify_sources=True)
    records = [builder.ActivityEvidence(**item) for item in bundle["activities"]]
    groups, ledger = group_evidence(records)

    result = apply_label_policy(groups, ledger=ledger, entity_bundle_path=bundle_path)

    clean_labels = [row for row in result.confirmed_labeled_pairs
                    if row.pair_key == "CHEMBL123456|9606|P12345"]
    luteolin_labels = [row for row in result.confirmed_labeled_pairs
                       if row.pair_key == "CHEMBL151|9606|P12345"]
    assert [(row.endpoint, row.label) for row in clean_labels] == [("Ki", "active")]
    assert luteolin_labels == []
    assert {row.endpoint for row in result.conflicts if row.pair_key == "CHEMBL151|9606|P12345"} == {"Ki", "Kd"}
    assert result.counts["confirmed_active"] == 1
    assert result.counts["confirmed_inactive"] == 0
    assert result.confirmed_labeled_pairs
    assert result.test_only_labeled_pairs == ()
    assert result.ledger.input_count == len(records)
    assert sum(result.ledger.counts.values()) == len(records)
    assert len(result.ledger.sha256) == 64
    assert result.training_ready is False


@pytest.mark.parametrize(
    "endpoint,value,unit,relation,confidence,reason",
    [
        ("Ki", "100", "nM", ">", None, "relation_not_exact"),
        ("Ki", "100", "furlong", "=", None, "unit_not_allowed"),
        ("Ki", "NaN", "nM", "=", None, "measurement_value_not_finite_positive"),
        ("Ki", 0, "nM", "=", None, "measurement_value_not_finite_positive"),
        ("Ki", 100, "nM", "=", 7, "confidence_below_minimum"),
        ("IC50", "100", "nM", "=", None, "endpoint_not_label_eligible"),
    ],
)
def test_invalid_evidence_is_excluded_with_reason_and_never_becomes_label(
    tmp_path, endpoint, value, unit, relation, confidence, reason,
):
    base = next(row for row in _producer_evidence() if row.activity_id == 700001)
    record = _with_measurement(
        base, 730001, endpoint, value, unit, relation=relation, confidence=confidence,
    )

    result = _apply_test_records([record], tmp_path)

    assert result.test_only_labeled_pairs == ()
    assert len(result.excluded) == 1
    assert reason in result.excluded[0].reasons
    assert result.excluded[0].evidence_ids


def test_self_reported_status_cannot_bypass_verified_bundle_and_synthetic_stays_test_only(tmp_path):
    bundle_root = tmp_path / "source-verified-fixture"
    ingest = builder.ingest_local_manifest(FIXTURE_PATH, bundle_root)
    assert ingest["status"] == "complete"
    bundle_path = bundle_root / "entities.json"
    bundle = builder.load_entity_bundle(bundle_path, verify_sources=True)
    records = [builder.ActivityEvidence(**item) for item in bundle["activities"]]
    groups, ledger = group_evidence(records)
    policy_path = _test_policy(tmp_path)

    test_result = apply_label_policy(
        groups, ledger=ledger, policy_path=policy_path,
        entity_bundle_path=bundle_path, allow_test_only=True,
    )
    assert test_result.confirmed_labeled_pairs == ()
    assert test_result.counts["confirmed_active"] == test_result.counts["confirmed_inactive"] == 0
    assert test_result.test_only_labeled_pairs
    assert test_result.training_ready is False

    claimed_online = replace(
        records[0], source_kind="online_snapshot", metadata_status="complete",
        evidence_status="observed", page_status="complete", pagination_status="complete",
    )
    forged_groups, forged_ledger = group_evidence([claimed_online])
    forged_result = apply_label_policy(
        forged_groups, ledger=forged_ledger, policy_path=policy_path,
        entity_bundle_path=bundle_path,
    )
    assert forged_result.confirmed_labeled_pairs == ()
    assert forged_result.unverified


def test_invalid_policy_number_fails_closed(tmp_path):
    policy_path = _test_policy(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["minimum_confidence"] = float("nan")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    base = next(row for row in _producer_evidence() if row.activity_id == 700001)
    groups, ledger = group_evidence([base])

    with pytest.raises(label_policy.LabelPolicyInputError):
        apply_label_policy(groups, ledger=ledger, policy_path=policy_path, allow_test_only=True)
