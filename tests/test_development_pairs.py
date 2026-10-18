"""Contract tests for the train-only Phase 03 development manifest (T3-01).

The suite uses a minimal synthetic source bundle for the behavioural contract
(shuffled entity/assignment order, duplicate keys, missing label, not-ready ESM,
new-file-only writes, blocked class coverage) and a read-only check that the
real frozen manifest / entities / selected feature snapshot produce a complete
one-disposition-per-train-assignment ledger with counts recomputed from the
actual bytes.

No training, no GPU, no writes outside pytest temporary directories.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from training import development_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_FROZEN_DIR = (
    PROJECT_ROOT
    / "P1_02_03_Artifacts_6142fd3706ff480a8aa4f013cfe1c381"
    / "a4a30ac7dcee4478a4c3edee39e3b984"
    / "frozen"
    / "corpus-formal-v1-45fdfee33b6d"
)
REAL_MANIFEST = REAL_FROZEN_DIR / "manifest.json"
REAL_ENTITIES = REAL_FROZEN_DIR / "entities.json"
REAL_FEATURES = (
    PROJECT_ROOT
    / "P1_02_04_Artifacts_847e7b27ed5942e19af64e2bd65b8d2f"
    / "01d118a1293346d38a58f116f7c86e2a"
    / "feature_manifest.json"
)

CONTENT_SHA = "1" * 64
ENTITY_VERSION = "entities-v2-" + "1" * 16
MODEL_REVISION = "c" * 40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_json(path: Path, value, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, allow_nan=allow_nan, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def _load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _embedding(seed: int, dim: int = development_manifest.ESM_EMBEDDING_DIM) -> list:
    # Deterministic, finite and unique per accession; no NumPy dependency needed.
    values = []
    for offset in range(dim):
        values.append(round(math.sin(seed + offset * 0.01) * 0.05, 8))
    return values


def _feature_row(accession: str, taxonomy_id: int, sequence_sha256: str) -> dict:
    return {
        "accession": accession,
        "taxonomy_id": taxonomy_id,
        "sequence_length": 40,
        "sequence_sha256": sequence_sha256,
        "entity_version": ENTITY_VERSION,
        "entity_bundle_hash": CONTENT_SHA,
        "bundle_file_sha256": None,  # patched once the entities bytes are written
        "model_id": development_manifest.ESM_MODEL_ID,
        "model_revision": MODEL_REVISION,
        "embedding_dim": development_manifest.ESM_EMBEDDING_DIM,
        "pooling_policy": "residue-mask-mean-v1:max_tokens=1024",
        "max_tokens": 1024,
        "status": "ready",
        "reason": None,
        "finite": True,
        "test_provenance": None,
        "features": {
            "esm_embedding": _embedding(int(accession[-2:])),
        },
        "cache_key": hashlib.sha256(("cache:" + accession).encode("utf-8")).hexdigest(),
        "cache_file_sha256": hashlib.sha256(
            ("cache-file:" + accession).encode("utf-8")
        ).hexdigest(),
    }


def _protein(accession: str, taxonomy_id: int = 9606) -> dict:
    sequence = "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLL" + accession
    return {
        "accession": accession,
        "taxonomy_id": taxonomy_id,
        "sequence_status": "verified",
        "identity_status": "verified",
        "full_sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
    }


def _molecule(molecule_id: str, smiles: str, *, structure_status: str = "verified") -> dict:
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "structure_status": structure_status,
        "structure_sha256": hashlib.sha256(molecule_id.encode("utf-8")).hexdigest(),
    }


def _atom(molecule_id: str, accession: str) -> str:
    return molecule_id + "|9606|" + accession


def _build_synthetic_sources(root: Path, *, only_active: bool = False) -> dict:
    """Minimal schema fixture with deliberate gaps and extra non-train ledgers."""
    proteins = [
        _protein("P91001"),
        _protein("P91002"),
        _protein("P91003"),
        _protein("P91004"),
        _protein("P91005"),
        _protein("P91006"),
        _protein("P91007"),
    ]
    molecules = [
        _molecule("CHEMBL910001", "CCO"),
        _molecule("CHEMBL910002", "c1ccccc1"),
        _molecule("CHEMBL910003", "CCN"),
        _molecule("CHEMBL910004", "CCC", structure_status="unverified"),
        _molecule("CHEMBL910005", "CCCl"),
        _molecule("CHEMBL910006", "CCBr"),
        _molecule("CHEMBL910007", "CCF"),
    ]
    entities = {
        "schema_version": 2,
        "content_sha256": CONTENT_SHA,
        "entity_version": ENTITY_VERSION,
        "molecules": molecules,
        "proteins": proteins,
    }
    frozen_dir = root / "frozen"
    entities_path = frozen_dir / "entities.json"
    _dump_json(entities_path, entities)
    entities_sha = _sha256(entities_path)

    ready_accessions = ("P91001", "P91002", "P91004", "P91006", "P91007")
    feature_rows = []
    for protein in proteins:
        if protein["accession"] == "P91005":
            continue
        if protein["accession"] == "P91003":
            feature_rows.append({
                "accession": "P91003",
                "taxonomy_id": 9606,
                "sequence_length": 1130,
                "sequence_sha256": protein["sequence_sha256"],
                "entity_version": ENTITY_VERSION,
                "entity_bundle_hash": CONTENT_SHA,
                "bundle_file_sha256": entities_sha,
                "model_id": development_manifest.ESM_MODEL_ID,
                "model_revision": MODEL_REVISION,
                "embedding_dim": development_manifest.ESM_EMBEDDING_DIM,
                "status": "unsupported_length",
                "reason": "full_sequence_exceeds_registered_token_domain",
                "finite": False,
                "features": {},
            })
            continue
        if protein["accession"] not in ready_accessions:
            continue
        feature_rows.append(_feature_row(
            protein["accession"], protein["taxonomy_id"], protein["sequence_sha256"],
        ))
    # Deliberately reverse the feature row order; status truth lives per row.
    feature_rows.reverse()
    for row in feature_rows:
        row["bundle_file_sha256"] = entities_sha
    feature_manifest = {
        "schema_version": 1,
        "mode": "local_only_feature_snapshot",
        "status": "blocked",
        "bundle_file_sha256": entities_sha,
        "entity_bundle_hash": CONTENT_SHA,
        "entity_version": ENTITY_VERSION,
        "model": {
            "model_id": development_manifest.ESM_MODEL_ID,
            "revision": MODEL_REVISION,
            "external_revision_verified": False,
            "embedding_dim": development_manifest.ESM_EMBEDDING_DIM,
        },
        "entities": feature_rows,
        "status_counts": {"ready": len(ready_accessions), "unsupported_length": 1},
    }
    feature_manifest_path = root / "features" / "feature_manifest.json"
    _dump_json(feature_manifest_path, feature_manifest)

    train_assignments = [
        {
            "pair_key": _atom("CHEMBL910001", "P91001"),
            "labels": ["active"],
            "split": "train",
            "evidence_ids": ["ev-m1-p1"],
        },
        {
            "pair_key": _atom("CHEMBL910002", "P91002"),
            "labels": ["inactive"],
            "split": "train",
            "evidence_ids": ["ev-m2-p2"],
        },
        {
            "pair_key": _atom("CHEMBL910003", "P91003"),
            "labels": ["active"],
            "split": "train",
            "evidence_ids": ["ev-m3-p3"],
        },
        {
            "pair_key": _atom("CHEMBL910004", "P91004"),
            "labels": ["inactive"],
            "split": "train",
            "evidence_ids": ["ev-m4-p4"],
        },
        {
            "pair_key": _atom("CHEMBL910005", "P91005"),
            "labels": ["active"],
            "split": "train",
            "evidence_ids": ["ev-m5-p5"],
        },
        {
            "pair_key": _atom("CHEMBL910006", "P91006"),
            "labels": [],
            "split": "train",
            "evidence_ids": ["ev-m6-p6"],
        },
    ]
    if only_active:
        train_assignments = [
            assignment for assignment in train_assignments
            if assignment["labels"] != ["inactive"]
        ]
    validation_assignment = {
        "pair_key": _atom("CHEMBL910007", "P91007"),
        "labels": ["active"],
        "split": "val",
        "evidence_ids": ["ev-m7-p7"],
    }
    # Deliberately reversed assignment order (val first, reverse train order).
    assignments = [validation_assignment] + list(reversed(train_assignments))
    manifest = {
        "schema_version": 2,
        "entities": {
            "path": "entities.json",
            "file_sha256": entities_sha,
            "content_sha256": CONTENT_SHA,
            "entity_version": ENTITY_VERSION,
            "schema_version": 2,
        },
        "split": {
            "pair_assignments": assignments,
            "ambiguous_label_quarantine": [
                {"pair_key": _atom("CHEMBL919998", "P919998"), "labels": ["active"]},
            ],
            "crosspair_quarantine": [
                {"pair_key": _atom("CHEMBL919997", "P919997"), "labels": ["inactive"]},
            ],
            "test_only_pair_assignments": [
                {"pair_key": _atom("CHEMBL919996", "P919996"), "labels": ["active"]},
            ],
            "unassigned_pairs": [
                {"pair_key": _atom("CHEMBL919995", "P919995")},
            ],
            "entity_bundle_hash": CONTENT_SHA,
            "entity_version": ENTITY_VERSION,
            "label_policy_sha256": "b" * 64,
            "split_policy_sha256": "d" * 64,
            "split_sha256": "e" * 64,
            "training_ready": False,
        },
    }
    manifest_path = frozen_dir / "manifest.json"
    _dump_json(manifest_path, manifest)
    return {
        "root": root,
        "manifest_path": manifest_path,
        "entities_path": entities_path,
        "feature_manifest_path": feature_manifest_path,
        "manifest": manifest,
        "feature_manifest": feature_manifest,
        "train_assignments": train_assignments,
        "included_keys": [
            _atom("CHEMBL910001", "P91001"),
            _atom("CHEMBL910002", "P91002"),
        ],
    }


def _build(synthetic: dict) -> dict:
    return development_manifest.build_development_manifest(
        synthetic["manifest_path"],
        synthetic["entities_path"],
        synthetic["feature_manifest_path"],
        expected_feature_manifest_path=synthetic["feature_manifest_path"],
        expected_feature_manifest_sha256=_sha256(synthetic["feature_manifest_path"]),
    )


@pytest.fixture()
def synthetic(tmp_path):
    return _build_synthetic_sources(tmp_path / "synthetic")


# ---------------------------------------------------------------------------
# Synthetic contract: complete ledger, stable order, identity joins
# ---------------------------------------------------------------------------

def test_ledger_has_one_disposition_per_train_candidate(synthetic):
    manifest = _build(synthetic)

    assert manifest["purpose"] == development_manifest.PURPOSE
    assert manifest["label_mapping"] == {"inactive": 0, "active": 1}
    assert manifest["eligibility_rules_version"] == (
        development_manifest.ELIGIBILITY_RULES_VERSION
    )

    ledger_keys = [row["pair_key"] for row in manifest["ledger"]]
    assert ledger_keys == sorted(ledger_keys)
    assert len(ledger_keys) == len(set(ledger_keys))
    assert set(ledger_keys) == {
        assignment["pair_key"] for assignment in synthetic["train_assignments"]
    }

    dispositions = [row["disposition"] for row in manifest["ledger"]]
    assert set(dispositions) == {"included", "excluded"}
    assert len(dispositions) == len(synthetic["train_assignments"])
    for row in manifest["ledger"]:
        assert row["purpose"] == development_manifest.PURPOSE
        assert row["split"] == "train"
        if row["disposition"] == "included":
            assert row["reason_code"] == "included_eligible"
        else:
            assert row["reason_code"]
        if row["label"] is not None:
            assert row["label"] in (0, 1)

    included = [row for row in manifest["ledger"] if row["disposition"] == "included"]
    excluded = [row for row in manifest["ledger"] if row["disposition"] == "excluded"]
    assert [row["pair_key"] for row in included] == synthetic["included_keys"]
    assert [row["row_index"] for row in included] == list(range(len(included)))
    assert manifest["row_order"] == [row["pair_key"] for row in manifest["pairs"]]
    assert manifest["row_order"] == synthetic["included_keys"]
    assert [row["pair_key"] for row in manifest["pairs"]] == synthetic["included_keys"]

    assert manifest["counts"]["candidates"] == len(synthetic["train_assignments"])
    assert manifest["counts"]["included"] == len(included) == 2
    assert manifest["counts"]["excluded"] == len(excluded) == 4
    assert manifest["counts"]["class_counts"] == {"inactive": 1, "active": 1}
    assert manifest["counts"]["molecule_count"] == 2
    assert manifest["counts"]["protein_count"] == 2
    assert manifest["source"]["skipped_non_train_assignments"] == 1
    assert manifest["source"]["frozen_pair_assignments"] == len(
        synthetic["manifest"]["split"]["pair_assignments"]
    )
    assert manifest["status"] == "ready"
    assert manifest["blocking_reasons"] == []


def test_every_excluded_candidate_keeps_a_reason(synthetic):
    manifest = _build(synthetic)
    reasons = {row["pair_key"]: row["reason_code"] for row in manifest["ledger"]}
    assert reasons[_atom("CHEMBL910003", "P91003")] == "feature_status_unsupported_length"
    assert reasons[_atom("CHEMBL910004", "P91004")] == "molecule_structure_not_verified"
    assert reasons[_atom("CHEMBL910005", "P91005")] == "feature_row_missing"
    assert reasons[_atom("CHEMBL910006", "P91006")] == (
        "label_missing_or_not_unique_binary"
    )
    missing_label = [
        row for row in manifest["ledger"]
        if row["pair_key"] == _atom("CHEMBL910006", "P91006")
    ][0]
    assert missing_label["label"] is None and missing_label["label_name"] is None
    assert manifest["exclusion_reason_counts"] == {
        "feature_status_unsupported_length": 1,
        "molecule_structure_not_verified": 1,
        "feature_row_missing": 1,
        "label_missing_or_not_unique_binary": 1,
    }
    assert manifest["source"]["feature_status_counts"] == {
        "ready": 5, "unsupported_length": 1,
    }


def test_source_hashes_and_bundle_bindings_are_recorded(synthetic):
    manifest = _build(synthetic)
    source = manifest["source"]
    assert source["frozen_manifest_path"] == str(synthetic["manifest_path"])
    assert source["frozen_manifest_sha256"] == _sha256(synthetic["manifest_path"])
    assert source["entities_path"] == str(synthetic["entities_path"])
    assert source["entities_sha256"] == _sha256(synthetic["entities_path"])
    assert source["entities_sha256"] == (
        synthetic["manifest"]["entities"]["file_sha256"]
    )
    assert source["feature_manifest_path"] == str(synthetic["feature_manifest_path"])
    assert source["feature_manifest_sha256"] == _sha256(
        synthetic["feature_manifest_path"]
    )
    assert source["entity_bundle_hash"] == CONTENT_SHA
    assert source["entity_version"] == ENTITY_VERSION
    assert source["bundle_file_sha256"] == source["entities_sha256"]
    assert source["external_revision_verified"] is False
    assert source["embedding_dim"] == development_manifest.ESM_EMBEDDING_DIM

    features = _load_json(synthetic["feature_manifest_path"])
    cache_by_accession = {
        row["accession"]: (row["cache_key"], row["cache_file_sha256"])
        for row in features["entities"] if row["status"] == "ready"
    }
    for row in manifest["pairs"]:
        assert row["entity_bundle_hash"] == CONTENT_SHA
        assert row["entity_version"] == ENTITY_VERSION
        assert row["bundle_file_sha256"] == source["entities_sha256"]
        assert row["feature_model_id"] == development_manifest.ESM_MODEL_ID
        assert (row["feature_cache_key"], row["feature_cache_file_sha256"]) == (
            cache_by_accession[row["accession"]]
        )
        assert len(row["evidence_ids"]) == 1
        assert row["accession"] in row["pair_key"]
        assert row["molecule_chembl_id"] in row["pair_key"]


def test_other_ledgers_are_never_read(synthetic):
    manifest = _build(synthetic)
    keys = {row["pair_key"] for row in manifest["ledger"]}
    for forbidden in (
        _atom("CHEMBL919998", "P919998"),
        _atom("CHEMBL919997", "P919997"),
        _atom("CHEMBL919996", "P919996"),
        _atom("CHEMBL919995", "P919995"),
        _atom("CHEMBL910007", "P91007"),
    ):
        assert forbidden not in keys
    assert "quarantine" not in json.dumps(manifest).lower()


def test_entity_and_assignment_shuffle_keeps_identity_joins(synthetic):
    first = _build(synthetic)
    manifest = _load_json(synthetic["manifest_path"])
    entities = _load_json(synthetic["entities_path"])
    features = _load_json(synthetic["feature_manifest_path"])
    entities["molecules"] = list(reversed(entities["molecules"]))
    entities["proteins"] = list(reversed(entities["proteins"]))
    _dump_json(synthetic["entities_path"], entities)
    entities_sha = _sha256(synthetic["entities_path"])
    manifest["entities"]["file_sha256"] = entities_sha
    manifest["split"]["pair_assignments"] = list(
        reversed(manifest["split"]["pair_assignments"])
    )
    _dump_json(synthetic["manifest_path"], manifest)
    features["bundle_file_sha256"] = entities_sha
    for row in features["entities"]:
        row["bundle_file_sha256"] = entities_sha
    features["entities"] = list(reversed(features["entities"]))
    _dump_json(synthetic["feature_manifest_path"], features)

    second = _build(synthetic)
    identity_fields = (
        "pair_key", "disposition", "reason_code", "evidence_ids",
        "molecule_chembl_id", "taxonomy_id", "accession", "label", "label_name",
        "molecule_structure_sha256", "protein_sequence_sha256",
        "feature_status", "feature_cache_key", "feature_cache_file_sha256",
        "feature_sequence_sha256", "entity_bundle_hash", "entity_version",
    )
    assert [pair["pair_key"] for pair in first["pairs"]] == [
        pair["pair_key"] for pair in second["pairs"]
    ]
    for left, right in zip(first["pairs"], second["pairs"]):
        for field in identity_fields:
            if field == "pair_key":
                continue
            assert left[field] == right[field], field
    first_ledger = {
        row["pair_key"]: {field: row[field] for field in identity_fields}
        for row in first["ledger"]
    }
    second_ledger = {
        row["pair_key"]: {field: row[field] for field in identity_fields}
        for row in second["ledger"]
    }
    assert second_ledger == first_ledger
    assert second["counts"] == first["counts"]


def test_duplicate_pair_key_is_rejected(synthetic):
    manifest = _load_json(synthetic["manifest_path"])
    manifest["split"]["pair_assignments"].append(
        dict(manifest["split"]["pair_assignments"][-1])
    )
    _dump_json(synthetic["manifest_path"], manifest)
    with pytest.raises(ValueError, match="duplicate_pair_key"):
        _build(synthetic)


def test_tampered_bundle_binding_is_rejected(synthetic):
    manifest = _load_json(synthetic["manifest_path"])
    manifest["entities"]["content_sha256"] = "f" * 64
    _dump_json(synthetic["manifest_path"], manifest)
    with pytest.raises(ValueError, match="entities_content_sha256_mismatch"):
        _build(synthetic)


def test_tampered_entities_bytes_are_rejected(synthetic):
    entities = _load_json(synthetic["entities_path"])
    entities["extra"] = "tamper"
    _dump_json(synthetic["entities_path"], entities)
    with pytest.raises(ValueError, match="entities_file_sha256_mismatch"):
        _build(synthetic)


def test_feature_snapshot_binding_is_pinned(synthetic):
    with pytest.raises(ValueError, match="feature_manifest_sha256_mismatch"):
        development_manifest.build_development_manifest(
            synthetic["manifest_path"],
            synthetic["entities_path"],
            synthetic["feature_manifest_path"],
            expected_feature_manifest_sha256="0" * 64,
            expected_feature_manifest_path=synthetic["feature_manifest_path"],
        )
    with pytest.raises(ValueError, match="feature_manifest_path_mismatch"):
        development_manifest.build_development_manifest(
            synthetic["manifest_path"],
            synthetic["entities_path"],
            synthetic["feature_manifest_path"],
            expected_feature_manifest_sha256=_sha256(
                synthetic["feature_manifest_path"]
            ),
            expected_feature_manifest_path=synthetic["feature_manifest_path"].parent
            / "other_manifest.json",
        )
    feature_manifest = _load_json(synthetic["feature_manifest_path"])
    feature_manifest["entity_bundle_hash"] = "f" * 64
    _dump_json(synthetic["feature_manifest_path"], feature_manifest)
    with pytest.raises(ValueError, match="feature_entity_bundle_hash_mismatch"):
        _build(synthetic)


def test_write_new_manifest_refuses_to_overwrite(synthetic, tmp_path):
    manifest = _build(synthetic)
    target = tmp_path / "new_root" / "development_manifest.json"
    target.parent.mkdir()
    record = development_manifest.write_new_manifest(target, manifest)
    assert record["path"] == str(target)
    assert record["sha256"] == _sha256(target)
    assert record["bytes"] == target.stat().st_size
    assert record["status"] == "ready"
    original_bytes = target.read_bytes()
    with pytest.raises(FileExistsError, match="manifest_already_exists"):
        development_manifest.write_new_manifest(target, manifest)
    assert target.read_bytes() == original_bytes
    stored = _load_json(target)
    assert stored["row_order"] == manifest["row_order"]
    assert "NaN" not in target.read_text(encoding="utf-8")
    assert "Infinity" not in target.read_text(encoding="utf-8")


def test_missing_class_blocks_training_status(tmp_path):
    only_active = _build_synthetic_sources(tmp_path / "synthetic_active", only_active=True)
    manifest = _build(only_active)
    assert manifest["counts"]["class_counts"]["inactive"] == 0
    assert manifest["status"] == "blocked"
    assert "missing_class:inactive" in manifest["blocking_reasons"]
    # The artifact is still writable for the audit trail.
    target = tmp_path / "blocked_root" / "development_manifest.json"
    target.parent.mkdir()
    development_manifest.write_new_manifest(target, manifest)
    assert _load_json(target)["status"] == "blocked"


def test_no_eligible_pairs_blocks_training_status(tmp_path):
    sources = _build_synthetic_sources(tmp_path / "synthetic_noneligible")
    manifest_json = _load_json(sources["manifest_path"])
    manifest_json["split"]["pair_assignments"] = [
        assignment for assignment in manifest_json["split"]["pair_assignments"]
        if assignment["split"] != "train"
    ]
    _dump_json(sources["manifest_path"], manifest_json)
    manifest = _build(sources)
    assert manifest["counts"]["candidates"] == 0
    assert manifest["status"] == "blocked"
    assert "no_eligible_pairs" in manifest["blocking_reasons"]
    assert manifest["pairs"] == []


def test_run_root_guard_requires_fresh_guid_root(tmp_path):
    run_root = development_manifest.PROJECT_ROOT / (
        "P1_03_Artifacts_" + "0123456789abcdef0123456789abcdef"
    )
    resolved = development_manifest.resolve_manifest_run_root(
        run_root / "development_manifest.json"
    )
    assert resolved == run_root / "development_manifest.json"
    assert run_root.is_dir()
    run_root.rmdir()
    with pytest.raises(ValueError, match="run_root_name_must_be"):
        development_manifest.resolve_manifest_run_root(
            tmp_path / "not-a-run-root" / "development_manifest.json"
        )
    with pytest.raises(ValueError, match="out_path_must_be_absolute"):
        development_manifest.resolve_manifest_run_root("relative/development_manifest.json")


# ---------------------------------------------------------------------------
# Real frozen sources: read-only complete-ledger export
# ---------------------------------------------------------------------------

def test_real_sources_produce_complete_source_bound_ledger(tmp_path):
    if not (REAL_MANIFEST.is_file() and REAL_ENTITIES.is_file() and REAL_FEATURES.is_file()):
        pytest.fail("real Phase 02 frozen sources / selected feature snapshot are required")

    manifest = development_manifest.build_development_manifest(
        REAL_MANIFEST, REAL_ENTITIES, REAL_FEATURES,
    )

    frozen = _load_json(REAL_MANIFEST)
    train_assignments = [
        assignment for assignment in frozen["split"]["pair_assignments"]
        if assignment.get("split") == "train"
    ]
    assert manifest["counts"]["candidates"] == len(train_assignments)
    assert manifest["counts"]["included"] == len(manifest["pairs"])
    assert manifest["counts"]["excluded"] == len(manifest["excluded"])
    assert manifest["counts"]["included"] + manifest["counts"]["excluded"] == (
        manifest["counts"]["candidates"]
    )

    ledger_keys = [row["pair_key"] for row in manifest["ledger"]]
    assert ledger_keys == sorted(ledger_keys)
    assert set(ledger_keys) == {
        assignment["pair_key"] for assignment in train_assignments
    }
    assert len(ledger_keys) == len(set(ledger_keys))

    # Independent class recount from the actual train labels of included rows.
    label_by_key = {
        assignment["pair_key"]: assignment["labels"][0]
        for assignment in train_assignments
    }
    expected_counts = {"inactive": 0, "active": 0}
    for row in manifest["pairs"]:
        assert row["label"] == development_manifest.LABEL_MAPPING[label_by_key[row["pair_key"]]]
        expected_counts[row["label_name"]] += 1
    assert manifest["counts"]["class_counts"] == expected_counts
    assert sum(expected_counts.values()) == manifest["counts"]["included"]

    # Source hashes bind the exact files consumed.
    assert manifest["source"]["frozen_manifest_sha256"] == _sha256(REAL_MANIFEST)
    assert manifest["source"]["entities_sha256"] == _sha256(REAL_ENTITIES)
    assert manifest["source"]["feature_manifest_sha256"] == (
        development_manifest.EXPECTED_FEATURE_MANIFEST_SHA256
    )
    assert manifest["source"]["feature_manifest_path"] == str(REAL_FEATURES)

    # Independent identity re-join: every included row must match its entities
    # record and ready feature row by key, never by array position.
    entities = _load_json(REAL_ENTITIES)
    features = _load_json(REAL_FEATURES)
    molecules = {m["molecule_chembl_id"]: m for m in entities["molecules"]}
    proteins = {p["accession"]: p for p in entities["proteins"]}
    feature_rows = {
        (row["accession"], row["taxonomy_id"]): row
        for row in features["entities"]
    }
    for row in manifest["pairs"]:
        molecule = molecules[row["molecule_chembl_id"]]
        protein = proteins[row["accession"]]
        feature_row = feature_rows[(row["accession"], row["taxonomy_id"])]
        assert row["molecule_structure_sha256"] == molecule["structure_sha256"]
        assert row["protein_sequence_sha256"] == protein["sequence_sha256"]
        assert row["feature_cache_key"] == feature_row["cache_key"]
        assert row["feature_cache_file_sha256"] == feature_row["cache_file_sha256"]
        assert row["feature_status"] == "ready"
        assert row["entity_bundle_hash"] == frozen["entities"]["content_sha256"]
        assert row["entity_version"] == frozen["entities"]["entity_version"]

    # Every exclusion carries a stable reason code and a non-null pair key.
    for row in manifest["excluded"]:
        assert row["reason_code"]
        assert row["pair_key"]
        assert row["disposition"] == "excluded"

    assert manifest["status"] == "ready"
    assert manifest["blocking_reasons"] == []
    assert manifest["source"]["external_revision_verified"] is False
    text = json.dumps(manifest, ensure_ascii=False)
    assert "NaN" not in text and "Infinity" not in text
