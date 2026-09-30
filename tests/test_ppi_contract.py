"""Offline PPI contract tests that exercise the existing consumer entry points."""

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import replace
import json
import math

from main import NetworkPharmacologyFramework
from knowledge_graph_module import KnowledgeGraph
from data_contracts import (
    CanonicalPpiEdge,
    CanonicalPpiEndpoint,
    PpiRecordProvenance,
    canonical_json_sha256,
    canonicalize_ppi,
    read_ppi_records,
)


ONLINE_EDGE = {
    "stringId_A": "9606.ENSP00000354587",
    "stringId_B": "9606.ENSP00000263967",
    "preferredName_A": "AKT1",
    "preferredName_B": "PIK3CA",
    "ncbiTaxonId": 9606,
    "score": 0.985,
    "retrieved_at": "2026-09-30T08:00:00Z",
}

OFFLINE_EDGE = {
    "protein_a": "AKT1",
    "protein_b": "PIK3CA",
    "ncbiTaxonId": 9606,
    "score": 0.985,
    "retrieved_at": "2026-09-30T09:00:00Z",
}

TRUSTED_TEST_MAPPING = {
    "AKT1": {
        "source_native_id": "9606.ENSP00000354587",
        "taxonomy_id": 9606,
        "verified": True,
        "mapping_source": "synthetic_test:explicit mapping fixture",
    },
    "PIK3CA": {
        "source_native_id": "9606.ENSP00000263967",
        "taxonomy_id": 9606,
        "verified": True,
        "mapping_source": "synthetic_test:explicit mapping fixture",
    },
}


class _StubLigandExtractor:
    def extract_all_features(self, smiles, compound_id):
        return {"status": "stubbed", "features": {}}


class _StubProteinExtractor:
    def load_esm_model(self, timeout):
        return False

    def extract_all_features(self, sequence, protein_id):
        return {"status": "stubbed", "features": {}}


class _CapturingNetworkAnalyzer:
    """Capture the production NetworkX graph without creating a learned model."""

    def __init__(self):
        self.graph = None

    def compute_centrality(self, graph):
        self.graph = graph.copy()
        return {}

    def compute_graph_statistics(self, graph):
        self.graph = graph.copy()
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        }


def _dataset(interactions, **extra):
    return {
        "compound": {"name": "木犀草素", "smiles": "C"},
        "targets": [],
        "proteins": [],
        "interactions": interactions,
        **extra,
    }


def _framework(tmp_path, label):
    # Avoid __init__: it constructs every feature/model component. The methods
    # under test are production methods, with only their dependencies replaced.
    framework = object.__new__(NetworkPharmacologyFramework)
    framework.results = {}
    framework.output_dir = str(tmp_path / label)
    framework.ligand_extractor = _StubLigandExtractor()
    framework.protein_extractor = _StubProteinExtractor()
    framework.gnn_analyzer = _CapturingNetworkAnalyzer()
    framework.cnn_module = None
    framework.kg_module = KnowledgeGraph(output_dir=str(tmp_path / f"{label}-kg"))
    framework.kg_embedder = None
    return framework


def _run_consumers(tmp_path, label, dataset):
    framework = _framework(tmp_path, label)
    features = framework.run_feature_extraction(dataset)
    kg_results = framework.run_knowledge_graph(dataset)
    return framework, features, kg_results


def _ppi_graph_edges(graph):
    return {
        (left, right, properties["weight"])
        for left, right, properties in graph.edges(data=True)
        if properties.get("relation") == "interacts_with"
    }


def test_online_offline_tracer_reaches_networkx_and_knowledge_graph(tmp_path):
    online = _dataset([ONLINE_EDGE], ppi_source_kind="synthetic_test")
    offline = _dataset(
        [OFFLINE_EDGE],
        ppi_source_kind="synthetic_test",
        ppi_alias_mappings=TRUSTED_TEST_MAPPING,
    )

    online_framework, online_features, online_kg = _run_consumers(
        tmp_path, "online", online
    )
    offline_framework, offline_features, offline_kg = _run_consumers(
        tmp_path, "offline", offline
    )

    online_ppi = online_features.get("ppi", {})
    offline_ppi = offline_features.get("ppi", {})
    assert online_ppi.get("status") == "ready"
    assert offline_ppi.get("status") == "ready"
    assert online_ppi.get("edge_count") == offline_ppi.get("edge_count") == 1
    assert online_ppi.get("semantic_sha256") == offline_ppi.get("semantic_sha256")
    assert online_ppi.get("provenance_sha256") != offline_ppi.get("provenance_sha256")
    assert online_ppi["edges"][0]["source_record_hashes"] != offline_ppi["edges"][0]["source_record_hashes"]

    expected_a = (9606, "9606.ENSP00000263967")
    expected_b = (9606, "9606.ENSP00000354587")
    expected_edge = (expected_a, expected_b, 0.985)
    assert online_framework.gnn_analyzer.graph.number_of_edges() == 1
    assert offline_framework.gnn_analyzer.graph.number_of_edges() == 1
    assert _ppi_graph_edges(online_framework.kg_module.graph) == {expected_edge}
    assert _ppi_graph_edges(offline_framework.kg_module.graph) == {expected_edge}
    assert online_kg["graph"]["stats"]["num_edges"] == 1
    assert offline_kg["graph"]["stats"]["num_edges"] == 1
    assert online_features["gnn"]["model_type"] == "network_statistics_only"


def test_symbol_only_offline_row_without_mapping_is_unresolved(tmp_path):
    dataset = _dataset(
        [{"protein_a": "AKT1", "protein_b": "PIK3CA", "ncbiTaxonId": 9606, "score": 0.985}],
        ppi_source_kind="legacy_unverified",
    )

    framework, features, kg_results = _run_consumers(tmp_path, "unresolved", dataset)

    ppi = features.get("ppi", {})
    assert ppi.get("status") == "unresolved"
    assert ppi.get("edge_count") == 0
    assert ppi.get("node_count") == 0
    assert ppi.get("issues", [])[0]["code"] == "unresolved_alias"
    assert ppi["issues"][0]["record_index"] == 0
    assert framework.gnn_analyzer.graph is None
    assert framework.kg_module.graph.number_of_nodes() == 0
    assert framework.kg_module.graph.number_of_edges() == 0
    assert kg_results["graph"]["stats"]["num_edges"] == 0


def test_empty_ppi_input_produces_empty_network_and_knowledge_graph(tmp_path):
    framework, features, kg_results = _run_consumers(
        tmp_path, "empty", _dataset([])
    )

    ppi = features.get("ppi", {})
    assert ppi.get("status") == "empty"
    assert ppi.get("node_count") == 0
    assert ppi.get("edge_count") == 0
    assert ppi.get("issues") == []
    assert features["gnn"]["status"] == "empty"
    assert features["gnn"]["num_nodes"] == 0
    assert features["gnn"]["num_edges"] == 0
    assert framework.kg_module.graph.number_of_nodes() == 0
    assert framework.kg_module.graph.number_of_edges() == 0
    assert kg_results["graph"]["stats"]["num_nodes"] == 0
    assert kg_results["graph"]["stats"]["num_edges"] == 0


def test_fixture_manifest_accounts_for_every_synthetic_and_legacy_record(ppi_fixture_manifest):
    assert ppi_fixture_manifest["purpose"] == "synthetic_test_only"
    assert ppi_fixture_manifest["schema_version"] == 2

    for fixture in ppi_fixture_manifest["fixtures"]:
        result = canonicalize_ppi(
            [fixture["record"]],
            source_kind=fixture["source_kind"],
            alias_mappings=fixture.get("alias_mappings"),
        )
        expected = fixture["expected"]
        assert result.input_count == 1, fixture["id"]
        assert result.status == expected["status"], fixture["id"]
        assert result.accepted_record_count + len({issue.record_index for issue in result.issues}) == 1, fixture["id"]
        if expected["issue_code"] is None:
            assert result.edge_count == 1, fixture["id"]
            assert result.issues == (), fixture["id"]
        else:
            matching = [issue for issue in result.issues if issue.code == expected["issue_code"]]
            assert matching, fixture["id"]
            assert matching[0].record_index == 0
            assert len(matching[0].source_record_hash) == 64


def test_score_boundaries_and_invalid_types_have_explicit_ledger_codes():
    base = {
        "source_native_id_a": "9606.ENSP00000354587",
        "source_native_id_b": "9606.ENSP00000263967",
        "taxonomy_id": 9606,
    }
    for score in (0, 1, 0.5):
        result = canonicalize_ppi([{**base, "score": score}], source_kind="synthetic_test")
        assert result.status == "ready"
        assert result.edges[0].score == float(score)

    invalid_cases = (
        (True, "invalid_score_type"),
        ("0.5", "invalid_score_type"),
        (math.nan, "non_finite_score"),
        (math.inf, "non_finite_score"),
        (-0.01, "score_out_of_range"),
        (1.01, "score_out_of_range"),
    )
    for score, expected_code in invalid_cases:
        result = canonicalize_ppi([{**base, "score": score}], source_kind="synthetic_test")
        assert result.edge_count == 0
        assert result.accepted_record_count == 0
        assert result.issues[0].code == expected_code
        assert result.issues[0].record_index == 0
        assert len(result.issues[0].source_record_hash) == 64


def test_reversed_duplicates_and_edge_order_have_stable_semantic_hashes():
    records = [
        {
            "source_native_id_a": "9606.PROTEIN_B",
            "source_native_id_b": "9606.PROTEIN_A",
            "taxonomy_id": 9606,
            "score": 0.5,
            "retrieved_at": "2026-09-30T08:00:00Z",
        },
        {
            "source_native_id_a": "9606.PROTEIN_A",
            "source_native_id_b": "9606.PROTEIN_B",
            "taxonomy_id": 9606,
            "score": 0.5,
            "retrieved_at": "2026-09-30T09:00:00Z",
        },
        {
            "source_native_id_a": "9606.PROTEIN_D",
            "source_native_id_b": "9606.PROTEIN_C",
            "taxonomy_id": 9606,
            "score": 0.8,
        },
    ]
    first = canonicalize_ppi(records, source_kind="synthetic_test")
    reordered = canonicalize_ppi(list(reversed(records)), source_kind="synthetic_test")

    assert first.status == reordered.status == "ready"
    assert first.edge_count == reordered.edge_count == 2
    assert [edge.edge_key for edge in first.edges] == [edge.edge_key for edge in reordered.edges]
    assert first.semantic_sha256 == reordered.semantic_sha256
    duplicate = next(edge for edge in first.edges if edge.score == 0.5)
    assert len(duplicate.provenance) == 2
    assert len(duplicate.source_record_hashes) == 2
    assert duplicate.provenance[0].retrieved_at != duplicate.provenance[1].retrieved_at
    assert canonical_json_sha256(
        {"edge": "same", "retrieved_at": "first"}, exclude_fields=("retrieved_at",)
    ) == canonical_json_sha256(
        {"edge": "same", "retrieved_at": "second"}, exclude_fields=("retrieved_at",)
    )


def test_chinese_path_json_and_utf8_sig_csv_round_trip_identity(tmp_path):
    output_dir = tmp_path / "木犀草素数据" / "PPI fixtures"
    output_dir.mkdir(parents=True)
    json_path = output_dir / "在线快照.json"
    csv_path = output_dir / "离线快照.csv"
    json_record = {
        "source_native_id_a": "9606.测试蛋白甲",
        "source_native_id_b": "9606.测试蛋白乙",
        "taxonomy_id": 9606,
        "display_name_a": "木犀草素相关靶点",
        "display_name_b": "蛋白乙",
        "score": 0.75,
        "retrieved_at": "2026-09-30T08:00:00Z",
    }
    with json_path.open("w", encoding="utf-8", newline="") as stream:
        json.dump([json_record], stream, ensure_ascii=False)

    csv_record = {
        **json_record,
        "display_name_a": "不同显示名",
        "display_name_b": "另一个显示名",
        "retrieved_at": "2026-09-30T09:00:00Z",
    }
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_record))
        writer.writeheader()
        writer.writerow(csv_record)

    json_result = canonicalize_ppi(read_ppi_records(json_path), source_kind="synthetic_test")
    csv_result = canonicalize_ppi(read_ppi_records(csv_path), source_kind="synthetic_test")
    assert json_result.status == csv_result.status == "ready"
    assert json_result.edges[0].endpoint_a.identity_key == csv_result.edges[0].endpoint_a.identity_key
    assert json_result.edges[0].endpoint_b.identity_key == csv_result.edges[0].endpoint_b.identity_key
    assert json_result.semantic_sha256 == csv_result.semantic_sha256
    assert json_result.provenance_sha256 != csv_result.provenance_sha256
    assert "测试蛋白甲" in json_result.edges[0].endpoint_a.source_native_id or "测试蛋白甲" in json_result.edges[0].endpoint_b.source_native_id


def test_concurrent_normalization_keeps_result_ledgers_isolated():
    valid = [{
        "stringId_A": "9606.ENSP00000354587",
        "stringId_B": "9606.ENSP00000263967",
        "ncbiTaxonId": 9606,
        "score": 0.5,
    }]
    invalid = [{
        "stringId_A": "9606.ENSP00000354587",
        "stringId_B": "9606.ENSP00000263967",
        "ncbiTaxonId": 9606,
        "score": True,
    }]

    def normalize(index):
        rows = valid if index % 2 == 0 else invalid
        return index % 2, canonicalize_ppi(rows, source_kind="synthetic_test")

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(normalize, range(24)))

    valid_results = [result for invalid_flag, result in results if invalid_flag == 0]
    invalid_results = [result for invalid_flag, result in results if invalid_flag == 1]
    assert {result.semantic_sha256 for result in valid_results} == {valid_results[0].semantic_sha256}
    assert all(result.issues == () and result.edge_count == 1 for result in valid_results)
    assert all(result.issues[0].code == "invalid_score_type" for result in invalid_results)
    assert all(result.edge_count == 0 for result in invalid_results)


def test_unknown_schema_fields_and_display_only_names_never_become_edges():
    record = {
        "stringId_A": "9606.ENSP00000354587",
        "stringId_B": "9606.ENSP00000263967",
        "ncbiTaxonId": 9606,
        "score": 0.5,
    }
    unknown_version = canonicalize_ppi(
        {"schema_version": 99, "interactions": [record]}, source_kind="synthetic_test"
    )
    unknown_field = canonicalize_ppi(
        [{**record, "unexpected_identity": "AKT1"}], source_kind="synthetic_test"
    )
    display_only = canonicalize_ppi(
        [{"preferredName_A": "AKT1", "preferredName_B": "PIK3CA", "score": 0.5}],
        source_kind="legacy_unverified",
    )

    assert unknown_version.edge_count == 0
    assert unknown_version.issues[0].code == "unsupported_schema_version"
    assert unknown_version.issues[0].record_index == 0
    assert unknown_field.edge_count == 0
    assert unknown_field.issues[0].code == "unknown_fields"
    assert display_only.edge_count == 0
    assert display_only.issues[0].code == "missing_endpoint"


def test_envelope_and_source_metadata_require_exact_supported_types():
    record = {
        "stringId_A": "9606.ENSP00000354587",
        "stringId_B": "9606.ENSP00000263967",
        "ncbiTaxonId": 9606,
        "score": 0.5,
    }
    legacy_record = {
        "protein_a": "AKT1",
        "protein_b": "PIK3CA",
        "ncbiTaxonId": 9606,
        "score": 0.5,
    }
    float_version = canonicalize_ppi(
        {"schema_version": 2.0, "interactions": [record]}, source_kind="synthetic_test"
    )
    invalid_source = canonicalize_ppi([record], source_kind={})
    invalid_mappings = canonicalize_ppi({
        "schema_version": 2,
        "source_kind": "legacy_unverified",
        "interactions": [legacy_record],
        "alias_mappings": [],
    })
    unasserted_origin = canonicalize_ppi([record])

    assert float_version.status == "invalid"
    assert float_version.edge_count == 0
    assert float_version.issues[0].code == "unsupported_schema_version"
    assert invalid_source.status == "invalid"
    assert invalid_source.edge_count == 0
    assert invalid_source.issues[0].code == "invalid_source_kind"
    assert invalid_mappings.status == "invalid"
    assert invalid_mappings.edge_count == 0
    assert invalid_mappings.issues[0].code == "invalid_alias_mappings"
    assert unasserted_origin.status == "ready"
    assert unasserted_origin.edges[0].provenance[0].source_kind == "legacy_unverified"


def test_untrusted_canonical_dataclass_edges_and_results_are_rejected():
    endpoint_a = CanonicalPpiEndpoint(9606, "9606.PROTEIN_A")
    endpoint_b = CanonicalPpiEndpoint(9606, "9606.PROTEIN_B")
    forged_edge = CanonicalPpiEdge(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        score=math.nan,
        provenance=(),
        semantic_sha256="f" * 64,
    )
    forged_edge_result = canonicalize_ppi([forged_edge], source_kind="synthetic_test")
    valid_result = canonicalize_ppi([{
        "stringId_A": endpoint_a.source_native_id,
        "stringId_B": endpoint_b.source_native_id,
        "ncbiTaxonId": 9606,
        "score": 0.5,
    }], source_kind="synthetic_test")
    forged_result = replace(valid_result, semantic_sha256="f" * 64)
    checked_result = canonicalize_ppi(forged_result)

    assert forged_edge_result.status == "invalid"
    assert forged_edge_result.edge_count == 0
    assert forged_edge_result.issues[0].code == "invalid_canonical_edge"
    assert checked_result.status == "invalid"
    assert checked_result.edge_count == 0
    assert checked_result.issues[0].code == "invalid_canonical_result"


def test_untrusted_alias_mapping_hashes_remain_in_issue_provenance():
    record = {
        "protein_a": "AKT1",
        "protein_b": "PIK3CA",
        "ncbiTaxonId": 9606,
        "score": 0.5,
    }
    mapping_one = {
        "AKT1": {
            "source_native_id": "9606.ENSP00000354587",
            "taxonomy_id": 9606,
            "verified": False,
            "mapping_source": "unverified-source-one",
        },
        "PIK3CA": {
            "source_native_id": "9606.ENSP00000263967",
            "taxonomy_id": 9606,
            "verified": False,
            "mapping_source": "unverified-source-one",
        },
    }
    mapping_two = {
        alias: {**mapping, "mapping_source": "unverified-source-two"}
        for alias, mapping in mapping_one.items()
    }
    first = canonicalize_ppi([record], source_kind="legacy_unverified", alias_mappings=mapping_one)
    second = canonicalize_ppi([record], source_kind="legacy_unverified", alias_mappings=mapping_two)

    issue_hashes = getattr(first.issues[0], "mapping_hashes", ())
    assert issue_hashes == (canonical_json_sha256(mapping_one["AKT1"]),)
    assert first.provenance_sha256 != second.provenance_sha256


def test_rebuilding_knowledge_graph_replaces_previous_ppi_state(tmp_path):
    first = canonicalize_ppi([{
        "stringId_A": "9606.ENSP00000000001",
        "stringId_B": "9606.ENSP00000000002",
        "ncbiTaxonId": 9606,
        "score": 0.6,
    }], source_kind="synthetic_test")
    second = canonicalize_ppi([{
        "stringId_A": "9606.ENSP00000000003",
        "stringId_B": "9606.ENSP00000000004",
        "ncbiTaxonId": 9606,
        "score": 0.8,
    }], source_kind="synthetic_test")
    graph = KnowledgeGraph(output_dir=str(tmp_path / "repeat-build"))

    graph.build_from_data({}, [], first)
    assert graph.compute_graph_statistics()["num_edges"] == 1
    graph.build_from_data({}, [], first)
    assert graph.compute_graph_statistics()["num_edges"] == 1
    assert len(graph.triples) == 1

    graph.build_from_data({}, [], second)
    ppi_edges = [
        (left, right, properties)
        for left, right, properties in graph.graph.edges(data=True)
        if properties.get("relation") == "interacts_with"
    ]
    assert len(ppi_edges) == 1
    assert {ppi_edges[0][0], ppi_edges[0][1]} == {
        (9606, "9606.ENSP00000000003"),
        (9606, "9606.ENSP00000000004"),
    }
    assert len(graph.triples) == 1
    assert graph.ppi_contract["semantic_sha256"] == second.semantic_sha256
