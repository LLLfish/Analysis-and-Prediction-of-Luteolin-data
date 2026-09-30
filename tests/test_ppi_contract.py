"""Offline PPI contract tests that exercise the existing consumer entry points."""

from main import NetworkPharmacologyFramework
from knowledge_graph_module import KnowledgeGraph


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
