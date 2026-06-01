import os
import tempfile
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _run_case(name, func):
    try:
        func()
        return name, True, ""
    except Exception as exc:
        return name, False, f"{exc}\n{traceback.format_exc()}"


def test_target_scoring():
    from target_scoring import organism_distribution, rank_targets

    targets = [
        {"target_name": "Cyclin-dependent kinase 2", "gene_symbol": "CDK2", "organism": "Homo sapiens", "activity_type": "IC50", "activity_value": "20", "activity_unit": "nM"},
        {"target_name": "Unchecked", "organism": None, "activity_type": "IC50", "activity_value": "1", "activity_unit": "nM"},
        {"target_name": "Cyclin-dependent kinase 6", "gene_symbol": "CDK6", "organism": "Homo sapiens", "activity_type": "IC50", "activity_value": "300000", "activity_unit": "nM"},
    ]
    ranked = rank_targets(targets)
    assert ranked[0][0] == "CDK2"
    assert all(org != "Unknown" for org, _ in organism_distribution(targets))


def test_data_cleaning():
    from data_collection import LuteolinDataCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = LuteolinDataCollector(output_dir=tmpdir)
        collector.raw_data = {
            "pubchem": {
                "cid": 5280445,
                "properties": {
                    "Molecular_Formula": "C15H10O6",
                    "Molecular_Weight": 286.24,
                    "Canonical_SMILES": "C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O",
                },
            },
            "chembl": {
                "chembl_id": "CHEMBL151",
                "targets": [
                    {"target_chembl_id": "CHEMBL301", "target_name": "Cyclin-dependent kinase 2", "organism": "Homo sapiens", "activity_type": "IC50", "activity_value": "20000", "activity_unit": "nM"},
                    {"target_chembl_id": "CHEMBL612545", "target_name": "Unchecked", "organism": None, "activity_type": "IC50", "activity_value": "10400", "activity_unit": "nM"},
                ],
            },
        }
        cleaned = collector.clean_and_standardize()
        assert cleaned["targets"][0]["gene_symbol"] == "CDK2"
        assert cleaned["targets"][1]["organism"] == "Unknown"
        assert cleaned["targets"][1]["is_valid_target"] is False


def test_gnn_module():
    import networkx as nx
    from gnn_module import GraphEncoder, NetworkAnalyzer

    encoder = GraphEncoder()
    encoded = encoder.encode_molecular_graph({
        "atoms": [{"atomic_num": 6}, {"atomic_num": 8}],
        "bonds": [{"begin": 0, "end": 1, "type": "SINGLE"}],
    })
    assert encoded is not None
    graph = nx.Graph()
    graph.add_edge("AKT1", "PIK3CA")
    stats = NetworkAnalyzer().compute_graph_statistics(graph)
    assert stats["num_nodes"] == 2


def test_ligand_and_cnn_modules():
    from ligand_features import LigandFeatureExtractor, _RDKIT_AVAILABLE
    from cnn_module import MolecularCNN, RDKIT_AVAILABLE

    if not _RDKIT_AVAILABLE:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = LigandFeatureExtractor(output_dir=os.path.join(tmpdir, "ligand"))
        features = extractor.extract_all_features("Oc1cc(O)c2c(c1)oc(cc2=O)c1ccc(O)c(O)c1", "luteolin")
        assert features["success"] is True
        assert os.path.exists(os.path.join(tmpdir, "ligand", "structures", "luteolin.pdb"))

        if RDKIT_AVAILABLE and extractor.mol_3d is not None:
            cnn = MolecularCNN(output_dir=os.path.join(tmpdir, "cnn"))
            grid = cnn.molecule_to_grid(extractor.mol_3d)
            assert grid is not None


def test_transformer_module():
    import numpy as np
    from transformer_module import FeatureFusion

    fusion = FeatureFusion()
    output = fusion.fuse_features(np.ones((1, 64)), np.ones((1, 64)), np.ones((1, 64)))
    assert output is not None


def test_docking_module():
    from docking_module import DockingValidator

    validator = DockingValidator(output_dir=tempfile.mkdtemp())
    result = validator.validate_docking({
        "status": "estimated",
        "binding_affinity": 7.2,
        "docking_modes": [{"mode": 1, "affinity": 7.2}],
    })
    assert result["binding_affinity"] == -7.2
    assert result["docking_modes"][0]["affinity"] == -7.2


def test_knowledge_graph_module():
    from knowledge_graph_module import KnowledgeGraph

    with tempfile.TemporaryDirectory() as tmpdir:
        kg = KnowledgeGraph(output_dir=tmpdir)
        kg.build_from_data(
            {"name": "Luteolin"},
            [{"target_name": "Cyclin-dependent kinase 2", "gene_symbol": "CDK2"}],
            [{"preferredName_A": "CDK2", "preferredName_B": "CDK6", "score": 0.8}],
        )
        stats = kg.compute_graph_statistics()
        triples_path = kg.save_triples()
        assert stats["num_nodes"] >= 3
        assert os.path.exists(triples_path)


def test_output_module():
    from output_module import OutputGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = OutputGenerator(output_dir=os.path.join(tmpdir, "output"))
        report = generator.generate_summary_report({
            "compound": {"name": "Luteolin", "name_cn": "木犀草素"},
            "targets": [{"target_name": "Cyclin-dependent kinase 2", "gene_symbol": "CDK2", "organism": "Homo sapiens", "activity_type": "IC50", "activity_value": "20", "activity_unit": "nM"}],
            "docking": {"binding_affinity": -7.2, "docking_modes": [{"mode": 1, "affinity": -7.2}]},
            "gnn": {"num_nodes": 1, "num_edges": 0},
            "knowledge_graph": {"graph": {"stats": {"num_nodes": 2, "num_edges": 1, "num_relations": 1}}},
        })
        path = generator.generate_dashboard_html(report)
        assert os.path.exists(path)


def main():
    cases = [
        ("靶点评分", test_target_scoring),
        ("数据清洗", test_data_cleaning),
        ("GNN/网络分析", test_gnn_module),
        ("配体/CNN", test_ligand_and_cnn_modules),
        ("Transformer融合", test_transformer_module),
        ("分子对接", test_docking_module),
        ("知识图谱", test_knowledge_graph_module),
        ("输出报告", test_output_module),
    ]
    results = [_run_case(name, func) for name, func in cases]
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"模块测试完成: {passed}/{len(results)} 通过")
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if detail:
            print(detail)
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
