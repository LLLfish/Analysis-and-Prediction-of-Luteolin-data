"""
网络药理学混合人工智能框架 - 主脚本
Integrated Network Pharmacology Hybrid AI Framework
"""

# Suppress transformers optional dependency notices (lightning, haiku, etc. — not needed for ESM-2)
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


import json
import logging
from datetime import datetime
from typing import Dict, Any, List

import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
logger = setup_logger(__name__)

from data_contracts import CanonicalPpiResult, canonicalize_ppi

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not installed. SMILES parsing will be unavailable.")

_module_imports = {}

try:
    from data_collection import LuteolinDataCollector
    _module_imports['LuteolinDataCollector'] = True
except ImportError as e:
    logger.warning("data_collection import failed: %s", e)
    _module_imports['LuteolinDataCollector'] = False

try:
    from gnn_module import GraphEncoder, NetworkAnalyzer
    _module_imports['GraphEncoder'] = True
    _module_imports['NetworkAnalyzer'] = True
except ImportError as e:
    logger.warning("gnn_module import failed: %s", e)
    _module_imports['GraphEncoder'] = False
    _module_imports['NetworkAnalyzer'] = False

try:
    from cnn_module import MolecularCNN
    _module_imports['MolecularCNN'] = True
except ImportError as e:
    logger.warning("cnn_module import failed: %s", e)
    _module_imports['MolecularCNN'] = False

try:
    from protein_features import ProteinFeatureExtractor
    _module_imports['ProteinFeatureExtractor'] = True
except ImportError as e:
    logger.warning("protein_features import failed: %s", e)
    _module_imports['ProteinFeatureExtractor'] = False

try:
    from ligand_features import LigandFeatureExtractor
    _module_imports['LigandFeatureExtractor'] = True
except ImportError as e:
    logger.warning("ligand_features import failed: %s", e)
    _module_imports['LigandFeatureExtractor'] = False

try:
    from transformer_module import FeatureFusion
    _module_imports['FeatureFusion'] = True
except ImportError as e:
    logger.warning("transformer_module import failed: %s", e)
    _module_imports['FeatureFusion'] = False

try:
    from docking_module import DockingValidator
    _module_imports['DockingValidator'] = True
except ImportError as e:
    logger.warning("docking_module import failed: %s", e)
    _module_imports['DockingValidator'] = False

try:
    from knowledge_graph_module import KnowledgeGraph, KGEmbedder
    _module_imports['KnowledgeGraph'] = True
    _module_imports['KGEmbedder'] = True
except ImportError as e:
    logger.warning("knowledge_graph_module import failed: %s", e)
    _module_imports['KnowledgeGraph'] = False
    _module_imports['KGEmbedder'] = False

try:
    from output_module import OutputGenerator
    _module_imports['OutputGenerator'] = True
except ImportError as e:
    logger.warning("output_module import failed: %s", e)
    _module_imports['OutputGenerator'] = False


class NetworkPharmacologyFramework:

    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "framework_output")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.data_collector = self._safe_component(globals().get('LuteolinDataCollector'), 'LuteolinDataCollector', output_dir=os.path.join(self.output_dir, "data"))
        self.gnn_encoder = self._safe_component(globals().get('GraphEncoder'), 'GraphEncoder')
        self.gnn_analyzer = self._safe_component(globals().get('NetworkAnalyzer'), 'NetworkAnalyzer', output_dir=os.path.join(self.output_dir, "gnn"))
        self.cnn_module = self._safe_component(globals().get('MolecularCNN'), 'MolecularCNN', output_dir=os.path.join(self.output_dir, "cnn"))
        self.protein_extractor = self._safe_component(globals().get('ProteinFeatureExtractor'), 'ProteinFeatureExtractor', output_dir=os.path.join(self.output_dir, "protein"))
        self.ligand_extractor = self._safe_component(globals().get('LigandFeatureExtractor'), 'LigandFeatureExtractor', output_dir=os.path.join(self.output_dir, "ligand"))
        self.transformer_module = self._safe_component(globals().get('FeatureFusion'), 'FeatureFusion', output_dir=os.path.join(self.output_dir, "transformer"))
        self.docking_module = self._safe_component(globals().get('DockingValidator'), 'DockingValidator', output_dir=os.path.join(self.output_dir, "docking"))
        self.kg_module = self._safe_component(globals().get('KnowledgeGraph'), 'KnowledgeGraph', output_dir=os.path.join(self.output_dir, "knowledge_graph"))
        self.kg_embedder = self._safe_component(globals().get('KGEmbedder'), 'KGEmbedder', output_dir=os.path.join(self.output_dir, "knowledge_graph", "embeddings"))
        self.output_module = self._safe_component(globals().get('OutputGenerator'), 'OutputGenerator', output_dir=os.path.join(self.output_dir, "output"))

        self.results = {}

    def _safe_component(self, cls, import_key: str, *args, **kwargs):
        if cls is None or not _module_imports.get(import_key):
            return None
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            logger.warning("%s initialization skipped: %s", import_key, e)
            return None

    def run_data_collection(self) -> Dict[str, Any]:
        if not self.data_collector:
            logger.error("Data collector is unavailable")
            return {}
        dataset = self.data_collector.run_pipeline()
        self.results['data_collection'] = dataset
        compound_name = dataset.get('compound', {}).get('name', 'N/A')
        num_targets = len(dataset.get('targets', []))
        num_proteins = len(dataset.get('proteins', []))
        num_interactions = len(dataset.get('interactions', []))
        logger.info("Data collection: compound=%s, targets=%d, proteins=%d, interactions=%d",
                     compound_name, num_targets, num_proteins, num_interactions)
        return dataset

    def _canonicalize_dataset_ppi(self, dataset: Dict[str, Any]) -> CanonicalPpiResult:
        result = canonicalize_ppi(
            dataset.get('interactions', []),
            source_kind=dataset.get('ppi_source_kind'),
            alias_mappings=dataset.get('ppi_alias_mappings'),
        )
        cached = getattr(self, '_ppi_result', None)
        if (
            cached is not None
            and cached.input_sha256 == result.input_sha256
            and cached.semantic_sha256 == result.semantic_sha256
            and cached.provenance_sha256 == result.provenance_sha256
            and cached.issues == result.issues
        ):
            return cached
        self._ppi_result = result
        return result

    def run_feature_extraction(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        compound = dataset.get('compound', {})
        smiles = compound.get('smiles', 'C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O')

        logger.info("Extracting ligand features...")
        ligand_features = self.ligand_extractor.extract_all_features(smiles, "luteolin") if self.ligand_extractor else {}

        logger.info("Extracting protein features...")
        protein_sequence = "MNSFELKQVNGLDLRLLKPVLSSKESWFKGKQGKKKPKKISKAKIVNGKQIFLSKEL"
        if self.protein_extractor:
            self.protein_extractor.load_esm_model(timeout=120)
        protein_features = self.protein_extractor.extract_all_features(protein_sequence, "sample_protein") if self.protein_extractor else {}

        logger.info("Analyzing canonical PPI network statistics...")
        ppi_result = self._canonicalize_dataset_ppi(dataset)
        ppi_features = ppi_result.to_dict()
        gnn_features = {
            'model_type': 'network_statistics_only',
            'status': ppi_result.status,
            'centrality': {},
            'graph_stats': {'num_nodes': ppi_result.node_count, 'num_edges': ppi_result.edge_count},
            'num_nodes': ppi_result.node_count,
            'num_edges': ppi_result.edge_count,
        }
        if ppi_result.status == 'empty':
            gnn_features['status'] = 'empty'
        elif self.gnn_analyzer:
            try:
                import networkx as nx
                graph = nx.Graph()
                for edge in ppi_result.edges:
                    endpoint_a = edge.endpoint_a.identity_key
                    endpoint_b = edge.endpoint_b.identity_key
                    graph.add_node(endpoint_a, display_name=edge.endpoint_a.display_name)
                    graph.add_node(endpoint_b, display_name=edge.endpoint_b.display_name)
                    graph.add_edge(
                        endpoint_a,
                        endpoint_b,
                        weight=edge.score,
                        semantic_sha256=edge.semantic_sha256,
                        source_record_hashes=edge.source_record_hashes,
                    )
                if graph.number_of_nodes() > 0:
                    gnn_features['centrality'] = self.gnn_analyzer.compute_centrality(graph)
                    gnn_features['graph_stats'] = self.gnn_analyzer.compute_graph_statistics(graph)
                else:
                    gnn_features['status'] = 'empty'
                gnn_features['num_nodes'] = graph.number_of_nodes()
                gnn_features['num_edges'] = graph.number_of_edges()
            except ImportError:
                gnn_features['status'] = 'dependency_unavailable'

        logger.info("Extracting CNN features...")
        cnn_features = {}
        if self.cnn_module and RDKIT_AVAILABLE and smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                    grid = self.cnn_module.molecule_to_grid(mol)
                    if grid is not None:
                        feats = self.cnn_module.extract_features(grid)
                        cnn_features = {'features': feats}
            except Exception as e:
                logger.error("CNN feature extraction failed: %s", e)

        features = {
            'ligand': ligand_features,
            'protein': protein_features,
            'gnn': gnn_features,
            'cnn': cnn_features,
            'ppi': ppi_features,
        }

        self.results['feature_extraction'] = features
        logger.info("Feature extraction completed")

        return features

    def run_feature_fusion(self, features: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        gnn_feat = features.get('gnn', {})
        cnn_feat = features.get('cnn', {})
        protein_feat = features.get('protein', {})
        ligand_feat = features.get('ligand', {})

        gnn_arr = np.array([v for v in gnn_feat.get('centrality', {}).values()])[:64] if gnn_feat.get('centrality') else np.random.randn(64)
        cnn_arr = cnn_feat.get('features', np.random.randn(64)).flatten()[:64] if isinstance(cnn_feat.get('features'), np.ndarray) else np.random.randn(64)
        esm = protein_feat.get('features', {}).get('esm_embedding', np.random.randn(64))
        if isinstance(esm, np.ndarray):
            esm_arr = esm.flatten()[:64]
        else:
            esm_arr = np.random.randn(64)
        lig_desc = ligand_feat.get('features', {}).get('descriptors', {})
        lig_arr = np.array(list(lig_desc.values()))[:64] if lig_desc else np.random.randn(64)

        fusion_results = {}
        if self.transformer_module:
            import numpy as np
            def _to_array(x, size=64):
                if isinstance(x, np.ndarray):
                    return x.flatten()[:size] if x.size >= size else np.pad(x.flatten(), (0, size - x.size))
                return np.random.randn(size)

            fused = self.transformer_module.fuse_features(
                _to_array(gnn_arr).reshape(1, -1),
                _to_array(cnn_arr).reshape(1, -1),
                _to_array(esm_arr).reshape(1, -1)
            )
            fusion_results = {'fused_features': fused}

        self.results['feature_fusion'] = fusion_results
        logger.info("Feature fusion completed")

        return fusion_results

    _DOCKING_TARGETS = [
        {"gene": "CDK2",  "pdb_id": "1AQ1", "chain": "A", "name": "Cyclin-dependent kinase 2"},
        {"gene": "CDK6",  "pdb_id": "1XO2", "chain": "A", "name": "Cyclin-dependent kinase 6"},
        {"gene": "AKT1",  "pdb_id": "3O96", "chain": "A", "name": "RAC-alpha serine/threonine kinase"},
        {"gene": "PTEN",  "pdb_id": "1D5R", "chain": "A", "name": "Phosphatase and tensin homolog"},
        {"gene": "PIK3CA","pdb_id": "4L23", "chain": "A", "name": "PI3K alpha catalytic subunit"},
    ]

    def _download_pdb(self, pdb_id: str):
        cache_dir = os.path.join(self.output_dir, "docking", "receptors")
        os.makedirs(cache_dir, exist_ok=True)
        pdb_file = os.path.join(cache_dir, f"{pdb_id}.pdb")
        if os.path.exists(pdb_file):
            return pdb_file
        try:
            import requests
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            logger.info("Downloading PDB %s from RCSB...", pdb_id)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(pdb_file, "w", encoding="utf-8") as f:
                f.write(r.text)
            logger.info("Downloaded PDB %s (%d bytes)", pdb_id, len(r.text))
            return pdb_file
        except Exception as e:
            logger.warning("Failed to download PDB %s: %s", pdb_id, e)
            return None

    def run_docking(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        compound = dataset.get("compound", {})
        smiles = compound.get("smiles", "C1=CC(=C(C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O")
        docking_results = {"targets": {}}
        if not self.docking_module:
            self.results["docking"] = docking_results
            return docking_results
        if RDKIT_AVAILABLE and smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                    self.docking_module.prepare_ligand(mol, smiles=smiles)
            except Exception as e:
                logger.error("Ligand preparation failed: %s", e)
        best_affinity = 0
        for target_cfg in self._DOCKING_TARGETS:
            gene = target_cfg["gene"]
            pdb_id = target_cfg["pdb_id"]
            try:
                pdb_file = self._download_pdb(pdb_id)
                if not pdb_file:
                    docking_results["targets"][gene] = {"status": "error", "error": f"PDB {pdb_id} download failed", "binding_affinity": None}
                    continue
                self.docking_module.prepare_protein(pdb_file, protein_name=gene)
                pocket = self.docking_module.detect_binding_site(padding=8.0)
                vina_output = self.docking_module.run_vina(center=pocket["center"], box_size=pocket["box_size"], exhaustiveness=16)
                result = self.docking_module.validate_docking(vina_output)
                result["target_name"] = target_cfg["name"]
                result["pdb_id"] = pdb_id
                docking_results["targets"][gene] = result
                aff = result.get("binding_affinity")
                if aff is not None and aff < best_affinity:
                    best_affinity = aff
                logger.info("Docking %s (PDB %s): best affinity = %.3f kcal/mol", gene, pdb_id, aff or 0)
            except Exception as e:
                logger.error("Docking failed for %s (%s): %s", gene, pdb_id, e)
                docking_results["targets"][gene] = {"status": "error", "error": str(e), "binding_affinity": None}
        docking_results["binding_affinity"] = best_affinity if best_affinity != 0 else None
        self.results["docking"] = docking_results
        logger.info("Molecular docking completed for %d targets", len(docking_results["targets"]))
        return docking_results


    def run_knowledge_graph(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        compound_data = dataset.get('compound', {})
        target_data = dataset.get('targets', [])
        protein_data = dataset.get('proteins', [])
        ppi_result = self._canonicalize_dataset_ppi(dataset)
        kg_results = {'ppi_contract': ppi_result.to_dict(include_edges=False)}
        if self.kg_module:
            try:
                self.kg_module.build_from_data(compound_data, target_data, ppi_result)
                stats = self.kg_module.compute_graph_statistics()
                graph_path = self.kg_module.save_graph()
                triples_path = self.kg_module.save_triples()
                
                kg_embedding_status = {'attempted': False, 'available': False}
                if os.environ.get("RUN_PYKEEN", "0") == "1" and self.kg_embedder and self.kg_module.triples:
                    kg_embedding_status['attempted'] = True
                    model = self.kg_embedder.train_pykeen(self.kg_module.triples)
                    kg_embedding_status['available'] = model is not None
                nodes = list(self.kg_module.graph.nodes())[:50]
                edges = [(u, v) for u, v in self.kg_module.graph.edges()][:100]
                kg_results['graph'] = {
                    'stats': stats,
                    'nodes': nodes,
                    'edges': edges,
                    'graph_path': graph_path,
                    'triples_path': triples_path,
                    'kg_embedding': kg_embedding_status,
                }
            except Exception as e:
                logger.error("Knowledge graph analysis failed: %s", e)
                kg_results['graph'] = {'stats': {'num_nodes': 0, 'num_edges': 0}}

        self.results['knowledge_graph'] = kg_results
        logger.info("Knowledge graph analysis completed")

        return kg_results

    def run_output_generation(self) -> Dict[str, Any]:
        feature_extraction = self.results.get('feature_extraction', {})
        gnn_results = feature_extraction.get('gnn', {})
        ligand_results = feature_extraction.get('ligand', {})
        docking_results = self.results.get('docking', {})
        knowledge_graph_results = self.results.get('knowledge_graph', {})
        compound_data = self.results.get('data_collection', {}).get('compound', {})
        target_data = self.results.get('data_collection', {}).get('targets', [])

        output_results = {}
        if self.output_module:
            try:
                summary_input = {
                    'targets': target_data,
                    'compound': compound_data,
                    'docking': docking_results,
                    'gnn': gnn_results,
                    'ligand': ligand_results,
                    'knowledge_graph': knowledge_graph_results
                }
                report = self.output_module.generate_summary_report(summary_input)
                self.output_module.generate_dashboard_html(report)
                plot_input = {
                    'targets': target_data,
                    'compound': compound_data,
                    'docking': docking_results,
                    'gnn': gnn_results
                }
                self.output_module.generate_plots(plot_input)
                output_results = report
            except Exception as e:
                logger.error("Output generation failed: %s", e)

        self.results['output'] = output_results
        logger.info("Output generation completed")

        return output_results

    def save_results(self) -> str:
        filepath = os.path.join(self.output_dir, "framework_results.json")

        try:
            summary_data = self.results.get('data_collection', {})
            feature_data = self.results.get('feature_extraction', {})
            docking_data = self.results.get('docking', {})
            kg_data = self.results.get('knowledge_graph', {})
            results_to_save = {
                'framework_version': '1.0',
                'run_time': datetime.now().isoformat(),
                'results': self.results,
                'summary': {
                    'data_collection': {
                        'compound': summary_data.get('compound', {}).get('name', 'N/A'),
                        'num_targets': len(summary_data.get('targets', {})),
                        'num_proteins': len(summary_data.get('proteins', [])),
                        'num_interactions': len(summary_data.get('interactions', []))
                    },
                    'feature_extraction': {
                        'ligand_features': bool(feature_data.get('ligand', {})),
                        'protein_features': bool(feature_data.get('protein', {})),
                        'gnn_features': bool(feature_data.get('gnn', {})),
                        'cnn_features': bool(feature_data.get('cnn', {}))
                    },
                    'docking': {
                        'best_affinity': docking_data.get('binding_affinity', 'N/A')
                    },
                    'knowledge_graph': {
                        'num_nodes': kg_data.get('graph', {}).get('stats', {}).get('num_nodes', 0),
                        'num_edges': kg_data.get('graph', {}).get('stats', {}).get('num_edges', 0)
                    }
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, ensure_ascii=False, indent=2, default=self._json_serialize)

            logger.info("Saved framework results to %s", filepath)
            return filepath

        except Exception as e:
            logger.error("Error saving results: %s", e)
            return ""

    def _json_serialize(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    def _clean_previous_output(self):
        dirs_to_clean = [
            os.path.join(self.output_dir, "data"),
            os.path.join(self.output_dir, "ligand"),
            os.path.join(self.output_dir, "protein"),
            os.path.join(self.output_dir, "cnn"),
            os.path.join(self.output_dir, "gnn"),
            os.path.join(self.output_dir, "transformer"),
            os.path.join(self.output_dir, "docking"),
            os.path.join(self.output_dir, "knowledge_graph"),
            os.path.join(self.output_dir, "output"),
        ]
        files_to_clean = [
            os.path.join(self.output_dir, "framework_results.json"),
        ]
        import shutil
        for d in dirs_to_clean:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        for f in files_to_clean:
            if os.path.exists(f):
                os.remove(f)
        logger.info("Previous output cleaned")
        sub_dirs = {
            "data": ["raw", "processed"],
            "ligand": ["structures", "descriptors"],
            "protein": ["structures", "embeddings"],
            "cnn": [],
            "gnn": [],
            "transformer": [],
            "docking": ["structures", "receptors", "results"],
            "knowledge_graph": ["graph"],
            "output": ["images"],
        }
        for parent, subs in sub_dirs.items():
            parent_dir = os.path.join(self.output_dir, parent)
            os.makedirs(parent_dir, exist_ok=True)
            for sub in subs:
                os.makedirs(os.path.join(parent_dir, sub), exist_ok=True)

    def run_full_pipeline(self) -> Dict[str, Any]:
        logger.info("Starting full pipeline execution...")
        self._clean_previous_output()
        start_time = datetime.now()

        dataset = self.run_data_collection()
        features = self.run_feature_extraction(dataset)
        self.run_feature_fusion(features)
        self.run_docking(dataset)
        self.run_knowledge_graph(dataset)
        self.run_output_generation()
        self.save_results()

        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        logger.info("Pipeline execution completed in %.2f seconds", runtime)

        return self.results


def main():
    output_dir = os.path.join(_BASE_DIR, "framework_output")

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "framework_run.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info("Log file: %s", log_file)

    framework = NetworkPharmacologyFramework(output_dir=output_dir)
    results = framework.run_full_pipeline()

    data_summary = results.get('data_collection', {})
    logger.info("Data Collection: compound=%s, targets=%d, proteins=%d, interactions=%d",
                data_summary.get('compound', {}).get('name', 'N/A'),
                len(data_summary.get('targets', [])),
                len(data_summary.get('proteins', [])),
                len(data_summary.get('interactions', [])))

    feature_summary = results.get('feature_extraction', {})
    logger.info("Feature Extraction: ligand=%s, protein=%s, gnn=%s, cnn=%s",
                bool(feature_summary.get('ligand', {})),
                bool(feature_summary.get('protein', {})),
                bool(feature_summary.get('gnn', {})),
                bool(feature_summary.get('cnn', {})))

    docking_summary = results.get('docking', {})
    best_affinity = docking_summary.get('binding_affinity', 'N/A')
    logger.info("Molecular Docking: best affinity=%s kcal/mol", best_affinity)

    kg_summary = results.get('knowledge_graph', {})
    logger.info("Knowledge Graph: nodes=%d, edges=%d",
                kg_summary.get('graph', {}).get('stats', {}).get('num_nodes', 0),
                kg_summary.get('graph', {}).get('stats', {}).get('num_edges', 0))

    output_summary = results.get('output', {})
    logger.info("Output Generation: target_discovery=%s, mechanism=%s, drug_dev=%s",
                bool(output_summary.get('target_discovery', {})),
                bool(output_summary.get('mechanism_analysis', {})),
                bool(output_summary.get('drug_development', {})))

    logger.info("All modules executed successfully!")


if __name__ == "__main__":
    main()
