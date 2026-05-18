import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
logger = setup_logger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. KG embedding will be limited.")

NETWORKX_AVAILABLE = False
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    logger.warning("NetworkX not installed. Graph operations will be limited.")

_PYKEEN_AVAILABLE = False
try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    _PYKEEN_AVAILABLE = True
except ImportError:
    logger.warning("PyKEEN not installed. KG embedding training will be unavailable.")

_DGL_KE_AVAILABLE = False
try:
    import dgl
    _DGL_KE_AVAILABLE = True
except ImportError:
    logger.warning("DGL not installed. DGL-KE operations will be unavailable.")


class KnowledgeGraph:
    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "knowledge_graph")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.join(self.output_dir, "graph"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "embeddings"), exist_ok=True)
        self.graph = nx.MultiDiGraph() if NETWORKX_AVAILABLE else None
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.triples = []

    def add_compound_target_interaction(self, compound: str, target: str, relation: str = "binds_to",
                                        weight: float = 1.0, properties: Dict = None):
        if not NETWORKX_AVAILABLE:
            return
        try:
            self.graph.add_edge(compound, target, relation=relation, weight=weight,
                              **(properties or {}))
            self.triples.append((compound, relation, target))
            logger.debug("Added triple: (%s, %s, %s)", compound, relation, target)
        except Exception as e:
            logger.error("Failed to add triple: %", e)

    def build_from_data(self, compound_data: Dict, target_data: Dict, ppi_data: Dict = None):
        logger.info("Building knowledge graph from data...")
        try:
            compound_name = compound_data.get('name', 'Luteolin')
            targets = target_data.get('targets', [])
            for target in targets:
                target_name = target.get('target_name', '') or target.get('gene_symbol', '')
                if not target_name:
                    continue
                gene_symbol = target.get('gene_symbol', '')
                self.add_compound_target_interaction(compound_name, target_name)
                if gene_symbol:
                    self.add_compound_target_interaction(compound_name, gene_symbol, "targets")
            if ppi_data:
                interactions = ppi_data.get('interactions', [])
                for interaction in interactions:
                    preferredName_A = interaction.get('preferredName_A', '')
                    preferredName_B = interaction.get('preferredName_B', '')
                    if gene1 and gene2:
                        score = float(interaction.get('score', 0))
                        self.graph.add_edge(gene1, gene2ge2, relation="interacts_with", weight=score)
            logger.info("Knowledge graph built: %d nodes, %d edges",
                       self.graph.number_of_nodes(), self.graph.number_of_edges())
        except Exception as e:
            logger.error("Knowledge graph construction failed: %s", e)

    def compute_graph_statistics(self) -> Dict[str, Any]:
        stats = {'num_nodes': 0, 'num_edges': 0, 'num_relations': 0}
        if not NETWORKX_AVAILABLE or self.graph is None:
            return stats
        try:
            stats['num_nodes'] = self.graph.number_of_nodes()
            stats['num_edges'] = self.graph.number_of_edges()
            relations = set()
            for _, _, data in self.graph.edges(data=True):
                relations.add(data.get('relation', ''))
            stats['num_relations'] = len(relations)
            logger.info("KG statistics: %s", stats)
        except Exception as e:
            logger.error("KG statistics failed: %s", e)
        return stats

    def save_graph(self, filename: str = "knowledge_graph.json"):
        if not NETWORKX_AVAILABLE:
            return
        try:
            data = nx.node_link_data(self.graph)
            filepath = os.path.join(self.output_dir, "graph", filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Saved knowledge graph: %s", filepath)
        except Exception as e:
            logger.error("Failed to save graph: %s", e)

    def load_graph(self, filename: str = "knowledge_graph.json"):
        filepath = os.path.join(self.output_dir, "graph", filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if NETWORKX_AVAILABLE:
                self.graph = nx.node_link_graph(data)
                logger.info("Loaded knowledge graph: %s", filepath)
        except Exception as e:
            logger.error("Failed to load graph: %s", e)


class KGEmbedder:
    def __init__(self, embedding_dim: int = 64, output_dir: str = ""):
        self.embedding_dim = embedding_dim
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "knowledge_graph", "embeddings")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None

    def train_pykeen(self, triples: List[tuple]) -> Optional[Any]:
        if not _PYKEEN_AVAILABLE:
            return None
        try:
            tf = TriplesFactory.from_labeled_triples(triples)
            result = pipeline(
                training=tf,
                model='RotatE',
                training_kwargs=dict(num_epochs=100, batch_size=64),
                optimizer_kwargs=dict(lr=0.01)
            )
            self.model = result.model
            logger.info("PyKEEN model trained successfully")
            return self.model
        except Exception as e:
            logger.error("PyKEEN training failed: %s", e)
            return None

    def extract_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        if self.model is None:
            return None
        embeddings = {}
        try:
            entity_embeddings = self.model.entity_representations[0]
            relation_embeddings = self.model.relation_representations[0]
            embeddings['entities'] = entity_embeddings.detach().numpy()
            embeddings['relations'] = relation_embeddings.detach().numpy()
            logger.info("Extracted KG embeddings: entities %s, relations %s",
                       str(embeddings['entities'].shape), str(embeddings['relations'].shape))
        except Exception as e:
            logger.error("Embedding extraction failed: %s", e)
        return embeddings

    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filename: str = "kg_embeddings.npz"):
        filepath = os.path.join(self.output_dir, filename)
        np.savez(filepath, **embeddings)
        logger.info("Saved KG embeddings: %s", filepath)