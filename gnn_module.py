import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('PYG_LIB_DISABLED', '1')

from logging_config import setup_logger
logger =setup_logger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. GNN features will be limited.")

NETWORKX_AVAILABLE = False
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    logger.warning("NetworkX not installed. Graph operations will be limited.")

TORCH_GEOMETRIC_AVAILABLE = False
try:
    import torch_geometric.typing
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning("PyTorch Geometric not available (%s). GNN operations will be limited.", type(e).__name__)

GENSIM_AVAILABLE = False
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.warning("Gensim not installed. Node2Vec will be unavailable.")


class GNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(in_dim, hidden_dim, dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_dimhidden_dim, dropout))
        self.layers.append(GNNLayer(hidden_dim, out_dim, dropout))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GraphEncoder:
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def encode_molecular_graph(self, mol_graph: Dict[str, Any]) -> Optional[Any]:
        if not NETWORKX_AVAILABLE or not TORCH_AVAILABLE:
            return None
        try:
            G = nx.Graph()
            atoms = mol_graph.get('atoms', [])
            for i, atom in enumerate(atoms):
                G.add_node(i, **atom)
            bonds = mol_graph.get('bonds', [])
            for bond in bonds:
                G.add_edge(bond['begin'], bond['end'], type=bond['type'])

            node_features = []
            for node in G.nodes(datarrue):
                feat = [node[1].get('atomic_num', 0) / 100.0]
                node_features.append(feat)
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(list(G.edges)).t().contiguous()
            if edge_index.numel() > 0:
                edge_index = edge_index.to(torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            logger.info("Encoded graph: %d nodes, %d edges", x.shape[0], edge_index.shape[1])
            return {'x': x, 'edge_index': edge_index, 'graph': G}
        except Exception as e:
            logger.error("Graph encoding failed: %s", e)
            return None


class DeepWalkEmbedder:
    def __init__(self, dimensions: int = 64, walk_length: int = 20, num_walks: int = 10):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks

    def fit(self, graph: nx.Graph) -> Optional[np.ndarray]:
        if not GENSIM_AVAILABLE:
            logger.warning("Gensim not available for DeepWalk")
            return None
        try:
            walks = []
            nodes = list(graph.nodes())
            for _ in range(self.num_walks):
                np.random.shuffle(nodes)
                for node in nodes:
                    walk = [node]
                    for _ in range(self.walk_length - 1):
                        neighbors = list(graph.neighbors(walk[-1]))
                        if not neighbors:
                            break
                        walk.append(np.random.choice(neighbors))
                    walks.append([str(n) for n in walk])

            model = Word2Vec(walks, vector_size=self.dimensions,
                           window=5, min_count=0, sg=1, workers=1, epochs=10)
            embeddings = np.zeros((len(nodes), self.dimensions))
            for i, node in enumerate(nodes):
                if str(node) in model.wv:
                    embeddings[i] = model.wv[str(node)]
            logger.info("DeepWalk embeddings: %d nodes x %d dims", len(nodes), self.dimensions)
            return embeddings
        except Bution as e:
            logger.error("DeepWalk failed: %s", e)
            return None


class Node2VecEmbedder:
    def __init__(self, dimensions: int = 64, walk_length: int = 20, num_walks: int = 10, p: float = 1.0, q: float = 1.0):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q

    def fit(self, graph: nx.Graph) -> Optional[np.ndarray]:
        if not GENSIM_AVAILABLE:
            logger.warning("Gensim not available for Node2Vec")
            return None
        try:
            walks = []
            nodes = list(graph.nodes())
            for _ in range(self.num_walks):
                np.random.shuffle(nodes)
                for start in nodes:
                    walk = [start]
                    curr = start
                    prev = None
                    for _ in range(self.walk_length - 1):
                        neighbors = list(graph.neighbors(curr))
                        if not neighbors:
                            break
                        if prev is None:
                            next_node = np.random.choice(neighbors)
                        else:
                            probs = []
                            for n in neighbors:
                                if n == prev:
                                    probs.append(1.0 / self.p)
                                elif graph.has_edge(n, prev):
                                    probs.append(1.0)
                                else:
                                    probs.append(1.0 / self.q)
                            probs = np.array(probs) / sum(probs)
                            next_node = np.random.choice(neighbors, p=probs)
                        walk.append(next_node)
                        prev, curr = curr, next_node
                    walks.append([str(n) for n in walk])

            model = Word2Vec(walks, vector_size=self.dimensions,
                           window=5, min_count=0, sg=1, workers=1, epochs=10)
            embeddings = np.zeros((len(nodes), self.dimensions))
            for i, node in enumerate(nodes):
                if str(node) in model.wv:
                    embeddings[i] = model.wv[str(node)]
            logger.info("Node2Vec embeddings: %d nodes x %d dims", len(nodes), self.dimensions)
            return embeddings
        except Exception as e:
            logger.error("Node2Vec failed: %s", e)
            return None


class NetworkAnalyzer:
    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features", "network")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_centrality(self, graph: nx.Graph) -> Dict[str, Any]:
        metrics = {}
        try:
            metrics['degree_centrality'] = nx.degree_centrality(graph)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(graph)
            metrics['closeness_centrality'] = nx.closeness_centrality(graph)
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(graph, max_iter=1000)
            logger.info("Computed centrality metrics")
        except Exception as e:
            logger.error("Centrality computation failed: %s", e)
        return metrics

    def compute_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        stats = {}
        try:
            stats['num_nodes'] = graph.number_of_nodes()
            stats['num_edges'] = graph.number_of_edges()
            stats['density'] = nx.density(graph)
            stats['average_clustering'] = nx.average_clustering(graph)
            stats['num_connected_components'] = nx.number_connected_components(graph)
            try:
                stats['average_shortest_path_length'] = nx.average_shortest_path_length(graph)
            except Exception:
                stats['average_shortest_path_length'] = None
            degrees = [d for _, d in graph.degree()]
            stats['average_degree'] = float(np.mean(degrees)) if degrees else 0
            stats['max_degree'] = int(max(degrees)) if degrees else 0
            logger.info("Computed graph statistics")
        except Exception as e:
            logger.error("Graph statistics computation failed: %s", e)
        return stats

    def save_results(self, results: Dict[str, Any], filename: str = "network_results.json"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        logger.info("Saved network results: %s", filepath)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)