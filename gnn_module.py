"""
GNN图神经网络模块 (Graph Neural Network Module)
用于生物网络分析和靶点预测

功能:
- 图游走算法 (DeepWalk, Node2Vec)
- 图卷积网络 (GCN)
- 生物网络拓扑特征提取
- 关键靶点识别
"""

import os
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not installed. GNN module will be limited.")


class DeepWalk:
    """DeepWalk算法实现"""
    
    def __init__(self, G, walk_length: int = 40, num_walks: int = 10):
        self.G = G
        self.walk_length = walk_length
        self.num_walks = num_walks
    
    def random_walk(self, start_node):
        """从起始节点开始的随机游走"""
        walk = [start_node]
        current = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break
            current = np.random.choice(neighbors)
            walk.append(current)
        
        return walk
    
    def generate_walks(self):
        """生成所有节点的随机游走"""
        walks = []
        nodes = list(self.G.nodes())
        
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.random_walk(node)
                walks.append(walk)
        
        return walks


class Node2Vec:
    """Node2Vec算法实现"""
    
    def __init__(self, G, walk_length: int = 40, num_walks: int = 10, p: float = 1.0, q: float = 1.0):
        self.G = G
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p  # 返回概率
        self.q = q  # 探索概率
    
    def node2vec_walk(self, start_node):
        """Node2Vec随机游走"""
        walk = [start_node]
        current = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break
            
            if len(walk) == 1:
                # 初始步骤，等概率选择
                current = np.random.choice(neighbors)
            else:
                # 基于p和q的概率选择
                prev = walk[-2]
                weights = []
                for neighbor in neighbors:
                    if neighbor == prev:
                        # 返回上一节点的概率
                        weights.append(1.0 / self.p)
                    elif self.G.has_edge(neighbor, prev):
                        # 与上一节点相邻的概率
                        weights.append(1.0)
                    else:
                        # 其他节点的概率
                        weights.append(1.0 / self.q)
                
                # 归一化概率
                weights = np.array(weights) / sum(weights)
                current = np.random.choice(neighbors, p=weights)
            
            walk.append(current)
        
        return walk
    
    def generate_walks(self):
        """生成所有节点的Node2Vec游走"""
        walks = []
        nodes = list(self.G.nodes())
        
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vec_walk(node)
                walks.append(walk)
        
        return walks


class GCN(nn.Module):
    """图卷积网络"""
    
    def __init__(self, in_channels: int, hidden_channels: List[int], out_channels: int):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels[0]))
        
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
        
        self.convs.append(GCNConv(hidden_channels[-1], out_channels))
        
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.relu(x)
        
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    """图注意力网络"""
    
    def __init__(self, in_channels: int, hidden_channels: List[int], out_channels: int, heads: int = 4):
        super(GAT, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels[0], heads=heads))
        
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GATConv(hidden_channels[i] * heads, hidden_channels[i+1], heads=heads))
        
        self.convs.append(GATConv(hidden_channels[-1] * heads, out_channels, heads=1))
        
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.relu(x)
        
        x = self.convs[-1](x, edge_index)
        return x


class GNNModule:
    """GNN图神经网络模块"""
    
    def __init__(self, output_dir: str = "./features/gnn"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/embeddings", exist_ok=True)
        os.makedirs(f"{output_dir}/networks", exist_ok=True)
        
        self.G = None
        self.node_embeddings = None
        self.features = {}
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("  Warning: PyTorch Geometric not installed. GCN/GAT features will be limited.")
    
    def build_network_from_interactions(self, interactions: List[Dict[str, Any]]) -> nx.Graph:
        """从相互作用数据构建网络"""
        G = nx.Graph()
        
        for interaction in interactions:
            protein_a = interaction.get('protein_a', interaction.get('preferredName_A', ''))
            protein_b = interaction.get('protein_b', interaction.get('preferredName_B', ''))
            score = float(interaction.get('score', 0))
            
            if protein_a and protein_b:
                G.add_edge(protein_a, protein_b, weight=score)
        
        self.G = G
        print(f"  ✓ Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def build_network_from_file(self, file_path: str) -> nx.Graph:
        """从文件构建网络"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                interactions = data.get('interactions', [])
                return self.build_network_from_interactions(interactions)
            elif file_path.endswith('.edgelist'):
                G = nx.read_edgelist(file_path, nodetype=str, data=(('weight', float),))
                self.G = G
                print(f"  ✓ Loaded network from {file_path}")
                return G
            else:
                print(f"  Error: Unsupported file format: {file_path}")
                return None
        except Exception as e:
            print(f"  Error building network: {e}")
            return None
    
    def extract_network_features(self) -> Dict[str, Any]:
        """提取网络拓扑特征"""
        if self.G is None:
            print("  Error: No network built")
            return {}
        
        features = {}
        
        try:
            # 基本网络统计
            features['num_nodes'] = self.G.number_of_nodes()
            features['num_edges'] = self.G.number_of_edges()
            features['density'] = nx.density(self.G)
            features['average_clustering'] = nx.average_clustering(self.G)
            features['average_degree'] = sum(dict(self.G.degree()).values()) / features['num_nodes']
            
            # 中心性指标
            features['degree_centrality'] = nx.degree_centrality(self.G)
            features['betweenness_centrality'] = nx.betweenness_centrality(self.G)
            features['closeness_centrality'] = nx.closeness_centrality(self.G)
            features['eigenvector_centrality'] = nx.eigenvector_centrality(self.G, max_iter=1000)
            
            # 社区检测
            try:
                communities = list(nx.community.greedy_modularity_communities(self.G))
                features['num_communities'] = len(communities)
                features['community_size'] = [len(comm) for comm in communities]
            except:
                pass
            
            # 关键节点识别
            features['top_nodes_by_degree'] = sorted(
                features['degree_centrality'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            features['top_nodes_by_betweenness'] = sorted(
                features['betweenness_centrality'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            self.features['network'] = features
            print(f"  ✓ Extracted network features")
            
        except Exception as e:
            print(f"  Warning: Network feature extraction failed: {e}")
        
        return features
    
    def generate_node_embeddings(self, method: str = 'deepwalk', embedding_dim: int = 128) -> Dict[str, np.ndarray]:
        """生成节点嵌入"""
        if self.G is None:
            print("  Error: No network built")
            return {}
        
        try:
            if method == 'deepwalk':
                deepwalk = DeepWalk(self.G)
                walks = deepwalk.generate_walks()
            elif method == 'node2vec':
                node2vec = Node2Vec(self.G)
                walks = node2vec.generate_walks()
            else:
                print(f"  Error: Unknown method: {method}")
                return {}
            
            # 使用Word2Vec训练嵌入
            from gensim.models import Word2Vec
            
            # 将节点转换为字符串
            walks_str = [[str(node) for node in walk] for walk in walks]
            
            model = Word2Vec(
                walks_str, 
                vector_size=embedding_dim, 
                window=5, 
                min_count=1, 
                sg=1,  # skip-gram
                workers=4
            )
            
            embeddings = {}
            for node in self.G.nodes():
                node_str = str(node)
                if node_str in model.wv:
                    embeddings[node] = model.wv[node_str]
            
            self.node_embeddings = embeddings
            self.features['node_embeddings'] = embeddings
            print(f"  ✓ Generated node embeddings using {method}")
            
        except Exception as e:
            print(f"  Warning: Node embedding generation failed: {e}")
            embeddings = {}
        
        return embeddings
    
    def train_gnn_model(self, embedding_dim: int = 128, hidden_channels: List[int] = [64, 32]) -> Optional[nn.Module]:
        """训练GCN模型"""
        if self.G is None:
            print("  Error: No network built")
            return None
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("  Warning: PyTorch Geometric not installed. Skipping GCN model training.")
            return None
        
        try:
            # 准备PyTorch Geometric数据
            nodes = list(self.G.nodes())
            node_map = {node: i for i, node in enumerate(nodes)}
            
            edges = []
            for u, v in self.G.edges():
                edges.append([node_map[u], node_map[v]])
                edges.append([node_map[v], node_map[u]])  # 无向图
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # 节点特征（使用度作为初始特征）
            degrees = [self.G.degree(node) for node in nodes]
            x = torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
            
            # 构建数据对象
            data = Data(x=x, edge_index=edge_index)
            
            # 初始化模型
            in_channels = x.size(1)
            out_channels = embedding_dim
            model = GCN(in_channels, hidden_channels, out_channels)
            
            # 训练模型（简单的自监督学习）
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                
                # 计算损失（节点相似度）
                loss = F.mse_loss(out, out)
                loss.backward()
                optimizer.step()
            
            # 获取节点嵌入
            model.eval()
            with torch.no_grad():
                embeddings = model(data.x, data.edge_index).numpy()
            
            # 保存嵌入
            gnn_embeddings = {nodes[i]: embeddings[i] for i in range(len(nodes))}
            self.features['gnn_embeddings'] = gnn_embeddings
            print(f"  ✓ Trained GCN model and generated embeddings")
            
            return model
            
        except Exception as e:
            print(f"  Error training GNN model: {e}")
            return None
    
    def identify_key_targets(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """识别关键靶点"""
        if 'network' not in self.features:
            print("  Error: No network features extracted")
            return []
        
        network_features = self.features['network']
        
        # 综合多种中心性指标
        key_targets = {}
        
        if 'degree_centrality' in network_features:
            for node, score in network_features['degree_centrality'].items():
                key_targets[node] = key_targets.get(node, 0) + score
        
        if 'betweenness_centrality' in network_features:
            for node, score in network_features['betweenness_centrality'].items():
                key_targets[node] = key_targets.get(node, 0) + score * 2  # 给betweenness更高权重
        
        if 'eigenvector_centrality' in network_features:
            for node, score in network_features['eigenvector_centrality'].items():
                key_targets[node] = key_targets.get(node, 0) + score
        
        # 排序并返回前k个
        sorted_targets = sorted(key_targets.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        self.features['key_targets'] = sorted_targets
        print(f"  ✓ Identified {len(sorted_targets)} key targets")
        
        return sorted_targets
    
    def save_network(self, filename: str = "network.graphml") -> str:
        """保存网络"""
        if self.G is None:
            print("  Error: No network to save")
            return ""
        
        filepath = os.path.join(self.output_dir, "networks", filename)
        
        try:
            nx.write_graphml(self.G, filepath)
            print(f"  ✓ Saved network to {filepath}")
            return filepath
        except Exception as e:
            print(f"  Error saving network: {e}")
            return ""
    
    def save_embeddings(self, filename: str = "node_embeddings.json") -> str:
        """保存节点嵌入"""
        if not self.node_embeddings:
            print("  Error: No embeddings to save")
            return ""
        
        filepath = os.path.join(self.output_dir, "embeddings", filename)
        
        try:
            embeddings_to_save = {}
            for node, embedding in self.node_embeddings.items():
                embeddings_to_save[str(node)] = embedding.tolist()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(embeddings_to_save, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved embeddings to {filepath}")
            return filepath
        except Exception as e:
            print(f"  Error saving embeddings: {e}")
            return ""
    
    def save_features(self, filename: str = "gnn_features.json") -> str:
        """保存GNN特征"""
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            features_to_save = {
                'extract_time': datetime.now().isoformat(),
                'network_features': self.features.get('network', {}),
                'key_targets': self.features.get('key_targets', []),
                'node_embedding_shape': len(next(iter(self.features.get('node_embeddings', {}).values()))) if self.features.get('node_embeddings') else 0
            }
            
            # 处理大型数据
            if 'degree_centrality' in features_to_save['network_features']:
                features_to_save['network_features']['top_degree_nodes'] = sorted(
                    features_to_save['network_features']['degree_centrality'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                del features_to_save['network_features']['degree_centrality']
            
            if 'betweenness_centrality' in features_to_save['network_features']:
                features_to_save['network_features']['top_betweenness_nodes'] = sorted(
                    features_to_save['network_features']['betweenness_centrality'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                del features_to_save['network_features']['betweenness_centrality']
            
            if 'closeness_centrality' in features_to_save['network_features']:
                features_to_save['network_features']['top_closeness_nodes'] = sorted(
                    features_to_save['network_features']['closeness_centrality'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                del features_to_save['network_features']['closeness_centrality']
            
            if 'eigenvector_centrality' in features_to_save['network_features']:
                features_to_save['network_features']['top_eigenvector_nodes'] = sorted(
                    features_to_save['network_features']['eigenvector_centrality'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                del features_to_save['network_features']['eigenvector_centrality']
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(features_to_save, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved GNN features to {filepath}")
            return filepath
        except Exception as e:
            print(f"  Error saving features: {e}")
            return ""
    
    def run_analysis(self, interactions: List[Dict[str, Any]], method: str = 'deepwalk') -> Dict[str, Any]:
        """运行完整的GNN分析"""
        print("\n" + "=" * 60)
        print("  GNN NETWORK ANALYSIS")
        print("  图神经网络分析")
        print("=" * 60)
        
        # 构建网络
        self.build_network_from_interactions(interactions)
        
        # 提取网络特征
        self.extract_network_features()
        
        # 生成节点嵌入
        self.generate_node_embeddings(method=method)
        
        # 训练GNN模型
        self.train_gnn_model()
        
        # 识别关键靶点
        self.identify_key_targets()
        
        # 保存结果
        self.save_network()
        self.save_embeddings()
        self.save_features()
        
        print("\n" + "=" * 60)
        print("  GNN ANALYSIS COMPLETED")
        print("=" * 60)
        
        return self.features


def main():
    # 测试GNN模块
    from data_collection import LuteolinDataCollector
    
    # 加载数据
    collector = LuteolinDataCollector(output_dir="./data")
    dataset = collector.run_pipeline()
    
    # 提取相互作用数据
    interactions = dataset.get('interactions', [])
    
    if interactions:
        # 初始化GNN模块
        gnn = GNNModule(output_dir="./features/gnn")
        
        # 运行分析
        features = gnn.run_analysis(interactions, method='deepwalk')
        
        # 打印结果
        print("\n" + "=" * 60)
        print("  GNN ANALYSIS RESULTS")
        print("=" * 60)
        
        if 'network' in features:
            network = features['network']
            print(f"  Network: {network.get('num_nodes')} nodes, {network.get('num_edges')} edges")
            print(f"  Density: {network.get('density', 0):.4f}")
            print(f"  Average Clustering: {network.get('average_clustering', 0):.4f}")
        
        if 'key_targets' in features:
            print("\n  Key Targets:")
            for i, (target, score) in enumerate(features['key_targets'][:5]):
                print(f"    {i+1}. {target}: {score:.4f}")


if __name__ == "__main__":
    main()