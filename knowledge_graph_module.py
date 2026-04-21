"""
知识图谱构建模块 (Knowledge Graph Module)
用于构建动态可学习的生物知识图谱

功能:
- 多源生物数据集成
- 知识图谱构建
- 知识图谱可视化
- 知识图谱查询和推理
- 动态更新和学习
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import pykeen
    from pykeen.pipeline import pipeline
    from pykeen.models import TransE, DistMult, ComplEx
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    print("Warning: PyKEEN not installed. Knowledge graph learning will be limited.")

try:
    import dgl
    import dglke
    DGL_KE_AVAILABLE = True
except ImportError:
    DGL_KE_AVAILABLE = False
    print("Warning: DGL-KE not installed. Knowledge graph embedding will be limited.")


class KnowledgeGraph:
    """知识图谱类"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = []
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        """添加节点"""
        self.graph.add_node(node_id, type=node_type, **properties)
        self.nodes[node_id] = {
            'id': node_id,
            'type': node_type,
            'properties': properties
        }
    
    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0, properties: Dict[str, Any] = None):
        """添加边"""
        if properties is None:
            properties = {}
        
        self.graph.add_edge(source, target, type=edge_type, weight=weight, **properties)
        self.edges.append({
            'source': source,
            'target': target,
            'type': edge_type,
            'weight': weight,
            'properties': properties
        })
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_edges(self, source: str = None, target: str = None, edge_type: str = None) -> List[Dict[str, Any]]:
        """获取边"""
        filtered_edges = []
        for edge in self.edges:
            if source and edge['source'] != source:
                continue
            if target and edge['target'] != target:
                continue
            if edge_type and edge['type'] != edge_type:
                continue
            filtered_edges.append(edge)
        return filtered_edges
    
    def get_neighbors(self, node_id: str, edge_type: str = None) -> List[str]:
        """获取邻居节点"""
        neighbors = []
        for edge in self.edges:
            if edge['source'] == node_id:
                if edge_type and edge['type'] != edge_type:
                    continue
                neighbors.append(edge['target'])
        return neighbors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'nodes': list(self.nodes.values()),
            'edges': self.edges,
            'stats': {
                'num_nodes': len(self.nodes),
                'num_edges': len(self.edges),
                'node_types': list(set(node['type'] for node in self.nodes.values())),
                'edge_types': list(set(edge['type'] for edge in self.edges))
            }
        }
    
    def visualize(self, filename: str = "knowledge_graph.png") -> str:
        """可视化知识图谱"""
        try:
            plt.figure(figsize=(12, 10))
            
            # 节点颜色映射
            node_types = {node['type'] for node in self.nodes.values()}
            color_map = {}
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i, node_type in enumerate(node_types):
                color_map[node_type] = colors[i % len(colors)]
            
            # 节点颜色
            node_colors = [color_map[self.nodes[node]['type']] for node in self.graph.nodes()]
            
            # 布局
            pos = nx.spring_layout(self.graph, k=0.3, iterations=50)
            
            # 绘制节点
            nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color=node_colors, alpha=0.8)
            
            # 绘制边
            nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
            
            # 标签
            labels = {node: node for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)
            
            # 边标签
            edge_labels = {}
            for edge in self.edges:
                edge_labels[(edge['source'], edge['target'])] = edge['type']
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=8)
            
            plt.title('Biological Knowledge Graph')
            plt.axis('off')
            plt.tight_layout()
            
            # 保存
            filepath = os.path.join('./', filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            
            print(f"  ✓ Knowledge graph visualized: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  Error visualizing knowledge graph: {e}")
            return ""


class KnowledgeGraphModule:
    """知识图谱构建模块"""
    
    def __init__(self, output_dir: str = "./knowledge_graph"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/graphs", exist_ok=True)
        os.makedirs(f"{output_dir}/embeddings", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
        self.knowledge_graph = KnowledgeGraph()
        self.features = {}
    
    def build_from_data(self, compound_data: Dict[str, Any], 
                      target_data: List[Dict[str, Any]],
                      protein_data: List[Dict[str, Any]],
                      interaction_data: List[Dict[str, Any]]):
        """从数据构建知识图谱"""
        print("\n1. Building knowledge graph from data...")
        
        # 添加化合物节点
        if compound_data:
            compound_id = compound_data.get('name', 'compound')
            self.knowledge_graph.add_node(
                compound_id,
                'drug',
                {
                    'name': compound_data.get('name', ''),
                    'name_cn': compound_data.get('name_cn', ''),
                    'molecular_formula': compound_data.get('molecular_formula', ''),
                    'molecular_weight': compound_data.get('molecular_weight', 0),
                    'smiles': compound_data.get('smiles', '')
                }
            )
            print(f"  ✓ Added compound node: {compound_id}")
        
        # 添加靶点节点
        for target in target_data:
            target_id = target.get('gene_symbol', target.get('target_name', f"target_{len(self.knowledge_graph.nodes)}"))
            self.knowledge_graph.add_node(
                target_id,
                'target',
                {
                    'target_name': target.get('target_name', ''),
                    'gene_symbol': target.get('gene_symbol', ''),
                    'uniprot_id': target.get('uniprot_id', ''),
                    'activity_value': target.get('activity_value', 0),
                    'pchembl_value': target.get('pchembl_value', 0)
                }
            )
            
            # 添加药物-靶点边
            if compound_data:
                compound_id = compound_data.get('name', 'compound')
                weight = target.get('pchembl_value', 0) if target.get('pchembl_value') else 0.5
                self.knowledge_graph.add_edge(
                    compound_id,
                    target_id,
                    'targets',
                    weight=weight,
                    properties={'activity': target.get('activity_value', 0)}
                )
        
        print(f"  ✓ Added {len(target_data)} target nodes")
        
        # 添加蛋白质节点
        for protein in protein_data:
            protein_id = protein.get('gene_symbol', protein.get('uniprot_id', f"protein_{len(self.knowledge_graph.nodes)}"))
            self.knowledge_graph.add_node(
                protein_id,
                'protein',
                {
                    'protein_name': protein.get('protein_name', ''),
                    'gene_symbol': protein.get('gene_symbol', ''),
                    'uniprot_id': protein.get('uniprot_id', ''),
                    'length': protein.get('length', 0),
                    'mass': protein.get('mass', 0)
                }
            )
        
        print(f"  ✓ Added {len(protein_data)} protein nodes")
        
        # 添加相互作用边
        for interaction in interaction_data:
            protein_a = interaction.get('protein_a', interaction.get('preferredName_A', ''))
            protein_b = interaction.get('protein_b', interaction.get('preferredName_B', ''))
            score = float(interaction.get('score', 0))
            
            if protein_a and protein_b:
                self.knowledge_graph.add_edge(
                    protein_a,
                    protein_b,
                    'interacts_with',
                    weight=score,
                    properties={'score': score}
                )
        
        print(f"  ✓ Added {len(interaction_data)} interaction edges")
    
    def enhance_with_pathways(self, pathways: List[Dict[str, Any]]):
        """用通路信息增强知识图谱"""
        print("\n2. Enhancing knowledge graph with pathways...")
        
        for pathway in pathways:
            pathway_id = pathway.get('id', f"pathway_{len(self.knowledge_graph.nodes)}")
            self.knowledge_graph.add_node(
                pathway_id,
                'pathway',
                {
                    'name': pathway.get('name', ''),
                    'description': pathway.get('description', '')
                }
            )
            
            # 添加通路-靶点边
            for target in pathway.get('targets', []):
                if target in self.knowledge_graph.nodes:
                    self.knowledge_graph.add_edge(
                        target,
                        pathway_id,
                        'in_pathway',
                        weight=1.0
                    )
        
        print(f"  ✓ Added {len(pathways)} pathway nodes")
    
    def learn_embeddings(self) -> Optional[Dict[str, Any]]:
        """学习知识图谱嵌入"""
        if not PYKEEN_AVAILABLE:
            print("  Warning: PyKEEN not installed. Skipping embedding learning.")
            return self._mock_embeddings()
        
        try:
            print("\n3. Learning knowledge graph embeddings...")
            
            # 准备训练数据
            triples = []
            for edge in self.knowledge_graph.edges:
                triples.append((edge['source'], edge['type'], edge['target']))
            
            if not triples:
                print("  Error: No triples available for embedding learning")
                return self._mock_embeddings()
            
            # 运行PyKEEN pipeline
            result = pipeline(
                model=TransE,
                dataset=triples,
                training_kwargs={'num_epochs': 100},
                evaluation_kwargs={'use_tqdm': False}
            )
            
            # 获取嵌入
            model = result.model
            entity_embeddings = model.entity_representations[0](indices=None).detach().numpy()
            
            # 映射实体到嵌入
            entity_to_id = result.training.triples_factory.entity_to_id
            embeddings = {}
            for entity, idx in entity_to_id.items():
                embeddings[entity] = entity_embeddings[idx].tolist()
            
            self.features['embeddings'] = embeddings
            print(f"  ✓ Learned embeddings for {len(embeddings)} entities")
            return embeddings
            
        except Exception as e:
            print(f"  Error learning embeddings: {e}")
            return self._mock_embeddings()
    
    def _mock_embeddings(self) -> Dict[str, Any]:
        """生成模拟嵌入"""
        print("  Using mock embeddings...")
        embeddings = {}
        for node_id in self.knowledge_graph.nodes:
            embeddings[node_id] = [0.1] * 128
        return embeddings
    
    def query_graph(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查询知识图谱"""
        print("\n4. Querying knowledge graph...")
        
        results = []
        
        # 支持的查询类型
        if 'target' in query:
            # 查询靶点的邻居
            target = query['target']
            neighbors = self.knowledge_graph.get_neighbors(target)
            for neighbor in neighbors:
                edges = self.knowledge_graph.get_edges(source=target, target=neighbor)
                for edge in edges:
                    results.append({
                        'source': target,
                        'target': neighbor,
                        'relation': edge['type'],
                        'weight': edge['weight']
                    })
        
        elif 'drug' in query:
            # 查询药物的靶点
            drug = query['drug']
            edges = self.knowledge_graph.get_edges(source=drug, edge_type='targets')
            for edge in edges:
                results.append({
                    'drug': drug,
                    'target': edge['target'],
                    'relation': edge['type'],
                    'weight': edge['weight']
                })
        
        print(f"  ✓ Found {len(results)} results for query")
        return results
    
    def predict_interactions(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """预测潜在的相互作用"""
        print("\n5. Predicting potential interactions...")
        
        predictions = []
        
        # 简单的基于相似度的预测
        nodes = list(self.knowledge_graph.nodes.keys())
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue
                
                # 检查是否已有边
                existing_edges = self.knowledge_graph.get_edges(source=node1, target=node2)
                if existing_edges:
                    continue
                
                # 计算相似度（基于共同邻居）
                neighbors1 = set(self.knowledge_graph.get_neighbors(node1))
                neighbors2 = set(self.knowledge_graph.get_neighbors(node2))
                common_neighbors = neighbors1.intersection(neighbors2)
                
                if common_neighbors:
                    similarity = len(common_neighbors) / max(len(neighbors1), len(neighbors2))
                    if similarity >= threshold:
                        predictions.append({
                            'source': node1,
                            'target': node2,
                            'similarity': similarity,
                            'common_neighbors': list(common_neighbors)
                        })
        
        print(f"  ✓ Predicted {len(predictions)} potential interactions")
        return predictions
    
    def save_graph(self, filename: str = "knowledge_graph.json") -> str:
        """保存知识图谱"""
        filepath = os.path.join(self.output_dir, "graphs", filename)
        
        try:
            graph_data = self.knowledge_graph.to_dict()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved knowledge graph to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  Error saving knowledge graph: {e}")
            return ""
    
    def save_embeddings(self, filename: str = "embeddings.json") -> str:
        """保存嵌入"""
        filepath = os.path.join(self.output_dir, "embeddings", filename)
        
        try:
            if 'embeddings' in self.features:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.features['embeddings'], f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ Saved embeddings to {filepath}")
                return filepath
            else:
                print("  Error: No embeddings to save")
                return ""
                
        except Exception as e:
            print(f"  Error saving embeddings: {e}")
            return ""
    
    def visualize_graph(self, filename: str = "knowledge_graph.png") -> str:
        """可视化知识图谱"""
        filepath = os.path.join(self.output_dir, "visualizations", filename)
        return self.knowledge_graph.visualize(filepath)
    
    def run_analysis(self, compound_data: Dict[str, Any], 
                     target_data: List[Dict[str, Any]],
                     protein_data: List[Dict[str, Any]],
                     interaction_data: List[Dict[str, Any]],
                     pathways: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """运行完整的知识图谱分析"""
        print("\n" + "=" * 60)
        print("  KNOWLEDGE GRAPH ANALYSIS")
        print("  知识图谱分析")
        print("=" * 60)
        
        # 构建知识图谱
        self.build_from_data(compound_data, target_data, protein_data, interaction_data)
        
        # 增强知识图谱
        if pathways:
            self.enhance_with_pathways(pathways)
        
        # 学习嵌入
        self.learn_embeddings()
        
        # 预测相互作用
        predictions = self.predict_interactions()
        
        # 保存结果
        self.save_graph()
        self.save_embeddings()
        self.visualize_graph()
        
        print("\n" + "=" * 60)
        print("  KNOWLEDGE GRAPH ANALYSIS COMPLETED")
        print("=" * 60)
        
        return {
            'graph': self.knowledge_graph.to_dict(),
            'predictions': predictions,
            'embeddings': self.features.get('embeddings', {})
        }


def main():
    # 测试知识图谱模块
    # 模拟数据
    compound_data = {
        'name': 'Luteolin',
        'name_cn': '木犀草素',
        'molecular_formula': 'C15H10O6',
        'molecular_weight': 286.24,
        'smiles': 'C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O'
    }
    
    target_data = [
        {
            'target_name': 'Protein Kinase B',
            'gene_symbol': 'AKT1',
            'uniprot_id': 'P31749',
            'activity_value': 100.0,
            'pchembl_value': 7.0
        },
        {
            'target_name': 'Phosphoinositide-3-Kinase Catalytic Subunit Alpha',
            'gene_symbol': 'PIK3CA',
            'uniprot_id': 'P42336',
            'activity_value': 200.0,
            'pchembl_value': 6.7
        },
        {
            'target_name': 'Phosphatase And Tensin Homolog',
            'gene_symbol': 'PTEN',
            'uniprot_id': 'P60484',
            'activity_value': 150.0,
            'pchembl_value': 6.8
        }
    ]
    
    protein_data = [
        {
            'protein_name': 'AKT1',
            'gene_symbol': 'AKT1',
            'uniprot_id': 'P31749',
            'length': 480,
            'mass': 55000.0
        },
        {
            'protein_name': 'PIK3CA',
            'gene_symbol': 'PIK3CA',
            'uniprot_id': 'P42336',
            'length': 1068,
            'mass': 124000.0
        },
        {
            'protein_name': 'PTEN',
            'gene_symbol': 'PTEN',
            'uniprot_id': 'P60484',
            'length': 403,
            'mass': 47000.0
        }
    ]
    
    interaction_data = [
        {'protein_a': 'AKT1', 'protein_b': 'PIK3CA', 'score': 0.9},
        {'protein_a': 'AKT1', 'protein_b': 'PTEN', 'score': 0.85},
        {'protein_a': 'PIK3CA', 'protein_b': 'PTEN', 'score': 0.8}
    ]
    
    pathways = [
        {
            'id': 'pathway_1',
            'name': 'PI3K-AKT signaling pathway',
            'description': 'Phosphoinositide 3-kinase-AKT signaling pathway',
            'targets': ['AKT1', 'PIK3CA', 'PTEN']
        }
    ]
    
    # 初始化知识图谱模块
    kg_module = KnowledgeGraphModule(output_dir="./test_knowledge_graph")
    
    # 运行分析
    results = kg_module.run_analysis(
        compound_data, target_data, protein_data, interaction_data, pathways
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("  KNOWLEDGE GRAPH RESULTS")
    print("=" * 60)
    
    graph_stats = results['graph']['stats']
    print(f"  Nodes: {graph_stats['num_nodes']}")
    print(f"  Edges: {graph_stats['num_edges']}")
    print(f"  Node types: {', '.join(graph_stats['node_types'])}")
    print(f"  Edge types: {', '.join(graph_stats['edge_types'])}")
    
    print(f"\n  Predicted interactions: {len(results['predictions'])}")
    
    if results['embeddings']:
        print(f"  Embeddings: {len(results['embeddings'])} entities")


if __name__ == "__main__":
    main()