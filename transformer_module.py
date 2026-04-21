"""
Transformer特征融合模块 (Transformer Feature Fusion Module)
用于整合多源特征

功能:
- 多源特征融合
- Multi-Head Attention机制
- 动态生物知识图谱构建
- 特征交互学习
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Transformer module will be limited.")


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """将输入分割成多个头"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, q, k, v, mask=None):
        # 线性变换
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        # 分割多头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)
        
        # 拼接多头
        batch_size, num_heads, seq_len, d_k = output.size()
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出线性变换
        output = self.W_o(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    """Transformer层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerLayer, self).__init__()
        
        # 多头注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        #  dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights


class TransformerFusion(nn.Module):
    """Transformer融合模型"""
    
    def __init__(self, d_model: int = 128, num_heads: int = 4, num_layers: int = 2, d_ff: int = 256):
        super(TransformerFusion, self).__init__()
        
        self.d_model = d_model
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(TransformerLayer(d_model, num_heads, d_ff))
        
        # 输出层
        self.output_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        attn_weights = []
        
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_weights.append(attn)
        
        x = self.output_layer(x)
        return x, attn_weights


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        """添加节点"""
        self.nodes[node_id] = {
            'id': node_id,
            'type': node_type,
            'properties': properties
        }
    
    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0):
        """添加边"""
        self.edges.append({
            'source': source,
            'target': target,
            'type': edge_type,
            'weight': weight
        })
    
    def build_graph(self) -> Dict[str, Any]:
        """构建知识图谱"""
        return {
            'nodes': list(self.nodes.values()),
            'edges': self.edges
        }


class TransformerModule:
    """Transformer特征融合模块"""
    
    def __init__(self, output_dir: str = "./features/transformer"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/fusion", exist_ok=True)
        os.makedirs(f"{output_dir}/knowledge_graph", exist_ok=True)
        
        self.features = {}
        self.fusion_model = None
        
        if not TORCH_AVAILABLE:
            print("  Warning: PyTorch not installed. Transformer features will be limited.")
    
    def load_features(self, gnn_features: Dict[str, Any], cnn_features: Dict[str, Any], 
                     protein_features: Dict[str, Any], ligand_features: Dict[str, Any]):
        """加载多源特征"""
        self.features['gnn'] = gnn_features
        self.features['cnn'] = cnn_features
        self.features['protein'] = protein_features
        self.features['ligand'] = ligand_features
        
        print(f"  ✓ Loaded features: GNN={bool(gnn_features)}, CNN={bool(cnn_features)}, Protein={bool(protein_features)}, Ligand={bool(ligand_features)}")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
        if len(features) == 0:
            return features
        
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        return (features - mean) / std
    
    def prepare_feature_vectors(self) -> Optional[torch.Tensor]:
        """准备特征向量"""
        try:
            feature_vectors = []
            
            # GNN特征
            if 'node_embeddings' in self.features.get('gnn', {}):
                gnn_embeddings = self.features['gnn']['node_embeddings']
                if gnn_embeddings:
                    # 取第一个节点的嵌入作为示例
                    gnn_vec = next(iter(gnn_embeddings.values()))
                    feature_vectors.append(gnn_vec)
            
            # CNN特征
            if 'cnn_features' in self.features.get('cnn', {}):
                cnn_vec = self.features['cnn']['cnn_features']
                if cnn_vec:
                    feature_vectors.append(cnn_vec)
            
            # 蛋白质特征
            if 'sequence' in self.features.get('protein', {}):
                protein_vec = self.features['protein']['sequence'].get('mean_embedding', [])
                if protein_vec:
                    feature_vectors.append(protein_vec)
            
            # 配体特征
            if 'descriptors' in self.features.get('ligand', {}):
                ligand_desc = self.features['ligand']['descriptors']
                if ligand_desc:
                    # 提取关键描述符
                    key_descriptors = ['MolecularWeight', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']
                    ligand_vec = [ligand_desc.get(key, 0) for key in key_descriptors]
                    feature_vectors.append(ligand_vec)
            
            if not feature_vectors:
                print("  Error: No feature vectors available")
                return None
            
            # 统一特征维度
            max_len = max(len(vec) for vec in feature_vectors)
            padded_vectors = []
            
            for vec in feature_vectors:
                if len(vec) < max_len:
                    # 填充到最大长度
                    padded = np.pad(vec, (0, max_len - len(vec)), 'constant')
                else:
                    # 截断到最大长度
                    padded = vec[:max_len]
                padded_vectors.append(padded)
            
            # 归一化
            normalized = [self.normalize_features(np.array(vec)) for vec in padded_vectors]
            
            # 转换为张量
            features_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            
            print(f"  ✓ Prepared feature vectors: {features_tensor.shape}")
            return features_tensor
            
        except Exception as e:
            print(f"  Error preparing feature vectors: {e}")
            return None
    
    def fuse_features(self) -> Optional[Dict[str, Any]]:
        """融合特征"""
        if not TORCH_AVAILABLE:
            print("  Warning: PyTorch not installed. Skipping feature fusion.")
            return {}
        
        try:
            # 准备特征向量
            features_tensor = self.prepare_feature_vectors()
            if features_tensor is None:
                return {}
            
            # 初始化融合模型
            d_model = features_tensor.size(-1)
            self.fusion_model = TransformerFusion(d_model=d_model)
            
            # 执行融合
            fused_features, attn_weights = self.fusion_model(features_tensor)
            
            # 提取融合特征
            fused_features_np = fused_features.squeeze().detach().numpy()
            
            # 提取注意力权重
            attn_weights_np = [w.squeeze().detach().numpy().tolist() for w in attn_weights]
            
            fusion_results = {
                'fused_features': fused_features_np.tolist(),
                'attention_weights': attn_weights_np,
                'feature_importance': np.mean(fused_features_np, axis=0).tolist()
            }
            
            self.features['fusion'] = fusion_results
            print(f"  ✓ Fused features: {fused_features_np.shape}")
            return fusion_results
            
        except Exception as e:
            print(f"  Error fusing features: {e}")
            return {}
    
    def build_knowledge_graph(self) -> Dict[str, Any]:
        """构建知识图谱"""
        try:
            builder = KnowledgeGraphBuilder()
            
            # 添加药物节点
            if 'ligand' in self.features:
                ligand_desc = self.features['ligand'].get('descriptors', {})
                builder.add_node(
                    'ligand',
                    'drug',
                    {
                        'name': 'Luteolin',
                        'molecular_weight': ligand_desc.get('MolecularWeight', 0),
                        'logp': ligand_desc.get('LogP', 0),
                        'tpsa': ligand_desc.get('TPSA', 0)
                    }
                )
            
            # 添加靶点节点
            if 'gnn' in self.features and 'key_targets' in self.features['gnn']:
                for target, score in self.features['gnn']['key_targets'][:5]:
                    builder.add_node(
                        target,
                        'target',
                        {'importance': score}
                    )
                    # 添加药物-靶点边
                    builder.add_edge('ligand', target, 'targets', weight=score)
            
            # 添加蛋白质节点
            if 'protein' in self.features:
                protein_name = self.features['protein'].get('protein_name', 'protein')
                builder.add_node(
                    protein_name,
                    'protein',
                    {
                        'length': self.features['protein'].get('sequence_length', 0),
                        'molecular_weight': self.features['protein'].get('physicochemical', {}).get('molecular_weight', 0)
                    }
                )
            
            # 构建图谱
            knowledge_graph = builder.build_graph()
            self.features['knowledge_graph'] = knowledge_graph
            
            print(f"  ✓ Built knowledge graph: {len(knowledge_graph['nodes'])} nodes, {len(knowledge_graph['edges'])} edges")
            return knowledge_graph
            
        except Exception as e:
            print(f"  Error building knowledge graph: {e}")
            return {}
    
    def calculate_feature_similarity(self) -> Dict[str, float]:
        """计算特征相似度"""
        try:
            similarities = {}
            
            # 计算配体和蛋白质的相似度
            if 'ligand' in self.features and 'protein' in self.features:
                ligand_desc = self.features['ligand'].get('descriptors', {})
                protein_props = self.features['protein'].get('physicochemical', {})
                
                # 计算分子量比例
                if 'MolecularWeight' in ligand_desc and 'molecular_weight' in protein_props:
                    mw_ratio = ligand_desc['MolecularWeight'] / protein_props['molecular_weight']
                    similarities['mw_ratio'] = mw_ratio
                
                # 计算疏水性匹配
                if 'LogP' in ligand_desc and 'mean_hydrophobicity' in protein_props:
                    hydrophobicity_diff = abs(ligand_desc['LogP'] - protein_props['mean_hydrophobicity'])
                    similarities['hydrophobicity_match'] = max(0, 1 - hydrophobicity_diff / 10)
            
            self.features['similarity'] = similarities
            print(f"  ✓ Calculated feature similarities")
            return similarities
            
        except Exception as e:
            print(f"  Error calculating feature similarity: {e}")
            return {}
    
    def save_fusion_results(self, filename: str = "fusion_results.json") -> str:
        """保存融合结果"""
        filepath = os.path.join(self.output_dir, "fusion", filename)
        
        try:
            fusion_results = {
                'extract_time': datetime.now().isoformat(),
                'fusion': self.features.get('fusion', {}),
                'similarity': self.features.get('similarity', {}),
                'feature_sources': list(self.features.keys())
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(fusion_results, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved fusion results to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  Error saving fusion results: {e}")
            return ""
    
    def save_knowledge_graph(self, filename: str = "knowledge_graph.json") -> str:
        """保存知识图谱"""
        filepath = os.path.join(self.output_dir, "knowledge_graph", filename)
        
        try:
            if 'knowledge_graph' in self.features:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.features['knowledge_graph'], f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ Saved knowledge graph to {filepath}")
                return filepath
            else:
                print("  Error: No knowledge graph to save")
                return ""
                
        except Exception as e:
            print(f"  Error saving knowledge graph: {e}")
            return ""
    
    def run_fusion(self, gnn_features: Dict[str, Any], cnn_features: Dict[str, Any], 
                   protein_features: Dict[str, Any], ligand_features: Dict[str, Any]) -> Dict[str, Any]:
        """运行完整的特征融合流程"""
        print("\n" + "=" * 60)
        print("  TRANSFORMER FEATURE FUSION")
        print("  特征融合")
        print("=" * 60)
        
        # 加载特征
        self.load_features(gnn_features, cnn_features, protein_features, ligand_features)
        
        # 融合特征
        self.fuse_features()
        
        # 构建知识图谱
        self.build_knowledge_graph()
        
        # 计算特征相似度
        self.calculate_feature_similarity()
        
        # 保存结果
        self.save_fusion_results()
        self.save_knowledge_graph()
        
        print("\n" + "=" * 60)
        print("  FEATURE FUSION COMPLETED")
        print("=" * 60)
        
        return self.features


def main():
    # 测试Transformer模块
    from gnn_module import GNNModule
    from cnn_module import CNNModule
    from protein_features import ProteinFeatureExtractor
    from ligand_features import LigandFeatureExtractor
    
    # 测试数据
    luteolin_smiles = "C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O"
    protein_sequence = "MNSFELKQVNGLDLRLLKPVLSSKESWFKGKQGKKKPKKISKAKIVNGKQIFLSKEL"
    
    # 初始化模块
    gnn = GNNModule(output_dir="./test_features/gnn")
    cnn = CNNModule(output_dir="./test_features/cnn")
    protein_extractor = ProteinFeatureExtractor(output_dir="./test_features/protein")
    ligand_extractor = LigandFeatureExtractor(output_dir="./test_features/ligand")
    
    # 提取特征
    test_interactions = [
        {'protein_a': 'AKT1', 'protein_b': 'PIK3CA', 'score': 0.9},
        {'protein_a': 'AKT1', 'protein_b': 'PTEN', 'score': 0.85},
        {'protein_a': 'PIK3CA', 'protein_b': 'PTEN', 'score': 0.8}
    ]
    gnn_features = gnn.run_analysis(test_interactions)
    
    cnn_features = cnn.run_analysis(luteolin_smiles, "luteolin")
    
    protein_features = protein_extractor.extract_all_features(protein_sequence, "sample_protein")
    
    ligand_features = ligand_extractor.extract_all_features(luteolin_smiles, "luteolin")
    
    # 初始化Transformer模块
    transformer = TransformerModule(output_dir="./test_features/transformer")
    
    # 运行融合
    fusion_results = transformer.run_fusion(gnn_features, cnn_features, protein_features, ligand_features)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("  FUSION RESULTS")
    print("=" * 60)
    
    if 'fusion' in fusion_results:
        fusion = fusion_results['fusion']
        print(f"  Fused features dimension: {len(fusion.get('fused_features', []))}")
    
    if 'knowledge_graph' in fusion_results:
        kg = fusion_results['knowledge_graph']
        print(f"  Knowledge graph: {len(kg.get('nodes', []))} nodes, {len(kg.get('edges', []))} edges")
    
    if 'similarity' in fusion_results:
        similarity = fusion_results['similarity']
        print("  Feature similarities:")
        for key, value in similarity.items():
            print(f"    {key}: {value:.4f}")


if __name__ == "__main__":
    main()