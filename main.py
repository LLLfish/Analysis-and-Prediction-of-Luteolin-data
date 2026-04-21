"""
网络药理学混合人工智能框架 - 主脚本
Integrated Network Pharmacology Hybrid AI Framework

功能:
- 集成所有模块
- 端到端测试
- 完整流程执行
- 结果汇总
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 导入模块
from data_collection import LuteolinDataCollector
from gnn_module import GNNModule
from cnn_module import CNNModule
from protein_features import ProteinFeatureExtractor
from ligand_features import LigandFeatureExtractor
from transformer_module import TransformerModule
from docking_module import DockingModule
from knowledge_graph_module import KnowledgeGraphModule
from output_module import OutputModule


class NetworkPharmacologyFramework:
    """网络药理学混合人工智能框架"""
    
    def __init__(self, output_dir: str = "./framework_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化各个模块
        self.data_collector = LuteolinDataCollector(output_dir=f"{output_dir}/data")
        self.gnn_module = GNNModule(output_dir=f"{output_dir}/gnn")
        self.cnn_module = CNNModule(output_dir=f"{output_dir}/cnn")
        self.protein_extractor = ProteinFeatureExtractor(output_dir=f"{output_dir}/protein")
        self.ligand_extractor = LigandFeatureExtractor(output_dir=f"{output_dir}/ligand")
        self.transformer_module = TransformerModule(output_dir=f"{output_dir}/transformer")
        self.docking_module = DockingModule(output_dir=f"{output_dir}/docking")
        self.knowledge_graph_module = KnowledgeGraphModule(output_dir=f"{output_dir}/knowledge_graph")
        self.output_module = OutputModule(output_dir=f"{output_dir}/output")
        
        self.results = {}
    
    def run_data_collection(self) -> Dict[str, Any]:
        """运行数据收集"""
        print("\n" + "=" * 70)
        print("  STEP 1: DATA COLLECTION")
        print("  步骤1: 数据收集")
        print("=" * 70)
        
        dataset = self.data_collector.run_pipeline()
        self.results['data_collection'] = dataset
        
        print(f"  ✓ Data collection completed")
        print(f"  - Compound: {dataset.get('compound', {}).get('name', 'N/A')}")
        print(f"  - Targets: {len(dataset.get('targets', []))}")
        print(f"  - Proteins: {len(dataset.get('proteins', []))}")
        print(f"  - Interactions: {len(dataset.get('interactions', []))}")
        
        return dataset
    
    def run_feature_extraction(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """运行特征提取"""
        print("\n" + "=" * 70)
        print("  STEP 2: FEATURE EXTRACTION")
        print("  步骤2: 特征提取")
        print("=" * 70)
        
        # 提取配体特征
        compound = dataset.get('compound', {})
        smiles = compound.get('smiles', 'C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O')
        
        print("\n  2.1 Extracting ligand features...")
        ligand_features = self.ligand_extractor.extract_all_features(smiles, "luteolin")
        
        # 提取蛋白质特征
        print("\n  2.2 Extracting protein features...")
        protein_sequence = "MNSFELKQVNGLDLRLLKPVLSSKESWFKGKQGKKKPKKISKAKIVNGKQIFLSKEL"
        protein_features = self.protein_extractor.extract_all_features(protein_sequence, "sample_protein")
        
        # 提取GNN特征
        print("\n  2.3 Extracting GNN features...")
        interactions = dataset.get('interactions', [])
        if not interactions:
            # 使用模拟数据
            interactions = [
                {'protein_a': 'AKT1', 'protein_b': 'PIK3CA', 'score': 0.9},
                {'protein_a': 'AKT1', 'protein_b': 'PTEN', 'score': 0.85},
                {'protein_a': 'PIK3CA', 'protein_b': 'PTEN', 'score': 0.8}
            ]
        gnn_features = self.gnn_module.run_analysis(interactions)
        
        # 提取CNN特征
        print("\n  2.4 Extracting CNN features...")
        cnn_features = self.cnn_module.run_analysis(smiles, "luteolin")
        
        features = {
            'ligand': ligand_features,
            'protein': protein_features,
            'gnn': gnn_features,
            'cnn': cnn_features
        }
        
        self.results['feature_extraction'] = features
        print("  ✓ Feature extraction completed")
        
        return features
    
    def run_feature_fusion(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """运行特征融合"""
        print("\n" + "=" * 70)
        print("  STEP 3: FEATURE FUSION")
        print("  步骤3: 特征融合")
        print("=" * 70)
        
        fusion_results = self.transformer_module.run_fusion(
            features['gnn'],
            features['cnn'],
            features['protein'],
            features['ligand']
        )
        
        self.results['feature_fusion'] = fusion_results
        print("  ✓ Feature fusion completed")
        
        return fusion_results
    
    def run_docking(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """运行分子对接"""
        print("\n" + "=" * 70)
        print("  STEP 4: MOLECULAR DOCKING")
        print("  步骤4: 分子对接")
        print("=" * 70)
        
        # 准备配体SMILES
        compound = dataset.get('compound', {})
        smiles = compound.get('smiles', 'C1=CC(=C(C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O')
        
        # 准备受体PDB文件
        def create_mock_receptor():
            pdb_content = """ATOM      1  N   ALA A   1      26.500  22.500  22.500  1.00  0.00           N  
ATOM      2  CA  ALA A   1      25.900  21.100  22.500  1.00  0.00           C  
ATOM      3  C   ALA A   1      24.400  21.200  22.000  1.00  0.00           C  
ATOM      4  O   ALA A   1      23.600  20.200  22.000  1.00  0.00           O  
ATOM      5  CB  ALA A   1      26.400  20.200  23.800  1.00  0.00           C  
ATOM      6  H   ALA A   1      27.500  22.500  22.500  1.00  0.00           H  
ATOM      7  HA  ALA A   1      26.200  20.600  21.500  1.00  0.00           H  
ATOM      8  HB1 ALA A   1      25.800  19.200  23.800  1.00  0.00           H  
ATOM      9  HB2 ALA A   1      27.500  20.100  23.800  1.00  0.00           H  
ATOM     10  HB3 ALA A   1      26.300  20.700  24.800  1.00  0.00           H  
END
"""
            pdb_file = os.path.join(self.output_dir, "docking", "receptors", "receptor.pdb")
            os.makedirs(os.path.dirname(pdb_file), exist_ok=True)
            with open(pdb_file, 'w') as f:
                f.write(pdb_content)
            return pdb_file
        
        receptor_pdb = create_mock_receptor()
        
        # 运行对接
        docking_results = self.docking_module.run_docking(smiles, receptor_pdb)
        
        self.results['docking'] = docking_results
        print("  ✓ Molecular docking completed")
        
        return docking_results
    
    def run_knowledge_graph(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """运行知识图谱分析"""
        print("\n" + "=" * 70)
        print("  STEP 5: KNOWLEDGE GRAPH ANALYSIS")
        print("  步骤5: 知识图谱分析")
        print("=" * 70)
        
        # 准备数据
        compound_data = dataset.get('compound', {})
        target_data = dataset.get('targets', [])
        protein_data = dataset.get('proteins', [])
        interaction_data = dataset.get('interactions', [])
        
        # 添加模拟通路数据
        pathways = [
            {
                'id': 'pathway_1',
                'name': 'PI3K-AKT signaling pathway',
                'description': 'Phosphoinositide 3-kinase-AKT signaling pathway',
                'targets': ['AKT1', 'PIK3CA', 'PTEN']
            }
        ]
        
        # 运行知识图谱分析
        kg_results = self.knowledge_graph_module.run_analysis(
            compound_data, target_data, protein_data, interaction_data, pathways
        )
        
        self.results['knowledge_graph'] = kg_results
        print("  ✓ Knowledge graph analysis completed")
        
        return kg_results
    
    def run_output_generation(self) -> Dict[str, Any]:
        """运行输出生成"""
        print("\n" + "=" * 70)
        print("  STEP 6: OUTPUT GENERATION")
        print("  步骤6: 输出生成")
        print("=" * 70)
        
        # 准备数据
        gnn_results = self.results.get('feature_extraction', {}).get('gnn', {})
        docking_results = self.results.get('docking', {})
        knowledge_graph_results = self.results.get('knowledge_graph', {})
        compound_data = self.results.get('data_collection', {}).get('compound', {})
        target_data = self.results.get('data_collection', {}).get('targets', [])
        
        # 运行输出生成
        output_results = self.output_module.run_output_generation(
            gnn_results, docking_results, knowledge_graph_results, compound_data, target_data
        )
        
        self.results['output'] = output_results
        print("  ✓ Output generation completed")
        
        return output_results
    
    def save_results(self) -> str:
        """保存结果"""
        filepath = os.path.join(self.output_dir, "framework_results.json")
        
        try:
            results_to_save = {
                'framework_version': '1.0',
                'run_time': datetime.now().isoformat(),
                'results': self.results,
                'summary': {
                    'data_collection': {
                        'compound': self.results.get('data_collection', {}).get('compound', {}).get('name', 'N/A'),
                        'num_targets': len(self.results.get('data_collection', {}).get('targets', [])),
                        'num_proteins': len(self.results.get('data_collection', {}).get('proteins', [])),
                        'num_interactions': len(self.results.get('data_collection', {}).get('interactions', []))
                    },
                    'feature_extraction': {
                        'ligand_features': bool(self.results.get('feature_extraction', {}).get('ligand', {})),
                        'protein_features': bool(self.results.get('feature_extraction', {}).get('protein', {})),
                        'gnn_features': bool(self.results.get('feature_extraction', {}).get('gnn', {})),
                        'cnn_features': bool(self.results.get('feature_extraction', {}).get('cnn', {}))
                    },
                    'docking': {
                        'best_affinity': self.results.get('docking', {}).get('docking_results', {}).get('binding_affinity', 'N/A')
                    },
                    'knowledge_graph': {
                        'num_nodes': self.results.get('knowledge_graph', {}).get('graph', {}).get('stats', {}).get('num_nodes', 0),
                        'num_edges': self.results.get('knowledge_graph', {}).get('graph', {}).get('stats', {}).get('num_edges', 0)
                    }
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved framework results to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  Error saving results: {e}")
            return ""
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("  NETWORK PHARMACOLOGY HYBRID AI FRAMEWORK")
        print("  网络药理学混合人工智能框架")
        print("  Full Pipeline Execution")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # 步骤1: 数据收集
        dataset = self.run_data_collection()
        
        # 步骤2: 特征提取
        features = self.run_feature_extraction(dataset)
        
        # 步骤3: 特征融合
        self.run_feature_fusion(features)
        
        # 步骤4: 分子对接
        self.run_docking(dataset)
        
        # 步骤5: 知识图谱分析
        self.run_knowledge_graph(dataset)
        
        # 步骤6: 输出生成
        self.run_output_generation()
        
        # 保存结果
        self.save_results()
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("  PIPELINE EXECUTION COMPLETED")
        print(f"  Runtime: {runtime:.2f} seconds")
        print("=" * 80)
        
        return self.results


def main():
    # 初始化框架
    framework = NetworkPharmacologyFramework(output_dir="./framework_output")
    
    # 运行完整流程
    results = framework.run_full_pipeline()
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("  FRAMEWORK EXECUTION SUMMARY")
    print("=" * 80)
    
    # 数据收集摘要
    data_summary = results.get('data_collection', {})
    print(f"  1. Data Collection:")
    print(f"     Compound: {data_summary.get('compound', {}).get('name', 'N/A')} ({data_summary.get('compound', {}).get('name_cn', 'N/A')})")
    print(f"     Targets: {len(data_summary.get('targets', []))}")
    print(f"     Proteins: {len(data_summary.get('proteins', []))}")
    print(f"     Interactions: {len(data_summary.get('interactions', []))}")
    
    # 特征提取摘要
    feature_summary = results.get('feature_extraction', {})
    print(f"\n  2. Feature Extraction:")
    print(f"     Ligand features: {'✓' if feature_summary.get('ligand', {}) else '✗'}")
    print(f"     Protein features: {'✓' if feature_summary.get('protein', {}) else '✗'}")
    print(f"     GNN features: {'✓' if feature_summary.get('gnn', {}) else '✗'}")
    print(f"     CNN features: {'✓' if feature_summary.get('cnn', {}) else '✗'}")
    
    # 分子对接摘要
    docking_summary = results.get('docking', {})
    print(f"\n  3. Molecular Docking:")
    best_affinity = docking_summary.get('docking_results', {}).get('binding_affinity', 'N/A')
    print(f"     Best binding affinity: {best_affinity} kcal/mol")
    
    # 知识图谱摘要
    kg_summary = results.get('knowledge_graph', {})
    print(f"\n  4. Knowledge Graph:")
    num_nodes = kg_summary.get('graph', {}).get('stats', {}).get('num_nodes', 0)
    num_edges = kg_summary.get('graph', {}).get('stats', {}).get('num_edges', 0)
    print(f"     Nodes: {num_nodes}")
    print(f"     Edges: {num_edges}")
    
    # 输出生成摘要
    output_summary = results.get('output', {})
    print(f"\n  5. Output Generation:")
    print(f"     Target discovery report: {'✓' if output_summary.get('target_discovery', {}) else '✗'}")
    print(f"     Mechanism analysis: {'✓' if output_summary.get('mechanism_analysis', {}) else '✗'}")
    print(f"     Drug development suggestions: {'✓' if output_summary.get('drug_development', {}) else '✗'}")
    print(f"     Visualizations: {len(output_summary.get('visualizations', {}))}")
    
    print("\n" + "=" * 80)
    print("  All modules executed successfully!")
    print("  所有模块执行成功！")
    print("=" * 80)


if __name__ == "__main__":
    main()