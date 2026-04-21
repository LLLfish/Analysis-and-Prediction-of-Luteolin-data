"""
CNN卷积神经网络模块 (Convolutional Neural Network Module)
用于分子3D结构特征提取

功能:
- 分子3D结构处理
- 3D-CNN特征提取
- 空间特征编码
- 局部特征聚合
- 与配体特征提取模块集成
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Initializing CNN module...")

# 检查依赖
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("  ✓ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("  ✗ PyTorch not installed. CNN module will be limited.")

try:
    import deepchem as dc
    from deepchem.feat import AtomicConvFeaturizer, MolGraphConvFeaturizer
    from deepchem.models import GraphConvModel
    DEEPCHEM_AVAILABLE = True
    print("  ✓ DeepChem available")
except ImportError:
    DEEPCHEM_AVAILABLE = False
    print("  ✗ DeepChem not installed. 3D-CNN features will be limited.")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
    print("  ✓ RDKit available")
except ImportError:
    RDKIT_AVAILABLE = False
    print("  ✗ RDKit not installed. Molecular processing will be limited.")


class Molecular3DCNN(nn.Module):
    """3D-CNN模型用于分子结构特征提取"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 128):
        super(Molecular3DCNN, self).__init__()
        
        # 3D卷积层
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, out_channels)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch_size, channels, depth, height, width)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SchNet(nn.Module):
    """SchNet模型用于分子表示"""
    
    def __init__(self, hidden_channels: int = 64, num_filters: int = 128, num_interactions: int = 3, cutoff: float = 5.0):
        super(SchNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        
        # 输入嵌入
        self.embedding = nn.Embedding(100, hidden_channels)  # 假设最多100种元素
        
        # 交互层
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_interactions):
            self.interaction_layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        # 输出层
        self.output = nn.Linear(hidden_channels, 128)
    
    def forward(self, atomic_numbers, positions, batch=None):
        # 简化实现，实际SchNet需要更复杂的消息传递
        x = self.embedding(atomic_numbers)
        
        for layer in self.interaction_layers:
            x = F.relu(layer(x))
        
        # 全局池化
        if batch is not None:
            # 按批次聚合
            unique_batches = torch.unique(batch)
            pooled = []
            for b in unique_batches:
                mask = batch == b
                pooled.append(torch.mean(x[mask], dim=0))
            x = torch.stack(pooled)
        else:
            # 整体聚合
            x = torch.mean(x, dim=0, keepdim=True)
        
        x = self.output(x)
        return x


class CNNModule:
    """CNN卷积神经网络模块"""
    
    def __init__(self, output_dir: str = "./features/cnn"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/features", exist_ok=True)
        os.makedirs(f"{output_dir}/structures", exist_ok=True)
        
        self.mol = None
        self.mol_3d = None
        self.features = {}
        
        if not RDKIT_AVAILABLE:
            print("  Warning: RDKit not installed. Molecular processing will be limited.")
    
    def load_molecule_from_smiles(self, smiles: str, name: str = "molecule") -> bool:
        """从SMILES加载分子"""
        if not RDKIT_AVAILABLE:
            print("  Error: RDKit not installed")
            return False
        
        try:
            self.mol = Chem.MolFromSmiles(smiles)
            if self.mol is None:
                print(f"  Error: Invalid SMILES - {smiles}")
                return False
            
            self.mol = Chem.AddHs(self.mol)
            self.name = name
            self.smiles = smiles
            
            print(f"  ✓ Loaded molecule: {name}")
            print(f"    Atoms: {self.mol.GetNumAtoms()}, Bonds: {self.mol.GetNumBonds()}")
            return True
            
        except Exception as e:
            print(f"  Error loading SMILES: {e}")
            return False
    
    def generate_3d_conformation(self, num_conformers: int = 10) -> bool:
        """生成3D构象"""
        if self.mol is None:
            print("  Error: No molecule loaded")
            return False
        
        try:
            params = AllChem.ETKDGv3()
            conformer_ids = AllChem.EmbedMultipleConfs(
                self.mol, 
                numConfs=num_conformers,
                params=params
            )
            
            if len(conformer_ids) == 0:
                print("  Warning: ETKDG failed, trying random coordinates...")
                AllChem.EmbedMolecule(self.mol, randomSeed=42)
                conformer_ids = [0]
            
            # 能量最小化
            for conf_id in conformer_ids:
                try:
                    AllChem.MMFFOptimizeMolecule(self.mol, confId=conf_id)
                except:
                    AllChem.UFFOptimizeMolecule(self.mol, confId=conf_id)
            
            self.mol_3d = self.mol
            self.best_conformer_id = conformer_ids[0]
            
            print(f"  ✓ Generated {len(conformer_ids)} conformers")
            print(f"    Best conformer ID: {self.best_conformer_id}")
            return True
            
        except Exception as e:
            print(f"  Error generating 3D conformation: {e}")
            return False
    
    def extract_3d_coordinates(self) -> np.ndarray:
        """提取3D坐标"""
        if self.mol_3d is None:
            print("  Error: No 3D conformation available")
            return np.array([])
        
        try:
            conf = self.mol_3d.GetConformer(self.best_conformer_id)
            coords = []
            for i in range(conf.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            coords = np.array(coords)
            self.features['coordinates'] = coords.tolist()
            print(f"  ✓ Extracted 3D coordinates: {coords.shape[0]} atoms")
            return coords
            
        except Exception as e:
            print(f"  Error extracting 3D coordinates: {e}")
            return np.array([])
    
    def create_voxel_grid(self, coords: np.ndarray, grid_size: int = 32, voxel_size: float = 0.5) -> np.ndarray:
        """创建体素网格"""
        try:
            # 计算边界
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = (min_coords + max_coords) / 2
            
            # 创建网格
            grid = np.zeros((grid_size, grid_size, grid_size))
            
            # 填充体素
            for coord in coords:
                # 归一化到网格空间
                grid_coord = ((coord - center) / voxel_size) + grid_size / 2
                grid_coord = np.round(grid_coord).astype(int)
                
                # 检查边界
                if (0 <= grid_coord[0] < grid_size and
                    0 <= grid_coord[1] < grid_size and
                    0 <= grid_coord[2] < grid_size):
                    grid[grid_coord[0], grid_coord[1], grid_coord[2]] = 1.0
            
            self.features['voxel_grid'] = grid.tolist()
            print(f"  ✓ Created voxel grid: {grid.shape}")
            return grid
            
        except Exception as e:
            print(f"  Error creating voxel grid: {e}")
            return np.array([])
    
    def extract_deepchem_features(self) -> Dict[str, Any]:
        """使用DeepChem提取特征"""
        if not DEEPCHEM_AVAILABLE:
            print("  Warning: DeepChem not installed. Skipping DeepChem features.")
            return {}
        
        if self.mol is None:
            print("  Error: No molecule loaded")
            return {}
        
        try:
            # 使用AtomicConvFeaturizer
            featurizer = AtomicConvFeaturizer(
                labels=['atomic_number'],
                radii=[1.0, 2.0, 4.0],
                max_atoms=100
            )
            
            # 转换为DeepChem分子
            dc_mol = dc.Mol(self.mol)
            features = featurizer.featurize([dc_mol])[0]
            
            deepchem_features = {
                'atomic_features': features[0].tolist(),
                'neighbor_features': features[1].tolist(),
                'distance_features': features[2].tolist()
            }
            
            self.features['deepchem'] = deepchem_features
            print(f"  ✓ Extracted DeepChem features")
            return deepchem_features
            
        except Exception as e:
            print(f"  Error extracting DeepChem features: {e}")
            return {}
    
    def train_cnn_model(self, grid: np.ndarray) -> Optional[nn.Module]:
        """训练3D-CNN模型"""
        if not TORCH_AVAILABLE:
            print("  Warning: PyTorch not installed. Skipping CNN model training.")
            return None
        
        try:
            # 准备数据
            grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # 初始化模型
            model = Molecular3DCNN()
            
            # 训练（简单前向传播）
            model.eval()
            with torch.no_grad():
                features = model(grid)
            
            self.features['cnn_features'] = features.squeeze().numpy().tolist()
            print(f"  ✓ Extracted CNN features: {features.shape[1]} dimensions")
            return model
            
        except Exception as e:
            print(f"  Error training CNN model: {e}")
            return None
    
    def calculate_molecular_properties(self) -> Dict[str, float]:
        """计算分子性质"""
        if self.mol is None:
            print("  Error: No molecule loaded")
            return {}
        
        try:
            from rdkit.Chem import Descriptors
            
            properties = {
                'molecular_weight': Descriptors.MolWt(self.mol),
                'logp': Descriptors.MolLogP(self.mol),
                'tpsa': Descriptors.TPSA(self.mol),
                'num_h_donors': Descriptors.NumHDonors(self.mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(self.mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(self.mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(self.mol),
                'ring_count': Descriptors.RingCount(self.mol)
            }
            
            self.features['molecular_properties'] = properties
            print(f"  ✓ Calculated molecular properties")
            return properties
            
        except Exception as e:
            print(f"  Error calculating molecular properties: {e}")
            return {}
    
    def save_structure(self, format: str = 'sdf') -> str:
        """保存分子结构"""
        if self.mol_3d is None:
            print("  Error: No 3D structure to save")
            return ""
        
        filepath = os.path.join(self.output_dir, "structures", f"{self.name}.{format}")
        
        try:
            if format == 'sdf':
                writer = Chem.SDWriter(filepath)
                writer.write(self.mol_3d)
                writer.close()
            elif format == 'mol':
                Chem.MolToMolFile(self.mol_3d, filepath)
            elif format == 'pdb':
                Chem.MolToPDBFile(self.mol_3d, filepath)
            
            print(f"  ✓ Saved structure to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  Error saving structure: {e}")
            return ""
    
    def save_features(self, filename: str = "cnn_features.json") -> str:
        """保存CNN特征"""
        filepath = os.path.join(self.output_dir, "features", filename)
        
        try:
            features_to_save = {
                'extract_time': datetime.now().isoformat(),
                'molecule_name': self.name,
                'smiles': self.smiles,
                'molecular_properties': self.features.get('molecular_properties', {}),
                'coordinates_shape': len(self.features.get('coordinates', [])) if self.features.get('coordinates') else 0,
                'voxel_grid_shape': self.features.get('voxel_grid', [[[]]]).__len__() if self.features.get('voxel_grid') else 0,
                'cnn_features_dim': len(self.features.get('cnn_features', [])) if self.features.get('cnn_features') else 0
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(features_to_save, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved CNN features to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  Error saving features: {e}")
            return ""
    
    def run_analysis(self, smiles: str, name: str = "molecule") -> Dict[str, Any]:
        """运行完整的CNN分析"""
        print("\n" + "=" * 60)
        print("  CNN MOLECULAR ANALYSIS")
        print("  卷积神经网络分子分析")
        print("=" * 60)
        
        # 加载分子
        if not self.load_molecule_from_smiles(smiles, name):
            return {}
        
        # 生成3D构象
        self.generate_3d_conformation()
        
        # 提取3D坐标
        coords = self.extract_3d_coordinates()
        
        # 创建体素网格
        if len(coords) > 0:
            grid = self.create_voxel_grid(coords)
            
            # 训练CNN模型
            self.train_cnn_model(grid)
        
        # 提取DeepChem特征
        self.extract_deepchem_features()
        
        # 计算分子性质
        self.calculate_molecular_properties()
        
        # 保存结果
        self.save_structure()
        self.save_features()
        
        print("\n" + "=" * 60)
        print("  CNN ANALYSIS COMPLETED")
        print("=" * 60)
        
        return self.features


def main():
    # 测试CNN模块
    luteolin_smiles = "C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O"
    
    # 初始化CNN模块
    cnn = CNNModule(output_dir="./features/cnn")
    
    # 运行分析
    features = cnn.run_analysis(luteolin_smiles, "luteolin")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("  CNN ANALYSIS RESULTS")
    print("=" * 60)
    
    if 'molecular_properties' in features:
        props = features['molecular_properties']
        print("  Molecular Properties:")
        for key, value in props.items():
            print(f"    {key}: {value:.4f}")
    
    if 'cnn_features' in features:
        print(f"\n  CNN Features: {len(features['cnn_features'])} dimensions")
    
    if 'coordinates' in features:
        print(f"  3D Coordinates: {len(features['coordinates'])} atoms")


if __name__ == "__main__":
    main()