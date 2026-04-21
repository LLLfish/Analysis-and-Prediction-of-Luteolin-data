"""
配体特征提取模块 (Ligand Feature Extraction Module)
使用RDKit进行分子结构处理和特征提取

功能:
- 分子结构标准化
- 3D构象生成
- 能量最小化
- 分子描述符计算
- 分子指纹生成
- 分子图表示
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import Draw
    from rdkit import DataStructs
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not installed. Ligand feature extraction will be limited.")

try:
    import deepchem as dc
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False


class LigandFeatureExtractor:
    
    def __init__(self, output_dir: str = "./features/ligand"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/structures", exist_ok=True)
        os.makedirs(f"{output_dir}/descriptors", exist_ok=True)
        
        self.mol = None
        self.mol_3d = None
        self.features = {}
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ligand feature extraction")
    
    def load_from_smiles(self, smiles: str, name: str = "ligand") -> bool:
        if not smiles:
            print("Error: Empty SMILES string")
            return False
        
        self.name = name
        self.smiles = smiles
        
        try:
            self.mol = Chem.MolFromSmiles(smiles)
            if self.mol is None:
                print(f"Error: Invalid SMILES - {smiles}")
                return False
            
            self.mol = Chem.AddHs(self.mol)
            print(f"  ✓ Loaded molecule: {name}")
            print(f"    Atoms: {self.mol.GetNumAtoms()}, Bonds: {self.mol.GetNumBonds()}")
            return True
            
        except Exception as e:
            print(f"Error loading SMILES: {e}")
            return False
    
    def load_from_mol_file(self, mol_file: str) -> bool:
        try:
            if mol_file.endswith('.sdf'):
                suppl = Chem.SDMolSupplier(mol_file)
                self.mol = next(suppl)
            elif mol_file.endswith('.mol') or mol_file.endswith('.mol2'):
                self.mol = Chem.MolFromMolFile(mol_file)
            else:
                print(f"Unsupported file format: {mol_file}")
                return False
            
            if self.mol is None:
                print(f"Error: Could not load molecule from {mol_file}")
                return False
            
            self.mol = Chem.AddHs(self.mol)
            self.name = os.path.basename(mol_file).split('.')[0]
            print(f"  ✓ Loaded molecule from file: {self.name}")
            return True
            
        except Exception as e:
            print(f"Error loading molecule file: {e}")
            return False
    
    def standardize_structure(self) -> bool:
        if self.mol is None:
            print("Error: No molecule loaded")
            return False
        
        try:
            self.mol = Chem.RemoveHs(self.mol)
            self.mol = Chem.AddHs(self.mol)
            
            for atom in self.mol.GetAtoms():
                atom.SetProp('_Name', f"Atom_{atom.GetIdx()}")
            
            Chem.SanitizeMol(self.mol)
            
            print("  ✓ Structure standardized")
            return True
            
        except Exception as e:
            print(f"Error standardizing structure: {e}")
            return False
    
    def generate_3d_conformation(self, 
                                  method: str = "ETKDG",
                                  num_conformers: int = 10,
                                  optimize: bool = True) -> bool:
        if self.mol is None:
            print("Error: No molecule loaded")
            return False
        
        try:
            params = AllChem.ETKDGv3()
            
            conformer_ids = AllChem.EmbedMultipleConfs(
                self.mol, 
                numConfs=num_conformers,
                params=params
            )
            
            if len(conformer_ids) == 0:
                print("Warning: ETKDG failed, trying random coordinates...")
                AllChem.EmbedMolecule(self.mol, randomSeed=42)
                conformer_ids = [0]
            
            if optimize:
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
            print(f"Error generating 3D conformation: {e}")
            return False
    
    def calculate_molecular_descriptors(self) -> Dict[str, float]:
        if self.mol is None:
            return {}
        
        descriptors = {}
        
        try:
            descriptors['MolecularWeight'] = Descriptors.MolWt(self.mol)
            descriptors['LogP'] = Descriptors.MolLogP(self.mol)
            descriptors['TPSA'] = Descriptors.TPSA(self.mol)
            descriptors['NumHDonors'] = Descriptors.NumHDonors(self.mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(self.mol)
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(self.mol)
            descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(self.mol)
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(self.mol)
            descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(self.mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(self.mol)
            descriptors['RingCount'] = Descriptors.RingCount(self.mol)
            descriptors['FractionCSP3'] = Descriptors.FractionCSP3(self.mol)
            
            descriptors['MolMR'] = Descriptors.MolMR(self.mol)
            descriptors['LabuteASA'] = Descriptors.LabuteASA(self.mol)
            descriptors['BalabanJ'] = Descriptors.BalabanJ(self.mol) if self.mol.GetNumBonds() > 0 else 0
            descriptors['BertzCT'] = Descriptors.BertzCT(self.mol)
            
            descriptors['NumValenceElectrons'] = Descriptors.NumValenceElectrons(self.mol)
            descriptors['MaxPartialCharge'] = Descriptors.MaxPartialCharge(self.mol)
            descriptors['MinPartialCharge'] = Descriptors.MinPartialCharge(self.mol)
            
            descriptors['HallKierAlpha'] = Descriptors.HallKierAlpha(self.mol)
            descriptors['Kappa1'] = Descriptors.Kappa1(self.mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(self.mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(self.mol)
            
            descriptors['Chi0'] = Descriptors.Chi0(self.mol)
            descriptors['Chi1'] = Descriptors.Chi1(self.mol)
            descriptors['Chi0v'] = Descriptors.Chi0v(self.mol)
            descriptors['Chi1v'] = Descriptors.Chi1v(self.mol)
            descriptors['Chi2v'] = Descriptors.Chi2v(self.mol)
            descriptors['Chi3v'] = Descriptors.Chi3v(self.mol)
            
            self.features['descriptors'] = descriptors
            print(f"  ✓ Calculated {len(descriptors)} molecular descriptors")
            
        except Exception as e:
            print(f"Warning: Some descriptors could not be calculated: {e}")
        
        return descriptors
    
    def generate_fingerprints(self, 
                               fp_types: List[str] = ['Morgan', 'MACCS', 'RDKit']) -> Dict[str, np.ndarray]:
        if self.mol is None:
            return {}
        
        fingerprints = {}
        
        try:
            if 'Morgan' in fp_types:
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                    self.mol, radius=2, nBits=2048
                )
                fingerprints['Morgan'] = np.array(morgan_fp)
            
            if 'MACCS' in fp_types:
                from rdkit.Chem import MACCSkeys
                maccs_fp = MACCSkeys.GenMACCSKeys(self.mol)
                fingerprints['MACCS'] = np.array(maccs_fp)
            
            if 'RDKit' in fp_types:
                rdkit_fp = Chem.RDKFingerprint(self.mol)
                fingerprints['RDKit'] = np.array(rdkit_fp)
            
            if 'AtomPair' in fp_types:
                atompair_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(self.mol)
                fingerprints['AtomPair'] = np.array(atompair_fp)
            
            if 'TopologicalTorsion' in fp_types:
                tt_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(self.mol)
                fingerprints['TopologicalTorsion'] = np.array(tt_fp)
            
            self.features['fingerprints'] = fingerprints
            print(f"  ✓ Generated {len(fingerprints)} fingerprint types")
            
        except Exception as e:
            print(f"Warning: Some fingerprints could not be generated: {e}")
        
        return fingerprints
    
    def extract_graph_features(self) -> Dict[str, Any]:
        if self.mol is None:
            return {}
        
        graph_features = {}
        
        try:
            atom_features = []
            for atom in self.mol.GetAtoms():
                atom_feat = {
                    'idx': atom.GetIdx(),
                    'symbol': atom.GetSymbol(),
                    'atomic_num': atom.GetAtomicNum(),
                    'degree': atom.GetDegree(),
                    'formal_charge': atom.GetFormalCharge(),
                    'hybridization': str(atom.GetHybridization()),
                    'is_aromatic': atom.GetIsAromatic(),
                    'num_hydrogens': atom.GetTotalNumHs(),
                    'in_ring': atom.IsInRing(),
                    'num_radical_electrons': atom.GetNumRadicalElectrons(),
                    'chiral_tag': str(atom.GetChiralTag())
                }
                atom_features.append(atom_feat)
            
            bond_features = []
            for bond in self.mol.GetBonds():
                bond_feat = {
                    'idx': bond.GetIdx(),
                    'begin_atom': bond.GetBeginAtomIdx(),
                    'end_atom': bond.GetEndAtomIdx(),
                    'bond_type': str(bond.GetBondType()),
                    'is_conjugated': bond.GetIsConjugated(),
                    'in_ring': bond.IsInRing(),
                    'stereo': str(bond.GetStereo())
                }
                bond_features.append(bond_feat)
            
            adj_matrix = Chem.GetAdjacencyMatrix(self.mol)
            
            dist_matrix = Chem.GetDistanceMatrix(self.mol)
            
            graph_features = {
                'num_atoms': self.mol.GetNumAtoms(),
                'num_bonds': self.mol.GetNumBonds(),
                'atom_features': atom_features,
                'bond_features': bond_features,
                'adjacency_matrix': adj_matrix.tolist(),
                'distance_matrix': dist_matrix.tolist()
            }
            
            self.features['graph'] = graph_features
            print(f"  ✓ Extracted graph features: {len(atom_features)} atoms, {len(bond_features)} bonds")
            
        except Exception as e:
            print(f"Warning: Graph features could not be extracted: {e}")
        
        return graph_features
    
    def extract_3d_features(self) -> Dict[str, Any]:
        if self.mol_3d is None:
            print("Warning: No 3D conformation available")
            return {}
        
        features_3d = {}
        
        try:
            conf = self.mol_3d.GetConformer(self.best_conformer_id)
            
            coords = []
            for i in range(conf.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            features_3d['coordinates'] = coords
            
            if DEEPCHEM_AVAILABLE:
                try:
                    from deepchem.feat import RDKitDescriptors
                    featurizer = RDKitDescriptors()
                    rdkit_desc = featurizer.featurize(self.mol_3d)
                    features_3d['rdkit_descriptors'] = rdkit_desc.tolist() if len(rdkit_desc) > 0 else []
                except:
                    pass
            
            try:
                shape = rdMolDescriptors.CalcPBF(self.mol_3d, confId=self.best_conformer_id)
                features_3d['PBF'] = shape
                
                pmi = rdMolDescriptors.CalcPMI1(self.mol_3d), \
                      rdMolDescriptors.CalcPMI2(self.mol_3d), \
                      rdMolDescriptors.CalcPMI3(self.mol_3d)
                features_3d['PMI'] = list(pmi)
                
                radius = rdMolDescriptors.CalcExactMolWt(self.mol_3d)
                features_3d['spherocity'] = rdMolDescriptors.CalcSpherocityIndex(self.mol_3d)
                features_3d['asphericity'] = rdMolDescriptors.CalcAsphericity(self.mol_3d)
                features_3d['eccentricity'] = rdMolDescriptors.CalcEccentricity(self.mol_3d)
                features_3d['inertial_shape_factor'] = rdMolDescriptors.CalcInertialShapeFactor(self.mol_3d)
                
            except Exception as e:
                print(f"Warning: Some 3D descriptors could not be calculated: {e}")
            
            self.features['3d_features'] = features_3d
            print(f"  ✓ Extracted 3D features")
            
        except Exception as e:
            print(f"Warning: 3D features could not be extracted: {e}")
        
        return features_3d
    
    def save_molecule(self, format: str = 'sdf', filename: Optional[str] = None) -> str:
        if self.mol_3d is None:
            print("Warning: No 3D molecule to save")
            return ""
        
        if filename is None:
            filename = f"{self.name}.{format}"
        
        filepath = os.path.join(self.output_dir, "structures", filename)
        
        try:
            if format == 'sdf':
                writer = Chem.SDWriter(filepath)
                writer.write(self.mol_3d)
                writer.close()
            elif format == 'mol':
                Chem.MolToMolFile(self.mol_3d, filepath)
            elif format == 'pdb':
                Chem.MolToPDBFile(self.mol_3d, filepath)
            elif format == 'smiles':
                with open(filepath, 'w') as f:
                    f.write(Chem.MolToSmiles(self.mol))
            
            print(f"  ✓ Saved molecule to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving molecule: {e}")
            return ""
    
    def save_features(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"{self.name}_features.json"
        
        filepath = os.path.join(self.output_dir, "descriptors", filename)
        
        try:
            features_to_save = {
                'name': self.name,
                'smiles': self.smiles if hasattr(self, 'smiles') else '',
                'extract_time': datetime.now().isoformat(),
                'descriptors': self.features.get('descriptors', {}),
                'fingerprints': {k: v.tolist() for k, v in self.features.get('fingerprints', {}).items()},
                'graph': self.features.get('graph', {}),
                '3d_features': self.features.get('3d_features', {})
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(features_to_save, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved features to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving features: {e}")
            return ""
    
    def extract_all_features(self, smiles: str, name: str = "ligand") -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("  LIGAND FEATURE EXTRACTION")
        print("  配体特征提取")
        print("=" * 60)
        
        if not self.load_from_smiles(smiles, name):
            return {}
        
        self.standardize_structure()
        self.generate_3d_conformation()
        self.calculate_molecular_descriptors()
        self.generate_fingerprints()
        self.extract_graph_features()
        self.extract_3d_features()
        
        self.save_molecule(format='sdf')
        self.save_molecule(format='pdb')
        self.save_features()
        
        print("\n" + "=" * 60)
        print("  FEATURE EXTRACTION COMPLETED")
        print("=" * 60)
        
        return self.features


def main():
    luteolin_smiles = "C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O"
    
    extractor = LigandFeatureExtractor(output_dir="./features/ligand")
    features = extractor.extract_all_features(luteolin_smiles, "luteolin")
    
    if features:
        print("\n" + "=" * 60)
        print("  FEATURE SUMMARY")
        print("=" * 60)
        
        if 'descriptors' in features:
            print(f"\n  Molecular Descriptors ({len(features['descriptors'])}):")
            for key, value in list(features['descriptors'].items())[:5]:
                print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
            print("    ...")
        
        if 'fingerprints' in features:
            print(f"\n  Fingerprints:")
            for fp_name, fp_array in features['fingerprints'].items():
                print(f"    {fp_name}: {len(fp_array)} bits, {np.sum(fp_array)} set bits")


if __name__ == "__main__":
    main()
