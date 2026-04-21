"""
特征提取管道 (Feature Extraction Pipeline)
整合配体和蛋白质特征提取模块

Phase 1: 特征提取模块
- 配体特征提取 (RDKit)
- 蛋白质特征提取 (ESM-2 + ESMFold)
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ligand_features import LigandFeatureExtractor
from protein_features import ProteinFeatureExtractor


class FeatureExtractionPipeline:
    
    def __init__(self, 
                 output_dir: str = "./features",
                 use_gpu: bool = True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.ligand_extractor = LigandFeatureExtractor(
            output_dir=f"{output_dir}/ligand"
        )
        
        device = "auto" if use_gpu else "cpu"
        self.protein_extractor = ProteinFeatureExtractor(
            output_dir=f"{output_dir}/protein",
            device=device
        )
        
        self.ligand_features = {}
        self.protein_features = {}
        self.combined_features = {}
    
    def extract_ligand_features(self, 
                                 smiles: str, 
                                 ligand_name: str = "ligand") -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("  STEP 1: LIGAND FEATURE EXTRACTION")
        print("  步骤1: 配体特征提取")
        print("=" * 60)
        
        self.ligand_features = self.ligand_extractor.extract_all_features(
            smiles, ligand_name
        )
        
        return self.ligand_features
    
    def extract_protein_features(self, 
                                  sequence: str, 
                                  protein_name: str = "protein",
                                  predict_structure: bool = True) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("  STEP 2: PROTEIN FEATURE EXTRACTION")
        print("  步骤2: 蛋白质特征提取")
        print("=" * 60)
        
        self.protein_features = self.protein_extractor.extract_all_features(
            sequence, protein_name, predict_structure
        )
        
        return self.protein_features
    
    def combine_features(self) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("  STEP 3: FEATURE COMBINATION")
        print("  步骤3: 特征融合")
        print("=" * 60)
        
        combined = {
            'extract_time': datetime.now().isoformat(),
            'ligand': {},
            'protein': {},
            'interaction_features': {}
        }
        
        if self.ligand_features:
            combined['ligand'] = {
                'name': self.ligand_extractor.name,
                'smiles': self.ligand_extractor.smiles,
                'descriptors': self.ligand_features.get('descriptors', {}),
                'fingerprints': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in self.ligand_features.get('fingerprints', {}).items()
                },
                'graph_summary': {
                    'num_atoms': self.ligand_features.get('graph', {}).get('num_atoms', 0),
                    'num_bonds': self.ligand_features.get('graph', {}).get('num_bonds', 0)
                }
            }
            
            if '3d_features' in self.ligand_features:
                combined['ligand']['has_3d'] = True
        
        if self.protein_features:
            combined['protein'] = {
                'name': self.protein_extractor.protein_name,
                'sequence_length': len(self.protein_extractor.sequence) if self.protein_extractor.sequence else 0,
                'embedding_dim': self.protein_features.get('sequence', {}).get('embedding_dim', 0),
                'physicochemical': self.protein_features.get('physicochemical', {}),
                'secondary_structure': self.protein_features.get('secondary_structure', {}),
                'structure': self.protein_features.get('structure', {})
            }
        
        if self.ligand_features and self.protein_features:
            interaction = self._compute_interaction_features()
            combined['interaction_features'] = interaction
        
        self.combined_features = combined
        print(f"  ✓ Features combined successfully")
        
        return combined
    
    def _compute_interaction_features(self) -> Dict[str, Any]:
        interaction = {}
        
        try:
            ligand_desc = self.ligand_features.get('descriptors', {})
            protein_props = self.protein_features.get('physicochemical', {})
            
            if ligand_desc and protein_props:
                interaction['ligand_mw'] = ligand_desc.get('MolecularWeight', 0)
                interaction['ligand_logp'] = ligand_desc.get('LogP', 0)
                interaction['ligand_hbd'] = ligand_desc.get('NumHDonors', 0)
                interaction['ligand_hba'] = ligand_desc.get('NumHAcceptors', 0)
                interaction['ligand_tpsa'] = ligand_desc.get('TPSA', 0)
                
                interaction['protein_length'] = protein_props.get('length', 0)
                interaction['protein_mw'] = protein_props.get('molecular_weight', 0)
                interaction['protein_hydrophobicity'] = protein_props.get('mean_hydrophobicity', 0)
                interaction['protein_aromaticity'] = protein_props.get('aromaticity', 0)
                
                interaction['mw_ratio'] = (
                    ligand_desc.get('MolecularWeight', 1) / 
                    max(protein_props.get('molecular_weight', 1), 1)
                )
                
                interaction['size_compatibility'] = (
                    1.0 if ligand_desc.get('MolecularWeight', 0) < 500 else 0.5
                )
                
                interaction['hydrophobicity_match'] = (
                    1.0 if abs(ligand_desc.get('LogP', 0) - protein_props.get('mean_hydrophobicity', 0)) < 3 
                    else 0.5
                )
            
            print(f"  ✓ Computed interaction features")
            
        except Exception as e:
            print(f"  Warning: Could not compute interaction features: {e}")
        
        return interaction
    
    def save_combined_features(self, filename: str = "combined_features.json") -> str:
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.combined_features, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved combined features to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  Error saving combined features: {e}")
            return ""
    
    def run_pipeline(self, 
                     smiles: str, 
                     sequence: str,
                     ligand_name: str = "ligand",
                     protein_name: str = "protein",
                     predict_protein_structure: bool = True) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("  FEATURE EXTRACTION PIPELINE")
        print("  特征提取管道")
        print("  Phase 1: Ligand + Protein Feature Extraction")
        print("=" * 70)
        
        self.extract_ligand_features(smiles, ligand_name)
        
        self.extract_protein_features(sequence, protein_name, predict_protein_structure)
        
        self.combine_features()
        
        self.save_combined_features()
        
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        return self.combined_features
    
    def get_feature_summary(self) -> str:
        summary = []
        summary.append("\n" + "=" * 60)
        summary.append("  FEATURE SUMMARY")
        summary.append("=" * 60)
        
        if self.ligand_features:
            summary.append("\n  [LIGAND FEATURES]")
            desc = self.ligand_features.get('descriptors', {})
            summary.append(f"    Molecular Weight: {desc.get('MolecularWeight', 'N/A'):.2f}")
            summary.append(f"    LogP: {desc.get('LogP', 'N/A'):.2f}")
            summary.append(f"    TPSA: {desc.get('TPSA', 'N/A'):.2f}")
            summary.append(f"    H-Bond Donors: {desc.get('NumHDonors', 'N/A')}")
            summary.append(f"    H-Bond Acceptors: {desc.get('NumHAcceptors', 'N/A')}")
            summary.append(f"    Rotatable Bonds: {desc.get('NumRotatableBonds', 'N/A')}")
            
            fps = self.ligand_features.get('fingerprints', {})
            summary.append(f"    Fingerprints: {list(fps.keys())}")
        
        if self.protein_features:
            summary.append("\n  [PROTEIN FEATURES]")
            props = self.protein_features.get('physicochemical', {})
            summary.append(f"    Sequence Length: {props.get('length', 'N/A')}")
            summary.append(f"    Molecular Weight: {props.get('molecular_weight', 0):.2f} Da")
            summary.append(f"    Mean Hydrophobicity: {props.get('mean_hydrophobicity', 0):.2f}")
            summary.append(f"    Aromaticity: {props.get('aromaticity', 0):.3f}")
            summary.append(f"    Net Charge: {props.get('net_charge', 0):.1f}")
            
            seq_feat = self.protein_features.get('sequence', {})
            summary.append(f"    ESM-2 Embedding Dim: {seq_feat.get('embedding_dim', 'N/A')}")
            
            ss = self.protein_features.get('secondary_structure', {})
            summary.append(f"    Helix Content: {ss.get('helix_content', 0):.2%}")
            summary.append(f"    Sheet Content: {ss.get('sheet_content', 0):.2%}")
        
        if self.combined_features.get('interaction_features'):
            summary.append("\n  [INTERACTION FEATURES]")
            inter = self.combined_features['interaction_features']
            summary.append(f"    MW Ratio: {inter.get('mw_ratio', 0):.6f}")
            summary.append(f"    Size Compatibility: {inter.get('size_compatibility', 0):.2f}")
            summary.append(f"    Hydrophobicity Match: {inter.get('hydrophobicity_match', 0):.2f}")
        
        return "\n".join(summary)


def main():
    luteolin_smiles = "C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O"
    
    ptgs2_sequence = (
        "MNSFELKQVNGLDLRLLKPVLSSKESWFKGKQGKKKPKKISKAKIVNGKQIFLSKEL"
        "DRFQHRLQKQVNGLDLRLLKPVLSSKESWFKGKQGKKKPKKISKAKIVNGKQIFLSK"
    )
    
    pipeline = FeatureExtractionPipeline(output_dir="./features")
    
    combined_features = pipeline.run_pipeline(
        smiles=luteolin_smiles,
        sequence=ptgs2_sequence,
        ligand_name="luteolin",
        protein_name="PTGS2",
        predict_protein_structure=True
    )
    
    print(pipeline.get_feature_summary())


if __name__ == "__main__":
    main()
