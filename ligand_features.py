import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
logger = setup_logger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import Draw
    from rdkit import DataStructs
    from rdkit.Chem.Draw import rdMolDraw2D
    _RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not installed. Ligand feature extraction will be limited.")

_DEEPCHEM_AVAILABLE = False
try:
    import deepchem as dc
    _DEEPCHEM_AVAILABLE = True
except ImportError:
    logger.warning("DeepChem not installed. Some features will be unavailable.")


class LigandFeatureExtractor:

    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features", "ligand")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.join(self.output_dir, "structures"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "descriptors"), exist_ok=True)
        self.mol = None
        self.mol_3d = None
        self.features = {}
        if not _RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ligand feature extraction")

    def load_from_smiles(self, smiles: str, name: str = "ligand") -> bool:
        if not smiles:
            logger.error("Empty SMILES string")
            return False
        self.name = name
        self.smiles = smiles
        try:
            self.mol = Chem.MolFromSmiles(smiles)
            if self.mol is None:
                logger.error("Invalid SMILES: %s", smiles)
                return False
            self.mol = Chem.AddHs(self.mol)
            logger.info("Loaded molecule: %s", name)
            return True
        except Exception as e:
            logger.error("Failed to load SMILES: %s", e)
            return False

    def generate_3d_conformation(self, num_conformers: int = 50) -> Optional[Any]:
        if self.mol is None:
            logger.error("No molecule loaded")
            return None
        try:
            params = AllChem.EmbedMultipleConfs(self.mol, numConformers=num_conformers)
            if params < 1:
                logger.warning("No conformers generated, trying ETKDG")
                params = rdDistGeom.EmbedMultipleConfs(self.mol, numConformers=num_conformers)
            AllChem.MMFFOptimizeMoleculeConfs(self.mol)
            self.mol_3d = Chem.Mol(self.mol)
            logger.info("Generated %d conformers", num_conformers)
            return self.mol_3d
        except Exception as e:
            logger.error("3D conformation generation failed: %s", e)
            return None

    def compute_descriptors(self) -> Dict[str, float]:
        if self.mol is None:
            return {}
        desc = {}
        try:
            desc_list = [desc_name for desc_name in dir(Descriptors) if desc_name[0].islower()]
            for d in desc_list:
                try:
                    desc[d] = getattr(Descriptors, d)(self.mol)
                except Exception:
                    pass
            self.features['descriptors'] = desc
            logger.info("Computed %d molecular descriptors", len(desc))
        except Exception as e:
            logger.error("Descriptor computation failed: %s", e)
        return desc

    def compute_fingerprints(self) -> Dict[str, np.ndarray]:
        if self.mol is None:
            return {}
        fps = {}
        try:
            morgan = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=2048)
            arr = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(morgan, arr)
            fps['morgan'] = arr
            self.features['fingerprints'] = {k: v.tolist() for k, v in fps.items()}
            logger.info("Computed Morgan fingerprint (2048 bits)")
        except Exception as e:
            logger.error("Fingerprint computation failed: %s", e)
        return fps

    def generate_molecular_graph(self) -> Optional[Dict[str, Any]]:
        if self.mol is None:
            return None
        graph = {}
        try:
            conf = self.mol.GetConformer()
            atoms = []
            for atom in self.mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atoms.append({
                    'symbol': atom.GetSymbol(),
                    'atomic_num': atom.GetAtomicNum(),
                    'x': pos.x, 'y': pos.y, 'z': pos.z
                })
            graph['num_atoms'] = len(atoms)
            graph['atoms'] = atoms
            bonds = []
            for bond in self.mol.GetBonds():
                bonds.append({
                    'begin': bond.GetBeginAtomIdx(),
                    'end': bond.GetEndAtomIdx(),
                    'type': str(bond.GetBondType())
                })
            graph['num_bonds'] = len(bonds)
            graph['bonds'] = bonds
            self.features['molecular_graph'] = graph
            logger.info("Generated molecular graph: %d atoms, %d bonds", len(atoms), len(bonds))
        except Exception as e:
            logger.error("Molecular graph generation failed: %s", e)
        return graph

    def visualize_molecule(self, filename: str = "molecule.png") -> Optional[str]:
        if self.mol is None:
            return None
        try:
            filepath = os.path.join(self.output_dir, "structures", filename)
            img = Draw.MolToImage(self.mol, size=(600, 600))
            img.save(filepath)
            logger.info("Saved molecular image: %s", filepath)
            return filepath
        except Exception as e:
            logger.error("Molecular visualization failed: %s", e)
            return None

    def save_features(self, filename: str = "ligand_features.json") -> str:
        filepath = os.path.join(self.output_dir, "descriptors", filename)
        data = {
            'name': getattr(self, 'name', 'ligand'),
            'smiles': getattr(self, 'smiles', ''),
            'features': self.features
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        logger.info("Saved ligand features: %s", filepath)
        return filepath

    def extract_all_features(self, smiles: str, name: str = "ligand") -> Dict[str, Any]:
        result = {'name': name, 'smiles': smiles, 'success': False}
        if not self.load_from_smiles(smiles, name):
            return result
        self.generate_3d_conformation()
        self.compute_descriptors()
        self.compute_fingerprints()
        self.generate_molecular_graph()
        self.save_features()
        result['success'] = True
        result['features'] = self.features
        logger.info("All features extracted for %s", name)
        return result


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)