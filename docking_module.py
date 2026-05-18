import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
logger = setup_logger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms
    from rdkit.Chem import rdDistGeom
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not installed. Docking preparation will be limited.")

OPENBABEL_AVAILABLE = False
try:
    from openbabel import openbabel
    from openbabel import pybel
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    logger.warning("Open Babel not installed. Format conversion will be unavailable.")


class DockingValidator:
    def __init__(self, output_dir: str = "", vina_path: str = "vina"):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "docking")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.join(self.output_dir, "structures"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "results"), exist_ok=True)
        self.vina_path = vina_path
        self.ligand = None
        self.protein = None
        self.results = []

    def prepare_ligand(self, mol, ligand_name: str = "ligand") -> bool:
        if not RDKIT_AVAILABLE or mol is None:
            return False
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            self.ligand = mol
            self.ligand_name = ligand_name
            pdbqt_path = os.path.join(self.output_dir, "structures", f"{ligand_name}.pdbqt")
            logger.info("Ligand prepared for docking: %s", ligand_name)
            return True
        except Exception as e:
            logger.error("Ligand preparation failed: %s", e)
            return False

    def prepare_protein(self, protein_pdb: str, protein_name: str = "protein") -> bool:
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(protein_name, protein_pdb)
            self.protein = structure
            self.protein_name = protein_name
            logger.info("Protein loaded for docking: %s", protein_name)
            return True
        except ImportError:
            logger.warning("BioPython not installed, using PDB file directly")
            self.protein = protein_pdb
            self.protein_name = protein_name
            return True
        except Exception as e:
            logger.error("Protein preparation failed: %s", e)
            return False

    def validate_docking(self, vina_output: Dict[str, Any]) -> Dict[str, Any]:
        validation = {
            'timestamp': datetime.now().isoformat(),
            'ligand': self.ligand_name if hasattr(self, 'ligand_name') else 'unknown',
            'protein': self.protein_name if hasattr(self, 'protein_name') else 'unknown',
            'vina_results': vina_output,
            'binding_affinity': None,
            'validation_metrics': {}
        }
        try:
            if 'results' in vina_output:
                best_mode = vina_output['results'][0]
                validation['binding_affinity'] = best_mode.get('affinity', 0)
            logger.info("Docking validation completed, best affinity: %.2f kcal/mol",
                        validation['binding_affinity'] or 0)
        except Exception as e:
            logger.error("Docking validation failed: %s", e)
        return validation

    def save_results(self, results: Dict[str, Any], filename: str = "docking_results.json"):
        filepath = os.path.join(self.output_dir, "results", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved docking results: %s", filepath)