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

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Protein feature extraction will be limited.")

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModel, AutoModel, AutoTokenizer
    import transformers
    transformers.logging.set_verbosity_error()
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not installed. ESM-2 features will be unavailable.")


class ProteinFeatureExtractor:
    def __init__(self, output_dir: str = "", esm_model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = "auto"):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features", "protein")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.join(self.output_dir, "structures"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "embeddings"), exist_ok=True)
        self.esm_model_name = esm_model_name
        self.device = self._get_device(device) if TORCH_AVAILABLE else "cpu"
        self.sequence = None
        self.protein_name = None
        self.features = {}
        self.esm_model = None
        self.esm_tokenizer = None
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for protein feature extraction")

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def load_esm_model(self) -> bool:
        if not _TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not installed")
            return False
        try:
            logger.info("Loading ESM-2 model: %s", self.esm_model_name)
            self.esm_tokenizer = AutoTokenizer.from_pretrained(self.esm_model_name)
            self.esm_model = AutoModel.from_pretrained(self.esm_model_name)
            self.esm_model.to(self.device)
            self.esm_model.eval()
            logger.info("ESM-2 model loaded successfully")
            return True
        except Exception as e:
            logger.error("Error loading ESM-2 model: %s", e)
            return False

    def load_sequence(self, sequence: str, protein_name: str = "protein") -> bool:
        sequence = sequence.upper().replace(" ", "").replace("\n", "").replace("\t", "")
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = ''.join(c for c in sequence if c in valid_aa)
        if len(sequence) == len(sequence) == 0:
            logger.error("Empty sequence after validation")
            return False
        self.sequence = sequence
        self.protein_name = protein_name
        logger.info("Loaded protein: %s, length: %d", protein_name, len(sequence))
        return True

    def extract_sequence_features(self) -> Dict[str, Any]:
        features = {'length': 0, 'molecular_weight': 0.0, 'isoelectric_point': 7.0, 'amino_acid_composition': {}}
        if not self.sequence:
            return features
        try:
            aa_weights = {'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
                         'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.18,
                         'L': 131.18, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
                         'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15}
            features['length'] = len(self.sequence)
            mw = sum(aa_weights.get(aa, 0) for aa in self.sequence)
            features['molecular_weight'] = round(mw - 18.015 * (len(self.sequence) - 1), 2)
            composition = {}
            for aa in self.sequence:
                composition[aa] = composition.get(aa, 0) + 1
            features['amino_acid_composition'] = {k: round(v / len(self.sequence), 4) for k, v in composition.items()}
            pka_values = {'C': 8.18, 'D': 3.90, 'E': 4.07, 'H': 6.04, 'K': 10.79, 'R': 12.48, 'Y': 10.13}
            n_pos = sum(1 for aa in self.sequence if aa in 'KRH')
            n_neg = sum(1 for aa in self.sequence if aa in 'DE')
            pI = 7.0 + 0.3 * (n_pos - n_neg) / max(len(self.sequence), 1)
            features['isoelectric_point'] = round(min(max(pI, 0), 14), 2)
            self.features['sequence'] = features
            logger.info("Sequence features extracted: length=%d, MW=%.1f", features['length'], features['molecular_weight'])
        except Exception as e:
            logger.error("Sequence feature extraction failed: %s", e)
        return features

    def calculate_physicochemical_properties(self) -> Dict[str, float]:
        props = {}
        if not self.sequence:
            return props
        try:
            hydropathy = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
                         'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
                         'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5}
            props['hydropathy'] = round(sum(hydropathy.get(aa, 0) for aa in self.sequence) / len(self.sequence), 3)
            charge_aa = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
            props['net_charge_pH7'] = sum(charge_aa.get(aa, 0) for aa in self.sequence)
            aliphatic = sum(1 for aa in self.sequence if aa in 'AILV') / len(self.sequence)
            props['aliphatic_index'] = round(aliphatic * 100, 2)
            self.features['physicochemical'] = props
            logger.info("Physicochemical properties computed")
        except Exception as e:
            logger.error("Physicochemical computation failed: %s", e)
        return props

    def predict_secondary_structure(self) -> Dict[str, Any]:
        ss = {'helix_fraction': 0.0, 'sheet_fraction': 0.0, 'coil_fraction': 0.0}
        if not self.sequence:
            return ss
        try:
            helix_formers = set('AELM')
            sheet_formers = set('VIF')
            n_helix = sum(1 for aa in self.sequence if aa in helix_formers)
            n_sheet = sum(1 for aa in self.sequence if aa in sheet_formers)
            n_coil = len(self.sequence) - n_helix - n_sheet
            ss['helix_fraction'] = round(n_helix / len(self.sequence), 4)
            ss['sheet_fraction'] = round(n_sheet / len(self.sequence), 4)
            ss['coil_fraction'] = round(n_coil / len(self.sequence), 4)
            self.features['secondary_structure'] = ss
        except Exception as e:
            logger.error("Secondary structure prediction failed: %s", e)
        return ss

    def extract_esm_embeddings(self) -> Optional[np.ndarray]:
        if self.esm_model is None or self.sequence is None:
            return None
        try:
            inputs = self.esm_tokenizer(self.sequence, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            self.features['esm_embedding'] = embeddings
            logger.info("ESM embeddings extracted: %s", str(embeddings.shape))
            return embeddings
        except Exception as e:
            logger.error("ESM embedding extraction failed: %s", e)
            return None

    def predict_3d_structure_esmfold(self, sequence: str) -> Optional[str]:
        logger.warning("ESMFold requires Colab/GPU environment. Returning placeholder.")
        return None

    def extract_binding_site_features(self) -> Dict[str, Any]:
        return {'binding_sites': [], 'binding_score': 0.0}

    def extract_all_features(self, sequence: str, protein_name: str = "protein") -> Dict[str, Any]:
        result = {'name': protein_name, 'success': False}
        if not self.load_sequence(sequence, protein_name):
            return result
        self.load_esm_model()
        self.extract_sequence_features()
        self.calculate_physicochemical_properties()
        self.predict_secondary_structure()
        esm_emb = self.extract_esm_embeddings()
        result['success'] = True
        result['features'] = self.features
        result['embedding_shape'] = str(esm_emb.shape) if esm_emb is not None else None
        logger.info("All protein features extracted for %s", protein_name)
        return result

    def save_features(self, filename: str = "protein_features.json") -> str:
        filepath = os.path.join(self.output_dir, "embeddings", filename)
        data = {'protein': self.protein_name, 'features': self.features}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        logger.info("Saved protein features: %s", filepath)
        return filepath


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)