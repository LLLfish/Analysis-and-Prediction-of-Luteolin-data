import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
logger = setup_logger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not installed. Data handling will be limited.")

SKLEARN_AVAILABLE = False
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Feature scaling will be unavailable.")


class FeatureExtractionPipeline:
    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.join(self.output_dir, "ligand"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "protein"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "fusion"), exist_ok=True)
        self.ligand_extractor = None
        self.protein_extractor = None

    def set_ligand_extractor(self, extractor: Any):
        self.ligand_extractor = extractor

    def set_protein_extractor(self, extractor: Any):
        self.protein_extractor = extractor

    def run_pipeline(self, smiles: str, protein_sequence: str) -> Dict[str, Any]:
        results = {'success': False}
        try:
            if self.ligand_extractor:
                ligand_features = self.ligand_extractor.extract_all_features(smiles)
                results['ligand'] = ligand_features
                logger.info("Ligand features extracted")
            if self.protein_extractor:
                protein_features = self.protein_extractor.extract_all_features(protein_sequence)
                results['protein'] = protein_features
                logger.info("Protein features extracted")
            results['success'] = True
        except Exception as e:
            logger.error("Pipeline execution failed: %s", e)
        return results

    def scale_features(self, features: np.ndarray, method: str = "standard") -> Optional[np.ndarray]:
        if not SKLEARN_AVAILABLE:
            return features
        try:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                return features
            scaled = scaler.fit_transform(features)
            logger.info("Features scaled using %s", method)
            return scaled
        except Exception as e:
            logger.error("Feature scaling failed: %s", e)
            return None

    def reduce_dimensions(self, features: np.ndarray, n_components: int = 50) -> Optional[np.ndarray]:
        if not SKLEARN_AVAILABLE:
            return features
        try:
            pca = PCA(n_components=min(n_components, features.shape[1], features.shape[0]))
            reduced = pca.fit_transform(features)
            logger.info("Dimensions reduced: %s -> %s", str(features.shape), str(reduced.shape))
            return reduced
        except Exception as e:
            logger.error("Dimension reduction failed: %s", e)
            return None

    def save_results(self, results: Dict[str, Any], filename: str = "pipeline_results.json"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved pipeline results: %s", filepath)