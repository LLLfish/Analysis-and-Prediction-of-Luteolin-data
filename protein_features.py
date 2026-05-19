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

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
logger.info("HuggingFace endpoint: %s", os.environ.get("HF_ENDPOINT"))

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Protein feature extraction will be limited.")

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModel, AutoTokenizer
    import transformers
    transformers.logging.set_verbosity_error()
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not installed. ESM-2 features will be unavailable.")

_MODELSCOPE_AVAILABLE = False
try:
    from modelscope import snapshot_download
    _MODELSCOPE_AVAILABLE = True
except ImportError:
    pass


class ProteinFeatureExtractor:
    """Protein feature extractor with ESM-2 model support.

    ESM-2 Model Selection Guide:
        | Model                          | Params | Size   | Embed Dim | Use Case                  |
        |--------------------------------|--------|--------|-----------|---------------------------|
        | facebook/esm2_t6_8M_UR50D     | 8M     | ~30MB  | 320       | Default, fast screening    |
        | facebook/esm2_t12_35M_UR50D   | 35M    | ~130MB | 480       | Balanced accuracy/speed    |
        | facebook/esm2_t33_650M_UR50D  | 650M   | ~2.6GB | 1280      | High accuracy, GPU needed  |

    Switch model example:
        # Use larger model for better accuracy:
        extractor = ProteinFeatureExtractor(esm_model_name="facebook/esm2_t33_650M_UR50D")

        # Or add as fallback:
        extractor.add_fallback_model(
            "facebook/esm2_t33_650M_UR50D",
            "AI-ModelScope/esm2_t33_650M_UR50D"
        )

    Download sources (in order):
        1. ModelScope (国内优先): pip install modelscope
        2. HuggingFace Mirror: https://hf-mirror.com
        3. HuggingFace (original): set HF_ENDPOINT="" to use
    """

    def __init__(self, output_dir: str = "", esm_model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = "auto"):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features", "protein")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.join(self.output_dir, "structures"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "embeddings"), exist_ok=True)
        self.esm_model_name = esm_model_name
        self._model_cache_dir = os.path.join(_BASE_DIR, ".model_cache")
        os.makedirs(self._model_cache_dir, exist_ok=True)
        self.device = self._get_device(device) if TORCH_AVAILABLE else "cpu"
        self.sequence = None
        self.protein_name = None
        self.features = {}
        self.esm_model = None
        self.esm_tokenizer = None
        self._fallback_models = []
        self._modelscope_models = {
            "facebook/esm2_t6_8M_UR50D": "AI-ModelScope/esm2_t6_8M_UR50D",
        }
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for protein feature extraction")

    def add_fallback_model(self, model_name: str, modelscope_name: str = ""):
        if model_name not in self._fallback_models and model_name != self.esm_model_name:
            self._fallback_models.append(model_name)
            if modelscope_name:
                self._modelscope_models[model_name] = modelscope_name
            logger.info("Added fallback model: %s (modelscope: %s)", model_name, modelscope_name or "N/A")

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def load_esm_model(self, timeout: int = 120) -> bool:
        if not _TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not installed")
            return False
        models_to_try = [self.esm_model_name] + self._fallback_models
        for model_name in models_to_try:
            try:
                logger.info("Loading ESM-2 model: %s (cache: %s)", model_name, self._model_cache_dir)
                try:
                    local_dir = os.path.join(self._model_cache_dir, model_name.replace("/", "--"))
                    self.esm_tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
                    self.esm_model = AutoModel.from_pretrained(local_dir, local_files_only=True)
                    logger.info("ESM-2 model loaded from local cache: %s", model_name)
                except Exception:
                    loaded = self._download_from_modelscope(model_name, timeout)
                    if not loaded:
                        loaded = self._download_from_hf(model_name, timeout)
                    if not loaded:
                        raise RuntimeError(f"Failed to download {model_name} from all sources")

                self.esm_model.to(self.device)
                self.esm_model.eval()
                logger.info("ESM-2 model loaded successfully: %s", model_name)
                return True
            except Exception as e:
                if model_name == self.esm_model_name:
                    logger.warning("Primary ESM-2 model failed (%s), trying fallback...", e)
                else:
                    logger.warning("Fallback model %s also failed: %s", model_name, e)
        logger.error("All ESM-2 models failed to load, using rule-based features only")
        return False

    def _download_from_hf(self, model_name: str, timeout: int = 120) -> bool:
        hf_endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")
        logger.info("Downloading from HuggingFace (%s): %s", hf_endpoint, model_name)
        import threading
        result = {"success": False, "error": None, "local_dir": None}

        def _download():
            try:
                local_dir = os.path.join(self._model_cache_dir, model_name.replace("/", "--"))
                if os.path.exists(os.path.join(local_dir, "config.json")):
                    logger.info("Found cached model at: %s", local_dir)
                else:
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        model_name,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False
                    )
                    logger.info("Model downloaded to: %s", local_dir)
                self.esm_tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
                self.esm_model = AutoModel.from_pretrained(local_dir, local_files_only=True)
                result["success"] = True
                result["local_dir"] = local_dir
            except Exception as e:
                result["error"] = e

        dl_thread = threading.Thread(target=_download, daemon=True)
        dl_thread.start()
        dl_thread.join(timeout=timeout)

        if dl_thread.is_alive():
            logger.warning("HuggingFace download timed out (%ds): %s", timeout, model_name)
            return False
        if not result["success"]:
            logger.warning("HuggingFace download failed: %s", result["error"])
            return False
        return True

    def _download_from_modelscope(self, model_name: str, timeout: int = 120) -> bool:
        if not _MODELSCOPE_AVAILABLE:
            logger.info("ModelScope not installed, skipping")
            return False
        ms_model = self._modelscope_models.get(model_name)
        if not ms_model:
            logger.info("No ModelScope mapping for %s", model_name)
            return False
        logger.info("Trying ModelScope: %s -> %s", model_name, ms_model)
        import threading
        result = {"success": False, "error": None}

        def _download():
            try:
                local_dir = os.path.join(self._model_cache_dir, "ms--" + ms_model.replace("/", "--"))
                if os.path.exists(os.path.join(local_dir, "config.json")):
                    logger.info("Found cached ModelScope model at: %s", local_dir)
                else:
                    cache_dir = snapshot_download(ms_model, cache_dir=self._model_cache_dir)
                    logger.info("ModelScope model downloaded to: %s", cache_dir)
                self.esm_tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
                self.esm_model = AutoModel.from_pretrained(local_dir, local_files_only=True)
                result["success"] = True
            except Exception as e:
                result["error"] = e

        dl_thread = threading.Thread(target=_download, daemon=True)
        dl_thread.start()
        dl_thread.join(timeout=timeout)

        if dl_thread.is_alive():
            logger.warning("ModelScope download timed out (%ds): %s", timeout, model_name)
            return False
        if not result["success"]:
            logger.warning("ModelScope download failed: %s", result["error"])
            return False
        return True

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
        if self.esm_model is None:
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