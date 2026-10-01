import os
import json
import logging
import hashlib
import ntpath
import re
import threading
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
from data_contracts import FeatureResult, ProteinRecord, StageStatus, validate_protein_record
logger = setup_logger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_POOLING_POLICY = "residue-mask-mean-v1"


def _paths_intersect(first: str, second: str) -> bool:
    try:
        common = os.path.commonpath((first, second))
    except ValueError:
        return False
    normalized = os.path.normcase(common)
    return normalized in {os.path.normcase(first), os.path.normcase(second)}


def _assert_feature_output_path_is_unprotected(requested: str, resolved: str) -> None:
    protected_names = (
        "data", "framework_output", "offline_data.json", ".model_cache",
        "model_cache", "features", "history", "output_dashboard.html",
    )
    candidates = (os.path.abspath(requested), os.path.realpath(resolved))
    for name in protected_names:
        protected_path = os.path.join(_BASE_DIR, name)
        lexical_protected = os.path.abspath(protected_path)
        resolved_protected = os.path.realpath(lexical_protected)
        if any(
            _paths_intersect(candidate, protected)
            for candidate in candidates
            for protected in (lexical_protected, resolved_protected)
        ):
            raise ValueError("feature output path intersects protected historical data or model paths")


def _prepare_safe_output_directory(
    output_dir: Optional[str], subdirectory: Optional[str] = None,
) -> tuple[str, str]:
    if not output_dir:
        raise ValueError("an explicit safe output_dir is required")
    requested_root = os.path.abspath(os.path.expanduser(os.fspath(output_dir)))
    resolved_root = os.path.realpath(requested_root)
    _assert_feature_output_path_is_unprotected(requested_root, resolved_root)
    if os.path.normcase(requested_root) != os.path.normcase(resolved_root):
        raise ValueError("feature output root must not be a symlink or junction")
    if os.path.lexists(requested_root) and not os.path.isdir(requested_root):
        raise FileExistsError(f"feature output root is not a directory: {requested_root}")
    if not os.path.lexists(requested_root):
        os.makedirs(requested_root, exist_ok=False)
    resolved_root = os.path.realpath(requested_root)
    _assert_feature_output_path_is_unprotected(requested_root, resolved_root)
    if os.path.normcase(requested_root) != os.path.normcase(resolved_root):
        raise ValueError("feature output root changed resolution during creation")

    target_directory = requested_root if subdirectory is None else os.path.join(requested_root, subdirectory)
    resolved_target = os.path.realpath(target_directory)
    _assert_feature_output_path_is_unprotected(target_directory, resolved_target)
    try:
        if os.path.normcase(os.path.commonpath((resolved_root, resolved_target))) != os.path.normcase(resolved_root):
            raise ValueError("feature output directory resolves outside the configured root")
    except ValueError as exc:
        if "feature output directory resolves" in str(exc):
            raise
        raise ValueError("feature output directory resolves outside the configured root") from exc
    if os.path.normcase(target_directory) != os.path.normcase(resolved_target):
        raise ValueError("feature output directory must not be a symlink or junction")
    if os.path.lexists(target_directory) and not os.path.isdir(target_directory):
        raise FileExistsError(f"feature output path is not a directory: {target_directory}")
    if not os.path.lexists(target_directory):
        os.makedirs(target_directory, exist_ok=False)
    resolved_target = os.path.realpath(target_directory)
    _assert_feature_output_path_is_unprotected(target_directory, resolved_target)
    if os.path.normcase(target_directory) != os.path.normcase(resolved_target):
        raise ValueError("feature output directory changed resolution during creation")
    return resolved_root, resolved_target

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


def build_feature_cache_key(
    protein: ProteinRecord,
    *,
    model_id: str,
    model_revision: str,
    model_artifact_sha256: str,
    pooling_policy: str,
    entity_version: str,
    entity_bundle_hash: str,
) -> str:
    """Build a deterministic cache key for one identity- and bundle-bound embedding."""
    payload = {
        "identity": [protein.taxonomy_id, protein.accession],
        "sequence_sha256": protein.sequence_sha256,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_artifact_sha256": model_artifact_sha256,
        "pooling_policy": pooling_policy,
        "entity_version": entity_version,
        "entity_bundle_hash": entity_bundle_hash,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_sequence_identity(
    protein: ProteinRecord,
    bundle: Dict[str, Any],
    *,
    test_only: bool = False,
) -> StageStatus:
    """Require the exact shared protein record from a source-verified entity bundle."""
    if type(protein) is not ProteinRecord:
        return StageStatus("unverified_source", "shared_protein_record_required")
    try:
        validate_protein_record(protein)
    except (TypeError, ValueError) as exc:
        return StageStatus("invalid_sequence", f"invalid_protein_record:{exc}")
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 2:
        return StageStatus("unverified_source", "verified_entity_bundle_required")
    content_hash = bundle.get("content_sha256")
    if (
        not isinstance(content_hash, str)
        or len(content_hash) != 64
        or any(char not in "0123456789abcdef" for char in content_hash)
        or bundle.get("entity_version") != f"entities-v2-{content_hash[:16]}"
    ):
        return StageStatus("unverified_source", "entity_bundle_version_or_hash_invalid")
    proteins = bundle.get("proteins")
    if not isinstance(proteins, list) or asdict(protein) not in proteins:
        return StageStatus("unverified_source", "protein_not_present_in_verified_entity_bundle")
    if protein.sequence_status == "missing":
        return StageStatus("unavailable", protein.sequence_status_reason or "sequence_missing")
    if protein.sequence_status == "error":
        return StageStatus("error", protein.sequence_status_reason or "sequence_source_error")
    if protein.sequence_status == "invalid":
        return StageStatus("invalid_sequence", protein.sequence_status_reason or "sequence_invalid")
    if protein.sequence_status != "verified" or protein.identity_status != "verified":
        return StageStatus("unverified_source", protein.sequence_status_reason or "protein_identity_unverified")
    if protein.source_kind == "synthetic_test" and not test_only:
        return StageStatus("unverified_source", "synthetic_test_source_not_allowed_for_runtime")
    if not protein.accession or type(protein.taxonomy_id) is not int:
        return StageStatus("unverified_source", "protein_accession_or_taxonomy_missing")
    return StageStatus("ready")


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

    Model loading is offline and local-only. A revision-verifiable snapshot must
    already exist in the configured cache directory; no download or fallback is
    attempted by the production feature path.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "auto",
        *,
        model_revision: Optional[str] = None,
        max_tokens: int = 1024,
        model=None,
        tokenizer=None,
        cache_dir: Optional[str] = None,
        test_only: bool = False,
    ):
        self.output_dir = os.path.abspath(os.fspath(output_dir)) if output_dir else None
        self.esm_model_name = esm_model_name
        self._model_cache_dir = os.path.abspath(cache_dir or os.path.join(_BASE_DIR, ".model_cache"))
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self.pooling_policy = _POOLING_POLICY
        self.test_only = bool(test_only)
        self.device = self._get_device(device) if TORCH_AVAILABLE else "cpu"
        self.sequence = None
        self.protein_name = None
        self.features = {}
        self.last_result: Optional[FeatureResult] = None
        self.model_id = esm_model_name
        self.esm_model = model if self.test_only else None
        self.esm_tokenizer = tokenizer if self.test_only else None
        self._runtime_model_provenance = None
        self._test_provenance = (
            "explicit offline test-only model/tokenizer injection; not pretrained evidence"
            if self.test_only and model is not None and tokenizer is not None else None
        )
        self.model_revision = (
            model_revision or self._model_revision(model) or "test-only-unpinned"
            if self.test_only else None
        )
        self.model_artifact_sha256 = (
            hashlib.sha256(
                json.dumps(
                    {"test_only": True, "model_id": self.model_id, "revision": self.model_revision},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if self.test_only and model is not None and tokenizer is not None else None
        )
        self._cache: Dict[str, FeatureResult] = {}
        self._feature_lock = threading.RLock()
        self._fallback_models = []
        self._modelscope_models = {
            "facebook/esm2_t6_8M_UR50D": "AI-ModelScope/esm2_t6_8M_UR50D",
        }

    @staticmethod
    def _model_revision(model) -> Optional[str]:
        config = getattr(model, "config", None)
        revision = getattr(config, "_commit_hash", None)
        return revision if isinstance(revision, str) and revision else None

    @staticmethod
    def _local_model_provenance(
        model_dir: Path,
        *,
        model_name: str,
    ) -> Optional[Dict[str, str]]:
        """Bind a local ESM snapshot revision to config, tokenizer, and weight bytes."""
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            return None
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        revision = config.get("_commit_hash") if isinstance(config, dict) else None
        model_type = config.get("model_type") if isinstance(config, dict) else None
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or not isinstance(model_type, str)
            or model_type.lower() != "esm"
            or not isinstance(model_name, str)
            or not model_name.startswith("facebook/esm2_")
        ):
            return None

        all_files = [path for path in model_dir.rglob("*") if path.is_file()]
        tokenizer_files = sorted(
            path for path in all_files
            if path.name in {
                "tokenizer.json", "vocab.txt", "tokenizer.model",
                "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
            }
        )
        tokenizer_core = {path.name for path in tokenizer_files} & {
            "tokenizer.json", "vocab.txt", "tokenizer.model",
        }
        weight_files = sorted(
            path for path in all_files
            if path.name in {"pytorch_model.bin", "model.safetensors"}
            or re.fullmatch(r"(?:pytorch_model-\d+-of-\d+\.bin|model-\d+-of-\d+\.safetensors)", path.name)
        )
        weight_indexes = sorted(
            path for path in all_files
            if path.name in {"pytorch_model.bin.index.json", "model.safetensors.index.json"}
        )
        if not tokenizer_core or not weight_files:
            return None

        def file_hash(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        config_sha256 = file_hash(config_path)
        tokenizer_manifest = {
            path.relative_to(model_dir).as_posix(): file_hash(path)
            for path in tokenizer_files
        }
        weight_manifest = {
            path.relative_to(model_dir).as_posix(): file_hash(path)
            for path in [*weight_files, *weight_indexes]
        }
        artifact_manifest = {
            "model_id": model_name,
            "revision": revision,
            "config_sha256": config_sha256,
            "tokenizer": tokenizer_manifest,
            "weights": weight_manifest,
        }
        canonical_json = lambda value: json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return {
            "model_id": model_name,
            "revision": revision,
            "config_sha256": config_sha256,
            "tokenizer_sha256": hashlib.sha256(canonical_json(tokenizer_manifest)).hexdigest(),
            "weights_sha256": hashlib.sha256(canonical_json(weight_manifest)).hexdigest(),
            "model_artifact_sha256": hashlib.sha256(canonical_json(artifact_manifest)).hexdigest(),
        }

    @classmethod
    def _local_model_revision(cls, model_dir: Path) -> Optional[str]:
        """Compatibility helper returning only a byte-bound local ESM revision."""
        provenance = cls._local_model_provenance(
            model_dir, model_name="facebook/esm2_t6_8M_UR50D",
        )
        return provenance["revision"] if provenance else None

    @staticmethod
    def _assert_complete_esm_weights(loading_info: Any) -> None:
        if not isinstance(loading_info, dict):
            raise ValueError("model loader did not return checkpoint loading diagnostics")
        missing = list(loading_info.get("missing_keys", []))
        mismatched = list(loading_info.get("mismatched_keys", []))
        errors = list(loading_info.get("error_msgs", []))
        unexpected = [
            str(key) for key in loading_info.get("unexpected_keys", [])
            if not str(key).startswith(("lm_head.", "contact_head."))
        ]
        if missing or mismatched or errors or unexpected:
            raise ValueError(
                "local ESM checkpoint is incomplete or incompatible "
                f"(missing={len(missing)}, mismatched={len(mismatched)}, "
                f"unexpected={len(unexpected)}, errors={len(errors)})"
            )

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
        """Load only a complete local model snapshot; never download or fall back online."""
        if self.test_only:
            return self.esm_model is not None and self.esm_tokenizer is not None
        if not _TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not installed; ESM is unavailable")
            return False
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed; ESM is unavailable")
            return False
        local_dir = Path(self._model_cache_dir) / self.esm_model_name.replace("/", "--")
        provenance = self._local_model_provenance(
            local_dir, model_name=self.esm_model_name,
        )
        if provenance is None:
            logger.info("No revision-verifiable local ESM snapshot at %s", local_dir)
            return False
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
            loaded_model = AutoModel.from_pretrained(
                str(local_dir), local_files_only=True, output_loading_info=True,
            )
            if not isinstance(loaded_model, tuple) or len(loaded_model) != 2:
                raise ValueError("local model loader did not return loading diagnostics")
            model, loading_info = loaded_model
            self._assert_complete_esm_weights(loading_info)
            loaded_revision = self._model_revision(model)
            if loaded_revision is not None and loaded_revision != provenance["revision"]:
                raise ValueError("loaded model revision does not match local snapshot config")
            model.to(self.device)
            model.eval()
            after_load = self._local_model_provenance(
                local_dir, model_name=self.esm_model_name,
            )
            if after_load != provenance:
                raise ValueError("local model snapshot changed while it was being loaded")
            self.esm_tokenizer = tokenizer
            self.esm_model = model
            self.model_revision = provenance["revision"]
            self.model_artifact_sha256 = provenance["model_artifact_sha256"]
            self._runtime_model_provenance = provenance
            logger.info(
                "Loaded local ESM snapshot %s at revision %s with artifact %s",
                self.esm_model_name, self.model_revision, self.model_artifact_sha256,
            )
            return True
        except Exception as exc:
            self.esm_model = None
            self.esm_tokenizer = None
            self.model_revision = None
            self.model_artifact_sha256 = None
            self._runtime_model_provenance = None
            logger.warning("Local ESM snapshot is unavailable: %s", exc)
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
        if not isinstance(sequence, str):
            logger.error("Protein sequence must be text")
            return False
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            logger.error("Empty sequence after validation")
            return False
        if any(residue not in _STANDARD_AMINO_ACIDS for residue in sequence):
            logger.error("Protein sequence contains nonstandard residues; preserving and rejecting it")
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

    def _token_limit(self) -> int:
        limits = [self.max_tokens]
        model_limit = getattr(getattr(self.esm_model, "config", None), "max_position_embeddings", None)
        if type(model_limit) is int and 0 < model_limit < 1_000_000:
            limits.append(model_limit)
        tokenizer_limit = getattr(self.esm_tokenizer, "model_max_length", None)
        if type(tokenizer_limit) is int and 0 < tokenizer_limit < 1_000_000:
            limits.append(tokenizer_limit)
        return min(limits)

    def _extract_residue_embedding(self, sequence: str) -> tuple[Optional[np.ndarray], str, Optional[str]]:
        if not TORCH_AVAILABLE:
            return None, "unavailable", "pytorch_unavailable"
        if self.esm_model is None or self.esm_tokenizer is None:
            return None, "unavailable", "esm_model_or_tokenizer_unavailable"
        try:
            encoded = self.esm_tokenizer(
                sequence,
                return_tensors="pt",
                truncation=False,
                padding=False,
                return_special_tokens_mask=True,
            )
            if "input_ids" not in encoded:
                return None, "error", "tokenizer_input_ids_missing"
            input_ids = encoded["input_ids"]
            if not torch.is_tensor(input_ids):
                input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                return None, "error", "tokenizer_input_shape_invalid"
            token_count = int(input_ids.shape[1])
            token_limit = self._token_limit()
            if token_count > token_limit:
                return None, "unsupported_length", f"token_count_exceeds_budget:{token_count}>{token_limit}"

            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                if not torch.is_tensor(attention_mask):
                    attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
                if attention_mask.ndim == 1:
                    attention_mask = attention_mask.unsqueeze(0)
            special_tokens_mask = encoded.get("special_tokens_mask")
            if special_tokens_mask is None:
                get_mask = getattr(self.esm_tokenizer, "get_special_tokens_mask", None)
                if not callable(get_mask):
                    return None, "error", "tokenizer_special_token_mask_unavailable"
                special_tokens_mask = torch.tensor(
                    [get_mask(input_ids[0].detach().cpu().tolist(), already_has_special_tokens=True)],
                    dtype=torch.long,
                )
            else:
                if not torch.is_tensor(special_tokens_mask):
                    special_tokens_mask = torch.as_tensor(special_tokens_mask, dtype=torch.long)
                if special_tokens_mask.ndim == 1:
                    special_tokens_mask = special_tokens_mask.unsqueeze(0)
            if attention_mask.shape != input_ids.shape or special_tokens_mask.shape != input_ids.shape:
                return None, "error", "tokenizer_mask_shape_mismatch"

            valid_residues = attention_mask.to(dtype=torch.bool) & ~special_tokens_mask.to(dtype=torch.bool)
            if int(valid_residues.sum().item()) != len(sequence):
                return None, "invalid_sequence", "tokenizer_residue_count_mismatch"
            model_inputs = {
                key: encoded[key].to(self.device) if torch.is_tensor(encoded.get(key)) else input_ids.to(self.device)
                for key in ("input_ids", "attention_mask")
                if key == "input_ids" or encoded.get(key) is not None
            }
            model_inputs["input_ids"] = input_ids.to(self.device)
            model_inputs["attention_mask"] = attention_mask.to(self.device)
            with torch.no_grad():
                outputs = self.esm_model(**model_inputs)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None or hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
                return None, "error", "model_hidden_state_shape_invalid"
            mask = valid_residues.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vector = pooled[0].detach().cpu().numpy()
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                return None, "error", "pooled_embedding_invalid"
            return vector, "ready", None
        except Exception as exc:
            logger.error("ESM residue pooling failed: %s", exc)
            return None, "error", f"esm_inference_failed:{type(exc).__name__}"

    @staticmethod
    def _clear_sample_state(extractor: "ProteinFeatureExtractor") -> None:
        extractor.sequence = None
        extractor.protein_name = None
        extractor.features = {}
        extractor.last_result = None

    def _result(
        self,
        protein: ProteinRecord,
        bundle: Dict[str, Any],
        status: str,
        reason: Optional[str],
        *,
        model_revision: Optional[str] = None,
        pooling_policy: Optional[str] = None,
        cache_key: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> FeatureResult:
        result = FeatureResult(
            status=status,
            reason=reason,
            features=features or {},
            accession=protein.accession,
            taxonomy_id=protein.taxonomy_id,
            sequence_length=protein.sequence_length,
            sequence_sha256=protein.sequence_sha256,
            entity_version=bundle.get("entity_version"),
            entity_bundle_hash=bundle.get("content_sha256"),
            source_kind=protein.source_kind,
            model_id=self.model_id if model_revision else None,
            model_revision=model_revision,
            model_artifact_sha256=(
                self.model_artifact_sha256 if model_revision else None
            ),
            pooling_policy=pooling_policy,
            cache_key=cache_key,
            test_provenance=self._test_provenance if status == "test_only" else None,
        )
        self.last_result = result
        return result

    def _extract_validated_record(
        self,
        protein: ProteinRecord,
        bundle: Dict[str, Any],
        *,
        test_only: bool,
    ) -> FeatureResult:
        self._clear_sample_state(self)
        if test_only != self.test_only:
            return self._result(protein, bundle, "unverified_source", "test_only_mode_mismatch")
        identity_status = validate_sequence_identity(protein, bundle, test_only=test_only)
        if identity_status.status != "ready":
            return self._result(protein, bundle, identity_status.status, identity_status.reason)
        sequence = protein.full_sequence or ""
        if len(sequence) > self.max_tokens:
            return self._result(
                protein, bundle, "unsupported_length",
                f"residue_count_exceeds_token_budget:{len(sequence)}>{self.max_tokens}",
            )
        if self.esm_model is None or self.esm_tokenizer is None:
            if not self.load_esm_model():
                return self._result(protein, bundle, "unavailable", "revision_verified_local_esm_unavailable")
        if not self.model_revision or not self.model_artifact_sha256:
            return self._result(protein, bundle, "unavailable", "esm_model_revision_unavailable")
        if not test_only and not self._runtime_model_provenance:
            return self._result(protein, bundle, "unavailable", "verified_local_model_provenance_unavailable")

        self.sequence = sequence
        self.protein_name = protein.accession
        pooling_policy = f"{self.pooling_policy}:max_tokens={self.max_tokens}"
        cache_key = build_feature_cache_key(
            protein,
            model_id=self.model_id,
            model_revision=self.model_revision,
            model_artifact_sha256=self.model_artifact_sha256,
            pooling_policy=pooling_policy,
            entity_version=bundle["entity_version"],
            entity_bundle_hash=bundle["content_sha256"],
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.features = deepcopy(cached.features)
            result = deepcopy(cached)
            self.last_result = result
            return result

        embedding, status, reason = self._extract_residue_embedding(sequence)
        if status != "ready" or embedding is None:
            return self._result(
                protein, bundle, status, reason,
                model_revision=self.model_revision,
                pooling_policy=pooling_policy,
                cache_key=cache_key,
            )
        features = {"esm_embedding": embedding.astype(float).tolist()}
        self.features = deepcopy(features)
        result_status = "test_only" if test_only else "ready"
        result = self._result(
            protein, bundle, result_status,
            "offline_test_only_mock_result" if test_only else None,
            model_revision=self.model_revision,
            pooling_policy=pooling_policy,
            cache_key=cache_key,
            features=features,
        )
        self._cache[cache_key] = deepcopy(result)
        return result

    def extract_from_bundle(
        self,
        bundle_path: str,
        *,
        test_only: bool = False,
    ) -> List[FeatureResult]:
        """Verify one entities.json and extract each source-bound protein independently."""
        from evidence_builder import load_entity_bundle

        with self._feature_lock:
            self._clear_sample_state(self)
            try:
                bundle = load_entity_bundle(bundle_path, verify_sources=True)
            except Exception:
                self.last_result = FeatureResult(
                    status="unavailable",
                    reason="entity_bundle_verification_failed",
                )
                raise
            results = []
            for protein_data in bundle.get("proteins", []):
                try:
                    protein = ProteinRecord(**protein_data)
                except (TypeError, ValueError) as exc:
                    result = FeatureResult(
                        status="invalid_sequence",
                        reason=f"invalid_protein_record:{exc}",
                        entity_version=bundle.get("entity_version"),
                        entity_bundle_hash=bundle.get("content_sha256"),
                        source_kind=protein_data.get("source_kind"),
                    )
                    self.last_result = result
                    results.append(result)
                    continue
                results.append(self._extract_validated_record(
                    protein, bundle, test_only=test_only,
                ))
            return results

    def extract_for_protein(
        self,
        protein: ProteinRecord,
        *,
        bundle_path: str,
        test_only: bool = False,
    ) -> FeatureResult:
        """Extract only when the exact record is present in a verified source bundle."""
        from evidence_builder import load_entity_bundle

        with self._feature_lock:
            self._clear_sample_state(self)
            try:
                bundle = load_entity_bundle(bundle_path, verify_sources=True)
            except Exception:
                self.last_result = FeatureResult(
                    status="unavailable",
                    reason="entity_bundle_verification_failed",
                )
                raise
            return self._extract_validated_record(
                protein, bundle, test_only=test_only,
            )

    def predict_3d_structure_esmfold(self, sequence: str) -> Optional[str]:
        logger.warning("ESMFold requires Colab/GPU environment. Returning placeholder.")
        return None

    def extract_binding_site_features(self) -> Dict[str, Any]:
        return {'binding_sites': [], 'binding_score': 0.0}

    def extract_all_features(self, sequence: str, protein_name: str = "protein") -> Dict[str, Any]:
        return {
            "name": protein_name,
            "status": "unverified_source",
            "reason": "entity_bundle_required_for_protein_features",
            "success": False,
            "features": {},
        }

    def save_features(
        self,
        filename: str = "protein_features.json",
        *,
        result: Optional[FeatureResult] = None,
    ) -> str:
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or os.path.basename(filename) != filename
            or ntpath.basename(filename) != filename
            or os.path.isabs(filename)
            or ntpath.isabs(filename)
            or "/" in filename
            or "\\" in filename
            or any(character in filename for character in '<>:"|?*')
            or filename.endswith((" ", "."))
        ):
            raise ValueError("filename must be a basename without path separators")
        _, embeddings_dir = _prepare_safe_output_directory(self.output_dir, "embeddings")
        filepath = os.path.join(embeddings_dir, filename)
        resolved_filepath = os.path.realpath(filepath)
        if os.path.normcase(os.path.commonpath((embeddings_dir, resolved_filepath))) != os.path.normcase(embeddings_dir):
            raise ValueError("feature path resolves outside the embeddings directory")
        _assert_feature_output_path_is_unprotected(filepath, resolved_filepath)
        if os.path.lexists(filepath):
            raise FileExistsError(f"feature snapshot already exists: {filepath}")
        if result is None:
            raise ValueError("save_features requires an explicit FeatureResult")
        if not isinstance(result, FeatureResult):
            raise TypeError("result must be a validated FeatureResult")
        data = result.to_dict()
        with open(filepath, 'x', encoding='utf-8') as f:
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
