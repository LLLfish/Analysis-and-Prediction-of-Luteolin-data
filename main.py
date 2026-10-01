"""
网络药理学混合人工智能框架 - 主脚本
Integrated Network Pharmacology Hybrid AI Framework
"""

# Suppress transformers optional dependency notices (lightning, haiku, etc. — not needed for ESM-2)
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
logger = setup_logger(__name__)

from data_contracts import CanonicalPpiResult, FeatureResult, ProteinRecord, StageStatus, canonicalize_ppi

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not installed. SMILES parsing will be unavailable.")

_module_imports = {}

try:
    from data_collection import LuteolinDataCollector
    _module_imports['LuteolinDataCollector'] = True
except ImportError as e:
    logger.warning("data_collection import failed: %s", e)
    _module_imports['LuteolinDataCollector'] = False

try:
    from gnn_module import GraphEncoder, NetworkAnalyzer
    _module_imports['GraphEncoder'] = True
    _module_imports['NetworkAnalyzer'] = True
except ImportError as e:
    logger.warning("gnn_module import failed: %s", e)
    _module_imports['GraphEncoder'] = False
    _module_imports['NetworkAnalyzer'] = False

try:
    from cnn_module import MolecularCNN
    _module_imports['MolecularCNN'] = True
except ImportError as e:
    logger.warning("cnn_module import failed: %s", e)
    _module_imports['MolecularCNN'] = False

try:
    from protein_features import ProteinFeatureExtractor
    _module_imports['ProteinFeatureExtractor'] = True
except ImportError as e:
    logger.warning("protein_features import failed: %s", e)
    _module_imports['ProteinFeatureExtractor'] = False

try:
    from ligand_features import LigandFeatureExtractor
    _module_imports['LigandFeatureExtractor'] = True
except ImportError as e:
    logger.warning("ligand_features import failed: %s", e)
    _module_imports['LigandFeatureExtractor'] = False

try:
    from transformer_module import FeatureFusion
    _module_imports['FeatureFusion'] = True
except ImportError as e:
    logger.warning("transformer_module import failed: %s", e)
    _module_imports['FeatureFusion'] = False

try:
    from docking_module import DockingValidator
    _module_imports['DockingValidator'] = True
except ImportError as e:
    logger.warning("docking_module import failed: %s", e)
    _module_imports['DockingValidator'] = False

try:
    from knowledge_graph_module import KnowledgeGraph, KGEmbedder
    _module_imports['KnowledgeGraph'] = True
    _module_imports['KGEmbedder'] = True
except ImportError as e:
    logger.warning("knowledge_graph_module import failed: %s", e)
    _module_imports['KnowledgeGraph'] = False
    _module_imports['KGEmbedder'] = False

try:
    from output_module import OutputGenerator
    _module_imports['OutputGenerator'] = True
except ImportError as e:
    logger.warning("output_module import failed: %s", e)
    _module_imports['OutputGenerator'] = False


class NetworkPharmacologyFramework:

    def __init__(self, output_dir: Optional[str] = None, *, allow_online: bool = False):
        if type(allow_online) is not bool:
            raise TypeError("allow_online must be an explicit boolean")
        if output_dir is None or output_dir == "":
            if allow_online:
                raise ValueError("online collection requires an explicit new output directory")
            self.output_dir = None
        else:
            self.output_dir = _resolve_new_output_dir(output_dir)
            os.makedirs(self.output_dir, exist_ok=False)
            created_output = os.path.realpath(self.output_dir)
            if os.path.normcase(created_output) != os.path.normcase(self.output_dir):
                raise ValueError("output directory changed resolution after creation")
            _assert_output_path_is_unprotected(created_output, created_output)
        self._output_dir_preexisted = False
        self._full_pipeline_started = False
        self.allow_online = allow_online

        if self.output_dir is None:
            self.data_collector = self._safe_component(
                globals().get("LuteolinDataCollector"), "LuteolinDataCollector",
            )
            self.gnn_encoder = None
            self.gnn_analyzer = None
            self.cnn_module = None
            self.protein_extractor = None
            self.ligand_extractor = None
            self.transformer_module = None
            self.docking_module = None
            self.kg_module = None
            self.kg_embedder = None
            self.output_module = None
        else:
            self.data_collector = self._safe_component(
                globals().get("LuteolinDataCollector"), "LuteolinDataCollector",
                output_dir=os.path.join(self.output_dir, "data"),
                allow_online=self.allow_online,
            )
            self.gnn_encoder = self._safe_component(globals().get("GraphEncoder"), "GraphEncoder")
            self.gnn_analyzer = self._safe_component(
                globals().get("NetworkAnalyzer"), "NetworkAnalyzer",
                output_dir=os.path.join(self.output_dir, "gnn"),
            )
            self.cnn_module = self._safe_component(
                globals().get("MolecularCNN"), "MolecularCNN",
                output_dir=os.path.join(self.output_dir, "cnn"),
            )
            self.protein_extractor = self._safe_component(
                globals().get("ProteinFeatureExtractor"), "ProteinFeatureExtractor",
                output_dir=os.path.join(self.output_dir, "protein"),
            )
            self.ligand_extractor = self._safe_component(
                globals().get("LigandFeatureExtractor"), "LigandFeatureExtractor",
                output_dir=os.path.join(self.output_dir, "ligand"),
            )
            self.transformer_module = self._safe_component(
                globals().get("FeatureFusion"), "FeatureFusion",
                output_dir=os.path.join(self.output_dir, "transformer"),
            )
            self.docking_module = self._safe_component(
                globals().get("DockingValidator"), "DockingValidator",
                output_dir=os.path.join(self.output_dir, "docking"),
            )
            self.kg_module = self._safe_component(
                globals().get("KnowledgeGraph"), "KnowledgeGraph",
                output_dir=os.path.join(self.output_dir, "knowledge_graph"),
            )
            self.kg_embedder = self._safe_component(
                globals().get("KGEmbedder"), "KGEmbedder",
                output_dir=os.path.join(self.output_dir, "knowledge_graph", "embeddings"),
            )
            self.output_module = self._safe_component(
                globals().get("OutputGenerator"),
                "OutputGenerator",
                output_dir=os.path.join(self.output_dir, "output"),
                run_root=self.output_dir,
            )

        self.results = {}
    def _safe_component(self, cls, import_key: str, *args, **kwargs):
        if cls is None or not _module_imports.get(import_key):
            return None
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            logger.warning("%s initialization skipped: %s", import_key, e)
            return None

    def run_data_collection(self) -> Dict[str, Any]:
        reason = None
        if self.output_dir is None:
            reason = "explicit_new_output_dir_required"
        elif not getattr(self, "allow_online", False):
            reason = "explicit_online_collection_opt_in_required"
        elif not self.data_collector:
            reason = "data_collector_unavailable"

        if reason is not None:
            dataset = {
                "compound": {},
                "targets": [],
                "activity_evidence": [],
                "proteins": [],
                "interactions": [],
                "ppi_source_kind": "legacy_unverified",
                "ppi_request_status": "not_requested",
                "ppi_trace": {
                    "source_kind": "legacy_unverified",
                    "request_status": "not_requested",
                    "snapshot_path": None,
                    "snapshot_sha256": None,
                },
                "collection_status": "not_requested",
                "status": "partial",
                "reason": reason,
            }
        else:
            try:
                dataset = self.data_collector.run_pipeline()
            except Exception as exc:
                logger.warning("Data collection stopped safely: %s", exc)
                dataset = {
                    "compound": {},
                    "targets": [],
                    "activity_evidence": [],
                    "proteins": [],
                    "interactions": [],
                    "ppi_source_kind": "legacy_unverified",
                    "ppi_request_status": "failed",
                    "ppi_trace": {
                        "source_kind": "legacy_unverified",
                        "request_status": "failed",
                        "snapshot_path": None,
                        "snapshot_sha256": None,
                    },
                    "collection_status": "incomplete",
                    "status": "partial",
                    "reason": f"collection_failed:{type(exc).__name__}",
                }
        self.results["data_collection"] = dataset
        compound_name = dataset.get("compound", {}).get("name", "N/A")
        num_targets = len(dataset.get("targets", []))
        num_proteins = len(dataset.get("proteins", []))
        num_interactions = len(dataset.get("interactions", []))
        logger.info(
            "Data collection: status=%s compound=%s targets=%d proteins=%d interactions=%d",
            dataset.get("status", "partial"), compound_name, num_targets, num_proteins, num_interactions,
        )
        return dataset
    @staticmethod
    def _json_safe_metric_keys(value):
        """Project tuple graph identities to reversible JSON object keys."""
        if isinstance(value, dict):
            projected = {}
            for key, item in value.items():
                if isinstance(key, tuple):
                    if len(key) != 2 or type(key[0]) is not int or not isinstance(key[1], str):
                        raise TypeError("Unsupported tuple key in network metrics")
                    key = json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))
                if key in projected:
                    raise ValueError("Network metric key projection is not collision-free")
                projected[key] = NetworkPharmacologyFramework._json_safe_metric_keys(item)
            return projected
        if isinstance(value, (list, tuple)):
            return [NetworkPharmacologyFramework._json_safe_metric_keys(item) for item in value]
        return value

    def _canonicalize_dataset_ppi(self, dataset: Dict[str, Any]) -> CanonicalPpiResult:
        metadata_names = {
            "ppi_source_kind": "source_kind",
            "ppi_request_status": "request_status",
            "ppi_snapshot_path": "snapshot_path",
            "ppi_snapshot_root": "snapshot_root",
            "ppi_snapshot_sha256": "snapshot_sha256",
        }
        top_level_metadata = {
            contract_name: dataset[dataset_name]
            for dataset_name, contract_name in metadata_names.items()
            if dataset_name in dataset
        }
        metadata_sources = []
        if top_level_metadata:
            metadata_sources.append(top_level_metadata)
        if "ppi_trace" in dataset:
            metadata_sources.append(dataset["ppi_trace"])
        result = canonicalize_ppi(
            dataset.get("interactions", []),
            alias_mappings=dataset.get("ppi_alias_mappings"),
            ppi_trace=metadata_sources or None,
        )
        cached = getattr(self, "_ppi_result", None)
        if (
            cached is not None
            and cached.input_sha256 == result.input_sha256
            and cached.semantic_sha256 == result.semantic_sha256
            and cached.provenance_sha256 == result.provenance_sha256
            and cached.request_status == result.request_status
            and cached.snapshot_path == result.snapshot_path
            and cached.snapshot_root == result.snapshot_root
            and cached.snapshot_sha256 == result.snapshot_sha256
            and cached.snapshot_records_sha256 == result.snapshot_records_sha256
            and cached.issues == result.issues
        ):
            return cached
        self._ppi_result = result
        return result

    @staticmethod
    def _feature_readiness(stage_statuses: Dict[str, StageStatus]) -> Dict[str, Any]:
        stage_data = {name: status.to_dict() for name, status in stage_statuses.items()}
        learned_names = ("protein", "cnn", "transformer")
        learned_statuses = [stage_statuses[name].status for name in learned_names if name in stage_statuses]
        ready_count = sum(status == "ready" for status in learned_statuses)
        if ready_count == len(learned_names) and learned_statuses:
            overall = "ready"
        elif ready_count > 0 or any(status == "partial" for status in learned_statuses):
            overall = "partial"
        else:
            overall = "unavailable"
        return {"overall_status": overall, "stages": stage_data}

    @staticmethod
    def _protein_readiness(results: List[Any]) -> StageStatus:
        if not results:
            return StageStatus("unavailable", "no_protein_records_in_entity_bundle")
        statuses = [getattr(result, "status", "error") for result in results]
        ready_count = sum(status == "ready" for status in statuses)
        details = {
            "protein_count": len(results),
            "statuses": statuses,
            "sequence_sha256": [getattr(result, "sequence_sha256", None) for result in results],
        }
        if ready_count == len(results):
            return StageStatus("ready", details=details)
        if ready_count:
            return StageStatus("partial", "some_protein_features_unavailable", details)
        if len(set(statuses)) == 1 and statuses[0] in {
            "skipped_untrained", "unavailable", "invalid_sequence",
            "unsupported_length", "unverified_source", "error", "empty", "test_only",
        }:
            reason = getattr(results[0], "reason", None) or "protein_features_not_ready"
            return StageStatus(statuses[0], reason, details)
        return StageStatus("unavailable", "no_protein_features_ready", details)

    @staticmethod
    def _bundle_bound_feature_results(results, bundle, extractor) -> List[FeatureResult]:
        """Accept only extractor results that re-match the independently verified bundle."""
        from protein_features import build_feature_cache_key, validate_sequence_identity

        candidates = list(results or [])
        runtime_provenance = getattr(extractor, "_runtime_model_provenance", None)
        bundled_proteins = bundle.get("proteins", []) if isinstance(bundle, dict) else []
        bound_results = []

        for protein_data in bundled_proteins:
            protein = ProteinRecord(**protein_data)
            identity_status = validate_sequence_identity(protein, bundle)
            identity = (protein.accession, protein.taxonomy_id)
            matching = [
                result for result in candidates
                if (
                    getattr(result, "accession", None),
                    getattr(result, "taxonomy_id", None),
                ) == identity
            ]

            def rejected(reason: str) -> FeatureResult:
                return FeatureResult(
                    status="unverified_source",
                    reason=reason,
                    accession=protein.accession,
                    taxonomy_id=protein.taxonomy_id,
                    sequence_length=protein.sequence_length,
                    sequence_sha256=protein.sequence_sha256,
                    entity_version=bundle.get("entity_version"),
                    entity_bundle_hash=bundle.get("content_sha256"),
                    source_kind=protein.source_kind,
                )

            if identity_status.status != "ready":
                bound_results.append(rejected(identity_status.reason or "protein_source_not_verified"))
                continue
            if len(matching) != 1 or not isinstance(matching[0], FeatureResult):
                bound_results.append(rejected("feature_result_missing_or_ambiguous_for_bundle_record"))
                continue
            candidate = matching[0]
            try:
                candidate = FeatureResult(**candidate.to_dict())
            except (TypeError, ValueError) as exc:
                bound_results.append(rejected(f"feature_result_contract_invalid:{type(exc).__name__}"))
                continue

            expected_identity = (
                candidate.accession == protein.accession
                and candidate.taxonomy_id == protein.taxonomy_id
                and candidate.sequence_length == protein.sequence_length
                and candidate.sequence_sha256 == protein.sequence_sha256
                and candidate.entity_version == bundle.get("entity_version")
                and candidate.entity_bundle_hash == bundle.get("content_sha256")
                and candidate.source_kind == protein.source_kind
            )
            if not expected_identity or protein.source_kind == "synthetic_test":
                bound_results.append(rejected("feature_result_identity_or_bundle_mismatch"))
                continue
            if candidate.status == "ready":
                provenance_matches = (
                    isinstance(runtime_provenance, dict)
                    and candidate.model_id == runtime_provenance.get("model_id")
                    and candidate.model_revision == runtime_provenance.get("revision")
                    and candidate.model_artifact_sha256 == runtime_provenance.get("model_artifact_sha256")
                )
                try:
                    expected_cache_key = build_feature_cache_key(
                        protein,
                        model_id=candidate.model_id,
                        model_revision=candidate.model_revision,
                        model_artifact_sha256=candidate.model_artifact_sha256,
                        pooling_policy=candidate.pooling_policy,
                        entity_version=bundle["entity_version"],
                        entity_bundle_hash=bundle["content_sha256"],
                    )
                except (KeyError, TypeError, ValueError):
                    expected_cache_key = None
                if not provenance_matches or candidate.cache_key != expected_cache_key:
                    bound_results.append(rejected("verified_local_model_or_cache_provenance_required"))
                    continue
            bound_results.append(candidate)

        return bound_results

    def run_feature_extraction(
        self,
        dataset: Dict[str, Any],
        entity_bundle_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        dataset = dataset or {}
        compound = dataset.get("compound", {}) or {}
        smiles = compound.get("smiles")

        logger.info("Extracting ligand features...")
        if self.ligand_extractor and smiles:
            try:
                ligand_features = self.ligand_extractor.extract_all_features(
                    smiles, compound.get("name", "compound"),
                )
                ligand_success = bool(ligand_features.get("success"))
                ligand_status = StageStatus(
                    "ready" if ligand_success else "unavailable",
                    None if ligand_success else "ligand_feature_extraction_failed",
                )
            except Exception as exc:
                ligand_features = {"features": {}, "status": "error", "reason": type(exc).__name__}
                ligand_status = StageStatus("error", f"ligand_feature_extraction_failed:{type(exc).__name__}")
        else:
            ligand_features = {
                "features": {},
                "status": "unavailable",
                "reason": "compound_smiles_or_extractor_missing",
            }
            ligand_status = StageStatus("unavailable", "compound_smiles_or_extractor_missing")
        ligand_features = dict(ligand_features or {})
        ligand_features.setdefault("status", ligand_status.status)
        ligand_features.setdefault("reason", ligand_status.reason)

        logger.info("Extracting identity-bound protein features...")
        bundle_path = entity_bundle_path or dataset.get("entity_bundle_path")
        protein_results = []
        if not bundle_path:
            protein_status = StageStatus("unavailable", "verified_entity_bundle_path_required")
            protein_features = {
                "status": protein_status.status,
                "reason": protein_status.reason,
                "results": [],
            }
        elif self.protein_extractor is None:
            protein_status = StageStatus("unavailable", "protein_feature_extractor_unavailable")
            protein_features = {
                "status": protein_status.status,
                "reason": protein_status.reason,
                "results": [],
                "entity_bundle_path": str(bundle_path),
            }
        else:
            try:
                from evidence_builder import load_entity_bundle

                verified_bundle = load_entity_bundle(str(bundle_path), verify_sources=True)
                extracted_results = self.protein_extractor.extract_from_bundle(str(bundle_path))
                protein_results = self._bundle_bound_feature_results(
                    extracted_results, verified_bundle, self.protein_extractor,
                )
                protein_status = self._protein_readiness(protein_results)
            except Exception as exc:
                logger.warning("Verified entity bundle was rejected by protein extraction: %s", exc)
                protein_status = StageStatus(
                    "unverified_source", f"entity_bundle_rejected:{type(exc).__name__}",
                )
            protein_features = {
                "status": protein_status.status,
                "reason": protein_status.reason,
                "results": [
                    result.to_dict() if hasattr(result, "to_dict") else dict(result)
                    for result in protein_results
                ],
                "entity_bundle_path": str(bundle_path),
            }
            if protein_results:
                protein_features["entity_version"] = getattr(protein_results[0], "entity_version", None)
                protein_features["entity_bundle_hash"] = getattr(
                    protein_results[0], "entity_bundle_hash", None,
                )

        logger.info("Analyzing canonical PPI network statistics...")
        ppi_result = self._canonicalize_dataset_ppi(dataset)
        ppi_features = ppi_result.to_dict()
        ppi_trace = ppi_features.get("ppi_trace") if isinstance(ppi_features, dict) else None
        if not isinstance(ppi_trace, dict):
            ppi_trace = {
                "source_kind": getattr(ppi_result, "source_kind", None),
                "request_status": getattr(ppi_result, "request_status", None),
                "snapshot_path": getattr(ppi_result, "snapshot_path", None),
                "snapshot_root": getattr(ppi_result, "snapshot_root", None),
                "snapshot_sha256": getattr(ppi_result, "snapshot_sha256", None),
            }
            if isinstance(ppi_features, dict):
                ppi_features["ppi_trace"] = ppi_trace
        request_status = getattr(ppi_result, "request_status", None)
        if request_status is None and isinstance(ppi_features, dict):
            request_status = ppi_features.get("request_status")
        snapshot_path = getattr(ppi_result, "snapshot_path", None)
        if snapshot_path is None and isinstance(ppi_features, dict):
            snapshot_path = ppi_features.get("snapshot_path")
        snapshot_sha256 = getattr(ppi_result, "snapshot_sha256", None)
        if snapshot_sha256 is None and isinstance(ppi_features, dict):
            snapshot_sha256 = ppi_features.get("snapshot_sha256")
        gnn_features = {
            "model_type": "network_statistics_only",
            "status": ppi_result.status,
            "centrality": {},
            "graph_stats": {"num_nodes": ppi_result.node_count, "num_edges": ppi_result.edge_count},
            "num_nodes": ppi_result.node_count,
            "num_edges": ppi_result.edge_count,
            "ppi_trace": ppi_trace,
            "ppi_contract": ppi_features,
        }
        if request_status in {"failed", "not_requested"}:
            gnn_features["status"] = "unavailable"
        elif ppi_result.status == "empty":
            gnn_features["status"] = "empty"
        elif self.gnn_analyzer:
            try:
                import networkx as nx
                graph = nx.Graph()
                graph.graph["ppi_trace"] = ppi_trace
                for edge in ppi_result.edges:
                    endpoint_a = edge.endpoint_a.identity_key
                    endpoint_b = edge.endpoint_b.identity_key
                    graph.add_node(endpoint_a, display_name=edge.endpoint_a.display_name)
                    graph.add_node(endpoint_b, display_name=edge.endpoint_b.display_name)
                    graph.add_edge(
                        endpoint_a,
                        endpoint_b,
                        weight=edge.score,
                        semantic_sha256=edge.semantic_sha256,
                        source_record_hashes=edge.source_record_hashes,
                        source_snapshot_path=snapshot_path,
                        source_snapshot_sha256=snapshot_sha256,
                        ppi_trace=ppi_trace,
                    )
                if graph.number_of_nodes() > 0:
                    centrality = self.gnn_analyzer.compute_centrality(graph)
                    gnn_features["centrality"] = self._json_safe_metric_keys(centrality)
                    gnn_features["graph_stats"] = self.gnn_analyzer.compute_graph_statistics(graph)
                else:
                    gnn_features["status"] = "empty"
                gnn_features["num_nodes"] = graph.number_of_nodes()
                gnn_features["num_edges"] = graph.number_of_edges()
            except ImportError:
                gnn_features["status"] = "dependency_unavailable"
        ppi_status = getattr(ppi_result, "status", "invalid")
        if request_status == "failed":
            network_status = StageStatus(
                "unavailable", "ppi_collection_failed", {"ppi_trace": ppi_trace},
            )
        elif request_status == "not_requested":
            network_status = StageStatus(
                "unavailable", "ppi_collection_not_requested", {"ppi_trace": ppi_trace},
            )
        elif ppi_status == "empty":
            empty_reason = (
                "successful_ppi_collection_without_edges"
                if request_status == "complete" else "no_canonical_ppi_edges"
            )
            network_status = StageStatus("empty", empty_reason, {"ppi_trace": ppi_trace})
        elif ppi_status == "partial":
            network_status = StageStatus("partial", "ppi_result_partial", {"ppi_trace": ppi_trace})
        elif ppi_status == "ready" and gnn_features.get("status") != "dependency_unavailable":
            network_status = StageStatus("ready", details={"ppi_trace": ppi_trace})
        else:
            failed_status = (
                "dependency_unavailable"
                if gnn_features.get("status") == "dependency_unavailable"
                else ppi_status
            )
            reason = {
                "invalid": "ppi_result_invalid",
                "unresolved": "ppi_result_unresolved",
                "unavailable": "ppi_collection_unavailable",
                "dependency_unavailable": "network_statistics_dependency_unavailable",
            }.get(failed_status, "network_statistics_not_ready")
            network_status = StageStatus("unavailable", reason, {"ppi_trace": ppi_trace})
        if self.cnn_module is None:
            cnn_status = StageStatus("unavailable", "cnn_module_unavailable")
        else:
            wrapper_status = getattr(self.cnn_module, "status", None)
            cnn_status = (
                wrapper_status
                if isinstance(wrapper_status, StageStatus)
                else StageStatus("skipped_untrained", "verified_trained_weights_not_loaded")
            )
        cnn_features = {
            "status": cnn_status.status,
            "reason": cnn_status.reason,
            "features": None,
        }

        transformer_module = getattr(self, "transformer_module", None)
        if transformer_module is None:
            transformer_status = StageStatus("unavailable", "transformer_module_unavailable")
        else:
            wrapper_status = getattr(transformer_module, "status", None)
            transformer_status = (
                wrapper_status
                if isinstance(wrapper_status, StageStatus)
                else StageStatus("skipped_untrained", "verified_trained_weights_not_loaded")
            )
        feature_readiness = self._feature_readiness({
            "protein": protein_status,
            "cnn": cnn_status,
            "transformer": transformer_status,
            "ligand": ligand_status,
            "network_statistics": network_status,
        })
        features = {
            "ligand": ligand_features,
            "protein": protein_features,
            "gnn": gnn_features,
            "cnn": cnn_features,
            "ppi": ppi_features,
            "feature_readiness": feature_readiness,
        }
        self.results["feature_extraction"] = features
        logger.info("Feature extraction completed with status=%s", feature_readiness["overall_status"])
        return features

    def run_feature_fusion(self, features: Dict[str, Any]) -> Dict[str, Any]:
        features = features or {}
        transformer = getattr(self, "transformer_module", None)
        if transformer is None:
            stage = StageStatus("unavailable", "transformer_module_unavailable")
        elif (
            not callable(getattr(transformer, "_has_verified_weights", None))
            or not transformer._has_verified_weights()
        ):
            stage = StageStatus("skipped_untrained", "verified_trained_weights_not_loaded")
        elif features.get("gnn", {}).get("model_type") != "learned_gnn":
            stage = StageStatus("skipped_untrained", "learned_gnn_not_available")
        else:
            stage = StageStatus("unavailable", "verified_fusion_inputs_not_available")

        fusion_results = {
            "status": stage.status,
            "reason": stage.reason,
            "fused_features": None,
        }
        readiness = features.get("feature_readiness", {})
        stages = dict(readiness.get("stages", {})) if isinstance(readiness, dict) else {}
        stages["transformer"] = stage.to_dict()
        stage_objects = {}
        for name, value in stages.items():
            if isinstance(value, StageStatus):
                stage_objects[name] = value
            elif isinstance(value, dict):
                status = value.get("status", "unavailable")
                reason = value.get("reason") or (None if status == "ready" else "stage_not_ready")
                stage_objects[name] = StageStatus(status, reason, value.get("details", {}))
        features["feature_readiness"] = self._feature_readiness(stage_objects)
        if "feature_extraction" in self.results:
            self.results["feature_extraction"]["feature_readiness"] = features["feature_readiness"]
        self.results["feature_fusion"] = fusion_results
        logger.info("Feature fusion completed with status=%s", stage.status)
        return fusion_results
    _DOCKING_TARGETS = [
        {"gene": "CDK2",  "pdb_id": "1AQ1", "chain": "A", "name": "Cyclin-dependent kinase 2"},
        {"gene": "CDK6",  "pdb_id": "1XO2", "chain": "A", "name": "Cyclin-dependent kinase 6"},
        {"gene": "AKT1",  "pdb_id": "3O96", "chain": "A", "name": "RAC-alpha serine/threonine kinase"},
        {"gene": "PTEN",  "pdb_id": "1D5R", "chain": "A", "name": "Phosphatase and tensin homolog"},
        {"gene": "PIK3CA","pdb_id": "4L23", "chain": "A", "name": "PI3K alpha catalytic subunit"},
    ]

    def _download_pdb(self, pdb_id: str):
        cache_dir = os.path.join(self.output_dir, "docking", "receptors")
        os.makedirs(cache_dir, exist_ok=True)
        pdb_file = os.path.join(cache_dir, f"{pdb_id}.pdb")
        if os.path.exists(pdb_file):
            return pdb_file
        try:
            import requests
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            logger.info("Downloading PDB %s from RCSB...", pdb_id)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(pdb_file, "w", encoding="utf-8") as f:
                f.write(r.text)
            logger.info("Downloaded PDB %s (%d bytes)", pdb_id, len(r.text))
            return pdb_file
        except Exception as e:
            logger.warning("Failed to download PDB %s: %s", pdb_id, e)
            return None

    def run_docking(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        compound = dataset.get("compound", {})
        smiles = compound.get("smiles", "C1=CC(=C(C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O")
        docking_results = {"targets": {}}
        if not self.docking_module:
            self.results["docking"] = docking_results
            return docking_results
        if RDKIT_AVAILABLE and smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                    self.docking_module.prepare_ligand(mol, smiles=smiles)
            except Exception as e:
                logger.error("Ligand preparation failed: %s", e)
        best_affinity = 0
        for target_cfg in self._DOCKING_TARGETS:
            gene = target_cfg["gene"]
            pdb_id = target_cfg["pdb_id"]
            try:
                pdb_file = self._download_pdb(pdb_id)
                if not pdb_file:
                    docking_results["targets"][gene] = {"status": "error", "error": f"PDB {pdb_id} download failed", "binding_affinity": None}
                    continue
                self.docking_module.prepare_protein(pdb_file, protein_name=gene)
                pocket = self.docking_module.detect_binding_site(padding=8.0)
                vina_output = self.docking_module.run_vina(center=pocket["center"], box_size=pocket["box_size"], exhaustiveness=16)
                result = self.docking_module.validate_docking(vina_output)
                result["target_name"] = target_cfg["name"]
                result["pdb_id"] = pdb_id
                docking_results["targets"][gene] = result
                aff = result.get("binding_affinity")
                if aff is not None and aff < best_affinity:
                    best_affinity = aff
                logger.info("Docking %s (PDB %s): best affinity = %.3f kcal/mol", gene, pdb_id, aff or 0)
            except Exception as e:
                logger.error("Docking failed for %s (%s): %s", gene, pdb_id, e)
                docking_results["targets"][gene] = {"status": "error", "error": str(e), "binding_affinity": None}
        docking_results["binding_affinity"] = best_affinity if best_affinity != 0 else None
        self.results["docking"] = docking_results
        logger.info("Molecular docking completed for %d targets", len(docking_results["targets"]))
        return docking_results


    def run_knowledge_graph(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        compound_data = dataset.get('compound', {})
        target_data = dataset.get('targets', [])
        protein_data = dataset.get('proteins', [])
        ppi_result = self._canonicalize_dataset_ppi(dataset)
        kg_contract = ppi_result.to_dict(include_edges=False)
        kg_trace = kg_contract.get("ppi_trace") if isinstance(kg_contract, dict) else None
        if not isinstance(kg_trace, dict):
            kg_trace = {
                "source_kind": getattr(ppi_result, "source_kind", None),
                "request_status": getattr(ppi_result, "request_status", None),
                "snapshot_path": getattr(ppi_result, "snapshot_path", None),
                "snapshot_root": getattr(ppi_result, "snapshot_root", None),
                "snapshot_sha256": getattr(ppi_result, "snapshot_sha256", None),
            }
            if isinstance(kg_contract, dict):
                kg_contract["ppi_trace"] = kg_trace
        kg_results = {'ppi_contract': kg_contract}
        if self.kg_module:
            try:
                self.kg_module.build_from_data(compound_data, target_data, ppi_result)
                self.kg_module.graph.graph["ppi_trace"] = kg_trace
                self.kg_module.graph.graph["ppi_contract"] = kg_results["ppi_contract"]
                stats = self.kg_module.compute_graph_statistics()
                graph_path = self.kg_module.save_graph()
                triples_path = self.kg_module.save_triples()
                
                kg_embedding_status = {'attempted': False, 'available': False}
                if os.environ.get("RUN_PYKEEN", "0") == "1" and self.kg_embedder and self.kg_module.triples:
                    kg_embedding_status['attempted'] = True
                    model = self.kg_embedder.train_pykeen(self.kg_module.triples)
                    kg_embedding_status['available'] = model is not None
                nodes = list(self.kg_module.graph.nodes())[:50]
                edges = [(u, v) for u, v in self.kg_module.graph.edges()][:100]
                kg_results['graph'] = {
                    'stats': stats,
                    'nodes': nodes,
                    'edges': edges,
                    'graph_path': graph_path,
                    'triples_path': triples_path,
                    'kg_embedding': kg_embedding_status,
                }
            except Exception as e:
                logger.error("Knowledge graph analysis failed: %s", e)
                kg_results['graph'] = {'stats': {'num_nodes': 0, 'num_edges': 0}}

        self.results['knowledge_graph'] = kg_results
        logger.info("Knowledge graph analysis completed")

        return kg_results

    def run_output_generation(self) -> Dict[str, Any]:
        feature_extraction = self.results.get('feature_extraction', {})
        gnn_results = feature_extraction.get('gnn', {})
        ligand_results = feature_extraction.get('ligand', {})
        docking_results = self.results.get('docking', {})
        knowledge_graph_results = self.results.get('knowledge_graph', {})
        compound_data = self.results.get('data_collection', {}).get('compound', {})
        target_data = self.results.get('data_collection', {}).get('targets', [])

        output_results = {}
        if self.output_module:
            try:
                summary_input = {
                    'targets': target_data,
                    'compound': compound_data,
                    'docking': docking_results,
                    'gnn': gnn_results,
                    'ligand': ligand_results,
                    'knowledge_graph': knowledge_graph_results,
                    'feature_readiness': feature_extraction.get('feature_readiness', {}),
                    'run_root': self.output_dir,
                    'data_files': self._available_output_files(),
                }
                report = self.output_module.generate_summary_report(summary_input)
                ppi_report = feature_extraction.get("ppi", {})
                if isinstance(report, dict) and isinstance(ppi_report, dict):
                    report["ppi_trace"] = ppi_report.get("ppi_trace")
                    report["ppi_contract"] = ppi_report
                if isinstance(report, dict):
                    report["pipeline_status"] = self.results.get("pipeline_status", "partial")
                    report["pipeline_status_reason"] = self.results.get("pipeline_status_reason")
                report_path = None
                save_report = getattr(self.output_module, "save_report", None)
                if callable(save_report):
                    report_path = save_report(report, "summary.json")
                    if not isinstance(report_path, str) or not report_path:
                        raise RuntimeError("Summary report persistence returned no path")
                self.output_module.generate_dashboard_html(report)
                plot_input = {
                    'targets': target_data,
                    'compound': compound_data,
                    'docking': docking_results,
                    'gnn': gnn_results
                }
                self.output_module.generate_plots(plot_input)
                output_results = report
                if report_path is not None:
                    output_results = dict(report)
                    output_results["summary_report_path"] = report_path
            except Exception as e:
                logger.error("Output generation failed: %s", e)

        self.results['output'] = output_results
        logger.info("Output generation completed")

        return output_results

    def _available_output_files(self) -> Dict[str, str]:
        files = {
            "靶点数据 (CSV)": "data/processed/targets.csv",
            "互作网络数据 (CSV)": "data/processed/interactions.csv",
            "化合物信息 (CSV)": "data/processed/compound_info.csv",
            "蛋白质数据 (CSV)": "data/processed/proteins.csv",
            "完整结果 (JSON)": "framework_results.json",
            "运行日志 (LOG)": "logs/framework_run.log",
        }
        return {
            label: relative_path
            for label, relative_path in files.items()
            if os.path.isfile(os.path.join(self.output_dir, relative_path))
        }

    def save_results(self) -> str:
        if self.output_dir is None:
            logger.warning("Framework results were not saved because no isolated output directory is configured")
            return ""
        output_root = os.path.abspath(os.fspath(self.output_dir))
        resolved_root = os.path.realpath(output_root)
        _assert_output_path_is_unprotected(output_root, resolved_root)
        if os.path.normcase(output_root) != os.path.normcase(resolved_root):
            logger.error("Framework output root changed through a symlink or junction")
            return ""
        filepath = os.path.join(output_root, "framework_results.json")
        resolved_file = os.path.realpath(filepath)
        try:
            if os.path.normcase(os.path.commonpath((resolved_root, resolved_file))) != os.path.normcase(resolved_root):
                raise ValueError("framework result path resolves outside the configured output directory")
            _assert_output_path_is_unprotected(filepath, resolved_file)
            summary_data = self.results.get("data_collection", {})
            feature_data = self.results.get("feature_extraction", {})
            fusion_data = self.results.get("feature_fusion", {})
            readiness = feature_data.get("feature_readiness", {})
            stage_data = readiness.get("stages", {}) if isinstance(readiness, dict) else {}
            stage_statuses = {
                name: value.get("status", "unavailable")
                for name, value in stage_data.items()
                if isinstance(value, dict)
            }
            docking_data = self.results.get("docking", {})
            kg_data = self.results.get("knowledge_graph", {})
            results_to_save = {
                "framework_version": "1.0",
                "run_time": datetime.now().isoformat(),
                "results": self.results,
                "summary": {
                    "data_collection": {
                        "compound": summary_data.get("compound", {}).get("name", "N/A"),
                        "num_targets": len(summary_data.get("targets", {})),
                        "num_proteins": len(summary_data.get("proteins", [])),
                        "num_interactions": len(summary_data.get("interactions", [])),
                    },
                    "feature_extraction": {
                        "status": readiness.get("overall_status", "unavailable"),
                        "stage_statuses": stage_statuses,
                        "ligand_features": stage_statuses.get("ligand") == "ready",
                        "protein_features": stage_statuses.get("protein") in {"ready", "partial"},
                        "gnn_features": bool(feature_data.get("gnn", {}).get("centrality")),
                        "cnn_features": stage_statuses.get("cnn") == "ready",
                    },
                    "feature_fusion": {
                        "status": fusion_data.get("status", "unavailable"),
                        "reason": fusion_data.get("reason"),
                    },
                    "docking": {
                        "best_affinity": docking_data.get("binding_affinity", "N/A"),
                    },
                    "knowledge_graph": {
                        "num_nodes": kg_data.get("graph", {}).get("stats", {}).get("num_nodes", 0),
                        "num_edges": kg_data.get("graph", {}).get("stats", {}).get("num_edges", 0),
                    },
                },
            }
            with open(filepath, "x", encoding="utf-8") as stream:
                json.dump(results_to_save, stream, ensure_ascii=False, indent=2, default=self._json_serialize)
            logger.info(
                "Saved framework results to %s with feature status=%s",
                filepath,
                readiness.get("overall_status", "unavailable"),
            )
            return filepath
        except Exception as exc:
            logger.error("Error saving results: %s", exc)
            return ""
    def _json_serialize(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    def _clean_previous_output(self):
        raise RuntimeError(
            "Destructive output cleanup is disabled; use a new run output directory."
        )
    def run_full_pipeline(self) -> Dict[str, Any]:
        if getattr(self, "_full_pipeline_started", False):
            raise FileExistsError("Full pipeline instances are single-use; create a new framework with a fresh output directory.")
        self._full_pipeline_started = True
        if getattr(self, "_output_dir_preexisted", True):
            raise FileExistsError(
                "Full pipeline requires a new output directory; existing framework outputs are read-only."
            )
        if self.output_dir is None:
            dataset = self.run_data_collection()
            self.results["pipeline_status"] = "partial"
            self.results["pipeline_status_reason"] = dataset.get("reason", "explicit_output_and_online_opt_in_required")
            logger.info("Pipeline remains partial because collection is not configured")
            return self.results

        if not getattr(self, "allow_online", False):
            dataset = self.run_data_collection()
            features = self.run_feature_extraction(dataset)
            self.run_feature_fusion(features)
            self.results["docking"] = {
                "status": "unavailable",
                "reason": "explicit_online_collection_opt_in_required_for_receptor_retrieval",
                "targets": {},
            }
            self.run_knowledge_graph(dataset)
            self.results["pipeline_status"] = "partial"
            self.results["pipeline_status_reason"] = dataset.get(
                "reason", "explicit_online_collection_opt_in_required",
            )
            self.run_output_generation()
            report_path = self.results.get("output", {}).get("summary_report_path")
            if not report_path:
                self.results["pipeline_status"] = "failed"
                self.results["pipeline_status_reason"] = "summary_report_persistence_failed"
                self.save_results()
                raise RuntimeError("Offline partial pipeline summary report could not be persisted")
            if not self.save_results():
                self.results["pipeline_status"] = "failed"
                self.results["pipeline_status_reason"] = "framework_results_persistence_failed"
                raise RuntimeError("Framework results could not be persisted; full pipeline did not complete")
            logger.info("Pipeline remains partial because online collection was not opted in")
            return self.results

        logger.info("Starting full pipeline execution...")
        start_time = datetime.now()

        dataset = self.run_data_collection()
        features = self.run_feature_extraction(dataset)
        self.run_feature_fusion(features)
        self.run_docking(dataset)
        self.run_knowledge_graph(dataset)
        status = self.results.get("feature_extraction", {}).get(
            "feature_readiness", {},
        ).get("overall_status", "unavailable")
        fusion_status = self.results.get("feature_fusion", {}).get("status", "unavailable")
        collection_status = dataset.get("collection_status", "incomplete")
        self.results["pipeline_status"] = (
            "complete"
            if collection_status == "complete" and status == "ready" and fusion_status == "ready"
            else "partial"
        )
        self.results.pop("pipeline_status_reason", None)
        self.run_output_generation()

        runtime = (datetime.now() - start_time).total_seconds()
        saved_path = self.save_results()
        if not saved_path:
            self.results["pipeline_status"] = "failed"
            self.results["pipeline_status_reason"] = "framework_results_persistence_failed"
            raise RuntimeError("Framework results could not be persisted; full pipeline did not complete")
        logger.info(
            "Pipeline execution completed in %.2f seconds with status=%s",
            runtime, self.results["pipeline_status"],
        )
        return self.results

def _paths_intersect(first: str, second: str) -> bool:
    try:
        common = os.path.commonpath((first, second))
    except ValueError:
        return False
    normalized = os.path.normcase(common)
    return normalized in {os.path.normcase(first), os.path.normcase(second)}


def _assert_output_path_is_unprotected(requested_output: str, resolved_output: str) -> None:
    protected_paths = [
        os.path.join(_BASE_DIR, "data"),
        os.path.join(_BASE_DIR, "framework_output"),
        os.path.join(_BASE_DIR, "offline_data.json"),
        os.path.join(_BASE_DIR, ".model_cache"),
        os.path.join(_BASE_DIR, "model_cache"),
        os.path.join(_BASE_DIR, "features"),
        os.path.join(_BASE_DIR, "history"),
        os.path.join(_BASE_DIR, "output_dashboard.html"),
    ]
    for protected_path in protected_paths:
        lexical_protected = os.path.abspath(protected_path)
        resolved_protected = os.path.realpath(lexical_protected)
        if any(
            _paths_intersect(candidate, protected)
            for candidate in (requested_output, resolved_output)
            for protected in (lexical_protected, resolved_protected)
        ):
            raise ValueError("output directory intersects protected historical data or model paths")


def _resolve_new_output_dir(output_dir: str) -> str:
    if not output_dir:
        raise ValueError("an explicit new output directory is required")
    requested_output = os.path.abspath(os.path.expanduser(os.fspath(output_dir)))
    resolved_output = os.path.realpath(requested_output)
    _assert_output_path_is_unprotected(requested_output, resolved_output)
    if os.path.lexists(requested_output):
        raise FileExistsError(f"refusing to overwrite existing output directory: {requested_output}")
    if os.path.lexists(resolved_output):
        raise FileExistsError(f"refusing to overwrite existing output directory: {resolved_output}")
    return resolved_output

def run_feature_pipeline(entity_bundle_path: str, output_dir: str) -> Dict[str, Any]:
    """Run the offline P0 feature/report path against one verified entities.json."""
    from evidence_builder import load_entity_bundle

    bundle_path = os.path.abspath(os.fspath(entity_bundle_path))
    if not os.path.isfile(bundle_path):
        raise FileNotFoundError(f"entity bundle not found: {bundle_path}")
    load_entity_bundle(bundle_path, verify_sources=True)

    if not output_dir:
        raise ValueError("an explicit new output directory is required")
    resolved_output = _resolve_new_output_dir(output_dir)
    dataset = {
        "entity_bundle_path": bundle_path,
        "compound": {},
        "targets": [],
        "proteins": [],
        "interactions": [],
        "ppi_source_kind": "legacy_unverified",
    }
    framework = NetworkPharmacologyFramework(output_dir=resolved_output)
    framework.results["data_collection"] = dataset
    features = framework.run_feature_extraction(dataset)
    framework.run_feature_fusion(features)
    framework.run_output_generation()
    results_path = framework.save_results()
    if not results_path:
        raise RuntimeError("framework results could not be saved")
    logger.info(
        "Local entity feature pipeline completed with status=%s; results=%s",
        framework.results.get("feature_extraction", {})
        .get("feature_readiness", {})
        .get("overall_status", "unavailable"),
        results_path,
    )
    return framework.results


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run offline protein feature extraction from a verified entity bundle."
    )
    parser.add_argument("--entity-bundle", required=True, help="Path to a source-verified entities.json")
    parser.add_argument("--output-dir", required=True, help="New output directory; existing paths are refused")
    args = parser.parse_args(argv)
    return run_feature_pipeline(args.entity_bundle, args.output_dir)


if __name__ == "__main__":
    main()
