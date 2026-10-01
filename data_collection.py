"""
木犀草素(Luteolin)数据清理与统一生物数据集构建
Data Cleaning and Unified Dataset Construction for Luteolin

数据流: PubChem -> ChEMBL -> UniProt -> STRING
"""

import hashlib
import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings

('ignore')

from logging_config import setup_logger
from target_scoring import classify_target, normalize_organism
logger = setup_logger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


_PROTECTED_OUTPUT_NAMES = (
    "data", "framework_output", "offline_data.json", ".model_cache",
    "model_cache", "features", "history", "output_dashboard.html",
)


def _paths_intersect(first: str, second: str) -> bool:
    try:
        common = os.path.commonpath((first, second))
    except ValueError:
        return False
    normalized = os.path.normcase(common)
    return normalized in {os.path.normcase(first), os.path.normcase(second)}


def _assert_output_path_is_unprotected(requested: str, resolved: str) -> None:
    protected_paths = [os.path.join(_BASE_DIR, name) for name in _PROTECTED_OUTPUT_NAMES]
    candidates = (os.path.abspath(requested), os.path.realpath(resolved))
    for protected_path in protected_paths:
        lexical_protected = os.path.abspath(protected_path)
        resolved_protected = os.path.realpath(lexical_protected)
        if any(
            _paths_intersect(candidate, protected)
            for candidate in candidates
            for protected in (lexical_protected, resolved_protected)
        ):
            raise ValueError("output directory intersects protected historical data or model paths")


def _resolve_new_output_dir(output_dir: Optional[str]) -> str:
    if not output_dir:
        raise ValueError("an explicit new output directory is required")
    requested = os.path.abspath(os.path.expanduser(os.fspath(output_dir)))
    resolved = os.path.realpath(requested)
    _assert_output_path_is_unprotected(requested, resolved)
    if os.path.lexists(requested):
        raise FileExistsError(f"refusing to overwrite existing output directory: {requested}")
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite existing output directory: {resolved}")
    return resolved

def _validate_existing_output_dir(output_dir: Optional[str]) -> str:
    if not output_dir:
        raise ValueError("an explicit new output directory is required")
    requested = os.path.abspath(os.fspath(output_dir))
    resolved = os.path.realpath(requested)
    _assert_output_path_is_unprotected(requested, resolved)
    if os.path.normcase(requested) != os.path.normcase(resolved):
        raise ValueError("configured output directory changed through a symlink or junction")
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"configured output directory is unavailable: {resolved}")
    return resolved


class LuteolinDataCollector:

    def __init__(self, output_dir: Optional[str] = None, *, allow_online: bool = False):
        if type(allow_online) is not bool:
            raise TypeError("allow_online must be an explicit boolean")
        if output_dir is None or output_dir == "":
            if allow_online:
                raise ValueError("online collection requires an explicit new output directory")
            self.output_dir = None
        else:
            self.output_dir = _resolve_new_output_dir(output_dir)
        self.allow_online = allow_online
        self._output_root_claimed = False
        self.compound_name = "Luteolin"
        self.compound_name_cn = "木犀草素"
        self.pubchem_cid = 5280445
        self.raw_data = {}
        self.processed_data = {}
        self._init_offline_data()

    def _prepare_output_root(self) -> str:
        if self.output_dir is None:
            raise ValueError("an explicit new output directory is required before writing")
        if self._output_root_claimed:
            self.output_dir = _validate_existing_output_dir(self.output_dir)
            return self.output_dir
        requested_output = os.path.abspath(os.fspath(self.output_dir))
        if os.path.normcase(os.path.realpath(requested_output)) != os.path.normcase(requested_output):
            raise ValueError("configured output directory changed through a symlink or junction")
        output_root = _resolve_new_output_dir(requested_output)
        os.makedirs(output_root, exist_ok=False)
        self.output_dir = _validate_existing_output_dir(output_root)
        self._output_root_claimed = True
        return self.output_dir

    def _assert_output_tree_safe(self, output_root: str) -> None:
        for current_root, directories, files in os.walk(output_root, followlinks=False):
            for name in [*directories, *files]:
                path = os.path.join(current_root, name)
                if os.path.normcase(os.path.realpath(path)) != os.path.normcase(os.path.abspath(path)):
                    raise ValueError("configured output tree contains a symlink or junction")

    def _require_online_collection(self) -> str:
        if self.output_dir is None:
            raise ValueError("online collection requires an explicit new output directory")
        if not self.allow_online:
            raise PermissionError("online collection requires allow_online=True")
        output_root = self._prepare_output_root()
        self._assert_output_tree_safe(output_root)
        return output_root

    def _write_immutable_bytes(self, relative_path: str, raw_bytes: bytes) -> tuple[str, str]:
        if not isinstance(raw_bytes, bytes):
            raise TypeError("snapshot content must be bytes")
        output_root = self._prepare_output_root()
        relative = Path(relative_path)
        if (
            relative.is_absolute()
            or relative.drive
            or relative.root
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("snapshot path must remain inside the configured output root")
        destination = Path(output_root) / relative
        resolved_root = os.path.realpath(output_root)
        resolved_parent = os.path.realpath(destination.parent)
        resolved_destination = os.path.realpath(destination)
        if os.path.normcase(resolved_parent) != os.path.normcase(os.path.abspath(destination.parent)):
            raise ValueError("snapshot parent must not be a symlink or junction")
        try:
            if (
                os.path.normcase(os.path.commonpath((resolved_root, resolved_destination)))
                != os.path.normcase(resolved_root)
                or os.path.normcase(os.path.commonpath((resolved_root, resolved_parent)))
                != os.path.normcase(resolved_root)
            ):
                raise ValueError("snapshot path resolves outside the configured output root")
        except ValueError as exc:
            if "snapshot path resolves" in str(exc):
                raise
            raise ValueError("snapshot path resolves outside the configured output root") from exc
        _assert_output_path_is_unprotected(str(destination), resolved_destination)
        _assert_output_path_is_unprotected(str(destination.parent), resolved_parent)

        if os.path.lexists(destination):
            if os.path.normcase(os.path.realpath(destination)) != os.path.normcase(os.path.abspath(destination)):
                raise ValueError("snapshot destination must not be a symlink or junction")
            if destination.read_bytes() != raw_bytes:
                raise FileExistsError(f"immutable snapshot conflict: {relative.as_posix()}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = os.path.realpath(destination.parent)
            resolved_destination = os.path.realpath(destination)
            if os.path.normcase(resolved_parent) != os.path.normcase(os.path.abspath(destination.parent)):
                raise ValueError("snapshot parent changed through a symlink or junction")
            if (
                os.path.normcase(os.path.commonpath((resolved_root, resolved_parent)))
                != os.path.normcase(resolved_root)
                or os.path.normcase(os.path.commonpath((resolved_root, resolved_destination)))
                != os.path.normcase(resolved_root)
            ):
                raise ValueError("snapshot path changed resolution during directory creation")
            _assert_output_path_is_unprotected(str(destination), resolved_destination)
            with destination.open("xb") as stream:
                stream.write(raw_bytes)
        return relative.as_posix(), hashlib.sha256(raw_bytes).hexdigest()
    def _save_json_snapshot(self, folder: str, prefix: str, payload: Any) -> tuple[str, str]:
        raw_bytes = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        relative_path = (Path(folder) / f"{prefix}-{digest}.json").as_posix()
        return self._write_immutable_bytes(relative_path, raw_bytes)

    def _save_csv_snapshot(self, folder: str, prefix: str, dataframe) -> tuple[str, str]:
        csv_text = dataframe.to_csv(index=False)
        raw_bytes = csv_text.encode("utf-8-sig")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        relative_path = (Path(folder) / f"{prefix}-{digest}.csv").as_posix()
        return self._write_immutable_bytes(relative_path, raw_bytes)
    def _init_offline_data(self):
        json_path = os.path.join(os.path.dirname(__file__), 'offline_data.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.offline_compound = data['compound']
            self.offline_compound['fetch_time'] = datetime.now().isoformat()
            self.offline_targets = data['targets']
            geneticsur = {}
            for gene, info in data['proteins'].items():
                geneticsur[gene] = {**info, 'gene_symbol': gene}
            self.offline_proteins = geneticsur
            self.offline_interactions = [
                {**inter, 'ncbiTaxonId': 9606}
                for inter in data['interactions']
            ]
            self.gene_mapping = data['gene_mapping']
            self.offline_tcmsp = data.get('tcmsp', {})
        except FileNotFoundError:
            logger.info("Offline data file not found; using embedded fallback in memory: %s", json_path)
            self._init_fallback_data()
        self.gene_mapping = {
            **self._default_gene_mapping(),
            **getattr(self, "gene_mapping", {}),
        }
        # The embedded fallback remains in memory; offline_data.json is read-only.

    def _default_gene_mapping(self) -> Dict[str, str]:
        return {
            "aldo-keto reductase family 1 member b1": "AKR1B1",
            "tyrosinase": "TYR",
            "cyclin-dependent kinase 5": "CDK5",
            "cyclin-dependent kinase 1": "CDK1",
            "cyclin-dependent kinase 6": "CDK6",
            "cyclin-dependent kinase 2": "CDK2",
            "glycogen synthase kinase-3": "GSK3B",
            "serine/threonine-protein kinase pim-1": "PIM1",
            "fatty acid synthase": "FASN",
            "dna topoisomerase 1": "TOP1",
            "dna topoisomerase 2-alpha": "TOP2A",
            "lactoylglutathione lyase": "GLO1",
            "tyrosine-protein kinase lck": "LCK",
            "adenosine receptor a1": "ADORA1",
            "enoyl-acyl-carrier protein reductase": "ENR",
            "3-oxoacyl-acyl-carrier protein reductase": "KAR",
            "serine protease 1": "PRSS1",
        }

    def _fetch_pubchem_via_pubchempy(self) -> Dict[str, Any]:
        try:
            import pubchempy as pcp
            c = pcp.Compound.from_cid(self.pubchem_cid)
            if c is None:
                raise ValueError("PubChemPy returned None")
            result = {
                'cid': self.pubchem_cid,
                'name': getattr(c, 'iupac_name', self.compound_name) or self.compound_name,
                'name_cn': self.compound_name_cn,
                'fetch_time': datetime.now().isoformat(),
                'source': 'pubchempy',
                'properties': {
                    'IUPAC_Name': getattr(c, 'iupac_name', ''),
                    'Molecular_Formula': getattr(c, 'molecular_formula', ''),
                    'Molecular_Weight': getattr(c, 'molecular_weight', 0),
                    'Canonical_SMILES': getattr(c, 'canonical_smiles', ''),
                    'Isomeric_SMILES': getattr(c, 'isomeric_smiles', ''),
                    'InChI': getattr(c, 'inchi', ''),
                    'InChIKey': getattr(c, 'inchikey', ''),
                    'XLogP': getattr(c, 'xlogp', ''),
                    'H_Bond_Donor': getattr(c, 'h_bond_donor_count', ''),
                    'H_Bond_Acceptor': getattr(c, 'h_bond_acceptor_count', ''),
                    'Rotatable_Bonds': getattr(c, 'rotatable_bond_count', ''),
                    'TPSA': getattr(c, 'tpsa', ''),
                    'Complexity': getattr(c, 'complexity', ''),
                }
            }
            result['properties'] = {k: v for k, v in result['properties'].items() if v != '' and v is not None}
            logger.info("PubChemPy: CID=%d, Formula=%s",
                        result['cid'], result['properties'].get('Molecular_Formula', 'N/A'))
            return result
        except Exception as e:
            raise RuntimeError(f"PubChemPy failed: {e}") from e

    def fetch_pubchem_data(self) -> Dict[str, Any]:
        self._require_online_collection()
        logger.info("Step 1: Fetching data from PubChem...")
        try:
            try:
                result = self._fetch_pubchem_via_pubchempy()
            except Exception as py_err:
                logger.info("PubChemPy unavailable (%s), trying NCBI E-utilities...", py_err)
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LuteolinAnalyzer/1.0"}
                eutils_url = (
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    f"?db=pccompound&id={self.pubchem_cid}&retmode=json"
                )
                response = requests.get(eutils_url, headers=headers, timeout=30)
                response.raise_for_status()
                payload = response.json()
                uids = payload.get("result", {}).get("uids", [])
                if not uids:
                    raise ValueError("E-utilities returned no compound data")
                summary = payload["result"][uids[0]]
                result = {
                    "cid": self.pubchem_cid,
                    "name": self.compound_name,
                    "name_cn": self.compound_name_cn,
                    "fetch_time": datetime.now().isoformat(),
                    "source": "eutils",
                    "properties": {
                        "Molecular_Formula": summary.get("molecularformula", ""),
                        "Molecular_Weight": float(summary.get("molecularweight", 0)),
                        "Canonical_SMILES": summary.get("canonicalsmiles", ""),
                        "InChIKey": summary.get("inchikey", ""),
                    },
                }
            snapshot_path, snapshot_sha256 = self._save_json_snapshot(
                "raw/pubchem", "pubchem_luteolin", result,
            )
            result["raw_snapshot_path"] = snapshot_path
            result["raw_snapshot_sha256"] = snapshot_sha256
            self.raw_data["pubchem"] = result
            return result
        except Exception as exc:
            logger.info("PubChem retrieval failed; retaining the local legacy record: %s", exc)
            result = dict(self.offline_compound)
            result["source_kind"] = "legacy_unverified"
            result["fetch_time"] = datetime.now().isoformat()
            snapshot_path, snapshot_sha256 = self._save_json_snapshot(
                "raw/pubchem", "pubchem_luteolin", result,
            )
            result["raw_snapshot_path"] = snapshot_path
            result["raw_snapshot_sha256"] = snapshot_sha256
            self.raw_data["pubchem"] = result
            return result
    def fetch_chembl_data(self) -> Dict[str, Any]:
        """Capture bounded ChEMBL pages and linked metadata without projection loss."""
        self._require_online_collection()
        logger.info("Step 2: Fetching bounded activity evidence from ChEMBL...")
        from evidence_builder import _collect_chembl, activity_evidence_from_collector_payload, LegacyDisplayAdapter

        requested_kind = getattr(self, "source_kind", "online_snapshot")
        source_kind = requested_kind if requested_kind in {"online_snapshot", "synthetic_test"} else "online_snapshot"
        molecule_ids = getattr(self, "chembl_molecule_ids", ["CHEMBL151"])
        try:
            result = _collect_chembl(
                molecule_ids,
                max_pages=10,
                max_records=1000,
                timeout=30,
                max_requests=128,
                max_response_bytes=8_000_000,
            )
        except Exception as exc:
            logger.warning("ChEMBL retrieval could not start: %s", exc)
            result = {
                "molecule_ids": list(molecule_ids) if isinstance(molecule_ids, (list, tuple)) else [],
                "molecule_records": {},
                "target_records": {},
                "assay_records": {},
                "document_records": {},
                "activity_pages": [],
                "page_receipts": [],
                "pagination_status": "incomplete",
                "metadata_status": "incomplete",
                "status": "incomplete",
                "incomplete_reasons": [str(exc)],
                "request_count": 0,
                "page_count": 0,
                "activity_count": 0,
            }

        result["chembl_id"] = molecule_ids[0] if isinstance(molecule_ids, (list, tuple)) and len(molecule_ids) == 1 else None
        result["source_kind"] = source_kind
        result["fetch_time"] = datetime.now().isoformat()
        snapshot_failures = []
        for page in result.get("activity_pages", []):
            page["source_kind"] = source_kind
            molecule_id = page.get("molecule_chembl_id")
            raw_text = page.get("raw_response_text")
            if not isinstance(molecule_id, str) or not isinstance(raw_text, str):
                snapshot_failures.append("activity page missing explicit molecule identity or raw body")
                continue
            page_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            relative_path = f"raw/chembl/activities/{molecule_id}/{page_hash}.json"
            snapshot_path = Path(self.output_dir) / Path(relative_path)
            try:
                saved_path, saved_hash = self._write_immutable_bytes(
                    relative_path, raw_text.encode("utf-8"),
                )
                if saved_hash != page_hash:
                    raise ValueError(f"immutable snapshot hash mismatch: {relative_path}")
                page["source_path"] = saved_path
                page["source_sha256"] = saved_hash
                for receipt in result.get("page_receipts", []):
                    if receipt.get("source_id", "").startswith(f"chembl_activity:{molecule_id}:") and receipt.get("response_sha256") == page.get("response_sha256"):
                        receipt["source_kind"] = source_kind
                        receipt["source_path"] = saved_path
                        receipt["source_sha256"] = saved_hash
            except Exception as exc:
                snapshot_failures.append(f"{relative_path}:{exc}")
        for receipt in result.get("page_receipts", []):
            source_id = receipt.get("source_id", "")
            if not source_id.startswith(("chembl_molecule:", "chembl_target:", "chembl_assay:", "chembl_document:")):
                continue
            category, identifier = source_id.split(":", 1)
            if not identifier.startswith("CHEMBL") or not identifier[6:].isdigit():
                snapshot_failures.append(f"invalid source record identifier: {source_id}")
                continue
            raw_text = receipt.get("raw_response_text")
            if not isinstance(raw_text, str):
                snapshot_failures.append(f"raw response missing: {source_id}")
                continue
            source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            relative_path = f"raw/chembl/{category.removeprefix('chembl_')}/{identifier}/{source_hash}.json"
            snapshot_path = Path(self.output_dir) / Path(relative_path)
            try:
                saved_path, saved_hash = self._write_immutable_bytes(
                    relative_path, raw_text.encode("utf-8"),
                )
                if saved_hash != source_hash:
                    raise ValueError(f"immutable source snapshot hash mismatch: {relative_path}")
                receipt["source_kind"] = source_kind
                receipt["source_path"] = saved_path
                receipt["source_sha256"] = saved_hash
            except Exception as exc:
                snapshot_failures.append(f"{relative_path}:{exc}")
        if snapshot_failures:
            result.setdefault("incomplete_reasons", []).extend(snapshot_failures)
            result["metadata_status"] = "incomplete"

        result["activities"] = [
            activity
            for page in result.get("activity_pages", [])
            for activity in page.get("activities", [])
        ]
        self.raw_data["chembl"] = result
        try:
            evidence_rows = activity_evidence_from_collector_payload(result)
            result["targets"] = LegacyDisplayAdapter.targets(evidence_rows)
            result["projection_status"] = "complete"
        except Exception as exc:
            # A display conversion failure must never erase captured pages or IDs.
            logger.warning("ChEMBL evidence projection failed; raw capture retained: %s", exc)
            result["targets"] = []
            result["projection_status"] = "error"
            result["projection_errors"] = [str(exc)]
            result.setdefault("incomplete_reasons", []).append(f"evidence_projection:{exc}")
        result["total_activities"] = len(result["activities"])
        result["collection_status"] = (
            "complete"
            if result.get("pagination_status") == "complete"
            and result.get("metadata_status") == "complete"
            and result.get("projection_status") == "complete"
            else "incomplete"
        )
        result["status"] = result["collection_status"]
        logger.info(
            "ChEMBL: molecules=%d, pages=%d, activities=%d, pagination=%s, metadata=%s",
            len(result.get("molecule_ids", [])), result.get("page_count", 0),
            result["total_activities"], result.get("pagination_status"), result.get("metadata_status"),
        )

        try:
            snapshot_path, snapshot_sha256 = self._save_json_snapshot(
                "raw/chembl", "chembl_luteolin_targets", result,
            )
            result["raw_snapshot_path"] = snapshot_path
            result["raw_snapshot_sha256"] = snapshot_sha256
        except Exception as exc:
            result["raw_snapshot_status"] = "error"
            result.setdefault("incomplete_reasons", []).append(f"aggregate_snapshot:{exc}")
            result["collection_status"] = "incomplete"
            result["status"] = "incomplete"
        self.raw_data["chembl"] = result
        return result

    def fetch_tcmsp_data(self) -> Dict[str, Any]:
        """Return the embedded TCMSP reference without calling it online evidence."""
        self._require_online_collection()
        logger.info("Step 2.5: Recording TCMSP static reference provenance...")
        request_status = "not_attempted"
        http_status = None
        response = None
        try:
            tcmsp_url = "https://tcmsp-e.com/tcmspsearch.php?qs=compound_name&qr=luteolin"
            response = requests.get(
                tcmsp_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                },
                timeout=30,
                stream=True,
                allow_redirects=False,
            )
            final_url = getattr(response, "url", tcmsp_url)
            if not isinstance(final_url, str) or not final_url.startswith("https://tcmsp-e.com/"):
                raise ValueError("TCMSP response escaped the configured host")
            http_status = getattr(response, "status_code", None)
            if type(http_status) is not int or not 200 <= http_status < 300:
                raise ValueError(f"TCMSP static probe returned HTTP {http_status!r}")
            response.raise_for_status()
            request_status = "http_success_static_payload_not_used"
        except Exception as e:
            request_status = "http_unavailable_static_payload_used"
            logger.info("TCMSP page request status=%s: %s", request_status, e)
        finally:
            close = getattr(response, "close", None) if response is not None else None
            if callable(close):
                try:
                    close()
                except Exception as close_error:
                    logger.debug("TCMSP static probe close failed: %s", close_error)
        static_path = os.path.join(os.path.dirname(__file__), "offline_data.json")
        static_sha256 = hashlib.sha256(Path(static_path).read_bytes()).hexdigest() if os.path.isfile(static_path) else None
        result = {
            "mol_id": self.offline_tcmsp.get("mol_id"),
            "name": "luteolin",
            "name_cn": "木犀草素",
            "fetch_time": datetime.now().isoformat(),
            "adme": self.offline_tcmsp.get("adme", {}),
            "herbs": self.offline_tcmsp.get("herbs", []),
            "targets": self.offline_tcmsp.get("targets", []),
            "diseases": self.offline_tcmsp.get("diseases", []),
            "source": "static_reference",
            "source_kind": "static_reference",
            "evidence_status": "unverified",
            "source_path": "offline_data.json",
            "source_sha256": static_sha256,
            "http_request_status": request_status,
            "http_status": http_status,
        }
        self.raw_data["tcmsp"] = result
        logger.info(
            "TCMSP retained as static_reference: herbs=%d, targets=%d",
            len(result["herbs"]), len(result["targets"]),
        )
        snapshot_path, snapshot_sha256 = self._save_json_snapshot(
            "raw/tcmsp", "tcmsp_luteolin", result,
        )
        result["raw_snapshot_path"] = snapshot_path
        result["raw_snapshot_sha256"] = snapshot_sha256
        return result

    def fetch_uniprot_data(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """Reject symbol-only target guesses; sequence hydration needs exact components."""
        self._prepare_output_root()
        logger.info("Step 3: UniProt symbol search disabled pending verified target components...")
        unresolved = []
        for value in gene_symbols or []:
            if isinstance(value, str) and value.strip():
                unresolved.append({
                    "query": value,
                    "reason": "symbol_search_disabled",
                    "resolution_status": "unverified",
                })
        result = {
            "retrieval_status": "not_requested",
            "resolution_status": "unverified" if unresolved else "empty",
            "source_kind": "legacy_unverified",
            "proteins": [],
            "total_proteins": 0,
            "unresolved_requests": unresolved,
            "reason": "verified ChEMBL target component accession and taxonomy are required",
        }
        self.raw_data["uniprot"] = result
        snapshot_path, snapshot_sha256 = self._save_json_snapshot(
            "raw/uniprot", "uniprot_proteins", result,
        )
        result["raw_snapshot_path"] = snapshot_path
        result["raw_snapshot_sha256"] = snapshot_sha256
        return result

    def _save_string_response(self, response) -> tuple[str, str]:
        raw_body = getattr(response, "content", None)
        if not isinstance(raw_body, (bytes, bytearray)):
            raise ValueError("STRING response did not expose exact raw response bytes")
        raw_bytes = bytes(raw_body)
        digest = hashlib.sha256(raw_bytes).hexdigest()
        relative_path = (Path("raw") / "string" / f"{digest}.json").as_posix()
        return self._write_immutable_bytes(relative_path, raw_bytes)

    def fetch_string_data(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """Fetch STRING for explicit seeds and retain exact response-byte provenance."""
        symbols = sorted({
            value.strip() for value in (gene_symbols or [])
            if isinstance(value, str) and value.strip()
        })
        if not symbols:
            result = {
                "species": "Homo sapiens",
                "species_id": 9606,
                "fetch_time": datetime.now().isoformat(),
                "input_genes": [],
                "interactions": [],
                "total_interactions": 0,
                "source_kind": "legacy_unverified",
                "resolution_status": "empty",
                "request_status": "not_requested",
                "snapshot_path": None,
                "snapshot_sha256": None,
            }
            self.raw_data["string"] = result
            return result

        self._require_online_collection()
        logger.info("Step 4: Fetching STRING PPI for %d explicit seed symbols...", len(symbols))
        response = None
        snapshot_path = None
        snapshot_sha256 = None
        try:
            url = "https://string-db.org/api/json/network"
            response = requests.post(
                url,
                data={
                    "identifiers": "\r".join(symbols),
                    "species": 9606,
                    "limit": 20,
                    "caller_identity": "luteolin_analysis",
                },
                timeout=60,
            )
            response.raise_for_status()
            interactions = response.json()
            if not isinstance(interactions, list) or any(not isinstance(row, dict) for row in interactions):
                raise ValueError("STRING response must be a list of interaction objects")
            snapshot_path, snapshot_sha256 = self._save_string_response(response)
            result = {
                "species": "Homo sapiens",
                "species_id": 9606,
                "fetch_time": datetime.now().isoformat(),
                "input_genes": symbols,
                "interactions": interactions,
                "total_interactions": len(interactions),
                "source_kind": "online_snapshot",
                "request_status": "complete",
                "resolution_status": "resolved" if interactions else "empty",
                "snapshot_path": snapshot_path,
                "snapshot_sha256": snapshot_sha256,
                "snapshot_root": self.output_dir,
                "raw_response_sha256": snapshot_sha256,
            }
        except Exception as exc:
            logger.warning("STRING retrieval incomplete: %s", exc)
            if response is not None:
                try:
                    snapshot_path, snapshot_sha256 = self._save_string_response(response)
                except Exception as snapshot_error:
                    logger.warning("STRING failure snapshot could not be saved: %s", snapshot_error)
            result = {
                "species": "Homo sapiens",
                "species_id": 9606,
                "fetch_time": datetime.now().isoformat(),
                "input_genes": symbols,
                "interactions": [],
                "total_interactions": 0,
                "source_kind": "legacy_unverified",
                "resolution_status": "incomplete",
                "request_status": "failed",
                "snapshot_path": snapshot_path,
                "snapshot_sha256": snapshot_sha256,
                "snapshot_root": self.output_dir if snapshot_path is not None else None,
                "reason": str(exc),
            }
        self.raw_data["string"] = result
        return result
    def _map_target_to_gene(self, target_name: str) -> str:
        for key, gene in self.gene_mapping.items():
            if key.lower() in target_name.lower():
                return gene
        return ''

    def clean_and_standardize(self) -> Dict[str, Any]:
        """Project raw collector output while retaining separate evidence rows."""
        logger.info("Step 5: Building evidence and legacy display projections...")
        from dataclasses import asdict
        from evidence_builder import activity_evidence_from_collector_payload, LegacyDisplayAdapter

        chembl_payload = self.raw_data.get("chembl", {})
        ppi_payload = self.raw_data.get("string", {})
        ppi_trace = {
            "source_kind": ppi_payload.get("source_kind", "legacy_unverified"),
            "request_status": ppi_payload.get("request_status", "not_requested"),
            "snapshot_path": ppi_payload.get("snapshot_path"),
            "snapshot_sha256": ppi_payload.get("snapshot_sha256"),
        }
        if ppi_payload.get("snapshot_root") is not None:
            ppi_trace["snapshot_root"] = ppi_payload["snapshot_root"]
        activity_rows = activity_evidence_from_collector_payload(chembl_payload)
        serialized_evidence = [asdict(row) for row in activity_rows]
        cleaned_data = {
            "compound": {},
            "targets": [],
            "activity_evidence": serialized_evidence,
            "interactions": [],
            "ppi_trace": ppi_trace,
            "ppi_source_kind": ppi_trace["source_kind"],
            "ppi_request_status": ppi_trace["request_status"],
            "ppi_snapshot_path": ppi_trace["snapshot_path"],
            "ppi_snapshot_sha256": ppi_trace["snapshot_sha256"],
            "ppi_snapshot_root": ppi_trace.get("snapshot_root"),
            "proteins": [],
            "metadata": {
                "processing_time": datetime.now().isoformat(),
                "data_sources": list(self.raw_data.keys()),
                "activity_pagination_status": chembl_payload.get("pagination_status", "unknown"),
                "activity_collection_status": chembl_payload.get("collection_status", "legacy_unverified"),
                "activity_source_kind": chembl_payload.get("source_kind", "legacy_unverified"),
            },
        }
        if "pubchem" in self.raw_data:
            pubchem = self.raw_data["pubchem"]
            props = pubchem.get("properties", {})
            smiles = (
                props.get("Canonical_SMILES")
                or props.get("SMILES_Absolute")
                or props.get("SMILES_Connectivity")
            )
            cleaned_data["compound"] = {
                "name": pubchem.get("name") or self.compound_name,
                "name_cn": pubchem.get("name_cn") or self.compound_name_cn,
                "pubchem_cid": pubchem.get("cid"),
                "chembl_id": chembl_payload.get("chembl_id"),
                "molecular_formula": props.get("Molecular_Formula"),
                "molecular_weight": props.get("Molecular_Weight"),
                "smiles": smiles,
                "inchi": props.get("InChI") or props.get("InChI_Standard"),
                "inchikey": props.get("InChIKey") or props.get("InChIKey_Standard"),
                "raw_record": pubchem,
                "structure_status": "present" if isinstance(smiles, str) and smiles.strip() else "missing",
            }

        display_rows = LegacyDisplayAdapter.targets(activity_rows)
        for target in display_rows:
            components = target.get("target_components")
            if not isinstance(components, list):
                matching_evidence = next((
                    item for item in serialized_evidence
                    if item.get("activity_id") == target.get("activity_id")
                    and item.get("target_chembl_id") == target.get("target_chembl_id")
                ), {})
                components = matching_evidence.get("target_components", [])
            accessions = [
                component.get("accession")
                for component in components
                if isinstance(component, dict) and isinstance(component.get("accession"), str)
            ]
            organism = normalize_organism(target.get("organism"))
            target_class = classify_target(target)
            target_type = next((
                item.get("target_type") for item in serialized_evidence
                if item.get("activity_id") == target.get("activity_id")
                and item.get("target_chembl_id") == target.get("target_chembl_id")
            ), None)
            is_confirmed_single_human = (
                target_type == "SINGLE PROTEIN"
                and organism == "Homo sapiens"
                and len(accessions) == 1
                and all(
                    component.get("component_tax_id", component.get("tax_id")) in (9606, "9606")
                    for component in components if isinstance(component, dict)
                )
            )
            cleaned_data["targets"].append({
                **target,
                "target_name": target.get("target_name"),
                "gene_symbol": target.get("gene_symbol"),
                "uniprot_id": accessions[0] if is_confirmed_single_human else None,
                "organism": organism,
                "target_type": target_type,
                "target_components": components,
                "target_class": target_class,
                "is_valid_target": False,
            })

        raw_proteins = self.raw_data.get("uniprot", {}).get("proteins", [])
        for protein in raw_proteins:
            if not isinstance(protein, dict):
                continue
            copied = dict(protein)
            if not copied.get("sequence"):
                copied["sequence_status"] = "missing"
                copied["sequence_status_reason"] = "legacy_payload_has_metadata_without_sequence"
            cleaned_data["proteins"].append(copied)
        cleaned_data["interactions"] = ppi_payload.get("interactions", [])
        self.processed_data = cleaned_data
        return cleaned_data
    def save_processed_data(self) -> str:
        self._prepare_output_root()
        if not self.processed_data:
            self.processed_data = self.clean_and_standardize()
        json_relative, _ = self._save_json_snapshot(
            "processed", "unified_luteolin_dataset", self.processed_data,
        )
        compound_df = pd.DataFrame([self.processed_data["compound"]])
        self._save_csv_snapshot("processed", "compound_info", compound_df)
        if self.processed_data["targets"]:
            self._save_csv_snapshot(
                "processed", "targets", pd.DataFrame(self.processed_data["targets"]),
            )
        self._save_csv_snapshot(
            "processed", "proteins", pd.DataFrame(self.processed_data["proteins"]),
        )
        if self.processed_data["interactions"]:
            self._save_csv_snapshot(
                "processed", "interactions", pd.DataFrame(self.processed_data["interactions"]),
            )
        summary = {
            "compound_name": "Luteolin (木犀草素)",
            "pubchem_cid": self.processed_data["compound"].get("pubchem_cid"),
            "molecular_formula": self.processed_data["compound"].get("molecular_formula"),
            "molecular_weight": self.processed_data["compound"].get("molecular_weight"),
            "total_targets": len(self.processed_data["targets"]),
            "total_proteins": len(self.processed_data["proteins"]),
            "total_interactions": len(self.processed_data["interactions"]),
            "processing_time": datetime.now().isoformat(),
        }
        self._save_json_snapshot("processed", "dataset_summary", summary)

        logger.info("Processed data saved to: %s", os.path.join(self.output_dir, "processed"))
        logger.info(
            "Summary: %d targets, %d proteins, %d interactions",
            summary["total_targets"], summary["total_proteins"], summary["total_interactions"],
        )
        return os.path.join(self.output_dir, json_relative)
    def run_pipeline(self) -> Dict[str, Any]:
        self._require_online_collection()
        self.fetch_pubchem_data()
        self.fetch_chembl_data()
        self.fetch_tcmsp_data()
        targets = self.raw_data.get("chembl", {}).get("targets", [])
        gene_symbols = sorted({
            target.get("gene_symbol").strip()
            for target in targets
            if isinstance(target, dict)
            and isinstance(target.get("gene_symbol"), str)
            and target.get("gene_symbol").strip()
            and normalize_organism(target.get("organism")) == "Homo sapiens"
        })
        self.fetch_uniprot_data(gene_symbols)
        self.fetch_string_data(gene_symbols)
        self.clean_and_standardize()
        chembl_status = self.raw_data.get("chembl", {}).get("collection_status")
        ppi_status = self.raw_data.get("string", {}).get("request_status")
        overall_status = "complete" if chembl_status == "complete" and ppi_status == "complete" else "partial"
        self.processed_data["status"] = overall_status
        self.processed_data["collection_status"] = overall_status
        self.processed_data.setdefault("metadata", {})["collection_status"] = overall_status
        self.save_processed_data()
        logger.info("Pipeline completed with status=%s", overall_status)
        return self.processed_data
    def _init_fallback_data(self):
        self.offline_compound = {
            'cid': 5280445,
            'name': 'Luteolin',
            'name_cn': '木犀草素',
            'properties': {
                'Molecular_Formula': 'C15H10O6',
                'Molecular_Weight': 286.24,
                'Canonical_SMILES': 'C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O',
                'InChIKey': 'IQPNAANSBPBGFQ-UHFFFAOYSA-N'
            }
        }
        self.offline_targets = []
        self.offline_proteins = {}
        self.offline_interactions = []
        self.gene_mapping = {}
        self.offline_tcmsp = {}
