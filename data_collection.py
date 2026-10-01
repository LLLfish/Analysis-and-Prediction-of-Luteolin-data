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


class LuteolinDataCollector:

    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "data")
        self.output_dir = os.path.abspath(output_dir)
        self.compound_name = "Luteolin"
        self.compound_name_cn = "木犀草素"
        self.pubchem_cid = 5280445

        os.makedirs(os.path.join(self.output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "processed"), exist_ok=True)

        self.raw_data = {}
        self.processed_data = {}
        self._init_offline_data()

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
            logger.info("Offline data file not found, generating from embedded fallback: %s", json_path)
            self._init_fallback_data()
        self.gene_mapping = {
            **self._default_gene_mapping(),
            **getattr(self, "gene_mapping", {}),
        }
        # Auto-generate offline_data.json if it does not exist yet
        if not os.path.exists(json_path):
            try:
                offline_save = {
                    'compound': self.offline_compound,
                    'targets': self.offline_targets,
                    'proteins': self.offline_proteins,
                    'interactions': self.offline_interactions,
                    'gene_mapping': self.gene_mapping,
                    'tcmsp': self.offline_tcmsp,
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(offline_save, f, ensure_ascii=False, indent=2)
                logger.info("Offline data file auto-generated: %s", json_path)
            except Exception as save_err:
                logger.warning("Failed to save offline data file: %s", save_err)

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
        logger.info("Step 1: Fetching data from PubChem...")
        try:
            # Primary: try PubChemPy (richer data, same API but better parsing)
            try:
                result = self._fetch_pubchem_via_pubchempy()
                self.raw_data['pubchem'] = result
                return result
            except Exception as py_err:
                logger.info("PubChemPy unavailable (%s), trying direct REST API...", py_err)

            # Secondary: try NCBI E-utilities (often works when PUG-REST is IP-blocked)
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LuteolinAnalyzer/1.0"}
            eutils_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pccompound&id={self.pubchem_cid}&retmode=json"
            eu_response = requests.get(eutils_url, headers=headers, timeout=30)
            eu_response.raise_for_status()
            eu_data = eu_response.json()
            uids = eu_data.get('result', {}).get('uids', [])
            if not uids:
                raise ValueError("E-utilities returned no compound data")
            summary = eu_data['result'][uids[0]]
            result = {
                'cid': self.pubchem_cid,
                'name': self.compound_name,
                'name_cn': self.compound_name_cn,
                'fetch_time': datetime.now().isoformat(),
                'source': 'eutils',
                'properties': {
                    'Molecular_Formula': summary.get('molecularformula', ''),
                    'Molecular_Weight': float(summary.get('molecularweight', 0)),
                    'Canonical_SMILES': summary.get('canonicalsmiles', ''),
                    'InChIKey': summary.get('inchikey', ''),
                }
            }
            self.raw_data['pubchem'] = result
            logger.info("E-utilities: CID=%d, Formula=%s, Weight=%s",
                        result['cid'], result['properties']['Molecular_Formula'],
                        result['properties']['Molecular_Weight'])
            return result
            compound = data['PC_Compounds'][0]
            props = compound.get('props', [])
            result = {
                'cid': self.pubchem_cid,
                'name': self.compound_name,
                'name_cn': self.compound_name_cn,
                'fetch_time': datetime.now().isoformat(),
                'properties': {}
            }
            for prop in props:
                urn = prop.get('urn', {})
                label = urn.get('label', '')
                name = urn.get('name', '')
                if 'value' in prop:
                    if 'fval' in prop['value']:
                        value = prop['value']['fval']
                    elif 'sval' in prop['value']:
                        value = prop['value']['sval']
                    elif 'ival' in prop['value']:
                        value = prop['value']['ival']
                    else:
                        value = str(prop['value'])
                    key = f"{label}_{name}" if name else label
                    result['properties'][key] = value

            smiles_url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{self.pubchem_cid}"
                "/property/CanonicalSMILES,IsomericSMILES,MolecularFormula,"
                "MolecularWeight,InChI,InChIKey/JSON"
            )
            smiles_response = requests.get(smiles_url, headers=headers, timeout=30)
            if smiles_response.status_code == 200:
                smiles_data = smiles_response.json()
                props = smiles_data.get('PropertyTable', {}).get('Properties', [{}])[0]
                result['properties']['Canonical_SMILES'] = props.get('CanonicalSMILES', '')
                result['properties']['Isomeric_SMILES'] = props.get('IsomericSMILES', '')
                result['properties']['Molecular_Formula'] = props.get('MolecularFormula', '')
                result['properties']['Molecular_Weight'] = props.get('MolecularWeight', 0)
                result['properties']['InChI'] = props.get('InChI', '')
                result['properties']['InChIKey'] = props.get('InChIKey', '')

            self.raw_data['pubchem'] = result
            logger.info("PubChem: CID=%s, Formula=%s, Weight=%s",
                        result['cid'], result['properties'].get('Molecular_Formula', 'N/A'),
                        result['properties'].get('Molecular_Weight', 'N/A'))
        except Exception as e:
            logger.info("PubChem network error (offline data will be used): %s", e)
            self.raw_data['pubchem'] = self.offline_compound.copy()
            self.raw_data['pubchem']['fetch_time'] = datetime.now().isoformat()
            logger.info("Offline data loaded successfully")

        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "pubchem_luteolin.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.raw_data['pubchem'], f, ensure_ascii=False, indent=2)
        return self.raw_data['pubchem']

    def fetch_chembl_data(self) -> Dict[str, Any]:
        """Capture bounded ChEMBL pages and linked metadata without projection loss."""
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
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                if snapshot_path.exists():
                    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != page_hash:
                        raise ValueError(f"immutable snapshot collision: {relative_path}")
                else:
                    with snapshot_path.open("xb") as snapshot:
                        snapshot.write(raw_text.encode("utf-8"))
                page["source_path"] = relative_path
                page["source_sha256"] = page_hash
                for receipt in result.get("page_receipts", []):
                    if receipt.get("source_id", "").startswith(f"chembl_activity:{molecule_id}:") and receipt.get("response_sha256") == page.get("response_sha256"):
                        receipt["source_kind"] = source_kind
                        receipt["source_path"] = relative_path
                        receipt["source_sha256"] = page_hash
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
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                if snapshot_path.exists():
                    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != source_hash:
                        raise ValueError(f"immutable snapshot collision: {relative_path}")
                else:
                    with snapshot_path.open("xb") as snapshot:
                        snapshot.write(raw_text.encode("utf-8"))
                receipt["source_kind"] = source_kind
                receipt["source_path"] = relative_path
                receipt["source_sha256"] = source_hash
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

        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "chembl_luteolin_targets.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result["raw_snapshot_path"] = os.path.relpath(json_path, self.output_dir).replace(os.sep, "/")
            result["raw_snapshot_sha256"] = hashlib.sha256(Path(json_path).read_bytes()).hexdigest()
        except Exception as exc:
            result["raw_snapshot_status"] = "error"
            result.setdefault("incomplete_reasons", []).append(f"aggregate_snapshot:{exc}")
            result["collection_status"] = "incomplete"
            result["status"] = "incomplete"
        self.raw_data["chembl"] = result
        return result

    def fetch_tcmsp_data(self) -> Dict[str, Any]:
        """Return the embedded TCMSP reference without calling it online evidence."""
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
        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "tcmsp_luteolin.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def fetch_uniprot_data(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """Reject symbol-only target guesses; sequence hydration needs exact components."""
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
        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "uniprot_proteins.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def _save_string_response(self, response) -> tuple[str, str]:
        raw_body = getattr(response, "content", None)
        if isinstance(raw_body, (bytes, bytearray)):
            raw_bytes = bytes(raw_body)
        else:
            response_text = getattr(response, "text", None)
            if not isinstance(response_text, str):
                raise ValueError("STRING response did not expose raw response bytes")
            raw_bytes = response_text.encode("utf-8")
        snapshot_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        relative_path = Path("raw") / "string" / f"{snapshot_sha256}.json"
        snapshot_path = Path(self.output_dir) / relative_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if snapshot_path.exists():
            if snapshot_path.read_bytes() != raw_bytes:
                raise ValueError(f"immutable STRING snapshot collision: {relative_path.as_posix()}")
        else:
            with snapshot_path.open("xb") as stream:
                stream.write(raw_bytes)
        return relative_path.as_posix(), snapshot_sha256

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
        processed_dir = os.path.join(self.output_dir, "processed")
        if not self.processed_data:
            self.processed_data = self.clean_and_standardize()
        json_path = os.path.join(processed_dir, "unified_luteolin_dataset.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        compound_df = pd.DataFrame([self.processed_data['compound']])
        compound_df.to_csv(os.path.join(processed_dir, "compound_info.csv"),
                          index=False, encoding='utf-8-sig')
        if self.processed_data['targets']:
            targets_df = pd.DataFrame(self.processed_data['targets'])
            targets_df.to_csv(os.path.join(processed_dir, "targets.csv"),
                            index=False, encoding='utf-8-sig')
        proteins_df = pd.DataFrame(self.processed_data['proteins'])
        proteins_df.to_csv(os.path.join(processed_dir, "proteins.csv"),
                    index=False, encoding='utf-8-sig')
        if self.processed_data['interactions']:
            interactions_df = pd.DataFrame(self.processed_data['interactions'])
            interactions_df.to_csv(os.path.join(processed_dir, "interactions.csv"),
                                  index=False, encoding='utf-8-sig')
        summary = {
            'compound_name': 'Luteolin (木犀草素)',
            'pubchem_cid': self.processed_data['compound'].get('pubchem_cid'),
            'molecular_formula': self.processed_data['compound'].get('molecular_formula'),
            'molecular_weight': self.processed_data['compound'].get('molecular_weight'),
            'total_targets': len(self.processed_data['targets']),
            'total_proteins': len(self.processed_data['proteins']),
            'total_interactions': len(self.processed_data['interactions']),
            'processing_time': datetime.now().isoformat()
        }
        summary_path = os.path.join(processed_dir, "dataset_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Processed data saved to: %s", processed_dir)
        logger.info("Summary: %d targets, %d proteins, %d interactions",
                    summary['total_targets'], summary['total_proteins'],
                    summary['total_interactions'])
        return json_path

    def run_pipeline(self) -> Dict[str, Any]:
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
        self.save_processed_data()
        logger.info("Pipeline completed successfully")
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
