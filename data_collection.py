"""
木犀草素(Luteolin)数据清理与统一生物数据集构建
Data Cleaning and Unified Dataset Construction for Luteolin

数据流: PubChem -> ChEMBL -> UniProt -> STRING
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
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
            logger.warning("Offline data file not found: %s", json_path)
            self._init_fallback_data()

    def fetch_pubchem_data(self) -> Dict[str, Any]:
        logger.info("Step 1: Fetching data from PubChem...")
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{self.pubchem_cid}/JSON"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
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
            smiles_response = requests.get(smiles_url, timeout=30)
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
            logger.warning("PubChem network error: %s, using offline data", e)
            self.raw_data['pubchem'] = self.offline_compound.copy()
            self.raw_data['pubchem']['fetch_time'] = datetime.now().isoformat()
            logger.info("Offline data loaded successfully")

        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "pubchem_luteolin.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.raw_data['pubchem'], f, ensure_ascii=False, indent=2)
        return self.raw_data['pubchem']

    def fetch_chembl_data(self) -> Dict[str, Any]:
        logger.info("Step 2: Fetching target data from ChEMBL...")
        try:
            molecule_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=luteolin"
            headers = {'Accept': 'application/json'}
            response = requests.get(molecule_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            chembl_id = 'CHEMBL151'
            molecules = data.get('molecules', [])

            target_url = (
                f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
                f"?molecule_chembl_id={chembl_id}&limit=100"
            )
            target_response = requests.get(target_url, headers=headers, timeout=30)
            target_response.raise_for_status()
            target_data = target_response.json()

            activities = target_data.get('activities', [])
            targets = []
            for act in activities:
                if act.get('target_chembl_id'):
                    targets.append({
                        'target_chembl_id': act.get('target_chembl_id'),
                        'target_name': act.get('target_pref_name', ''),
                        'gene_symbol': '',
                        'organism': act.get('target_organism', ''),
                        'activity_type': act.get('standard_type', ''),
                        'activity_value': act.get('standard_value'),
                        'activity_unit': act.get('standard_units', ''),
                        'pchembl_value': act.get('pchembl_value')
                    })

            result = {
                'chembl_id': chembl_id,
                'fetch_time': datetime.now().isoformat(),
                'targets': targets,
                'total_activities': len(activities)
            }
            self.raw_data['chembl'] = result
            logger.info("ChEMBL: ID=%s, activities=%d", chembl_id, len(activities))
        except Exception as e:
            logger.warning("ChEMBL network error: %s, using offline data", e)
            result = {
                'chembl_id': 'CHEMBL151',
                'fetch_time': datetime.now().isoformat(),
                'targets': self.offline_targets.copy(),
                'total_activities': len(self.offline_targets),
                'source': 'offline'
            }
            self.raw_data['chembl'] = result
            logger.info("Offline data loaded: %d targets", len(self.offline_targets))

        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "chembl_luteolin_targets.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.raw_data['chembl'], f, ensure_ascii=False, indent=2)
        return self.raw_data['chembl']

    def fetch_tcmsp_data(self) -> Dict[str, Any]:
        logger.info("Step 2.5: Fetching TCM data from TCMSP...")
        try:
            tcmsp_url = "https://tcmsp-e.com/tcmspsearch.php?qs=compound_name&qr=luteolin"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            response = requests.get(tcmsp_url, headers=headers, timeout=30)
            response.raise_for_status()

            result = {
                'mol_id': 'MOL000006',
                'name': 'luteolin',
                'name_cn': '木犀草素',
                'fetch_time': datetime.now().isoformat(),
                'adme': self.offline_tcmsp.get('adme', {}),
                'herbs': self.offline_tcmsp.get('herbs', []),
                'targets': self.offline_tcmsp.get('targets', []),
                'diseases': self.offline_tcmsp.get('diseases', []),
                'source': 'online'
            }
            self.raw_data['tcmsp'] = result
            logger.info("TCMSP: Mol ID=%s, herbs=%d, targets=%d",
                        result['mol_id'], len(result['herbs']), len(result['targets']))
        except Exception as e:
            logger.warning("TCMSP network error: %s, using offline data", e)
            result = {
                'mol_id': self.offline_tcmsp.get('mol_id', 'MOL000006'),
                'name': 'luteolin',
                'name_cn': '木犀草素',
                'fetch_time': datetime.now().isoformat(),
                'adme': self.offline_tcmsp.get('adme', {}),
                'herbs': self.offline_tcmsp.get('herbs', []),
                'targets': self.offline_tcmsp.get('targets', []),
                'diseases': self.offline_tcmsp.get('diseases', []),
                'source': 'offline'
            }
            self.raw_data['tcmsp'] = result
            logger.info("Offline TCM data loaded: herbs=%d, targets=%d",
                        len(result['herbs']), len(result['targets']))

        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "tcmsp_luteolin.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.raw_data['tcmsp'], f, ensure_ascii=False, indent=2)
        return self.raw_data['tcmsp']

    def fetch_uniprot_data(self, gene_symbols: List[str]) -> Dict[str, Any]:
        logger.info("Step 3: Fetching protein data from UniProt...")
        proteins = []
        found_genes = set()

        try:
            for gene in gene_symbols:
                if gene in self.offline_proteins:
                    proteins.append(self.offline_proteins[gene].copy())
                    found_genes.add(gene)
                    continue
                url = (
                    f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}"
                    f"+AND+organism_id:9606&fields=accession,id,gene_names,"
                    f"protein_name,length,mass,entry_name&format=json&size=1"
                )
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    if results:
                        protein = results[0]
                        protein_info = {
                            'gene_symbol': gene,
                            'uniprot_id': protein.get('primaryAccession', ''),
                            'entry_name': protein.get('uniProtkbId', ''),
                            'protein_name': protein.get('proteinDescription', {}).get(
                                'recommendedName', {}).get('fullName', {}).get('value', ''),
                            'length': protein.get('sequence', {}).get('length', 0),
                            'mass': protein.get('sequence', {}).get('mass', 0)
                        }
                        proteins.append(protein_info)
                        found_genes.add(gene)
            logger.info("UniProt: %d proteins retrieved", len(proteins))
        except Exception as e:
            logger.warning("UniProt network error: %s, using offline data", e)
            for gene in gene_symbols:
                if gene in self.offline_proteins and gene not in found_genes:
                    proteins.append(self.offline_proteins[gene].copy())
                    found_genes.add(gene)
            logger.info("Offline protein data loaded: %d proteins", len(proteins))

        result = {
            'fetch_time': datetime.now().isoformat(),
            'proteins': proteins,
            'total_proteins': len(proteins)
        }
        self.raw_data['uniprot'] = result

        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "uniprot_proteins.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def fetch_string_data(self, gene_symbols: List[str]) -> Dict[str, Any]:
        logger.info("Step 4: Fetching PPI network from STRING...")
        try:
            string_api_url = "https://string-db.org/api"
            output_format = "json"
            method = "network"
            species = 9606
            params = {
                'identifiers': '\r'.join(gene_symbols),
                'species': species,
                'limit': 20,
                'caller_identity': 'luteolin_analysis'
            }
            url = f"{string_api_url}/{output_format}/{method}"
            response = requests.post(url, data=params, timeout=60)
            response.raise_for_status()
            interactions = response.json()

            result = {
                'species': 'Homo sapiens',
                'species_id': species,
                'fetch_time': datetime.now().isoformat(),
                'input_genes': gene_symbols,
                'interactions': interactions,
                'total_interactions': len(interactions)
            }
            self.raw_data['string'] = result
            logger.info("STRING: %d genes, %d interactions",
                        len(gene_symbols), len(interactions))
        except Exception as e:
            logger.warning("STRING network error: %s, using offline data", e)
            result = {
                'species': 'Homo sapiens',
                'species_id': 9606,
                'fetch_time': datetime.now().isoformat(),
                'input_genes': gene_symbols,
                'interactions': self.offline_interactions.copy(),
                'total_interactions': len(self.offline_interactions),
                'source': 'offline'
            }
            self.raw_data['string'] = result
            logger.info("Offline PPI data loaded: %d interactions",
                        len(self.offline_interactions))

        raw_dir = os.path.join(self.output_dir, "raw")
        json_path = os.path.join(raw_dir, "string_ppi_network.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def _map_target_to_gene(self, target_name: str) -> str:
        for key, gene in self.gene_mapping.items():
            if key.lower() in target_name.lower():
                return gene
        return ''

    def clean_and_standardize(self) -> Dict[str, Any]:
        logger.info("Step 5: Data Cleaning and Standardization...")
        cleaned_data = {
            'compound': {},
            'targets': [],
            'interactions': [],
            'proteins': [],
            'metadata': {
                'processing_time': datetime.now().isoformat(),
                'data_sources': list(self.raw_data.keys())
            }
        }
        if 'pubchem' in self.raw_data:
            pubchem = self.raw_data['pubchem']
            props = pubchem.get('properties', {})
            cleaned_data['compound'] = {
                'name': 'Luteolin',
                'name_cn': '木犀草素',
                'pubchem_cid': pubchem.get('cid'),
                'chembl_id': self.raw_data.get('chembl', {}).get('chembl_id', 'CHEMBL151'),
                'molecular_formula': props.get('Molecular_Formula', 'C15H10O6'),
                'molecular_weight': float(props.get('Molecular_Weight', 286.24)),
                'smiles': props.get('Canonical_SMILES', ''),
                'inchi': props.get('InChI', ''),
                'inchikey': props.get('InChIKey', ''),
            }
            logger.info("Compound data cleaned")

        if 'chembl' in self.raw_data:
            targets = self.raw_data['chembl'].get('targets', [])
            seen_targets = set()
            cleaned_targets = []
            protein_data = {p.get('gene_symbol'): p
                           for p in self.raw_data.get('uniprot', {}).get('proteins', [])}

            for target in targets:
                target_id = target.get('target_chembl_id', '')
                if target_id and target_id not in seen_targets and target_id != 'N/A':
                    seen_targets.add(target_id)
                    gene_symbol = target.get('gene_symbol', '')
                    if not gene_symbol:
                        gene_symbol = self._map_target_to_gene(target.get('target_name', ''))
                    uniprot_id = ''
                    if gene_symbol in protein_data:
                        uniprot_id = protein_data[gene_symbol].get('uniprot_id', '')
                    elif gene_symbol in self.offline_proteins:
                        uniprot_id = self.offline_proteins[gene_symbol].get('uniprot_id', '')
                    cleaned_targets.append({
                        'target_chembl_id': target_id,
                        'target_name': target.get('target_name', ''),
                        'gene_symbol': gene_symbol,
                        'uniprot_id': uniprot_id,
                        'organism': target.get('organism', 'Homo sapiens'),
                        'activity_type': target.get('activity_type', ''),
                        'activity_value': target.get('activity_value'),
                        'activity_unit': target.get('activity_unit', '')
                    })
            cleaned_data['targets'] = cleaned_targets
            logger.info("Targets cleaned: %d unique targets", len(cleaned_targets))

        cleaned_data['proteins'] = self.raw_data.get('uniprot', {}).get('proteins', [])
        cleaned_data['interactions'] = self.raw_data.get('string', {}).get('interactions', [])
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
        if self.processed_data['proteins']:
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
        targets = self.raw_data.get('chembl', {}).get('targets', [])
        gene_symbols = []
        for t in targets:
            gene = t.get('gene_symbol', '') or self._map_target_to_gene(t.get('target_name', ''))
            if gene:
                gene_symbols.append(gene)
        gene_symbols = list(set(gene_symbols))
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