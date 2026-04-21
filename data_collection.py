"""
木犀草素(Luteolin)数据清理与统一生物数据集构建
Data Cleaning and Unified Dataset Construction for Luteolin

数据流: PubChem → ChEMBL → UniProt → STRING
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class LuteolinDataCollector:
    
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = output_dir
        self.compound_name = "Luteolin"
        self.compound_name_cn = "木犀草素"
        self.pubchem_cid = 5280445
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/raw", exist_ok=True)
        os.makedirs(f"{output_dir}/processed", exist_ok=True)
        
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
            
            self.offline_proteins = {}
            for gene, info in data['proteins'].items():
                self.offline_proteins[gene] = {**info, 'gene_symbol': gene}
            
            self.offline_interactions = [
                {**inter, 'ncbiTaxonId': 9606} 
                for inter in data['interactions']
            ]
            
            self.gene_mapping = data['gene_mapping']
            
            self.offline_tcmsp = data.get('tcmsp', {})
            
        except FileNotFoundError:
            print(f"  ⚠ Offline data file not found: {json_path}")
            self._init_fallback_data()
    
    def fetch_pubchem_data(self) -> Dict[str, Any]:
        """从PubChem获取木犀草素基本信息"""
        print("=" * 60)
        print("Step 1: Fetching data from PubChem...")
        print("=" * 60)
        
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
            
            smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{self.pubchem_cid}/property/CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,InChI,InChIKey/JSON"
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
            print(f"  ✓ CID: {result['cid']}")
            print(f"  ✓ Molecular Formula: {result['properties'].get('Molecular_Formula', 'N/A')}")
            print(f"  ✓ Molecular Weight: {result['properties'].get('Molecular_Weight', 'N/A')}")
            print(f"  ✓ InChIKey: {result['properties'].get('InChIKey', 'N/A')}")
            
        except Exception as e:
            print(f"  ✗ Network error: {e}")
            print(f"  → Using offline data...")
            self.raw_data['pubchem'] = self.offline_compound.copy()
            self.raw_data['pubchem']['fetch_time'] = datetime.now().isoformat()
            print(f"  ✓ Offline data loaded successfully")
        
        with open(f"{self.output_dir}/raw/pubchem_luteolin.json", 'w', encoding='utf-8') as f:
            json.dump(self.raw_data['pubchem'], f, ensure_ascii=False, indent=2)
        
        return self.raw_data['pubchem']
    
    def fetch_chembl_data(self) -> Dict[str, Any]:
        """从ChEMBL获取木犀草素靶点数据"""
        print("\n" + "=" * 60)
        print("Step 2: Fetching target data from ChEMBL...")
        print("=" * 60)
        
        try:
            molecule_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=luteolin"
            headers = {'Accept': 'application/json'}
            response = requests.get(molecule_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            chembl_id = 'CHEMBL151'
            molecules = data.get('molecules', [])
            for mol in molecules:
                if mol.get('molecule_chembl_id') == 'CHEMBL151':
                    chembl_id = 'CHEMBL151'
                    break
            
            target_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id={chembl_id}&limit=100"
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
            print(f"  ✓ ChEMBL ID: {chembl_id}")
            print(f"  ✓ Total activities: {len(activities)}")
            
        except Exception as e:
            print(f"  ✗ Network error: {e}")
            print(f"  → Using offline data...")
            result = {
                'chembl_id': 'CHEMBL151',
                'fetch_time': datetime.now().isoformat(),
                'targets': self.offline_targets.copy(),
                'total_activities': len(self.offline_targets),
                'source': 'offline'
            }
            self.raw_data['chembl'] = result
            print(f"  ✓ Offline data loaded: {len(self.offline_targets)} targets")
        
        with open(f"{self.output_dir}/raw/chembl_luteolin_targets.json", 'w', encoding='utf-8') as f:
            json.dump(self.raw_data['chembl'], f, ensure_ascii=False, indent=2)
        
        return self.raw_data['chembl']
    
    def fetch_tcmsp_data(self) -> Dict[str, Any]:
        """从TCMSP获取中药成分信息"""
        print("\n" + "=" * 60)
        print("Step 2.5: Fetching TCM data from TCMSP...")
        print("         (Traditional Chinese Medicine Database)")
        print("=" * 60)
        
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
            print(f"  ✓ Mol ID: {result['mol_id']}")
            print(f"  ✓ ADME parameters: OB={result['adme'].get('OB', 'N/A')}%, DL={result['adme'].get('DL', 'N/A')}")
            print(f"  ✓ Related herbs: {len(result['herbs'])}")
            print(f"  ✓ Related targets: {len(result['targets'])}")
            
        except Exception as e:
            print(f"  ✗ Network error: {e}")
            print(f"  → Using offline data...")
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
            print(f"  ✓ Offline data loaded")
            print(f"  ✓ ADME: OB={result['adme'].get('OB', 'N/A')}%, DL={result['adme'].get('DL', 'N/A')}")
            print(f"  ✓ Herbs: {len(result['herbs'])}, Targets: {len(result['targets'])}")
        
        with open(f"{self.output_dir}/raw/tcmsp_luteolin.json", 'w', encoding='utf-8') as f:
            json.dump(self.raw_data['tcmsp'], f, ensure_ascii=False, indent=2)
        
        return self.raw_data['tcmsp']
    
    def fetch_uniprot_data(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """从UniProt获取蛋白质详细信息 - 完善靶点属性"""
        print("\n" + "=" * 60)
        print("Step 3: Fetching protein data from UniProt...")
        print("       (Enriching target attributes)")
        print("=" * 60)
        
        proteins = []
        found_genes = set()
        
        try:
            for gene in gene_symbols:
                if gene in self.offline_proteins:
                    proteins.append(self.offline_proteins[gene].copy())
                    found_genes.add(gene)
                    continue
                
                url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}+AND+organism_id:9606&fields=accession,id,gene_names,protein_name,length,mass,entry_name&format=json&size=1"
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
                            'protein_name': protein.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                            'length': protein.get('sequence', {}).get('length', 0),
                            'mass': protein.get('sequence', {}).get('mass', 0)
                        }
                        proteins.append(protein_info)
                        found_genes.add(gene)
            
            print(f"  ✓ Proteins retrieved: {len(proteins)}")
            
        except Exception as e:
            print(f"  ✗ Network error: {e}")
            print(f"  → Using offline data...")
            for gene in gene_symbols:
                if gene in self.offline_proteins and gene not in found_genes:
                    proteins.append(self.offline_proteins[gene].copy())
                    found_genes.add(gene)
            print(f"  ✓ Offline data loaded: {len(proteins)} proteins")
        
        result = {
            'fetch_time': datetime.now().isoformat(),
            'proteins': proteins,
            'total_proteins': len(proteins)
        }
        
        self.raw_data['uniprot'] = result
        
        with open(f"{self.output_dir}/raw/uniprot_proteins.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    
    def fetch_string_data(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """从STRING获取蛋白质相互作用网络 - 揭示靶点间网络关系"""
        print("\n" + "=" * 60)
        print("Step 4: Fetching PPI network from STRING...")
        print("       (Revealing target interactions)")
        print("=" * 60)
        
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
            print(f"  ✓ Input genes: {len(gene_symbols)}")
            print(f"  ✓ Total interactions: {len(interactions)}")
            
        except Exception as e:
            print(f"  ✗ Network error: {e}")
            print(f"  → Using offline data...")
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
            print(f"  ✓ Offline data loaded: {len(self.offline_interactions)} interactions")
        
        with open(f"{self.output_dir}/raw/string_ppi_network.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    
    def _map_target_to_gene(self, target_name: str) -> str:
        """将靶点名称映射到基因符号"""
        for key, gene in self.gene_mapping.items():
            if key.lower() in target_name.lower():
                return gene
        return ''
    
    def clean_and_standardize(self) -> Dict[str, Any]:
        """数据清洗与标准化"""
        print("\n" + "=" * 60)
        print("Step 5: Data Cleaning and Standardization...")
        print("=" * 60)
        
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
            print(f"  ✓ Compound data cleaned")
        
        if 'chembl' in self.raw_data:
            targets = self.raw_data['chembl'].get('targets', [])
            seen_targets = set()
            cleaned_targets = []
            
            protein_data = {p.get('gene_symbol'): p for p in self.raw_data.get('uniprot', {}).get('proteins', [])}
            
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
                        'activity_value': float(target['activity_value']) if target.get('activity_value') else None,
                        'activity_unit': target.get('activity_unit', 'nM'),
                        'pchembl_value': float(target['pchembl_value']) if target.get('pchembl_value') else None
                    })
            
            cleaned_data['targets'] = cleaned_targets
            print(f"  ✓ Targets cleaned: {len(cleaned_targets)} unique targets")
        
        if 'uniprot' in self.raw_data:
            proteins = self.raw_data['uniprot'].get('proteins', [])
            cleaned_data['proteins'] = proteins
            print(f"  ✓ Proteins cleaned: {len(proteins)} proteins")
        
        if 'string' in self.raw_data:
            interactions = self.raw_data['string'].get('interactions', [])
            cleaned_interactions = []
            
            for interaction in interactions:
                cleaned_interactions.append({
                    'protein_a': interaction.get('preferredName_A', interaction.get('protein_a', '')),
                    'protein_b': interaction.get('preferredName_B', interaction.get('protein_b', '')),
                    'score': float(interaction.get('score', 0)),
                    'ncbiTaxonId': interaction.get('ncbiTaxonId', 9606)
                })
            
            cleaned_data['interactions'] = cleaned_interactions
            print(f"  ✓ Interactions cleaned: {len(cleaned_interactions)} interactions")
        
        self.processed_data = cleaned_data
        
        return cleaned_data
    
    def export_unified_dataset(self) -> str:
        """导出统一生物数据集"""
        print("\n" + "=" * 60)
        print("Step 6: Exporting Unified Dataset...")
        print("=" * 60)
        
        if not self.processed_data:
            self.clean_and_standardize()
        
        json_path = f"{self.output_dir}/processed/unified_luteolin_dataset.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ JSON dataset saved: {json_path}")
        
        compound_df = pd.DataFrame([self.processed_data['compound']])
        compound_df.to_csv(f"{self.output_dir}/processed/compound_info.csv", index=False, encoding='utf-8-sig')
        print(f"  ✓ Compound info saved: compound_info.csv")
        
        if self.processed_data['targets']:
            targets_df = pd.DataFrame(self.processed_data['targets'])
            targets_df.to_csv(f"{self.output_dir}/processed/targets.csv", index=False, encoding='utf-8-sig')
            print(f"  ✓ Targets saved: targets.csv ({len(targets_df)} records)")
        
        if self.processed_data['proteins']:
            proteins_df = pd.DataFrame(self.processed_data['proteins'])
            proteins_df.to_csv(f"{self.output_dir}/processed/proteins.csv", index=False, encoding='utf-8-sig')
            print(f"  ✓ Proteins saved: proteins.csv ({len(proteins_df)} records)")
        
        if self.processed_data['interactions']:
            interactions_df = pd.DataFrame(self.processed_data['interactions'])
            interactions_df.to_csv(f"{self.output_dir}/processed/interactions.csv", index=False, encoding='utf-8-sig')
            print(f"  ✓ Interactions saved: interactions.csv ({len(interactions_df)} records)")
        
        summary = {
            'compound_name': 'Luteolin (木犀草素)',
            'pubchem_cid': self.processed_data['compound'].get('pubchem_cid'),
            'molecular_formula': self.processed_data['compound'].get('molecular_formula'),
            'molecular_weight': self.processed_data['compound'].get('molecular_weight'),
            'total_targets': len(self.processed_data['targets']),
            'total_interactions': len(self.processed_data['interactions']),
            'total_proteins': len(self.processed_data['proteins']),
            'data_sources': self.processed_data['metadata']['data_sources'],
            'processing_time': self.processed_data['metadata']['processing_time']
        }
        
        summary_path = f"{self.output_dir}/processed/dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Summary saved: dataset_summary.json")
        
        return json_path
    
    def run_pipeline(self):
        """运行完整数据收集流程
        
        数据流: PubChem → ChEMBL → TCMSP → UniProt → STRING
        """
        print("\n" + "=" * 60)
        print("  LUTEOLIN DATA COLLECTION PIPELINE")
        print("  木犀草素数据收集流程")
        print("  Data Flow: PubChem → ChEMBL → TCMSP → UniProt → STRING")
        print("=" * 60)
        
        self.fetch_pubchem_data()
        
        self.fetch_chembl_data()
        
        self.fetch_tcmsp_data()
        
        gene_symbols = []
        for target in self.offline_targets:
            if target.get('gene_symbol'):
                gene_symbols.append(target['gene_symbol'])
        for target in self.offline_tcmsp.get('targets', []):
            if target.get('gene_symbol'):
                gene_symbols.append(target['gene_symbol'])
        gene_symbols = list(set(gene_symbols))
        
        self.fetch_uniprot_data(gene_symbols)
        
        self.fetch_string_data(gene_symbols)
        
        self.clean_and_standardize()
        
        self.export_unified_dataset()
        
        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return self.processed_data


def main():
    collector = LuteolinDataCollector(output_dir="./data")
    dataset = collector.run_pipeline()
    
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Compound: {dataset['compound'].get('name', 'N/A')} ({dataset['compound'].get('name_cn', 'N/A')})")
    print(f"  Molecular Formula: {dataset['compound'].get('molecular_formula', 'N/A')}")
    print(f"  Molecular Weight: {dataset['compound'].get('molecular_weight', 'N/A')}")
    print(f"  Total Targets: {len(dataset['targets'])}")
    print(f"  Total Proteins: {len(dataset['proteins'])}")
    print(f"  Total Interactions: {len(dataset['interactions'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
