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

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not installed. Visualization will be unavailable.")

SEABORN_AVAILABLE = False
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    logger.warning("Seaborn not installed. Statistical plots will be unavailable.")

NETWORKX_AVAILABLE = False
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    logger.warning("NetworkX not installed. Network visualization will be limited.")


class OutputGenerator:
    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "output")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        self.results = {}

    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        report = {
            'generation_time': datetime.now().isoformat(),
            'compound': results.get('compound', {}),
            'target_analysis': self._analyze_targets(results.get('targets', [])),
            'docking_summary': results.get('docking', {}),
            'gnn_summary': self._summarize_gnn(results.get('gnn', {})),
            'knowledge_graph_summary': results.get('knowledge_graph', {}),
            'data_quality': {}
        }
        report['data_quality']['targets_found'] = len(results.get('targets', []))
        report['data_quality']['features_extracted'] = bool(results.get('gnn'))
        report['data_quality']['docking_completed'] = bool(results.get('docking'))
        logger.info("Summary report generated")
        return report

    def _analyze_targets(self, targets: List[Dict]) -> Dict[str, Any]:
        analysis = {'total_targets': len(targets), 'gene_symbols': [], 'activities': []}
        for t in targets:
            if t.get('gene_symbol'):
                analysis['gene_symbols'].append(t['gene_symbol'])
            if t.get('activity_value'):
                analysis['activities'].append({
                    'type': t.get('activity_type', ''),
                    'value': t.get('activity_value'),
                    'unit': t.get('activity_unit', '')
                })
        return analysis

    def _summarize_gnn(self, gnn_results: Dict) -> Dict[str, Any]:
        summary = {'available': bool(gnn_results)}
        if gnn_results:
            summary['graph_nodes'] = gnn_results.get('num_nodes', 0)
            summary['graph_edges'] = gnn_results.get('num_edges', 0)
        return summary

    def generate_dashboard_html(self, report: Dict[str, Any]) -> str:
        html = []
        html.append('<!DOCTYPE html>')
        html.append('<html><head><meta charset="utf-8">')
        html.append('<title>Network Pharmacology Report</title>')
        html.append('<style>')
        html.append('body{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5}')
        html.append('.container{max-width:1200px;margin:0 auto}')
        html.append('.card{background:#fff;border-radius:8px;padding:20px;margin:20px0;box-shadow:0 2px 4px rgba(0,0,0,0.1)}')
        html.append('h1{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px}')
        html.append('h2{color:#34495e}')
        html.append('.metric{display:inline-block;margin:10px;padding:15px;background:#ecf0f1;border-radius:6px;min-width:150px}')
        html.append('.metric-value{font-size:24px;font-weight:bold;color:#2980b9}')
        html.append('.metric-label{font-size:12px;color:#7f8c8d}')
        html.append('</style></head><body>')
        html.append(f'<div class="container">')
        html.append(f'<h1>Network Pharmacology Analysis: {report["compound"].get("name","Luteolin")}</h1>')
        html.append(f'<p>Generated: {report["generation_time"]}</p>')

        html.append('<div class="card"><h2>Target Analysis</h2>')
        ta = report['target_analysis']
        html.append(f'<')
        html.append(f'<div class="metric"><div class="metric-value">{ta["total_targets"]}</div><div class="metric-label">Total Targets</div></div>')
        html.append(f'<div class="metric"><div class="metric-value">{len(ta["gene_symbols"])}</div><div class="metric-label">Unique Genes</div></div>')
        html.append('</div>')

        html.append('<div class="card"><h2>Docking Summary</h2>')
        ds = report['docking_summary']
        binding = ds.get('binding_affinity', 'N/A')
        html.append(f'<div class="metric"><div class="metric-value">{binding}</div><div class="metric-label">Binding Affinity (kcal/mol)</div></div>')
        html.append('</div>')

        html.append('<div class="card"><h2>GNN Analysis</h2>')
        gs = report['gnn_summary']
        html.append(f'<div class="metric"><div class="metric-value">{gs.get("graph_nodes", 0)}</div><div class="metric-label">Graph Nodes</div></div>')
        html.append(f'<div class="metric"><div class="metric-value">{gs.get("graph_edges", 0)}</div><div class="metric-label">Graph Edges</div></div>')
        html.append('</div></div></body></html>')

        content = '\n'.join(html)
        filepath = os.path.join(self.output_dir, "report.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Dashboard HTML generated: %s", filepath)
        return filepath

    def generate_plots(self, data: Dict[str, Any]) -> List[str]:
        plots = []
        if not MATPLOTLIB_AVAILABLE:
            return plots
        try:
            images_dir = os.path.join(self.output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            targets = data.get('targets', [])
            if targets:
                activities = [float(t.get('activity_value', 0)) for t in targets
                            if t.get('activity_value')]
                if activities:
                    axes[0, 0].hist(activities, bins=20, color='skyblue', edgecolor='black')
                    axes[0, 0].set_title('Activity Distribution')
                    axes[0, 0].set_xlabel('Activity Value')
                    axes[0, 0].set_ylabel('Frequency')
            compound_data = data.get('compound', {})
            if compound_data:
                meta_data = {
                    'MW': compound_data.get('molecular_weight', 0),
                    'LogP': compound_data.get('logP', 0),
                    'HBD': compound_data.get('h_bond_donors', 0),
                    'HBA': compound_data.get('h_bond_acceptors', 0)
                }
                labels = list(meta_data.keys())
                values = list(meta_data.values())
                axes[0, 1].bar(labels, values, color='lightcoral')
                axes[0, 1].set_title('Compound Properties')
            docking = data.get('docking', {})
            if docking and 'docking_modes' in docking:
                modes = docking['docking_modes']
                if modes:
                    affinities = [m.get('affinity', 0) for m in modes]
                    axes[1, 0].plot(affinities, 'o-', color='green')
                    axes[1, 0].set_title('Docking Scores')
                    axes[1, 0].set_xlabel('Mode')
                    axes[1, 0].set_ylabel('Affinity (kcal/mol)')

            gnn = data.get('gnn', {})
            if gnn and 'centrality' in gnn:
                centrality = gnn['centrality']
                dc = centrality.get('degree_centrality', {})
                if dc:
                    nodes = list(dc.keys())[:10]
                    values = [dc[n] for n in nodes]
                    axes[1, 1].barh(nodes, values, color='mediumpurple')
                    axes[1, 1].set_title('Top Nodes by Degree Centrality')

            plt.tight_layout()
            filepath = os.path.join(images_dir, "analysis_plots.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            plots.append(filepath)
            logger.info("Generated analysis plots: %s", filepath)
        except Exception as e:
            logger.error("Analysis plots generation failed: %s", e)

        try:
            images_dir = os.path.join(self.output_dir, "images")
            targets = data.get('targets', [])
            gene_scores = {}
            for t in targets:
                g = t.get('gene_symbol', '')
                if g:
                    score = abs(float(t.get('activity_value', 0))) if t.get('activity_value') else 0
                    gene_scores[g] = max(gene_scores.get(g, 0), score)
            if gene_scores:
                sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)[:15]
                fig, ax = plt.subplots(figsize=(10, 6))
                genes, scores = zip(*sorted_genes) if sorted_genes else ([], [])
                bars = ax.bar(range(len(genes)), scores, color='steelblue', edgecolor='navy')
                ax.set_xticks(range(len(genes)))
                ax.set_xticklabels(genes, rotation=45, ha='right')
                ax.set_title('Target Priority Score', fontsize=14, fontweight='bold')
                ax.set_xlabel('Gene Symbol')
                ax.set_ylabel('Activity Score')
                for bar, s in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scores)*0.01,
                            f'{s:.2f}', ha='center', va='bottom', fontsize=8)
                plt.tight_layout()
                filepath = os.path.join(images_dir, "target_priority.png")
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                plots.append(filepath)
                logger.info("Generated target priority plot: %s", filepath)
        except Exception as e:
            logger.error("Target priority plot failed: %s", e)

        try:
            images_dir = os.path.join(self.output_dir, "images")
            fig, ax = plt.subplots(figsize=(10, 6))
            target_types = {}
            for t in data.get('targets', []):
                org = t.get('organism', 'Unknown') or 'Unknown'
                target_types[org] = target_types.get(org, 0) + 1
            if target_types:
                orgs = list(target_types.keys())[:10]
                counts = [target_types[o] for o in orgs]
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(orgs)))
                path_bars = ax.barh(range(len(orgs)), counts, color=colors, edgecolor='black')
                ax.set_yticks(range(len(orgs)))
                ax.set_yticklabels(orgs, fontsize=9)
                ax.set_xlabel('Target Count')
                ax.set_title('Target Organism Distribution / Pathway Analysis', fontsize=14, fontweight='bold')
                for bar, c in zip(path_bars, counts):
                    ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                            str(c), ha='left', va='center', fontsize=9)
                plt.tight_layout()
                filepath = os.path.join(images_dir, "pathway_analysis.png")
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                plots.append(filepath)
                logger.info("Generated pathway analysis plot: %s", filepath)
        except Exception as e:
            logger.error("Pathway analysis plot failed: %s", e)

        return plots

    def save_report(self, report: Dict[str, Any], filename: str = "pharmacology_report.json"):
        filepath = os.path.join(self.output_dir, "reports", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Saved report: %s", filepath)