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
            'data_quality': {},
            'targets_raw': results.get('targets', []),
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
        try:
            dt = datetime.fromisoformat(report["generation_time"])
            display_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            display_time = report["generation_time"]

        compound = report.get('compound', {})
        ta = report.get('target_analysis', {})
        ds = report.get('docking_summary', {})
        gs = report.get('gnn_summary', {})
        kg = report.get('knowledge_graph_summary', {})
        dq = report.get('data_quality', {})

        compound_name = compound.get('name', 'Luteolin')
        compound_name_cn = compound.get('name_cn', '木犀草素')
        total_targets = ta.get('total_targets', 0)
        unique_genes = len(set(ta.get('gene_symbols', [])))
        binding_affinity = ds.get('binding_affinity', 'N/A')
        graph_nodes = gs.get('graph_nodes', 0) or kg.get('graph', {}).get('stats', {}).get('num_nodes', 0)
        graph_edges = gs.get('graph_edges', 0) or kg.get('graph', {}).get('stats', {}).get('num_edges', 0)
        graph_relations = kg.get('graph', {}).get('stats', {}).get('num_relations', 0)

        gene_scores = {}
        for t in report.get('targets_raw', []):
            g = t.get('gene_symbol', '')
            if g:
                try:
                    score = abs(float(t.get('activity_value', 0))) if t.get('activity_value') else 0
                    gene_scores[g] = max(gene_scores.get(g, 0), score)
                except (ValueError, TypeError):
                    pass
        sorted_targets = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        organism_dist = {}
        for t in report.get('targets_raw', []):
            org = t.get('organism', 'Unknown') or 'Unknown'
            organism_dist[org] = organism_dist.get(org, 0) + 1
        sorted_organisms = sorted(organism_dist.items(), key=lambda x: x[1], reverse=True)[:10]

        activity_types = {}
        for t in report.get('targets_raw', []):
            at = t.get('activity_type', 'Unknown') or 'Unknown'
            activity_types[at] = activity_types.get(at, 0) + 1
        sorted_activities = sorted(activity_types.items(), key=lambda x: x[1], reverse=True)[:8]

        target_rows = ""
        for i, (gene, score) in enumerate(sorted_targets, 1):
            target_rows += f'<tr><td>{i}</td><td>{gene}</td><td>{score:.4f}</td></tr>\n'

        org_rows = ""
        for org, count in sorted_organisms:
            org_rows += f'<li><strong>{org}</strong>: {count} targets</li>\n'

        act_rows = ""
        for at, count in sorted_activities:
            act_rows += f'<li><strong>{at}</strong>: {count} records</li>\n'

        chart_target_priority = self._generate_chart_base64('target_priority', report)
        chart_organism = self._generate_chart_base64('organism', report)
        chart_activity = self._generate_chart_base64('activity', report)
        chart_kg_network = self._generate_chart_base64('kg_network', report)

        html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{compound_name_cn}网络药理学分析结果</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #1a5276, #2e86c1); color: white; padding: 40px 30px; border-radius: 12px; margin-bottom: 30px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }}
        h1 {{ font-size: 2.2em; margin-bottom: 8px; text-shadow: 1px 1px 3px rgba(0,0,0,0.3); }}
        .subtitle {{ font-size: 1.1em; opacity: 0.9; margin-bottom: 5px; }}
        .gen-time {{ font-size: 0.9em; opacity: 0.8; }}
        .section {{ background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        h2 {{ color: #1a5276; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #2e86c1; font-size: 1.5em; }}
        h3 {{ color: #2e86c1; margin: 15px 0 10px 0; font-size: 1.15em; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #2e86c1; transition: transform 0.2s; }}
        .stat-card:hover {{ transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #2e86c1; }}
        .stat-label {{ color: #666; margin-top: 5px; font-size: 0.9em; }}
        .compound-info {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 15px 0; }}
        .info-item {{ background: #f8f9fa; padding: 12px; border-radius: 6px; border-left: 3px solid #27ae60; }}
        .info-label {{ font-size: 0.8em; color: #888; text-transform: uppercase; }}
        .info-value {{ font-size: 1em; font-weight: 600; color: #2c3e50; word-break: break-all; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e9ecef; }}
        th {{ background-color: #f1f3f5; font-weight: 600; color: #1a5276; }}
        tr:hover {{ background-color: #f8f9fa; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; margin: 20px 0; }}
        .chart-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .chart-card img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .chart-title {{ margin-top: 10px; font-weight: 600; color: #2e86c1; font-size: 1em; }}
        .list-styled {{ list-style: none; margin: 15px 0; }}
        .list-styled li {{ padding: 10px 15px; background: #f8f9fa; margin-bottom: 8px; border-radius: 5px; border-left: 4px solid #2e86c1; }}
        .data-files {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }}
        .data-file {{ display: inline-block; padding: 10px 18px; background: linear-gradient(135deg, #2e86c1, #1a5276); color: white; text-decoration: none; border-radius: 6px; font-size: 0.9em; transition: all 0.2s; }}
        .data-file:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(46,134,193,0.3); color: white; }}
        .pipeline-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .pipeline-step {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-top: 3px solid #27ae60; }}
        .step-icon {{ font-size: 1.5em; margin-bottom: 8px; }}
        .step-title {{ font-weight: 600; color: #1a5276; margin-bottom: 5px; }}
        .step-desc {{ font-size: 0.85em; color: #666; }}
        footer {{ text-align: center; margin-top: 40px; padding: 20px; background: #1a5276; color: white; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{compound_name_cn}网络药理学分析结果</h1>
            <p class="subtitle">基于深度学习和图神经网络的多维度分析</p>
            <p class="gen-time">生成时间: {display_time}</p>
        </header>

        <div class="section">
            <h2>分析摘要</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_targets}</div>
                    <div class="stat-label">识别的靶点数量</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{unique_genes}</div>
                    <div class="stat-label">唯一基因数量</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{graph_nodes}</div>
                    <div class="stat-label">知识图谱节点</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{graph_edges}</div>
                    <div class="stat-label">知识图谱边数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{graph_relations}</div>
                    <div class="stat-label">关系类型数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{binding_affinity}</div>
                    <div class="stat-label">结合能 (kcal/mol)</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>化合物信息</h2>
            <div class="compound-info">
                <div class="info-item"><div class="info-label">名称</div><div class="info-value">{compound_name} ({compound_name_cn})</div></div>
                <div class="info-item"><div class="info-label">分子式</div><div class="info-value">{compound.get('molecular_formula', 'N/A')}</div></div>
                <div class="info-item"><div class="info-label">分子量</div><div class="info-value">{compound.get('molecular_weight', 'N/A')}</div></div>
                <div class="info-item"><div class="info-label">PubChem CID</div><div class="info-value">{compound.get('pubchem_cid', 'N/A')}</div></div>
                <div class="info-item"><div class="info-label">ChEMBL ID</div><div class="info-value">{compound.get('chembl_id', 'N/A')}</div></div>
                <div class="info-item"><div class="info-label">InChIKey</div><div class="info-value">{compound.get('inchikey', 'N/A')}</div></div>
                <div class="info-item" style="grid-column: 1 / -1;"><div class="info-label">SMILES</div><div class="info-value">{compound.get('smiles', 'N/A')}</div></div>
            </div>
        </div>

        <div class="section">
            <h2>靶点优先级分析</h2>
            <table>
                <thead><tr><th>排名</th><th>基因符号</th><th>活性得分</th></tr></thead>
                <tbody>{target_rows}</tbody>
            </table>
        </div>

        <div class="section">
            <h2>数据可视化</h2>
            <div class="charts-grid">
                <div class="chart-card">
                    <img src="data:image/png;base64,{chart_target_priority}" alt="靶点优先级分布图">
                    <div class="chart-title">靶点优先级分布图</div>
                </div>
                <div class="chart-card">
                    <img src="data:image/png;base64,{chart_organism}" alt="物种分布图">
                    <div class="chart-title">靶点物种分布图</div>
                </div>
                <div class="chart-card">
                    <img src="data:image/png;base64,{chart_activity}" alt="活性类型分布图">
                    <div class="chart-title">活性类型分布图</div>
                </div>
                <div class="chart-card">
                    <img src="data:image/png;base64,{chart_kg_network}" alt="知识图谱网络">
                    <div class="chart-title">知识图谱网络可视化</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>物种分布</h2>
            <ul class="list-styled">{org_rows}</ul>
        </div>

        <div class="section">
            <h2>活性类型统计</h2>
            <ul class="list-styled">{act_rows}</ul>
        </div>

        <div class="section">
            <h2>数据文件</h2>
            <div class="data-files">
                <a href="framework_output/data/processed/targets.csv" class="data-file">靶点数据 (CSV)</a>
                <a href="framework_output/data/processed/interactions.csv" class="data-file">互作网络数据 (CSV)</a>
                <a href="framework_output/data/processed/compound_info.csv" class="data-file">化合物信息 (CSV)</a>
                <a href="framework_output/data/processed/proteins.csv" class="data-file">蛋白质数据 (CSV)</a>
                <a href="framework_output/framework_results.json" class="data-file">完整结果 (JSON)</a>
                <a href="framework_output/logs/framework_run.log" class="data-file">运行日志 (LOG)</a>
            </div>
        </div>

        <div class="section">
            <h2>分析流程</h2>
            <div class="pipeline-steps">
                <div class="pipeline-step"><div class="step-icon">&#128218;</div><div class="step-title">数据收集</div><div class="step-desc">PubChem, ChEMBL, TCMSP, UniProt, STRING</div></div>
                <div class="pipeline-step"><div class="step-icon">&#129514;</div><div class="step-title">特征提取</div><div class="step-desc">分子描述符, ESM-2嵌入, 3D-CNN</div></div>
                <div class="pipeline-step"><div class="step-icon">&#127919;</div><div class="step-title">特征融合</div><div class="step-desc">Transformer多源特征融合</div></div>
                <div class="pipeline-step"><div class="step-icon">&#128300;</div><div class="step-title">分子对接</div><div class="step-desc">配体-蛋白质相互作用预测</div></div>
                <div class="pipeline-step"><div class="step-icon">&#128202;</div><div class="step-title">知识图谱</div><div class="step-desc">生物知识网络构建与分析</div></div>
            </div>
        </div>

        <footer>
            <p>&#169; 2026 {compound_name_cn}网络药理学分析系统</p>
            <p>基于深度学习和网络药理学技术</p>
        </footer>
    </div>
</body>
</html>'''

        dashboard_path = os.path.join(_BASE_DIR, "output_dashboard.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info("Dashboard HTML generated: %s", dashboard_path)

        report_path = os.path.join(self.output_dir, "report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info("Report HTML generated: %s", report_path)

        return dashboard_path

    def _generate_chart_base64(self, chart_type: str, report: Dict[str, Any]) -> str:
        if not MATPLOTLIB_AVAILABLE:
            return ""
        try:
            import io
            import base64

            if chart_type == 'target_priority':
                gene_scores = {}
                for t in report.get('targets_raw', []):
                    g = t.get('gene_symbol', '')
                    if g:
                        try:
                            score = abs(float(t.get('activity_value', 0))) if t.get('activity_value') else 0
                            gene_scores[g] = max(gene_scores.get(g, 0), score)
                        except (ValueError, TypeError):
                            pass
                if not gene_scores:
                    return ""
                sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)[:15]
                fig, ax = plt.subplots(figsize=(8, 5))
                genes, scores = zip(*sorted_genes)
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(genes)))
                bars = ax.barh(range(len(genes)), scores, color=colors, edgecolor='white', linewidth=0.5)
                ax.set_yticks(range(len(genes)))
                ax.set_yticklabels(genes, fontsize=9)
                ax.set_xlabel('Activity Score', fontsize=10)
                ax.set_title('Target Priority Distribution', fontsize=13, fontweight='bold', pad=10)
                ax.invert_yaxis()
                for bar, s in zip(bars, scores):
                    ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{s:.2f}', ha='left', va='center', fontsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()

            elif chart_type == 'organism':
                organism_dist = {}
                for t in report.get('targets_raw', []):
                    org = t.get('organism', 'Unknown') or 'Unknown'
                    organism_dist[org] = organism_dist.get(org, 0) + 1
                if not organism_dist:
                    return ""
                sorted_orgs = sorted(organism_dist.items(), key=lambda x: x[1], reverse=True)[:10]
                fig, ax = plt.subplots(figsize=(8, 5))
                orgs, counts = zip(*sorted_orgs)
                colors = plt.cm.Set2(np.linspace(0, 1, len(orgs)))
                wedges, texts, autotexts = ax.pie(counts, labels=None, autopct='%1.1f%%',
                                                    colors=colors, startangle=90, pctdistance=0.85)
                for t in autotexts:
                    t.set_fontsize(8)
                ax.legend(orgs, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
                ax.set_title('Target Organism Distribution', fontsize=13, fontweight='bold', pad=10)
                plt.tight_layout()

            elif chart_type == 'activity':
                activity_types = {}
                for t in report.get('targets_raw', []):
                    at = t.get('activity_type', 'Unknown') or 'Unknown'
                    activity_types[at] = activity_types.get(at, 0) + 1
                if not activity_types:
                    return ""
                sorted_acts = sorted(activity_types.items(), key=lambda x: x[1], reverse=True)[:8]
                fig, ax = plt.subplots(figsize=(8, 5))
                acts, counts = zip(*sorted_acts)
                colors = plt.cm.Paired(np.linspace(0, 1, len(acts)))
                bars = ax.bar(range(len(acts)), counts, color=colors, edgecolor='white', linewidth=0.5)
                ax.set_xticks(range(len(acts)))
                ax.set_xticklabels(acts, rotation=30, ha='right', fontsize=9)
                ax.set_ylabel('Count', fontsize=10)
                ax.set_title('Activity Type Distribution', fontsize=13, fontweight='bold', pad=10)
                for bar, c in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                            str(c), ha='center', va='bottom', fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()

            elif chart_type == 'kg_network':
                kg = report.get('knowledge_graph_summary', {})
                graph_data = kg.get('graph', {})
                nodes = graph_data.get('nodes', [])
                edges = graph_data.get('edges', [])
                kg_stats = kg.get('stats', {})
                n_nodes = kg_stats.get('num_nodes', 0)
                n_edges = kg_stats.get('num_edges', 0)
                if not nodes and not NETWORKX_AVAILABLE:
                    return ""
                fig, ax = plt.subplots(figsize=(8, 6))
                if NETWORKX_AVAILABLE and nodes:
                    G = nx.Graph()
                    for node in nodes[:50]:
                        G.add_node(node)
                    for edge in edges[:100]:
                        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                            G.add_edge(str(edge[0]), str(edge[1]))
                    if len(G.nodes()) > 0:
                        pos = nx.spring_layout(G, k=1.5 / (len(G.nodes()) ** 0.5), iterations=50, seed=42)
                        node_sizes = [max(50, 300 * G.degree(n) / max(dict(G.degree()).values(), default=1)) for n in G.nodes()]
                        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color='#2e86c1', alpha=0.7)
                        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5)
                        labels = {n: n[:6] for n in G.nodes() if G.degree(n) >= 2}
                        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6, font_color='#1a5276')
                    ax.set_title('Knowledge Graph Network', fontsize=13, fontweight='bold', pad=10)
                else:
                    ax.text(0.5, 0.5, f'KG: {n_nodes} nodes, {n_edges} edges\n(NetworkX unavailable)',
                            ha='center', va='center', fontsize=12, transform=ax.transAxes)
                    ax.set_title('Knowledge Graph', fontsize=13, fontweight='bold', pad=10)
                ax.axis('off')
                plt.tight_layout()
            else:
                return ""

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            logger.warning("Chart generation failed (%s): %s", chart_type, e)
            return ""

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