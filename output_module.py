"""
输出层模块 (Output Module)
用于生成靶点发现报告、作用机制解析和药物开发建议

功能:
- 靶点发现报告生成
- 作用机制解析
- 药物开发建议
- 结果可视化和交互
- 报告导出
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Matplotlib or Seaborn not installed. Visualization will be limited.")


class OutputModule:
    """输出层模块"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        self.results = {}
    
    def generate_target_discovery_report(self, gnn_results: Dict[str, Any], 
                                       docking_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成靶点发现报告"""
        print("\n1. Generating target discovery report...")
        
        report = {
            'title': 'Target Discovery Report',
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'key_targets': [],
            'docking_results': {},
            'confidence_scores': {}
        }
        
        # 从GNN结果提取关键靶点
        if 'key_targets' in gnn_results:
            key_targets = gnn_results['key_targets']
            report['key_targets'] = key_targets[:10]  # 取前10个
            report['summary']['total_targets'] = len(key_targets)
            
            # 计算置信度评分
            if key_targets:
                max_score = key_targets[0][1]
                confidence_scores = {}
                for target, score in key_targets:
                    confidence = (score / max_score) * 100
                    confidence_scores[target] = confidence
                report['confidence_scores'] = confidence_scores
        
        # 从对接结果提取信息
        if 'docking_results' in docking_results:
            docking = docking_results['docking_results']
            report['docking_results'] = {
                'best_affinity': docking.get('binding_affinity', 0),
                'average_affinity': docking.get('summary', {}).get('average_affinity', 0),
                'num_modes': docking.get('summary', {}).get('num_modes', 0)
            }
        
        # 分析靶点优先级
        report['target_prioritization'] = self._prioritize_targets(report['key_targets'], report['docking_results'])
        
        print(f"  ✓ Generated target discovery report with {len(report['key_targets'])} targets")
        return report
    
    def _prioritize_targets(self, key_targets: List[Tuple[str, float]], 
                          docking_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优先级排序靶点"""
        prioritized = []
        
        for target, score in key_targets:
            priority_score = score
            
            # 如果有对接结果，结合亲和力评分
            if docking_results.get('best_affinity'):
                affinity = docking_results['best_affinity']
                # 亲和力越负（结合越强），得分越高
                affinity_score = abs(affinity) / 10  # 归一化到0-1范围
                priority_score = (priority_score + affinity_score) / 2
            
            prioritized.append({
                'target': target,
                'gnn_score': score,
                'priority_score': priority_score,
                'priority': self._get_priority_level(priority_score)
            })
        
        # 按优先级排序
        prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
        return prioritized
    
    def _get_priority_level(self, score: float) -> str:
        """根据分数获取优先级级别"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def analyze_mechanism_of_action(self, knowledge_graph_results: Dict[str, Any], 
                                  pathway_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """分析作用机制"""
        print("\n2. Analyzing mechanism of action...")
        
        analysis = {
            'title': 'Mechanism of Action Analysis',
            'generated_at': datetime.now().isoformat(),
            'pathways': [],
            'key_interactions': [],
            'biological_functions': []
        }
        
        # 从知识图谱结果提取通路信息
        if 'graph' in knowledge_graph_results:
            graph = knowledge_graph_results['graph']
            edges = graph.get('edges', [])
            
            # 提取通路信息
            pathways = {}
            for edge in edges:
                if edge['type'] == 'in_pathway':
                    pathway = edge['target']
                    if pathway not in pathways:
                        pathways[pathway] = []
                    pathways[pathway].append(edge['source'])
            
            for pathway, targets in pathways.items():
                analysis['pathways'].append({
                    'pathway': pathway,
                    'targets': targets,
                    'num_targets': len(targets)
                })
        
        # 分析关键相互作用
        if 'predictions' in knowledge_graph_results:
            analysis['key_interactions'] = knowledge_graph_results['predictions'][:5]
        
        # 推断生物学功能
        analysis['biological_functions'] = self._infer_biological_functions(analysis['pathways'])
        
        print(f"  ✓ Analyzed mechanism of action with {len(analysis['pathways'])} pathways")
        return analysis
    
    def _infer_biological_functions(self, pathways: List[Dict[str, Any]]) -> List[str]:
        """推断生物学功能"""
        functions = []
        
        # 基于通路推断功能
        pathway_functions = {
            'PI3K-AKT signaling pathway': ['Cell proliferation', 'Cell survival', 'Apoptosis regulation'],
            'MAPK signaling pathway': ['Cell growth', 'Differentiation', 'Stress response'],
            'NF-kappa B signaling pathway': ['Inflammation', 'Immune response', 'Cell survival'],
            'Wnt signaling pathway': ['Development', 'Cell fate determination', 'Tumorigenesis'],
            'JAK-STAT signaling pathway': ['Cytokine signaling', 'Immune response', 'Cell growth']
        }
        
        for pathway_info in pathways:
            pathway_name = pathway_info['pathway']
            for key_pathway, pathway_functions_list in pathway_functions.items():
                if key_pathway.lower() in pathway_name.lower():
                    functions.extend(pathway_functions_list)
        
        # 去重
        return list(set(functions))
    
    def generate_drug_development_suggestions(self, compound_data: Dict[str, Any], 
                                           target_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成药物开发建议"""
        print("\n3. Generating drug development suggestions...")
        
        suggestions = {
            'title': 'Drug Development Suggestions',
            'generated_at': datetime.now().isoformat(),
            'lead_optimization': [],
            'formulation': [],
            'preclinical_studies': [],
            'clinical_development': []
        }
        
        # 基于化合物性质的优化建议
        if compound_data:
            mol_weight = compound_data.get('molecular_weight', 0)
            logp = compound_data.get('logp', 0)
            tpsa = compound_data.get('tpsa', 0)
            
            # 先导化合物优化
            optimization_suggestions = []
            if mol_weight > 500:
                optimization_suggestions.append('Reduce molecular weight for better bioavailability')
            if logp > 5:
                optimization_suggestions.append('Reduce lipophilicity to improve solubility')
            if tpsa < 70:
                optimization_suggestions.append('Increase polar surface area to improve water solubility')
            
            if optimization_suggestions:
                suggestions['lead_optimization'] = optimization_suggestions
            else:
                suggestions['lead_optimization'].append('Compound properties are within favorable range')
        
        # 制剂建议
        suggestions['formulation'].extend([
            'Consider oral formulation if solubility permits',
            'Evaluate encapsulation strategies for improved delivery',
            'Investigate prodrug approaches to enhance bioavailability'
        ])
        
        # 临床前研究建议
        suggestions['preclinical_studies'].extend([
            'Perform in vitro efficacy studies on key targets',
            'Evaluate cytotoxicity and safety profiles',
            'Conduct pharmacokinetic studies in animal models',
            'Investigate drug-drug interactions'
        ])
        
        # 临床开发建议
        suggestions['clinical_development'].extend([
            'Phase I: Evaluate safety and pharmacokinetics in healthy volunteers',
            'Phase II: Assess efficacy in patient populations',
            'Phase III: Confirm efficacy in larger patient cohorts',
            'Monitor long-term safety and adverse effects'
        ])
        
        print("  ✓ Generated drug development suggestions")
        return suggestions
    
    def create_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """创建可视化"""
        print("\n4. Creating visualizations...")
        
        visualizations = {}
        
        if not VISUALIZATION_AVAILABLE:
            print("  Warning: Visualization libraries not available. Skipping visualizations.")
            return visualizations
        
        try:
            # 靶点优先级可视化
            if 'target_discovery' in results:
                prioritized = results['target_discovery'].get('target_prioritization', [])
                if prioritized:
                    filepath = self._plot_target_priority(prioritized)
                    visualizations['target_priority'] = filepath
            
            # 通路分析可视化
            if 'mechanism_analysis' in results:
                pathways = results['mechanism_analysis'].get('pathways', [])
                if pathways:
                    filepath = self._plot_pathway_analysis(pathways)
                    visualizations['pathway_analysis'] = filepath
            
            # 对接结果可视化
            if 'docking_results' in results:
                docking = results['docking_results'].get('docking_results', {})
                if docking:
                    filepath = self._plot_docking_results(docking)
                    visualizations['docking_results'] = filepath
            
            print(f"  ✓ Created {len(visualizations)} visualizations")
            return visualizations
            
        except Exception as e:
            print(f"  Error creating visualizations: {e}")
            return visualizations
    
    def _plot_target_priority(self, prioritized_targets: List[Dict[str, Any]]) -> str:
        """绘制靶点优先级图"""
        filepath = os.path.join(self.output_dir, "visualizations", "target_priority.png")
        
        try:
            targets = [t['target'] for t in prioritized_targets[:10]]
            scores = [t['priority_score'] for t in prioritized_targets[:10]]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=scores, y=targets, palette='viridis')
            plt.title('Target Priority Ranking')
            plt.xlabel('Priority Score')
            plt.ylabel('Target')
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"  Error plotting target priority: {e}")
            return ""
    
    def _plot_pathway_analysis(self, pathways: List[Dict[str, Any]]) -> str:
        """绘制通路分析图"""
        filepath = os.path.join(self.output_dir, "visualizations", "pathway_analysis.png")
        
        try:
            pathway_names = [p['pathway'] for p in pathways]
            target_counts = [p['num_targets'] for p in pathways]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=target_counts, y=pathway_names, palette='muted')
            plt.title('Pathway Analysis')
            plt.xlabel('Number of Targets')
            plt.ylabel('Pathway')
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"  Error plotting pathway analysis: {e}")
            return ""
    
    def _plot_docking_results(self, docking_results: Dict[str, Any]) -> str:
        """绘制对接结果图"""
        filepath = os.path.join(self.output_dir, "visualizations", "docking_results.png")
        
        try:
            modes = docking_results.get('modes', [])
            if modes:
                affinities = [mode['affinity'] for mode in modes]
                modes = [mode['mode'] for mode in modes]
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x=modes, y=affinities, palette='coolwarm')
                plt.title('Docking Binding Affinities')
                plt.xlabel('Mode')
                plt.ylabel('Binding Affinity (kcal/mol)')
                plt.tight_layout()
                plt.savefig(filepath, dpi=300)
                plt.close()
                
                return filepath
            
        except Exception as e:
            print(f"  Error plotting docking results: {e}")
        
        return ""
    
    def export_report(self, results: Dict[str, Any], filename: str = "comprehensive_report.json") -> str:
        """导出综合报告"""
        filepath = os.path.join(self.output_dir, "reports", filename)
        
        try:
            report = {
                'report_title': 'Comprehensive Network Pharmacology Report',
                'generated_at': datetime.now().isoformat(),
                'target_discovery': results.get('target_discovery', {}),
                'mechanism_analysis': results.get('mechanism_analysis', {}),
                'drug_development': results.get('drug_development', {}),
                'visualizations': results.get('visualizations', {})
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Exported comprehensive report to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  Error exporting report: {e}")
            return ""
    
    def export_csv_data(self, results: Dict[str, Any]) -> Dict[str, str]:
        """导出CSV数据"""
        csv_files = {}
        
        try:
            # 导出靶点数据
            if 'target_discovery' in results:
                prioritized = results['target_discovery'].get('target_prioritization', [])
                if prioritized:
                    df = pd.DataFrame(prioritized)
                    filepath = os.path.join(self.output_dir, "data", "targets.csv")
                    df.to_csv(filepath, index=False, encoding='utf-8-sig')
                    csv_files['targets'] = filepath
            
            # 导出通路数据
            if 'mechanism_analysis' in results:
                pathways = results['mechanism_analysis'].get('pathways', [])
                if pathways:
                    df = pd.DataFrame(pathways)
                    filepath = os.path.join(self.output_dir, "data", "pathways.csv")
                    df.to_csv(filepath, index=False, encoding='utf-8-sig')
                    csv_files['pathways'] = filepath
            
            print(f"  ✓ Exported {len(csv_files)} CSV files")
            return csv_files
            
        except Exception as e:
            print(f"  Error exporting CSV data: {e}")
            return csv_files
    
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """生成摘要"""
        summary = []
        summary.append("# Network Pharmacology Analysis Summary")
        summary.append("")
        summary.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # 靶点发现摘要
        if 'target_discovery' in results:
            target_report = results['target_discovery']
            summary.append("## Target Discovery")
            summary.append(f"Total targets identified: {target_report.get('summary', {}).get('total_targets', 0)}")
            summary.append("Top 5 targets:")
            for i, (target, score) in enumerate(target_report.get('key_targets', [])[:5]):
                summary.append(f"  {i+1}. {target}: {score:.4f}")
            summary.append("")
        
        # 作用机制摘要
        if 'mechanism_analysis' in results:
            mechanism = results['mechanism_analysis']
            summary.append("## Mechanism of Action")
            summary.append(f"Pathways involved: {len(mechanism.get('pathways', []))}")
            if mechanism.get('biological_functions'):
                summary.append("Biological functions:")
                for func in mechanism['biological_functions'][:5]:
                    summary.append(f"  - {func}")
            summary.append("")
        
        # 药物开发摘要
        if 'drug_development' in results:
            drug_dev = results['drug_development']
            summary.append("## Drug Development")
            if drug_dev.get('lead_optimization'):
                summary.append("Lead optimization suggestions:")
                for suggestion in drug_dev['lead_optimization'][:3]:
                    summary.append(f"  - {suggestion}")
            summary.append("")
        
        # 可视化摘要
        if 'visualizations' in results:
            viz = results['visualizations']
            summary.append("## Visualizations")
            for name, path in viz.items():
                summary.append(f"  - {name}: {path}")
            summary.append("")
        
        return "\n".join(summary)
    
    def run_output_generation(self, gnn_results: Dict[str, Any], 
                            docking_results: Dict[str, Any],
                            knowledge_graph_results: Dict[str, Any],
                            compound_data: Dict[str, Any],
                            target_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行完整的输出生成流程"""
        print("\n" + "=" * 60)
        print("  OUTPUT GENERATION")
        print("  输出生成")
        print("=" * 60)
        
        # 生成靶点发现报告
        target_report = self.generate_target_discovery_report(gnn_results, docking_results)
        
        # 分析作用机制
        mechanism_analysis = self.analyze_mechanism_of_action(knowledge_graph_results)
        
        # 生成药物开发建议
        drug_development = self.generate_drug_development_suggestions(compound_data, target_data)
        
        # 整合结果
        results = {
            'target_discovery': target_report,
            'mechanism_analysis': mechanism_analysis,
            'drug_development': drug_development
        }
        
        # 创建可视化
        results['visualizations'] = self.create_visualizations(results)
        
        # 导出报告
        self.export_report(results)
        
        # 导出CSV数据
        self.export_csv_data(results)
        
        # 生成摘要
        summary = self.generate_summary(results)
        
        # 保存摘要
        summary_path = os.path.join(self.output_dir, "reports", "summary.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print("  ✓ Saved summary to summary.md")
        print("\n" + "=" * 60)
        print("  OUTPUT GENERATION COMPLETED")
        print("=" * 60)
        
        return results


def main():
    # 测试输出模块
    # 模拟数据
    gnn_results = {
        'key_targets': [
            ('AKT1', 0.9),
            ('PIK3CA', 0.85),
            ('PTEN', 0.8),
            ('MAPK1', 0.75),
            ('ERK1', 0.7)
        ]
    }
    
    docking_results = {
        'docking_results': {
            'binding_affinity': -8.5,
            'summary': {
                'best_affinity': -8.5,
                'average_affinity': -8.17,
                'num_modes': 3
            },
            'modes': [
                {'mode': 1, 'affinity': -8.5},
                {'mode': 2, 'affinity': -8.2},
                {'mode': 3, 'affinity': -7.8}
            ]
        }
    }
    
    knowledge_graph_results = {
        'graph': {
            'edges': [
                {'source': 'AKT1', 'target': 'pathway_1', 'type': 'in_pathway'},
                {'source': 'PIK3CA', 'target': 'pathway_1', 'type': 'in_pathway'},
                {'source': 'PTEN', 'target': 'pathway_1', 'type': 'in_pathway'}
            ]
        },
        'predictions': []
    }
    
    compound_data = {
        'name': 'Luteolin',
        'molecular_weight': 286.24,
        'logp': 2.13,
        'tpsa': 107.22
    }
    
    target_data = [
        {'target_name': 'Protein Kinase B', 'gene_symbol': 'AKT1'},
        {'target_name': 'Phosphoinositide-3-Kinase', 'gene_symbol': 'PIK3CA'},
        {'target_name': 'Phosphatase And Tensin Homolog', 'gene_symbol': 'PTEN'}
    ]
    
    # 初始化输出模块
    output = OutputModule(output_dir="./test_output")
    
    # 运行输出生成
    results = output.run_output_generation(
        gnn_results, docking_results, knowledge_graph_results, compound_data, target_data
    )
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("  REPORT SUMMARY")
    print("=" * 60)
    print(output.generate_summary(results))


if __name__ == "__main__":
    main()