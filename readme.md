# 木犀草素数据分析与预测系统

基于网络药理学的木犀草素（Luteolin）多维度数据分析与预测框架。系统从 PubChem、ChEMBL、TCMSP 等公共数据库中获取化合物与靶点数据，结合深度学习和图神经网络技术，实现靶点预测、分子对接模拟和知识图谱分析。

---

## 项目概述

### 系统目标

本系统旨在通过网络药理学方法，对天然黄酮类化合物 **木犀草素（Luteolin）** 进行系统的药理活性分析，实现以下目标：

- **靶点预测**：基于多源数据融合，预测木犀草素的潜在作用靶点
- **分子对接**：模拟化合物与靶点蛋白的结合亲和力
- **知识图谱**：构建化合物-靶点-蛋白质关联网络，揭示作用机制
- **可视化报告**：自动生成包含数据图表的综合分析报告

### 分析流程

```
数据采集 ──> 特征提取 ──> 特征融合 ──> 分子对接 ──> 知识图谱 ──> 报告生成
(PubChem)    (RDKit)    (Transformer)   (AutoDock)    (NetworkX)    (HTML+图表)
(ChEMBL)     (ESM-2)
(TCMSP)      (GNN/CNN)
(UniProt)    (3D-CNN)
(STRING)
```

### 技术栈

```
数据层:     Pandas + requests + PubChemPy
                ↓
分子处理:   RDKit（分子描述符、指纹、构象生成）
                ↓
蛋白模型:   ESM-2 (HuggingFace Transformers)
                ↓
GNN模块:    PyTorch Geometric (GAT/GCN) — 可选
CNN模块:    DeepChem (3D-CNN) — 可选
                ↓
特征融合:   Transformer (Multi-Head Attention)
                ↓
分子对接:   AutoDock Vina — 外部工具
                ↓
知识图谱:   NetworkX + PyKEEN/DGL-KE — 可选
                ↓
可视化:     Matplotlib + Seaborn → HTML Dashboard
```

---

## 环境要求

### 基础环境

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11、Linux、macOS |
| Python | **3.10.8**（推荐，其他 3.10.x 也可） |
| 包管理器 | Conda（推荐 Anaconda 或 Miniconda） |
| 硬盘空间 | 至少 2GB（含模型缓存） |
| 内存 | 至少 8GB（推荐 16GB） |
| GPU（可选） | CUDA 兼容 GPU 加速模型推理 |

---

## 安装步骤

### 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt
```

> **注意**：如果使用 Anaconda 的 base 环境或其他自定义环境，请确保激活正确的环境后再执行以上命令。

### 验证安装

```bash
# 验证核心依赖是否安装成功
python -c "import numpy; print('numpy:', numpy.__version__); import pandas; print('pandas:', pandas.__version__); import torch; print('torch:', torch.__version__); import rdkit; from rdkit import Chem; print('rdkit: OK'); import matplotlib; print('matplotlib:', matplotlib.__version__)"
```

预期输出类似：

```
numpy: 1.23.5
pandas: 2.0.3
torch: 2.5.1+cu121
rdkit: OK
matplotlib: 3.7.2
```
---

## 基本使用方法

### 快速运行

激活环境后，在项目根目录执行：

```bash
# 如果使用了conda虚拟环境
conda activate luteolin

# 运行完整分析流程
python main.py
```

系统将自动执行以下流程，**无需任何手动配置**：

1. **数据采集** — 从 PubChem、ChEMBL、TCMSP、UniProt、STRING 获取数据
2. **配体特征提取** — 使用 RDKit 计算分子描述符、指纹、生成构象
3. **蛋白质特征提取** — 使用 ESM-2 模型提取蛋白质序列嵌入
4. **GNN 分析** — 基于蛋白质互作网络计算中心性指标
5. **CNN 分析** — 基于分子三维网格提取深度特征
6. **特征融合** — 使用 Transformer 多注意力机制融合多模态特征
7. **分子对接** — 模拟化合物与靶点的结合
8. **知识图谱** — 构建化合物-靶点-蛋白质关联网络
9. **报告生成** — 生成综合 HTML 报告和可视化图表

### 查看结果

运行完成后，打开项目目录下的 `output_dashboard.html`：

```
/
└── output_dashboard.html     ← 用浏览器打开此文件
```

也可以直接查看输出目录中的文件：

```
/framework_output/
├── output/
│   └── report.html           ← 综合报告
├── data/                     ← 原始与处理后数据
├── logs/
│   └── framework_run.log     ← 详细运行日志
└── framework_results.json    ← 完整结果（JSON 格式）
```

### 输出目录结构

```
framework_output/
├── data/                          # 数据采集结果
│   ├── raw/                       # PubChem, ChEMBL, UniProt 等原始数据
│   └── processed/                 # 清洗后的 CSV/JSON 数据
│       ├── targets.csv            # 靶点数据
│       ├── interactions.csv       # 蛋白质互作数据
│       ├── compound_info.csv      # 化合物信息
│       └── proteins.csv           # 蛋白质数据
├── ligand/                        # 配体特征
│   └── descriptors/               # 分子描述符与指纹（JSON）
├── protein/                       # 蛋白质特征
│   ├── embeddings/                # ESM-2 嵌入向量
│   └── structures/                # 蛋白质结构文件
├── docking/                       # 分子对接结果
├── knowledge_graph/               # 知识图谱数据
├── output/                        # 可视化输出
│   ├── images/                    # 图表（PNG 格式）
│   │   ├── analysis_plots.png     # 分析综合图
│   │   ├── target_priority.png    # 靶点优先级图
│   │   └── pathway_analysis.png   # 通路分析图
│   └── report.html                # 综合报告
├── logs/                          # 运行日志
│   └── framework_run.log          # 完整运行日志
└── framework_results.json         # 所有结果的汇总 JSON
```

### 输出 Dashboard 内容

`output_dashboard.html` 包含以下模块：

| 模块 | 内容说明 |
|------|----------|
| 分析摘要 | 统计卡片：靶点数量、蛋白质数量、知识图谱节点/边数、对接亲和力 |
| 化合物信息 | 名称、分子式、分子量、SMILES、PubChem CID |
| 靶点优先级 | 排名表格：基因符号、活性类型、活性值、物种、综合得分 |
| 数据可视化 | 4 种图表：靶点优先级分布、物种分布、活性类型分布、知识图谱网络 |
| 物种分布 | 按物种统计的靶点数量列表 |
| 活性类型 | 按活性类型统计的靶点数量列表 |
| 数据下载 | CSV/JSON/LOG 文件的下载链接 |

---

## ESM-2 模型配置

### 默认模型

框架默认使用轻量级模型 `facebook/esm2_t6_8M_UR50D`（8M 参数，约 30MB），适合在 CPU 环境下快速运行。

### 切换模型

编辑 main.py，找到特征提取部分的初始化代码，修改 `esm_model_name` 参数：

```python
# 使用 35M 模型（平衡精度与速度）
self.protein_extractor = ProteinFeatureExtractor(
    output_dir=os.path.join(self.output_dir, "protein"),
    esm_model_name="facebook/esm2_t12_35M_UR50D"
)

# 使用 650M 模型（最高精度，需 GPU）
self.protein_extractor = ProteinFeatureExtractor(
    output_dir=os.path.join(self.output_dir, "protein"),
    esm_model_name="facebook/esm2_t33_650M_UR50D"
)
```
### 添加降级模型

```python
# 当主模型下载失败时，自动使用备选模型
self.protein_extractor.add_fallback_model(
    "facebook/esm2_t33_650M_UR50D",
    "AI-ModelScope/esm2_t33_650M_UR50D"
)
```

> **注意**：模型缓存目录为 `/.model_cache/`，每次运行不会自动清理该目录，避免重复下载。

---

---

## 许可

本项目仅供学术研究使用。木犀草素（Luteolin）是一种天然黄酮类化合物，本系统的分析结果仅供参考，不构成医学建议。

---

*文档版本: v1.0 | 最后更新: 2026-05-19*
