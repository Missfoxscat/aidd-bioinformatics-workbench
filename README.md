# AIDD Bioinformatics Workbench
AI-Driven Drug Discovery Transcriptomics Analysis Pipeline
乳腺癌转录组差异基因挖掘 + 机器学习AI诊断全闭环生信流水线
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)]

## 📌 项目简介
本仓库是一套标准化、可复用的**肿瘤转录组+AI制药**分析闭环工具链，基于GEO公共数据集GSE183947（乳腺癌FPKM表达矩阵）实现完整分析流程：
1. 批量表达矩阵读取、自动分组（正常CAP/乳腺癌CA）
2. 差异表达基因DEG筛选（t检验、log2FC、p值多重过滤）
3. 候选药物靶点自动筛选 & 格式化Excel临床报告输出
4. 随机森林机器学习肿瘤诊断模型构建
    - 分层划分训练/测试集、混淆矩阵可视化
    - 5折分层交叉验证ROC曲线绘制
    - 置换检验Permutation Test验证靶点显著性
5. Enrichr在线KEGG通路富集分析+气泡图可视化
6. 封装通用底层生信工具库，支持一键复用至其他GEO/TCGA数据集

本项目适配AI制药交叉赛道学习、本科大创、硕士横向药企项目、简历作品集展示，兼顾**生物信息学统计**与**工业级机器学习工程化**，规避新手环境配置坑，支持云端Colab/本地服务器双运行。

## 📂 仓库目录结构
aidd-bioinformatics-workbench/
├── notebooks/          # Jupyter交互式分析笔记
├── scripts/           # 完整可运行流水线脚本 pipeline.py
├── src/                # 底层通用工具库 my_bio_tools.py
├── data/               # 数据集说明、GSE183947下载指引
├── results/            # 自动输出：Excel靶点报告、ROC/混淆矩阵/富集图
├── hypotheses/         # 科学猜测与验证记录 ← 新增这行
├── README.md
├── LICENSE
└── .gitignore

## 🧰 环境依赖
### 基础科学计算与生信库
```bash
pip install numpy pandas scipy matplotlib seaborn openpyxl requests gzip
pip install scikit-learn
