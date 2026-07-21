# -*- coding: utf-8 -*-
"""
pipeline.py
乳腺癌差异基因挖掘与机器学习分类器标准闭环流水线（高性能优化版）
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from openpyxl.styles import PatternFill, Font

# 导入自定义底层工具库
import my_bio_tools as mbt

def create_output_dir(dir_name="results"):
    """创建统一的输出结果文件夹"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"📁 已自动创建结果输出文件夹：{dir_name}/")
    return dir_name

def main():
    print("=" * 60)
    print("🚀 乳腺癌多组学数据挖掘与机器学习预测管线（正式启动）")
    print("=" * 60)
    
    # 0. 初始化输出路径
    out_dir = create_output_dir()
    data_path = os.path.join("data", "GSE183947_fpkm.csv.gz")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误：未在 ./data/ 文件夹找到数据集文件 '{data_path}'！")
        print("💡 请将 GSE183947_fpkm.csv.gz 放入项目下 data 文件夹，再重新运行。")
    sys.exit(1)
        
    # 1. 数据载入与全自动样本分组[span_0](start_span)[span_0](end_span)
    df_real = mbt.load_data(data_path)
    control_groups = [col for col in df_real.columns if col.startswith("CAP")]
    treat_groups = [col for col in df_real.columns if col.startswith("CA.")]
    
    print(f"📊 自动分组完成：对照组(CAP)共 {len(control_groups)} 样本，乳腺癌组(CA)共 {len(treat_groups)} 样本。")
    
    # 2. 运行差异表达分析[span_1](start_span)[span_1](end_span)
    df_analyzed = mbt.run_deg_analysis(df_real, control_groups, treat_groups)
    
    # 3. 筛选黄金药研靶点并输出精装版 Excel 报告[span_2](start_span)[span_2](end_span)
    final_targets = df_analyzed[
        (df_analyzed["log2_FC"] > 2) & 
        (df_analyzed["p_value"] < 0.01) & 
        (df_analyzed["Mean_Treat"] > 0.5)
    ].copy()
    final_targets = final_targets.sort_values(by="log2_FC", ascending=False)
    
    report_df = final_targets[["Mean_Control", "Mean_Treat", "Fold_Change", "log2_FC", "p_value"]].copy()
    report_df.columns = ["Normal Mean (FPKM)", "Breast Cancer Mean (FPKM)", "Fold Change", "log2(Fold Change)", "p-value (t-test)"]
    
    excel_file = os.path.join(out_dir, "GSE183947_Breast_Cancer_Target_Report.xlsx")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name="Candidate Targets", index_label="Gene Symbol")
        worksheet = writer.sheets["Candidate Targets"]
        
        # 自动调整列宽
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)
            
        highlight_fill = PatternFill(start_color="E6F7FF", end_color="E6F7FF", fill_type="solid")
        header_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
        bold_font = Font(name="Arial", size=10, bold=True)
        
        max_row = min(len(report_df) + 1, 11)
        for row_idx in range(2, max_row + 1):
            for col_idx in range(1, 7):
                worksheet.cell(row=row_idx, column=col_idx).fill = highlight_fill
        for col_idx in range(1, 7):
            cell = worksheet.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = bold_font
            
    print(f"💾 临床级精装靶点报告已输出至：{excel_file}")

    # 4. 机器学习特征提取与建模 (随机森林模型)[span_3](start_span)[span_3](end_span)
    top_10_genes = report_df.index[:10].tolist()
    X = df_real.loc[top_10_genes].T
    y = np.array([1 if col.startswith("CA.") else 0 for col in X.index])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 此处为单次建模训练，直接开启全核多线程暴击加速[span_4](start_span)[span_4](end_span)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"🏆 AI 诊断测试集闭卷考试得分: {accuracy_score(y_test, y_pred) * 100:.2f}% 准确率！")
    
    # 5. 可视化：混淆矩阵[span_5](start_span)[span_5](end_span)
    plt.figure(figsize=(6, 5), dpi=150)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Cancer"], yticklabels=["Normal", "Cancer"], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title("AI Diagnosis Confusion Matrix", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_confusion_matrix.png"), dpi=300)
    plt.close()

    # 6. 5折交叉验证 ROC 评估[span_6](start_span)[span_6](end_span)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(7, 6), dpi=150)
    
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.iloc[train], y[train])
        probas_ = classifier.predict_proba(X.iloc[test])
        fpr, tpr, _ = roc_curve(y[test], probas_[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f'Fold {fold + 1} (AUC = {aucs[-1]:.2f})')
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='#bfbfbf', label='Chance (AUC = 0.50)')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='#1890ff', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('5-Fold Cross-Validation ROC Curves')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_cross_validation_roc.png"), dpi=300)
    plt.close()

    # 7. 置换检验对决（Permutation Test）[span_7](start_span)[span_7](end_span)
    n_iters = 100
    negative_gene_pool = df_real.index.difference(top_10_genes).tolist()
    random_aucs = []
    
    print(f"🎲 正在执行 {n_iters} 次置换检验对决（优化版资源并发控制）...")
    for i in range(n_iters):
        random_genes = np.random.choice(negative_gene_pool, size=10, replace=False)
        X_random = df_real.loc[random_genes].T
        y_random = np.array([1 if col.startswith("CA.") else 0 for col in X_random.index])
        
        # 🌟 技术两亮点亮点：显式指定内层 n_jobs=1，避免多进程套多线程导致 CPU 上下文切换的严重内耗[span_8](start_span)[span_8](end_span)
        clf_random = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        
        # 🌟 外层交叉验证全核轰炸 (n_jobs=-1)，5折同时推进，计算效率达到最大化[span_9](start_span)[span_9](end_span)
        scores = cross_val_score(clf_random, X_random, y_random, cv=5, scoring='roc_auc', n_jobs=-1)
        random_aucs.append(scores.mean())
        
    better_than_gold = np.sum(np.array(random_aucs) >= mean_auc)
    decimals = max(int(np.log10(n_iters)), 2)
    p_label = f"p < {1/n_iters:.{decimals}f}" if better_than_gold == 0 else f"p = {(better_than_gold+1)/(n_iters+1):.{decimals+1}f}"
    
    plt.figure(figsize=(8, 5), dpi=150)
    sns.histplot(random_aucs, bins=15, kde=True, color='#8c8c8c', alpha=0.5, label='Random Genes Pool')
    plt.axvline(mean_auc, color='#f5222d', linestyle='--', linewidth=2, label=f'Gold Targets (AUC = {mean_auc:.3f})')
    plt.title(f'Empirical Hypothesis Testing ({p_label})')
    plt.xlabel('5-Fold CV AUC Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_permutation_test.png"), dpi=300)
    plt.close()

    # 8. 加装异常捕捉的 Enrichr 联网 KEGG 富集分析[span_10](start_span)[span_10](end_span)
    print("🌐 正在尝试与 Enrichr 数据库通信进行通路富集分析...")
    try:
        gene_list = report_df.index.tolist()
        genes_str = '\n'.join(gene_list)
        
        response = requests.post('https://maayanlab.cloud/Enrichr/addList', files={'list': (None, genes_str), 'description': (None, 'Gold Targets')}, timeout=15)
        if response.status_code == 200:
            user_list_id = response.json()['userListId']
            res = requests.get(f'https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType=KEGG_2021_Human', timeout=15)
            if res.status_code == 200:
                results = res.json()['KEGG_2021_Human']
                pathway_data = [{'Pathway': e[1], 'neg_log10_p': -np.log10(e[2]), 'Count': len(e[5])} for e in results[:15]]
                df_enrich = pd.DataFrame(pathway_data).sort_values(by='neg_log10_p', ascending=True)
                
                plt.figure(figsize=(9, 6), dpi=150)
                scatter = plt.scatter(df_enrich['neg_log10_p'], df_enrich['Pathway'], s=df_enrich['Count'] * 150, c=df_enrich['neg_log10_p'], cmap='viridis', alpha=0.85, edgecolors='#595959')
                plt.title("KEGG Pathway Enrichment Top 15")
                plt.xlabel("-log10 (p-value)")
                cbar = plt.colorbar(scatter, shrink=0.7)
                cbar.set_label("Significance")
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "04_kegg_enrichment.png"), dpi=300)
                plt.close()
                print(f"💾 KEGG 富集分析气泡图已保存。")
            else:
                print("⚠️ 提示：Enrichr 结果拉取失败，已跳过富集分析。")
        else:
            print("⚠️ 提示：Enrichr 基因列表提交失败，已跳过富集分析。")
    except Exception as network_error:
        print(f"⚠️ 网络安全拦截：Enrichr 联网请求超时或断开 ({network_error})。")
        print("💡 提示：核心上游机器学习与核心差异分析已完美结束并保存，不影响整体闭环。")

    print("\n" + "=" * 60)
    print(f"🎉 整个机器学习管线顺利运行完毕！全套成果已完美保存在 './{out_dir}/' 目录下！")
    print("=" * 60)

if __name__ == "__main__":
    main()