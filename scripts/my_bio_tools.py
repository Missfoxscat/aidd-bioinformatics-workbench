# -*- coding: utf-8 -*-
"""
my_bio_tools.py
底层生物信息学数据分析通用函数库
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import gzip

def load_data(file_path):
    """一键读取数据：自适应识别压缩包及分隔符"""
    print(f"📄 正在从库房读取数据: {file_path} ...")
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                first_line = f.readline()
        else:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                
        sep = '\t' if '\t' in first_line else ','
        print(f"🔍 自动检测到分隔符为: '{'Tab (TSV)' if sep == '\\t' else 'Comma (CSV)'}'")
    except Exception as e:
        sep = '\t'
        print(f"⚠️ 分隔符自动侦察失败 ({e})，采用默认制表符 '\\t'")
        
    df = pd.read_csv(file_path, sep=sep, compression="gzip" if file_path.endswith('.gz') else None, index_col=0)
    print(f"✅ 读取成功！共包含 {df.shape[0]} 个基因，{df.shape[1]} 个样本。")
    return df

def run_deg_analysis(df, control_cols, treat_cols):
    """一键差异表达分析（高度鲁棒防报错版）"""
    print("🧪 开始进行差异表达分析...")
    df_result = df.copy()
    
    # 引入微小伪计数，防止除以零和 log(0)
    epsilon = 1e-5
    
    df_result["Mean_Control"] = df_result[control_cols].mean(axis=1)
    df_result["Mean_Treat"] = df_result[treat_cols].mean(axis=1)
    
    df_result["Fold_Change"] = (df_result["Mean_Treat"] + epsilon) / (df_result["Mean_Control"] + epsilon)
    df_result["log2_FC"] = np.log2(df_result["Fold_Change"])
    
    p_values = []
    for index, row in df_result.iterrows():
        control_vals = row[control_cols].values
        treat_vals = row[treat_cols].values
        
        if np.all(control_vals == 0) and np.all(treat_vals == 0):
            p_val = 1.0
        else:
            _, p_val = stats.ttest_ind(control_vals, treat_vals, equal_var=False)
            if np.isnan(p_val):
                p_val = 1.0
        p_values.append(p_val)
        
    df_result["p_value"] = p_values
    df_result["p_value"] = df_result["p_value"].clip(lower=1e-300)
    df_result["neg_log10_p"] = -np.log10(df_result["p_value"])
    
    print("✅ 差异表达分析完成！")
    return df_result