import os
import glob
import pandas as pd
from pathlib import Path
import numpy as np
import argparse

def process_profiler_data(args):
    """
    读取性能数据写入test result表格
    """
    print("=============开始处理性能分析数据=============")
    
    # 检查是否存在PROF开头的文件夹
    prof_folders = [f for f in os.listdir() if os.path.isdir(f) and f.startswith('PROF')]

    if not prof_folders:
        print("PROF文件夹不存在")
        return
    
    prof_folder = prof_folders[0]
    print(f"PROF path: {prof_folder}")
    
    profiler_output_path = os.path.join(prof_folder, "mindstudio_profiler_output")
    if not os.path.exists(profiler_output_path) or not os.path.isdir(profiler_output_path):
        print(f"在 {prof_folder} 中不存在mindstudio_profiler_output文件夹")
        return
    
    # 检查是否存在op_summary开头的表格
    op_summary_files = []
    for ext in ['.csv']:
        pattern = os.path.join(profiler_output_path, f"op_summary*{ext}")
        op_summary_files.extend(glob.glob(pattern))
    
    if not op_summary_files:
        print(f"在 {profiler_output_path} 中未找到 op_summary 开头的表格文件，直接返回")
        return

    op_summary_file = op_summary_files[0]
    print(f"从 {op_summary_file}解析性能数据")
    
    # 读取表格A (op_summary)
    try:
        if op_summary_file.endswith('.csv'):
            df_a = pd.read_csv(op_summary_file)
        
    except Exception as e:
        print(f"读取表格A失败: {e}")
        return

    try:
        df_b = pd.read_excel(args.test_result_path)
        
    except Exception as e:
        print(f"读取表格B失败: {e}")
        return
    
    # 抓取duration数据存入表格B
    row_idx = 0 
    try:
        task_duration_col = "Task Duration(us)"
        df_b["metadata_duration"] = ""
        df_b["sas_duration"] = ""

        for i in range(df_b.shape[0]):
            a_row1_idx = 2 * row_idx          # 当前case metadata性能数据
            a_row2_idx = 2 * row_idx + 1      # 当前case sas性能数据
            
            if df_b.iloc[i]["result"] != "NPU ERROR":    # NPU ERROR时跳过
                metadata_perf = df_a.iloc[a_row1_idx][task_duration_col]
                df_b.at[i, "metadata_duration"] = metadata_perf
            
                sas_perf = df_a.iloc[a_row2_idx][task_duration_col]
                df_b.at[i, "sas_duration"] = sas_perf
                row_idx += 1
            
            if i % 10 == 0:
                print(f"已处理 {i}/{df_b.shape[0]} 行")
        
        print(f"数据合并完成，为表格B添加了 {df_b.shape[0]} 行的 metadata_duration 和 sas_duration 数据")
        
    except Exception as e:
        print(f"数据合并过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存处理后的表格B
    try:
        new_result_path = args.test_result_path.replace(".xlsx", "_perf.xlsx")
        df_b.to_excel(new_result_path, index=False)
        
        print(f"处理后的表格已保存为: {new_result_path}")
        
    except Exception as e:
        print(f"保存文件失败: {e}")
        return
    
    print("处理完成！")
    return new_result_path

# 命令行接口
def main():
    parser = argparse.ArgumentParser(description='处理性能数据')
    parser.add_argument('--test_result_path', type=str, default="result.xlsx", help='结果表格路径')
    args = parser.parse_args()
    
    process_profiler_data(args)

if __name__ == "__main__":
    main()