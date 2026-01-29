import os
import glob
import pandas as pd
from pathlib import Path
import numpy as np
import argparse

def process_profiler_data(args):
    """
    处理性能分析数据的完整流程
    1. 检查是否存在PROF开头的文件夹
    2. 检查是否存在mindstudio_profiler_output文件夹
    3. 检查是否存在op_summary开头的表格
    4. 读取并合并表格数据
    5. 保存处理后的表格
    """
    print("开始处理性能分析数据...")
    
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
        # 根据文件扩展名选择读取方式
        if op_summary_file.endswith('.csv'):
            df_a = pd.read_csv(op_summary_file)
        print(f"成功读取表格A: {op_summary_file}")
        print(f"表格A形状: {df_a.shape}")
        print(f"表格A列名: {list(df_a.columns)}")
        
    except Exception as e:
        print(f"读取表格A失败: {e}")
        return

    try:
        df_b = pd.read_excel(args.test_result_path)
        
        print(f"成功读取表格B: {args.test_result_path}")
        print(f"表格B形状: {df_b.shape}")
        print(f"表格B列名: {list(df_b.columns)}")
        
    except Exception as e:
        print(f"读取表格B失败: {e}")
        return
    
    # 步骤6: 验证表格A是否有足够的行数
    # B的每一行对应A的两行（除表头外）
    # 所以需要满足: (df_a.shape[0] - 1) >= 2 * (df_b.shape[0] - 1)
    # 假设都有表头
    
    a_data_rows = df_a.shape[0]  # 减去表头
    b_data_rows = df_b.shape[0]  # 减去表头
    
    # 合并数据
    row_idx = 0
    try:
        task_duration_col = "Task Duration(us)"
        
        # 创建两个新列来存储从A中提取的数据
        df_b["metadata_duration"] = ""
        df_b["sas_duration"] = ""
        
        # 遍历表格B的每一行（从第1行开始，跳过表头）
        for i in range(df_b.shape[0]):
            a_row1_idx = 2 * row_idx          # A的第1行对应数据
            a_row2_idx = 2 * row_idx + 1      # A的第2行对应数据
            
            if df_b.iloc[i]["result"] != "NPU ERROR":
                metadata_perf = df_a.iloc[a_row1_idx][task_duration_col]
                df_b.at[i, "metadata_duration"] = metadata_perf
            
                sas_perf = df_a.iloc[a_row2_idx][task_duration_col]
                df_b.at[i, "sas_duration"] = sas_perf
                row_idx += 1
            
            # 打印进度（可选）
            if i % 10 == 0:
                print(f"已处理 {i}/{b_data_rows} 行")
        
        print(f"数据合并完成，为表格B添加了 {b_data_rows} 行的 metadata_duration 和 sas_duration 数据")
        
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