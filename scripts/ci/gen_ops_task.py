# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import csv
import argparse


def get_sh_files(build_dir):
    """获取目录中所有 .sh 文件名（不包含路径）"""
    if not os.path.isdir(build_dir):
        raise ValueError(f"Provided path is not a valid directory: {build_dir}")
    
    sh_files = []
    for item in os.listdir(build_dir):
        item_path = os.path.join(build_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.sh'):
            sh_files.append(item)
    return sh_files


def parse_opname_from_filename(filename):
    """
    从文件名解析 op_name。
    要求格式：xxx-<opname>-<digits>.sh
    成功返回 op_name，失败返回 None
    """
    parts = filename.split('-')
    if len(parts) < 3:
        return None

    return parts[1]


def count_opnames(sh_filenames):
    """统计每个 op_name 出现的次数"""
    opname_to_count = {}
    for filename in sh_filenames:
        op_name = parse_opname_from_filename(filename)
        if op_name is not None:
            opname_to_count[op_name] = opname_to_count.get(op_name, 0) + 1
    return opname_to_count


def write_csv(output_path, opname_to_count):
    """将结果写入 CSV 文件"""
    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['op'])
        for op_name, count in opname_to_count.items():
            for i in range(count):
                row_string = f"{op_name},{count}-{i}"
                writer.writerow([row_string])


def main(soc, output_csv='single_column_data.csv'):
    """主流程：协调各步骤"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "build", "binary", soc, "gen"))
    sh_files = get_sh_files(build_dir)
    op_counts = count_opnames(sh_files)
    write_csv(output_csv, op_counts)
    print(f"CSV file '{output_csv}' has been generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .sh files in a directory and generate a CSV.")
    parser.add_argument("soc", help="Path to the directory containing .sh files")
    parser.add_argument("-o", "--output", default="single_column_data.csv",
                        help="Output CSV file path (default: single_column_data.csv)")
    args = parser.parse_args()
    
    main(args.soc, args.output)
