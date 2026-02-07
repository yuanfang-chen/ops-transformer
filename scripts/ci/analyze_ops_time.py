import os
import re
import csv
from pathlib import Path

def parse_txt_content(content):
    """解析txt文件内容，提取所需字段"""
    # 使用正则表达式匹配内容
    pattern = r'Build \[([^\]]+)\] op_file \[([^\]]+)\] bin_file \[([^\]]+)\] index \[([^\]]+)\] duration \[([^\]]+)\] start_time \[([^\]]+)\] end_time \[([^\]]+)\]'
    match = re.search(pattern, content)
    
    if not match:
        return None
    
    return {
        'soc': match.group(1),
        'op_file': match.group(2),
        'bin_file': match.group(3),
        'index': match.group(4),
        'duration': match.group(5),
        'start_time': match.group(6),
        'end_time': match.group(7)
    }

def get_bin_file_size(bin_file_path):
    """获取bin_file.o的文件大小（单位：字节）"""
    # 添加.o后缀
    o_file_path = bin_file_path + '.o'
    
    if os.path.exists(o_file_path):
        return os.path.getsize(o_file_path)
    else:
        print(f"警告: 文件 {o_file_path} 不存在")
        return 0

def main():
    # 获取当前目录下所有txt文件
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    data_list = []
    
    for txt_file in txt_files:
        try:
            # 从文件名解析op_name, soc, index
            filename_match = re.match(r'([^_]+)_([^_]+)_(\d+)\.txt', txt_file)
            if not filename_match:
                print(f"警告: 文件名格式不匹配: {txt_file}")
                continue
            
            # 读取文件内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 解析内容
            parsed_data = parse_txt_content(content)
            if not parsed_data:
                print(f"警告: 文件内容格式不匹配: {txt_file}")
                continue
            
            # 获取bin_file.o的文件大小
            bin_file = parsed_data['bin_file']
            size = get_bin_file_size(bin_file)
            
            # 构建数据行
            data_row = {
                'op_file': parsed_data['op_file'],
                'index': parsed_data['index'],
                'duration': parsed_data['duration'],
                'start_time': parsed_data['start_time'],
                'end_time': parsed_data['end_time'],
                'size': size
            }
            
            data_list.append(data_row)
            
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {e}")
            continue
    
    # 按照index排序（如果需要）
    data_list.sort(key=lambda x: (x['op_file'], int(x['index'])))
    
    # 写入CSV文件
    if data_list:
        csv_filename = 'output.csv'
        fieldnames = ['op_file', 'index', 'duration', 'start_time', 'end_time', 'size']
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)
        
        print(f"成功处理 {len(data_list)} 个文件，结果已保存到 {csv_filename}")
        
        # 打印统计信息
        print(f"\n统计信息:")
        print(f"总记录数: {len(data_list)}")
        total_size = sum(row['size'] for row in data_list)
        print(f"总文件大小: {total_size} 字节 ({total_size/1024:.2f} KB)")
    else:
        print("未找到有效的数据文件")

if __name__ == "__main__":
    main()
