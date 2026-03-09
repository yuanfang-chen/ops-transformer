def parse_list_string(s):
    """
    解析形如 "{ge::DT_type1, ge::DT_type2, ge::DT_type3}" 的字符串为列表
    
    Args:
        s: 包含花括号和逗号分隔元素的字符串
    
    Returns:
        元素列表（字符串）
    """
    # 去除首尾空白
    s = s.strip()
    # 去除首尾的花括号
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1].strip()
    else:
        raise ValueError("输入字符串必须用花括号包裹")
    
    # 按逗号分割，并去除每个元素的空白
    if s == "":
        return []
    items = [item.strip() for item in s.split(',')]
    return items

def extract_name(item):
    """
    从 "ge::DT_type1" 中提取 "type1"
    
    规则：
      1. 取 "::" 之后的部分（如果没有 "::" 则用原字符串）
      2. 如果该部分以 "DT_" 开头，则去掉 "DT_"
    """
    # 取 "::" 之后的部分
    after_colon = item.split('::')[-1] if '::' in item else item
    # 去掉 "DT_" 前缀
    if after_colon.startswith('DT_'):
        return after_colon[3:]  # 去掉前3个字符 "DT_"
    else:
        return after_colon

def generate_macros_from_strings(list_str1, list_str2, list_str3, output_file):
    """
    从三个字符串解析列表，生成宏定义并写入文件
    """
    list1 = parse_list_string(list_str1)
    list2 = parse_list_string(list_str2)
    list3 = parse_list_string(list_str3)
    
    # 检查列表长度是否一致
    if not (len(list1) == len(list2) == len(list3)):
        raise ValueError("三个列表的长度不一致")
    
    seen = set()
    with open(output_file, 'w') as f:
        cnt = 0
        for i in range(len(list1)):
            name1 = extract_name(list1[i])
            name2 = extract_name(list2[i])
            name3 = extract_name(list3[i])
            macro_name = f"{name1}_{name2}_{name3}"
        
           # 检查是否重复
            if macro_name in seen:
                print("macro_name: ", macro_name, "repeated")
                continue
            seen.add(macro_name)
            cnt += 1
            f.write(f"#define {macro_name} {cnt}\n")

# 示例用法
if __name__ == "__main__":
    # 你的输入字符串（严格按照格式）
    list_str1 = "{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,\
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,\
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,\
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,\
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,\
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,\
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,\
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,\
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,\
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,\
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_BF16,\
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,\
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,\
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16}"
    list_str2 = "{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,\
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,\
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,\
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,\
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,\
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,\
                       ge::DT_INT4,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8}"
    list_str3 = "{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,\
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,\
                       ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,\
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,\
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,\
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT4,\
                       ge::DT_INT4,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,\
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8}"
    
    # 输出文件路径
    output_file = "macros.h"
    
    # 生成宏定义
    generate_macros_from_strings(list_str1, list_str2, list_str3, output_file)
    
    print(f"宏定义已生成到文件: {output_file}")