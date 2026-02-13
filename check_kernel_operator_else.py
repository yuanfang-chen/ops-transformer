#!/usr/bin/env python3
"""
检查 #include "kernel_operator.h" 是否被正确地包裹在：
    #if ASC_DEVKIT_MAJOR >= 9
    ... (任意内容)
    #else
    #include "kernel_operator.h"
    #endif
的 else 分支中。

正确性判断标准（同时满足）：
1. 该行位于 #else 分支内。
2. 该 #else 对应的顶层 #if 表达式完全等于 ASC_DEVKIT_MAJOR >= 9（忽略空白）。
"""

import os
import re
import sys
import argparse

SOURCE_EXTENSIONS = {
    '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hh', '.hxx',
    '.inl', '.inc', '.tpp', '.C', '.H'
}

EXPECTED_IF_EXPR = "ASC_DEVKIT_MAJOR >= 9"
NORMALIZED_EXPECTED = re.sub(r'\s+', '', EXPECTED_IF_EXPR)


def normalize_expression(expr):
    return re.sub(r'\s+', '', expr)


def is_expected_if(expr):
    return normalize_expression(expr) == NORMALIZED_EXPECTED


class PreprocStack:
    """预处理条件栈，记录分支类型和顶层 if 表达式"""

    def __init__(self):
        self.stack = []  # 每个元素: {'type': str, 'if_expr': str}

    def _push(self, typ, if_expr):
        self.stack.append({
            'type': typ,
            'if_expr': if_expr   # 整个 if 结构的原始表达式（仅对 if/elif/else 有意义）
        })

    def process_directive(self, directive, expr, line_num):
        if directive in ('if', 'ifdef', 'ifndef'):
            # 新条件结构，顶层 if_expr 即为当前指令的表达式
            self._push(directive, expr)
        elif directive == 'elif':
            if self.stack:
                # 保持原 if_expr 不变
                prev = self.stack[-1]
                self._push('elif', prev['if_expr'])
        elif directive == 'else':
            if self.stack:
                prev = self.stack[-1]
                self._push('else', prev['if_expr'])
        elif directive == 'endif':
            if self.stack:
                self.stack.pop()

    def current_branch(self):
        return self.stack[-1] if self.stack else None

    def is_in_else(self):
        branch = self.current_branch()
        return branch is not None and branch['type'] == 'else'

    def get_if_expr(self):
        """返回当前所在条件结构的顶层 if 表达式（任何分支均可）"""
        branch = self.current_branch()
        return branch['if_expr'] if branch else ''


def is_source_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SOURCE_EXTENSIONS


def check_file(file_path, verbose=False):
    """返回问题列表，每个问题为 (行号, 行内容, 原因)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取失败 {file_path}: {e}", file=sys.stderr)
        return []

    stack = PreprocStack()
    problems = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('#'):
            parts = stripped[1:].split(None, 1)
            if parts:
                directive = parts[0].lower()
                expr = parts[1].strip() if len(parts) > 1 else ''
                stack.process_directive(directive, expr, i)

        # 检查目标 include
        if re.match(r'^\s*#\s*include\s*"kernel_operator\.h"\s*$', line):
            if not stack.is_in_else():
                problems.append((i, line.rstrip(), "不在 #else 分支内"))
                continue

            if_expr = stack.get_if_expr()
            if not is_expected_if(if_expr):
                problems.append((i, line.rstrip(), f"对应的 #if 表达式不是 '{EXPECTED_IF_EXPR}' (实际: {if_expr})"))
                continue

            # 所有检查通过，正确包裹
            if verbose:
                print(f"  正确: {file_path}:{i+1}")

    return problems


def scan_directories(directories, verbose=False, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    exclude_dirs = [os.path.abspath(d) for d in exclude_dirs]

    all_problems = {}
    total_files = 0
    for d in directories:
        if not os.path.isdir(d):
            print(f"警告: 目录不存在，跳过 {d}", file=sys.stderr)
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [dn for dn in dirs if os.path.abspath(os.path.join(root, dn)) not in exclude_dirs]
            for f in files:
                full_path = os.path.join(root, f)
                if is_source_file(full_path):
                    total_files += 1
                    problems = check_file(full_path, verbose)
                    if problems:
                        all_problems[full_path] = problems
    return all_problems, total_files


def print_report(problems_dict, total_files):
    print("\n" + "=" * 80)
    print(f"扫描完成，共检查 {total_files} 个源代码文件。")
    if not problems_dict:
        print("🎉 所有 #include \"kernel_operator.h\" 均已被正确包裹！")
        print("=" * 80)
        return

    print(f"❌ 发现 {len(problems_dict)} 个文件存在未被正确包裹的 #include \"kernel_operator.h\"：\n")
    for file_path, issues in problems_dict.items():
        print(f"文件: {file_path}")
        for line_num, line_content, reason in issues:
            print(f"  行 {line_num + 1}: {line_content}")
            print(f"      原因: {reason}")
        print()
    print("=" * 80)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="检查 #include \"kernel_operator.h\" 是否位于正确的 #else 分支中（#if ASC_DEVKIT_MAJOR >= 9）。"
    )
    parser.add_argument(
        'directories',
        nargs='+',
        help='要扫描的目录路径'
    )
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        help='要排除的目录（多个用空格分隔）'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细信息，包括已正确包裹的位置'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f"扫描目录: {args.directories}")
    if args.exclude:
        print(f"排除目录: {args.exclude}")
    print(f"期望的 #if 表达式: {EXPECTED_IF_EXPR}")
    print("-" * 80)

    problems, total = scan_directories(args.directories, args.verbose, args.exclude)
    print_report(problems, total)


if __name__ == '__main__':
    main()