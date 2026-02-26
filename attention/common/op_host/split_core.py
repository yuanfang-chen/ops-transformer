
"""
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

from typing import List, Tuple, Dict, Any
import math

# 定义常量
FA_TOLERANCE_RATIO = 2
FD_TOLERANCE_RATIO = 2

# 定义块类型枚举
class BlockType:
    NORMAL_BLOCK = 0
    TAIL_BLOCK = 1
    BLOCK_MAX_TYPE = 2

# 定义稀疏模式枚举
class SparseMode:
    DEFAULT_MASK = 0
    ALL_MASK = 1
    LEFT_UP_CAUSAL = 2
    RIGHT_DOWN_CAUSAL = 3
    BAND = 4
    SPARSE_BUTT = 5

def clip(value, min_value, max_value):
    """
    限制值在指定范围内
    
    Args:
        value: 输入值
        min_value: 最小值
        max_value: 最大值
    
    Returns:
        限制后的值
    """
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value

def is_within_tolerance(limit, tolerance, value):
    """
    判断值是否在容差范围内
    
    Args:
        limit: 限制值
        tolerance: 容差
        value: 待检查值
    
    Returns:
        是否在容差范围内
    """
    return limit + tolerance >= value

class BaseInfo:
    """
    分核功能模块输入：输入case的基本信息
    """
    def __init__(self):
        self.b_size = 0  # batch size
        self.n2_size = 0  # N2维度大小
        self.g_size = 0  # G维度大小
        self.s1_size = 0  # S1维度大小
        self.s2_size = 0  # S2维度大小
        self.is_s1g = True  # 是否为S1G模式
        self.is_accum_seq_s1 = False  # S1序列是否累积
        self.is_accum_seq_s2 = False  # S2序列是否累积
        self.actual_seq_s1_size = []  # 实际S1序列大小
        self.actual_seq_s2_size = []  # 实际S2序列大小
        self.actual_len_q_dims = 0  # Q维度实际长度
        self.actual_len_kv_dims = 0  # KV维度实际长度
        self.atten_mask_flag = False  # 注意力掩码标志
        self.sparse_mode = 0  # 稀疏模式
        self.pre_token = 0  # 前置token数
        self.next_token = 0  # 后置token数
        self.actual_seq_prefix_size = 0  # 实际序列前缀大小

class SplitParam:
    """
    分核功能模块输入：切分属性，预留接口，可作为切分方案的参数入口
    """
    def __init__(self):
        self.m_base_size = 1  # M轴基本块大小
        self.s2_base_size = 1  # S2轴基本块大小
        self.g_s1_base_size_of_fd = 8  # FD阶段分核，m轴切分基本块大小

class FlashDecodeResult:
    """
    分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
    """
    def __init__(self, core_num, vec_cube_ratio):
        # 1、归约任务的索引信息
        self.b_n2_idx_of_fd_head = [0] * core_num  # 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
        self.g_s1_idx_of_fd_head = [0] * core_num  # 每个归约任务的GS1索引，脚标为归约任务的序号
        self.s2_split_num_of_fd_head = [0] * core_num  # 每个归约任务的S2核间切分份数，脚标为归约任务的序号
        
        # 2、FD负载均衡阶段，归约任务的分核（vec）信息
        self.g_s1_split_num_of_fd_head = [0] * core_num  # 每个归约任务m轴切分份数，脚标为归约任务的序号
        self.g_s1_last_part_size_of_fd_head = [0] * core_num  # 每个归约任务m轴切分的最后一份的大小，脚标为归约任务的序号
        self.g_s1_idx_end_of_fd_head = [0] * (core_num * vec_cube_ratio)  # FD负载均衡阶段，每个vector的一级索引，脚标为vector ID，值为归约任务的ID
        self.g_s1_idx_end_of_fd_head_split = [0] * (core_num * vec_cube_ratio)  # FD负载均衡阶段，每个vector的二级索引，脚标为vector ID，值为归约任务的m轴切分ID
        
        # 3、每个core处理的第1个归约任务的数据应存放的workspace位置
        self.s2_split_start_idx_of_core = [0] * core_num  # 每个core的S2切分起始索引

class SplitResult:
    """
    分核功能模块输出：FA阶段的核间分核信息
    """
    def __init__(self, core_num, ratio):
        self.used_core_num = 0  # 使用的核数量
        self.vec_cube_ratio = ratio  # vec 与 cube 核数比例
        self.b_n2_end = [0] * core_num  # 每个核处理数据的BN2结束点
        self.g_s1_end = [0] * core_num  # 每个核处理数据的GS1结束点
        self.s2_end = [0] * core_num  # 每个核处理数据的S2结束点
        self.max_cost = 0  # 慢核开销
        self.num_of_fd_head = 0  # 归约任务数量
        self.max_s2_split_num = 0  # 单个归约任务最大分核数量
        self.used_vec_num_of_fd = 0  # 归约过程使用的vector数量
        self.fd_res = FlashDecodeResult(0, 0)  # FD信息

class SplitInfo:
    """
    分核功能模块内部使用：记录切分信息
    """
    def __init__(self, batch_size):
        self.s1_g_base_num = [0] * batch_size  # S1G方向，切了多少个基本块
        self.s2_base_num = [0] * batch_size  # S2方向，切了多少个基本块
        self.s1_g_tail_size = [0] * batch_size  # S1G方向，尾块size
        self.s2_tail_size = [0] * batch_size  # S2方向，尾块size
        self.is_kv_seq_all_zero = True  # KV序列是否全零

class CostInfo:
    """
    分核功能模块内部使用：记录batch的开销信息
    """
    def __init__(self, batch_size):
        self.b_n2_cost_of_each_batch = [0] * batch_size  # 整个batch的开销
        self.b_n2_block_of_each_batch = [0] * batch_size  # 整个batch的块数
        self.b_n2_last_block_cost_of_each_batch = [0] * batch_size  # batch最后一块的开销
        self.total_block_num = 0  # 总块数
        self.total_cost = 0  # 总开销

class SplitContext:
    """
    分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
    """
    def __init__(self, base_info, split_param):
        self.base_info = base_info
        self.split_param = split_param
        self.split_info = SplitInfo(base_info.b_size)
        self.cost_info = CostInfo(base_info.b_size)

class BatchCache:
    """
    分核功能模块内部使用：记录batch相关的临时信息
    """
    def __init__(self):
        self.b_idx = 0  # batch索引
        self.s1_size = 0  # S1大小
        self.s2_size = 0  # S2大小
        self.pre_token_left_up = 0  # 前置token左上角
        self.next_token_left_up = 0  # 后置token左上角
        self.type_cost = [[0 for _ in range(BlockType.BLOCK_MAX_TYPE)] for _ in range(BlockType.BLOCK_MAX_TYPE)]  # 类型开销

class S1GCache:
    """
    分核功能模块内部使用：记录当前行（S1G）的临时信息
    """
    def __init__(self):
        self.b_idx = 0  # batch索引
        self.s1_g_idx = 0  # S1G索引
        self.s2_start = 0  # S2起始位置
        self.s2_end = 0  # S2结束位置
        self.s1_g_cost = 0  # S1G开销
        self.s1_g_last_block_cost = 0  # S1G最后一个块的开销
        self.s1_g_block = 0  # S1G块数
        self.s1_g_normal_block_cost = 0  # S1G普通块开销

class CoreCache:
    """
    分核功能模块内部使用：记录分配过程中，当前核的负载信息
    """
    def __init__(self):
        self.cost_limit = 0  # 负载上限
        self.cost = 0  # 已分配负载
        self.block = 0  # 已分配块数

class AssignContext:
    """
    分核功能模块内部使用：记录分配过程中的上下文信息
    """
    def __init__(self):
        self.cur_b_idx = 0  # 当前batch索引
        self.cur_b_n2_idx = 0  # 当前BN2索引
        self.cur_s1_g_idx = 0  # 当前S1G索引
        self.cur_s2_idx = 0  # 当前S2索引
        self.cur_core_idx = 0  # 当前核索引
        self.unassigned_cost = 0  # 未分配开销
        self.used_core_num = 0  # 使用的核数
        self.cur_kv_split_part = 1  # 当前KV切分部分

        self.b_n2_cost = 0  # BN2开销
        self.b_n2_block = 0  # BN2块数
        self.is_finished = False  # 是否完成
        self.batch_cache = BatchCache()  # batch缓存
        self.s1_g_cache = S1GCache()  # S1G缓存
        self.core_cache = CoreCache()  # 核缓存

def get_s1_seq_size(b_idx, base_info):
    """
    获取S1序列大小
    
    Args:
        b_idx: batch索引
        base_info: 基本信息
    
    Returns:
        S1序列大小
    """
    if not base_info.actual_seq_s1_size:
        return base_info.s1_size

    if base_info.actual_len_q_dims == 1:
        return int(base_info.actual_seq_s1_size[0])

    if not base_info.is_accum_seq_s1:
        return int(base_info.actual_seq_s1_size[b_idx])

    if b_idx == 0:
        return int(base_info.actual_seq_s1_size[b_idx])
    else:
        return int(base_info.actual_seq_s1_size[b_idx] - base_info.actual_seq_s1_size[b_idx - 1])

def get_s2_seq_size(b_idx, base_info):
    """
    获取S2序列大小
    
    Args:
        b_idx: batch索引
        base_info: 基本信息
    
    Returns:
        S2序列大小
    """
    prefix = int(base_info.actual_seq_prefix_size)
    if not base_info.actual_seq_s2_size:
        return prefix + base_info.s2_size

    if base_info.actual_len_kv_dims == 1:
        return prefix + int(base_info.actual_seq_s2_size[0])

    if not base_info.is_accum_seq_s2:
        return prefix + int(base_info.actual_seq_s2_size[b_idx])

    if b_idx == 0:
        return prefix + int(base_info.actual_seq_s2_size[b_idx])
    else:
        return prefix + int(base_info.actual_seq_s2_size[b_idx] - base_info.actual_seq_s2_size[b_idx - 1])

def calc_pre_token_left_up(s1_size, s2_size, base_info):
    """
    计算前置token左上角
    
    Args:
        s1_size: S1大小
        s2_size: S2大小
        base_info: 基本信息
    
    Returns:
        前置token左上角值
    """
    mode = base_info.sparse_mode
    if mode == SparseMode.BAND:
        return int(s1_size) - int(s2_size) + base_info.pre_token
    return base_info.pre_token

def calc_next_token_left_up(s1_size, s2_size, base_info):
    """
    计算后置token左上角
    
    Args:
        s1_size: S1大小
        s2_size: S2大小
        base_info: 基本信息
    
    Returns:
        后置token左上角值
    """
    mode = base_info.sparse_mode
    if mode in [SparseMode.DEFAULT_MASK, SparseMode.ALL_MASK, SparseMode.LEFT_UP_CAUSAL]:
        return base_info.next_token
    elif mode == SparseMode.RIGHT_DOWN_CAUSAL:
        return int(s2_size) - int(s1_size)
    elif mode == SparseMode.BAND:
        return int(s2_size) - int(s1_size) + base_info.next_token
    else:
        return base_info.next_token

def calc_cost(basic_m, basic_s2):
    """
    计算开销
    
    Args:
        basic_m: M轴基本块大小
        basic_s2: S2轴基本块大小
    
    Returns:
        开销值
    """
    align_coef_m = 16
    align_coef_s2 = 64
    # 按align_coef_m对齐，向上取整
    align_basic_m = (basic_m + align_coef_m - 1) >> 4
    # 按align_coef_s2对齐，向上取整
    align_basic_s2 = (basic_s2 + align_coef_s2 - 1) >> 6
    # 6：M轴系数，10：S2轴系数
    return 6 * align_basic_m + 10 * align_basic_s2

def calc_cost_table(s1_normal_size, s2_normal_size, s1_g_tail_size, s2_tail_size):
    """
    计算开销表
    
    Args:
        s1_normal_size: S1正常块大小
        s2_normal_size: S2正常块大小
        s1_g_tail_size: S1G尾块大小
        s2_tail_size: S2尾块大小
    
    Returns:
        开销表
    """
    type_cost = [[0 for _ in range(BlockType.BLOCK_MAX_TYPE)] for _ in range(BlockType.BLOCK_MAX_TYPE)]
    type_cost[BlockType.NORMAL_BLOCK][BlockType.NORMAL_BLOCK] = calc_cost(s1_normal_size, s2_normal_size)
    type_cost[BlockType.TAIL_BLOCK][BlockType.NORMAL_BLOCK] = 0 if s1_g_tail_size == 0 else calc_cost(s1_g_tail_size, s2_normal_size)
    type_cost[BlockType.NORMAL_BLOCK][BlockType.TAIL_BLOCK] = 0 if s2_tail_size == 0 else calc_cost(s1_normal_size, s2_tail_size)
    type_cost[BlockType.TAIL_BLOCK][BlockType.TAIL_BLOCK] = 0 if s1_g_tail_size == 0 or s2_tail_size == 0 else calc_cost(s1_g_tail_size, s2_tail_size)
    return type_cost

def calc_s2_range(s1_g_idx, base_info, split_param, batch_cache):
    """
    计算S2范围
    
    Args:
        s1_g_idx: S1G索引
        base_info: 基本信息
        split_param: 切分参数
        batch_cache: batch缓存
    
    Returns:
        S2范围（起始，结束）
    """
    s2_start = 0
    s2_end = 0

    # actual seq == 0
    if batch_cache.s1_size == 0 or batch_cache.s2_size == 0:
        return (s2_start, s2_end)

    # no mask
    if not base_info.atten_mask_flag:
        s2_start = 0
        s2_end = (batch_cache.s2_size + split_param.s2_base_size - 1) // split_param.s2_base_size
        return (s2_start, s2_end)

    # 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    s1_g_first_token = s1_g_idx * split_param.m_base_size
    s1_g_last_token = min(s1_g_first_token + split_param.m_base_size,
                          batch_cache.s1_size * base_info.g_size) - 1

    s1_first_token = 0
    s1_last_token = 0
    if base_info.is_s1g:
        s1_first_token = s1_g_first_token // base_info.g_size
        s1_last_token = s1_g_last_token // base_info.g_size
    else:
        if s1_g_first_token // batch_cache.s1_size == s1_g_last_token // batch_cache.s1_size:
            # start and end locate in one G
            s1_first_token = s1_g_first_token % batch_cache.s1_size
            s1_last_token = s1_g_last_token % batch_cache.s1_size
        else:
            # start and end locate in tow or more G, but working same as crossing a complete block
            s1_first_token = 0
            s1_last_token = batch_cache.s1_size

    s2_first_token = s1_first_token - batch_cache.pre_token_left_up
    s2_last_token = s1_last_token + batch_cache.next_token_left_up

    # 2. trans index of token to index of block
    # no valid token
    if s2_first_token >= batch_cache.s2_size or s2_last_token < 0 or s2_last_token < s2_first_token:
        s2_start = 0
        s2_end = 0
        return (s2_start, s2_end)

    # get valid range
    s2_first_token = clip(s2_first_token, 0, batch_cache.s2_size - 1)
    s2_last_token = clip(s2_last_token, 0, batch_cache.s2_size - 1)
    s2_start = int(s2_first_token) // split_param.s2_base_size
    s2_end = int(s2_last_token) // split_param.s2_base_size + 1  # end of block index, +1 for Right-open interval

    return (s2_start, s2_end)

def calc_split_info(split_context):
    """
    计算切分信息
    
    Args:
        split_context: 切分上下文
    """
    base_info = split_context.base_info
    split_param = split_context.split_param
    split_info = split_context.split_info

    # 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    for b_idx in range(base_info.b_size):
        s1_size = get_s1_seq_size(b_idx, base_info)
        s2_size = get_s2_seq_size(b_idx, base_info)

        split_info.s1_g_base_num[b_idx] = (s1_size * base_info.g_size + (split_param.m_base_size - 1)) // split_param.m_base_size
        split_info.s1_g_tail_size[b_idx] = (s1_size * base_info.g_size) % split_param.m_base_size
        split_info.s2_base_num[b_idx] = (s2_size + split_param.s2_base_size - 1) // split_param.s2_base_size
        split_info.s2_tail_size[b_idx] = s2_size % split_param.s2_base_size
        if split_info.s1_g_base_num[b_idx] != 0 and split_info.s2_base_num[b_idx] != 0:
            split_info.is_kv_seq_all_zero = False

def calc_batch_cache(b_idx, split_context, batch_cache):
    """
    计算batch缓存
    
    Args:
        b_idx: batch索引
        split_context: 切分上下文
        batch_cache: batch缓存
    """
    base_info = split_context.base_info
    split_param = split_context.split_param
    split_info = split_context.split_info

    batch_cache.b_idx = b_idx
    batch_cache.s1_size = get_s1_seq_size(b_idx, base_info)
    batch_cache.s2_size = get_s2_seq_size(b_idx, base_info)
    batch_cache.pre_token_left_up = calc_pre_token_left_up(batch_cache.s1_size, batch_cache.s2_size, base_info)
    batch_cache.next_token_left_up = calc_next_token_left_up(batch_cache.s1_size, batch_cache.s2_size, base_info)
    batch_cache.type_cost = calc_cost_table(split_param.m_base_size, split_param.s2_base_size, 
                                            split_info.s1_g_tail_size[b_idx], split_info.s2_tail_size[b_idx])

def calc_s1_g_cache(s1_g_idx, split_context, batch_cache, s1_g_cache):
    """
    计算S1G缓存
    
    Args:
        s1_g_idx: S1G索引
        split_context: 切分上下文
        batch_cache: batch缓存
        s1_g_cache: S1G缓存
    """
    base_info = split_context.base_info
    split_param = split_context.split_param
    split_info = split_context.split_info

    s1_g_cache.b_idx = batch_cache.b_idx
    s1_g_cache.s1_g_idx = s1_g_idx

    s2_range = calc_s2_range(s1_g_idx, base_info, split_param, batch_cache)
    s1_g_cache.s2_start = s2_range[0]
    s1_g_cache.s2_end = s2_range[1]

    if s1_g_cache.s2_start >= s1_g_cache.s2_end:
        s1_g_cache.s1_g_block = 0
        s1_g_cache.s1_g_cost = 0
        s1_g_cache.s1_g_last_block_cost = 0
        s1_g_cache.s1_g_normal_block_cost = 0
        return

    # 计算S2方向满块、尾块数量
    s1_g_cache.s1_g_block = s1_g_cache.s2_end - s1_g_cache.s2_start
    cur_tail_s2_num = 1 if (split_info.s2_tail_size[batch_cache.b_idx] != 0 and
        s1_g_cache.s2_end == split_info.s2_base_num[batch_cache.b_idx]) else 0
    cur_normal_s2_num = s1_g_cache.s1_g_block - cur_tail_s2_num
    if split_info.s1_g_base_num[batch_cache.b_idx] == 0:
        s1_g_cache.s1_g_cost = 0
        s1_g_cache.s1_g_last_block_cost = 0
        s1_g_cache.s1_g_normal_block_cost = 0
    elif (s1_g_idx == (split_info.s1_g_base_num[batch_cache.b_idx] - 1) and 
          split_info.s1_g_tail_size[batch_cache.b_idx] != 0):
        s1_g_cache.s1_g_cost = (batch_cache.type_cost[BlockType.TAIL_BLOCK][BlockType.NORMAL_BLOCK] * cur_normal_s2_num +
            batch_cache.type_cost[BlockType.TAIL_BLOCK][BlockType.TAIL_BLOCK] * cur_tail_s2_num)
        s1_g_cache.s1_g_last_block_cost = (
            batch_cache.type_cost[BlockType.TAIL_BLOCK][BlockType.TAIL_BLOCK] if cur_tail_s2_num > 0 
            else batch_cache.type_cost[BlockType.TAIL_BLOCK][BlockType.NORMAL_BLOCK])
        s1_g_cache.s1_g_normal_block_cost = batch_cache.type_cost[BlockType.TAIL_BLOCK][BlockType.NORMAL_BLOCK]
    else:
        s1_g_cache.s1_g_cost = (batch_cache.type_cost[BlockType.NORMAL_BLOCK][BlockType.NORMAL_BLOCK] * cur_normal_s2_num +
            batch_cache.type_cost[BlockType.NORMAL_BLOCK][BlockType.TAIL_BLOCK] * cur_tail_s2_num)
        s1_g_cache.s1_g_last_block_cost = (
            batch_cache.type_cost[BlockType.NORMAL_BLOCK][BlockType.TAIL_BLOCK] if cur_tail_s2_num > 0 
            else batch_cache.type_cost[BlockType.NORMAL_BLOCK][BlockType.NORMAL_BLOCK])
        s1_g_cache.s1_g_normal_block_cost = batch_cache.type_cost[BlockType.NORMAL_BLOCK][BlockType.NORMAL_BLOCK]

def copy_tmp_result(tmp_res, split_res):
    """
    复制临时结果到最终结果
    
    Args:
        tmp_res: 临时结果
        split_res: 最终结果
    """
    length = len(tmp_res.b_n2_end)
    split_res.used_core_num = tmp_res.used_core_num
    split_res.max_cost = tmp_res.max_cost
    split_res.num_of_fd_head = tmp_res.num_of_fd_head
    split_res.max_s2_split_num = tmp_res.max_s2_split_num

    for i in range(length):
        split_res.b_n2_end[i] = tmp_res.b_n2_end[i]
        split_res.g_s1_end[i] = tmp_res.g_s1_end[i]
        split_res.s2_end[i] = tmp_res.s2_end[i]

        split_res.fd_res.b_n2_idx_of_fd_head[i] = tmp_res.fd_res.b_n2_idx_of_fd_head[i]
        split_res.fd_res.g_s1_idx_of_fd_head[i] = tmp_res.fd_res.g_s1_idx_of_fd_head[i]
        split_res.fd_res.s2_split_num_of_fd_head[i] = tmp_res.fd_res.s2_split_num_of_fd_head[i]
        split_res.fd_res.s2_split_start_idx_of_core[i] = tmp_res.fd_res.s2_split_start_idx_of_core[i]
        split_res.fd_res.g_s1_split_num_of_fd_head[i] = tmp_res.fd_res.g_s1_split_num_of_fd_head[i]
        split_res.fd_res.g_s1_last_part_size_of_fd_head[i] = tmp_res.fd_res.g_s1_last_part_size_of_fd_head[i]

def clear_tmp_result(tmp_result):
    """
    清空临时结果
    
    Args:
        tmp_result: 临时结果
    """
    length = len(tmp_result.b_n2_end)
    tmp_result.used_core_num = 0
    tmp_result.max_cost = 0
    tmp_result.num_of_fd_head = 0
    tmp_result.max_s2_split_num = 0
    tmp_result.used_vec_num_of_fd = 0

    for i in range(length):
        tmp_result.b_n2_end[i] = 0
        tmp_result.g_s1_end[i] = 0
        tmp_result.s2_end[i] = 0
        tmp_result.fd_res.b_n2_idx_of_fd_head[i] = 0
        tmp_result.fd_res.g_s1_idx_of_fd_head[i] = 0
        tmp_result.fd_res.s2_split_num_of_fd_head[i] = 0
        tmp_result.fd_res.s2_split_start_idx_of_core[i] = 0
        tmp_result.fd_res.g_s1_split_num_of_fd_head[i] = 0
        tmp_result.fd_res.g_s1_last_part_size_of_fd_head[i] = 0

def calc_batch_cost(b_idx, split_context, cost_info):
    """
    计算batch开销
    
    Args:
        b_idx: batch索引
        split_context: 切分上下文
        cost_info: 开销信息
    """
    base_info = split_context.base_info
    split_info = split_context.split_info

    cost_info.b_n2_cost_of_each_batch[b_idx] = 0
    cost_info.b_n2_block_of_each_batch[b_idx] = 0
    cost_info.b_n2_last_block_cost_of_each_batch[b_idx] = 0

    if get_s1_seq_size(b_idx, base_info) == 0 or get_s2_seq_size(b_idx, base_info) == 0:
        return

    b_cache = BatchCache()
    s1_g_cache = S1GCache()
    calc_batch_cache(b_idx, split_context, b_cache)
    for s1_g_idx in range(split_info.s1_g_base_num[b_idx]):
        calc_s1_g_cache(s1_g_idx, split_context, b_cache, s1_g_cache)
        cost_info.b_n2_cost_of_each_batch[b_idx] += s1_g_cache.s1_g_cost
        cost_info.b_n2_block_of_each_batch[b_idx] += s1_g_cache.s1_g_block

        if s1_g_cache.s1_g_block > 0:
            cost_info.b_n2_last_block_cost_of_each_batch[b_idx] = s1_g_cache.s1_g_last_block_cost

def calc_cost_info(split_context):
    """
    计算开销信息
    
    Args:
        split_context: 切分上下文
    """
    base_info = split_context.base_info
    split_info = split_context.split_info

    cost_info = split_context.cost_info

    if split_info.is_kv_seq_all_zero:
        cost_info.total_cost = 0
        cost_info.total_block_num = 0
        return

    # 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for b_idx in range(base_info.b_size):
        calc_batch_cost(b_idx, split_context, cost_info)
        cost_info.total_cost += cost_info.b_n2_cost_of_each_batch[b_idx] * base_info.n2_size
        cost_info.total_block_num += cost_info.b_n2_block_of_each_batch[b_idx] * base_info.n2_size

def update_cursor(split_context, assign_context):
    """
    更新游标
    
    Args:
        split_context: 切分上下文
        assign_context: 分配上下文
    """
    base_info = split_context.base_info
    split_info = split_context.split_info
    cost_info = split_context.cost_info

    update_s1_g = False
    update_batch = False

    # Update S2
    if assign_context.cur_s2_idx >= assign_context.s1_g_cache.s2_end:  # 边界assignInfo.s2End是取不到的开区间
        assign_context.cur_s2_idx = 0
        assign_context.cur_s1_g_idx += 1
        update_s1_g = True

    # Update S1G
    if assign_context.cur_s1_g_idx >= split_info.s1_g_base_num[assign_context.cur_b_idx]:
        assign_context.cur_s1_g_idx = 0
        assign_context.cur_b_n2_idx += 1

    # Update Batch
    if assign_context.cur_b_n2_idx == base_info.b_size * base_info.n2_size:  # 所有负载全部分配完，设置最后一个核的右开区间，返回
        assign_context.cur_s1_g_idx = 0
        assign_context.cur_s2_idx = 0
        assign_context.is_finished = True
        return

    if assign_context.cur_b_n2_idx // base_info.n2_size != assign_context.cur_b_idx:
        assign_context.cur_b_idx = assign_context.cur_b_n2_idx // base_info.n2_size
        assign_context.cur_s1_g_idx = 0
        update_batch = True
        update_s1_g = True

    # Update Cache
    if update_batch:
        calc_batch_cache(assign_context.cur_b_idx, split_context, assign_context.batch_cache)
        assign_context.b_n2_cost = cost_info.b_n2_cost_of_each_batch[assign_context.cur_b_idx]
        assign_context.b_n2_block = cost_info.b_n2_block_of_each_batch[assign_context.cur_b_idx]
    if update_s1_g:
        calc_s1_g_cache(assign_context.cur_s1_g_idx, split_context, assign_context.batch_cache, assign_context.s1_g_cache)
        assign_context.cur_s2_idx = assign_context.s1_g_cache.s2_start

def assign_by_batch(split_context, assign_context):
    """
    按batch分配
    
    Args:
        split_context: 切分上下文
        assign_context: 分配上下文
    """
    if assign_context.is_finished:
        return

    base_info = split_context.base_info
    cost_info = split_context.cost_info

    while (assign_context.b_n2_cost == 0 or 
           is_within_tolerance(assign_context.core_cache.cost_limit,
                               cost_info.b_n2_last_block_cost_of_each_batch[assign_context.cur_b_idx] // FA_TOLERANCE_RATIO,
                               assign_context.core_cache.cost + assign_context.b_n2_cost)):
        assign_context.core_cache.cost += assign_context.b_n2_cost
        assign_context.core_cache.block += assign_context.b_n2_block
        assign_context.cur_b_n2_idx += 1

        # to the end
        if assign_context.cur_b_n2_idx == base_info.b_size * base_info.n2_size:
            assign_context.cur_s1_g_idx = 0
            assign_context.cur_s2_idx = 0
            assign_context.is_finished = True
            return

        # next batch
        if assign_context.cur_b_n2_idx // base_info.n2_size != assign_context.cur_b_idx:
            assign_context.cur_b_idx = assign_context.cur_b_n2_idx // base_info.n2_size
            calc_batch_cache(assign_context.cur_b_idx, split_context, assign_context.batch_cache)

        assign_context.b_n2_cost = cost_info.b_n2_cost_of_each_batch[assign_context.cur_b_idx]
        assign_context.b_n2_block = cost_info.b_n2_block_of_each_batch[assign_context.cur_b_idx]
        assign_context.cur_s1_g_idx = 0
        calc_s1_g_cache(assign_context.cur_s1_g_idx, split_context, assign_context.batch_cache, assign_context.s1_g_cache)
        assign_context.cur_s2_idx = assign_context.s1_g_cache.s2_start

def assign_by_row(split_context, assign_context):
    """
    按行分配
    
    Args:
        split_context: 切分上下文
        assign_context: 分配上下文
    """
    if assign_context.is_finished:
        return

    while is_within_tolerance(assign_context.core_cache.cost_limit,
                              assign_context.s1_g_cache.s1_g_last_block_cost // FA_TOLERANCE_RATIO,
                              assign_context.core_cache.cost + assign_context.s1_g_cache.s1_g_cost):
        assign_context.core_cache.cost += assign_context.s1_g_cache.s1_g_cost
        assign_context.core_cache.block += assign_context.s1_g_cache.s1_g_block

        # 当前batch被分配一行出去，更新剩余负载
        assign_context.b_n2_cost = 0 if assign_context.b_n2_cost > assign_context.s1_g_cache.s1_g_cost else \
                                   assign_context.b_n2_cost - assign_context.s1_g_cache.s1_g_cost
        assign_context.b_n2_block = 0 if assign_context.b_n2_block > assign_context.s1_g_cache.s1_g_block else \
                                    assign_context.b_n2_block - assign_context.s1_g_cache.s1_g_block
        # 计算新一行的信息
        while True:
            assign_context.cur_s1_g_idx += 1
            calc_s1_g_cache(assign_context.cur_s1_g_idx, split_context, assign_context.batch_cache, assign_context.s1_g_cache)
            if assign_context.s1_g_cache.s1_g_block != 0:
                break
        assign_context.cur_s2_idx = assign_context.s1_g_cache.s2_start

def assign_by_block(split_context, assign_context):
    """
    按块分配
    
    Args:
        split_context: 切分上下文
        assign_context: 分配上下文
    """
    if assign_context.is_finished:
        return

    cur_cost = assign_context.s1_g_cache.s1_g_normal_block_cost
    if assign_context.cur_s2_idx == (assign_context.s1_g_cache.s2_end - 1):
        cur_cost = assign_context.s1_g_cache.s1_g_last_block_cost

    while is_within_tolerance(assign_context.core_cache.cost_limit, 
                              cur_cost // FA_TOLERANCE_RATIO, 
                              assign_context.core_cache.cost + cur_cost):  # (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assign_context.core_cache.cost += cur_cost
        assign_context.core_cache.block += 1
        assign_context.cur_s2_idx += 1
        # 当前batch被分配一块出去，更新剩余负载
        assign_context.b_n2_cost = assign_context.b_n2_cost - cur_cost
        # 当前行被分配一块出去，更新剩余负载
        assign_context.s1_g_cache.s1_g_cost = assign_context.s1_g_cache.s1_g_cost - cur_cost
        assign_context.b_n2_block -= 1
        assign_context.s1_g_cache.s1_g_block -= 1

def force_assign(split_context, assign_context):
    """
    强制分配
    
    Args:
        split_context: 切分上下文
        assign_context: 分配上下文
    """
    if assign_context.is_finished:
        return

    cur_cost = assign_context.s1_g_cache.s1_g_normal_block_cost
    if assign_context.cur_s2_idx == (assign_context.s1_g_cache.s2_end - 1):
        cur_cost = assign_context.s1_g_cache.s1_g_last_block_cost

    assign_context.core_cache.cost += cur_cost
    assign_context.core_cache.block += 1
    assign_context.cur_s2_idx += 1
    # 当前batch被分配一块出去，更新剩余负载
    assign_context.b_n2_cost = assign_context.b_n2_cost - cur_cost
    assign_context.b_n2_block -= 1
    # 当前行被分配一块出去，更新剩余负载
    assign_context.s1_g_cache.s1_g_cost = assign_context.s1_g_cache.s1_g_cost - cur_cost
    assign_context.s1_g_cache.s1_g_block -= 1
    update_cursor(split_context, assign_context)

def is_need_record_fd_info(assign_context, split_res):
    """
    判断是否需要记录FD信息
    
    Args:
        assign_context: 分配上下文
        split_res: 切分结果
    
    Returns:
        是否需要记录FD信息
    """
    # 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    # 核0无需处理
    if assign_context.cur_core_idx == 0:
        return False
    # 无跨核行，无需处理
    if assign_context.cur_kv_split_part <= 1:
        return False
    # 需要归约的行还未处理完
    if (assign_context.cur_b_n2_idx == split_res.b_n2_end[assign_context.cur_core_idx - 1] and
        assign_context.cur_s1_g_idx == split_res.g_s1_end[assign_context.cur_core_idx - 1]):
        return False
    return True

def record_fd_info(split_context, assign_context, result):
    """
    记录FD信息
    
    Args:
        split_context: 切分上下文
        assign_context: 分配上下文
        result: 结果
    """
    base_info = split_context.base_info
    split_param = split_context.split_param
    split_info = split_context.split_info
    # 需要规约的行是上一个核的切分点所在位置
    split_b_idx = result.b_n2_end[assign_context.cur_core_idx - 1] // base_info.n2_size
    split_s1_g_idx = result.g_s1_end[assign_context.cur_core_idx - 1]
    s1_size = get_s1_seq_size(split_b_idx, base_info)

    # 计算归约数据的FD均衡划分信息
    cur_fd_s1g_size = ((s1_size * base_info.g_size - split_s1_g_idx * split_param.m_base_size) 
                       if (split_s1_g_idx == split_info.s1_g_base_num[split_b_idx] - 1) 
                       else split_param.m_base_size)
    cur_fd_s1g_split_part = (cur_fd_s1g_size + split_param.g_s1_base_size_of_fd - 1) // split_param.g_s1_base_size_of_fd
    cur_fd_s1g_last_part_size = cur_fd_s1g_size - (split_param.g_s1_base_size_of_fd * (cur_fd_s1g_split_part - 1))
    # 记录
    result.max_s2_split_num = max(result.max_s2_split_num, assign_context.cur_kv_split_part)
    # 若存在头归约，则切分点一定为上一个核结束的位置
    result.fd_res.b_n2_idx_of_fd_head[result.num_of_fd_head] = result.b_n2_end[assign_context.cur_core_idx - 1]
    result.fd_res.g_s1_idx_of_fd_head[result.num_of_fd_head] = result.g_s1_end[assign_context.cur_core_idx - 1]
    result.fd_res.s2_split_num_of_fd_head[result.num_of_fd_head] = assign_context.cur_kv_split_part
    result.fd_res.g_s1_split_num_of_fd_head[result.num_of_fd_head] = cur_fd_s1g_split_part
    result.fd_res.g_s1_last_part_size_of_fd_head[result.num_of_fd_head] = cur_fd_s1g_last_part_size
    result.num_of_fd_head += 1

def calc_split_plan(core_num, cost_limit, split_context, result):
    """
    计算切分计划
    
    Args:
        core_num: 核数
        cost_limit: 开销限制
        split_context: 切分上下文
        result: 结果
    """
    cost_info = split_context.cost_info

    if core_num == 0:
        return
    result.max_cost = 0
    result.used_core_num = 0

    assign_context = AssignContext()
    assign_context.cur_b_idx = 0
    assign_context.cur_s1_g_idx = 0
    assign_context.unassigned_cost = cost_info.total_cost
    assign_context.b_n2_cost = cost_info.b_n2_cost_of_each_batch[assign_context.cur_b_idx]
    assign_context.b_n2_block = cost_info.b_n2_block_of_each_batch[assign_context.cur_b_idx]
    calc_batch_cache(assign_context.cur_b_idx, split_context, assign_context.batch_cache)
    calc_s1_g_cache(assign_context.cur_s1_g_idx, split_context, assign_context.batch_cache, assign_context.s1_g_cache)
    assign_context.cur_s2_idx = assign_context.s1_g_cache.s2_start

    for i in range(core_num):
        if result.max_cost > cost_limit:
            return
        if assign_context.is_finished or assign_context.unassigned_cost <= 0:
            break

        assign_context.cur_core_idx = i
        result.fd_res.s2_split_start_idx_of_core[assign_context.cur_core_idx] = assign_context.cur_kv_split_part - 1

        assign_context.core_cache = CoreCache()
        assign_context.core_cache.cost_limit = assign_context.unassigned_cost // (core_num - assign_context.cur_core_idx)

        # 1、按整batch分配
        assign_by_batch(split_context, assign_context)
        # 2、按行分配
        assign_by_row(split_context, assign_context)
        # 3、按块分配
        assign_by_block(split_context, assign_context)
        # 4、强制分配
        if assign_context.core_cache.block == 0:
            force_assign(split_context, assign_context)

        result.b_n2_end[i] = assign_context.cur_b_n2_idx
        result.g_s1_end[i] = assign_context.cur_s1_g_idx
        result.s2_end[i] = assign_context.cur_s2_idx
        result.max_cost = max(result.max_cost, assign_context.core_cache.cost)

        assign_context.unassigned_cost -= assign_context.core_cache.cost

        # 对之前的归约信息进行记录并清理
        if is_need_record_fd_info(assign_context, result):
            record_fd_info(split_context, assign_context, result)
            assign_context.cur_kv_split_part = 1

        # 更新S2切分信息
        if (assign_context.cur_s2_idx > assign_context.s1_g_cache.s2_start and
            assign_context.cur_s2_idx <= assign_context.s1_g_cache.s2_end):
            assign_context.cur_kv_split_part += 1

    result.used_core_num = assign_context.cur_core_idx + 1

def split_fd(result):
    """
    分割FD
    
    Args:
        result: 结果
    """
    total_fd_load = 0
    total_fd_head_split = 0
    # 计算FD的总数据量
    for i in range(result.num_of_fd_head):
        total_fd_load += result.fd_res.s2_split_num_of_fd_head[i] * result.fd_res.g_s1_split_num_of_fd_head[i]
        total_fd_head_split += result.fd_res.g_s1_split_num_of_fd_head[i]

    # 基于FA开核数量，计算每个Vector需要计算的FD数据量
    # FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    max_vector_num = min(total_fd_head_split, result.used_core_num * result.vec_cube_ratio)
    load_thr_of_vector = total_fd_load / max_vector_num  # 初始化vector的负载上限
    load_of_cur_vector = 0
    cur_core_index = 0
    pre_tmp_fd_index_end_of_fd_head = 0
    pre_tmp_fd_index_end_of_fd_head_split = 0
    for i in range(result.num_of_fd_head):
        f_d_kv_split_num = result.fd_res.s2_split_num_of_fd_head[i]
        for g_s1_split_idx in range(result.fd_res.g_s1_split_num_of_fd_head[i]):
            remain_space = load_thr_of_vector - load_of_cur_vector  # 计算当前vector剩余负载空间
            # 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if f_d_kv_split_num > remain_space * FD_TOLERANCE_RATIO:
                result.fd_res.g_s1_idx_end_of_fd_head[cur_core_index] = pre_tmp_fd_index_end_of_fd_head
                result.fd_res.g_s1_idx_end_of_fd_head_split[cur_core_index] = pre_tmp_fd_index_end_of_fd_head_split
                cur_core_index += 1
                total_fd_load -= int(load_of_cur_vector)  # 当前未分配的总负载
                # 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                load_thr_of_vector = total_fd_load / (max_vector_num - cur_core_index)
                load_of_cur_vector = 0
            load_of_cur_vector += f_d_kv_split_num
            pre_tmp_fd_index_end_of_fd_head = i
            pre_tmp_fd_index_end_of_fd_head_split = g_s1_split_idx

    result.fd_res.g_s1_idx_end_of_fd_head[cur_core_index] = pre_tmp_fd_index_end_of_fd_head
    result.fd_res.g_s1_idx_end_of_fd_head_split[cur_core_index] = pre_tmp_fd_index_end_of_fd_head_split
    result.used_vec_num_of_fd = cur_core_index + 1

def split_core(core_num, base_info, param, result):
    """
    分核主函数
    
    Args:
        core_num: 核数
        base_info: 基本信息
        param: 参数
        result: 结果
    """
    split_context = SplitContext(base_info, param)

    # 1、划分基本块，统计信息
    calc_split_info(split_context)
    # 全空case
    if split_context.split_info.is_kv_seq_all_zero:
        result.used_core_num = 1
        result.b_n2_end[0] = base_info.b_size * base_info.n2_size
        result.g_s1_end[0] = 0
        result.s2_end[0] = 0
        return

    calc_cost_info(split_context)

    # 2、获取每个核的分配方案
    max_core = min(core_num, split_context.cost_info.total_block_num)
    min_core = int(math.sqrt(split_context.cost_info.total_block_num + 0.25) + 0.5)
    min_core = min(min_core, max_core)

    result.max_cost = float('inf')
    result.used_core_num = 1

    tmp_result = SplitResult(core_num, result.vec_cube_ratio)
    for i in range(min_core, max_core + 1):
        calc_split_plan(i, result.max_cost, split_context, tmp_result)
        if tmp_result.max_cost < result.max_cost:
            copy_tmp_result(tmp_result, result)
        clear_tmp_result(tmp_result)

    # 3、存在FD任务，对FD进行负载均衡分配
    if result.num_of_fd_head > 0:
        split_fd(result)
    result.used_core_num = max(result.used_core_num, 1)  # 至少使用1个core

# 示例测试函数
def test_split_core():
    """测试分核函数"""
    # 创建基本参数
    base_info = BaseInfo()
    base_info.b_size = 2
    base_info.n2_size = 1
    base_info.g_size = 1
    base_info.s1_size = 100
    base_info.s2_size = 100
    base_info.atten_mask_flag = True
    base_info.sparse_mode = SparseMode.DEFAULT_MASK
    base_info.pre_token = 0
    base_info.next_token = 0
    
    # 设置切分参数
    split_param = SplitParam()
    split_param.m_base_size = 16
    split_param.s2_base_size = 32
    
    # 创建结果对象
    result = SplitResult(4, 1)
    
    # 执行分核
    split_core(4, base_info, split_param, result)
    
    print(f"使用的核数: {result.used_core_num}")
    print(f"最大开销: {result.max_cost}")
    print(f"归约任务数: {result.num_of_fd_head}")
    print(f"各核的结束点: BN2={result.b_n2_end[:result.used_core_num]}, GS1={result.g_s1_end[:result.used_core_num]}, S2={result.s2_end[:result.used_core_num]}")

if __name__ == "__main__":
    test_split_core()