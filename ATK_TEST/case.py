# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import logging
import numpy as np
import random
import torch
import torch_npu
from ml_dtypes import bfloat16
from dataclasses import dataclass
np.random.seed(1)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def assertRtolEqual(a, b, rtol=1e-3, atol=1e-3):
    try:
        assert_allclose(a, b, rtol=rtol, atol=atol)
        print("assertRtolEqual passed")
    except AssertionError as e:
        print("assertRtolEqual failed")
        print(e)

def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, batch: int):
    """生成变长序列长度列表"""
    # 生成变长的序列长度，模拟实际场景
    q_seqlen_list = []
    kv_seqlen_list = []
    
    for b in range(batch):
        # 生成变长的Q序列长度 (在max_q_seqlen的80%-100%之间)
        q_seqlen = int(max_q_seqlen * (0.8 + 0.2 * (b + 1) / batch))
        q_seqlen = max_q_seqlen
        q_seqlen_list.append(q_seqlen)
        
        # 生成变长的KV序列长度 (在max_kv_seqlen的80%-100%之间)
        kv_seqlen = int(max_kv_seqlen * (0.8 + 0.2 * (b + 1) / batch))
        kv_seqlen = max_kv_seqlen
        kv_seqlen_list.append(kv_seqlen)
    
    print(f"Q seqlen list: {q_seqlen_list}")
    print(f"KV seqlen list: {kv_seqlen_list}")
    return q_seqlen_list, kv_seqlen_list


def gen_select_idx_data(q_seqlen_list: list, kv_seqlen_list: list, s_block_x: int, s_block_y: int, batch: int, num_heads: int, sparsity_ratio: float = 0.3):
    """
    @brief 生成新的selectIdx数据格式，支持变长序列
    @param q_seqlen_list  每个batch的Q序列长度列表
    @param kv_seqlen_list 每个batch的KV序列长度列表
    @param s_block_x      Q方向分块大小
    @param s_block_y      KV方向分块大小
    @param batch          batch数
    @param num_heads      注意力头数
    @param sparsity_ratio 稀疏度比例，控制每个Q块选择多少比例的KV块 (默认0.3表示30%的KV块)
    @return select_idx_list, select_num_idx_list, total_q_blocks, max_kv_block_num
    """
    select_idx_list = []
    select_num_idx_list = []
    
    # 计算T: 所有batch中Q方向切块的总数 (直接按照x基本块计算)
    total_q_blocks = 0
    max_kv_block_num = 0
    
    # 首先计算总的Q块数和最大KV块数
    for b in range(batch):
        q_seqlen = q_seqlen_list[b]
        kv_seqlen = kv_seqlen_list[b]
        
        # Q方向分块计算: 直接按照x基本块计算
        s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
        s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
        
        total_q_blocks += s_block_num_q
        max_kv_block_num = max(max_kv_block_num, s_block_num_kv)
        
        print(f"Batch {b}: qSeqlen={q_seqlen}, qBlocks={s_block_num_q}, kvSeqlen={kv_seqlen}, kvBlocks={s_block_num_kv}")
    
    print(f"Total Q blocks (T): {total_q_blocks}")
    print(f"Max KV block num: {max_kv_block_num}")
    print(f"Sparsity ratio: {sparsity_ratio} ({sparsity_ratio*100:.1f}% of KV blocks selected per Q block)")
    print(f"SelectIdx shape: [{total_q_blocks}, {num_heads}, {max_kv_block_num}]")
    print(f"SelectNumIdx shape: [{total_q_blocks}, {num_heads}]")
    
    # 为每个Q块和每个头生成selectIdx
    q_block_offset = 0
    
    for b in range(batch):
        q_seqlen = q_seqlen_list[b]
        kv_seqlen = kv_seqlen_list[b]
        
        # 计算当前batch的分块数量
        s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
        s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
        
        for t_local in range(s_block_num_q):
            t_global = q_block_offset + t_local
            q_block_idx = t_local  # 当前batch内的Q块索引
            
            batch_select_idx = []
            batch_select_num = []
            
            for head in range(num_heads):
                # 为每个头生成稀疏的KV块选择（模拟实际稀疏注意力模式）
                selected_kv_blocks = []
                
                # 全部采用随机稀疏，随机选择若干KV块，不区分对角线/额外块
                if s_block_num_kv > 0:
                    # 为不同头和Q块设置不同的随机种子，确保多样性
                    random.seed(head * 1000 + q_block_idx)
                    # 随机选择1~int(s_block_num_kv * sparsity_ratio)个KV块
                    max_select = max(1, int(s_block_num_kv * sparsity_ratio))
                    # num_select = random.randint(1, max_select)
                    num_select = max_select
                    selected_kv_blocks = random.sample(range(s_block_num_kv), num_select)
                else:
                    selected_kv_blocks = []
                
                selected_kv_blocks.sort()
                
                # 填充到max_kv_block_num长度
                padded_blocks = selected_kv_blocks + [-1] * (max_kv_block_num - len(selected_kv_blocks))
                batch_select_idx.extend(padded_blocks)
                batch_select_num.append(len(selected_kv_blocks))
            
            select_idx_list.extend(batch_select_idx)
            select_num_idx_list.extend(batch_select_num)
            
            # print(f"Batch {b}, T={t_global}: Q block {q_block_idx}, selected KV blocks: {batch_select_num}")
            # print(f"Batch {b}, T={t_global}: Q block {q_block_idx}, batch_select_idx: {batch_select_idx}")
        
        q_block_offset += s_block_num_q
    
    return select_idx_list, select_num_idx_list, total_q_blocks, max_kv_block_num

class TestSparseAttentionInfer():

    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        mask_type: any
        shape_param: any
        select_idx_list: any
        select_num_idx_list: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int
        sparse_s_block_x: int
        sparse_s_block_y: int
        select_idx_list: list
        select_num_idx_list: list
        input_layout: str = "TND"  # 输入格式：TND 或 BNSD
        softmax_lse_flag: int = 0  # LSE输出标志：0=不输出，1=输出

    @classmethod
    def check_attr(cls, batch: int, q_seqlen: int, kv_seqlen: int, num_blocks: int, block_size: int, 
                   s_block_x: int, s_block_y: int):
        if s_block_x <= 0 or s_block_y <= 0:
            logging.error(f"[ERROR] s_block_x ({s_block_x}) and s_block_y ({s_block_y}) must be positive.")
            sys.exit()

    @classmethod
    def group_matmul(cls, head, kv_head, left, right):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :].astype(np.float32),
                                    right[i:(i + 1), :, :].astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        """标准 softmax 实现（用于非增量场景）"""
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res
    
    @classmethod
    def convert_tnd_to_bnsd(cls, tnd_tensor, seqlen_list, num_heads, head_size, max_seqlen=None):
        """
        将TND格式转换为BNSD格式
        
        Args:
            tnd_tensor: (total_tokens, num_heads, head_size) - TND格式的tensor
            seqlen_list: list - 每个batch的序列长度列表
            num_heads: int - 注意力头数
            head_size: int - 头维度大小
            max_seqlen: int - 最大序列长度（如果为None，则使用seqlen_list中的最大值）
            
        Returns:
            bnsd_tensor: (batch, num_heads, max_seqlen, head_size) - BNSD格式的tensor
        """
        batch_size = len(seqlen_list)
        if max_seqlen is None:
            max_seqlen = max(seqlen_list)
        
        # 转换为numpy进行操作（如果是torch tensor）
        if isinstance(tnd_tensor, torch.Tensor):
            tnd_array = tnd_tensor.cpu().numpy()
            is_torch = True
            device = tnd_tensor.device
        else:
            tnd_array = tnd_tensor
            is_torch = False
            device = None
        
        # 初始化BNSD格式的数组
        bnsd_shape = (batch_size, num_heads, max_seqlen, head_size)
        bnsd_array = np.zeros(bnsd_shape, dtype=tnd_array.dtype)
        
        # 根据seqlen_list切分并填充
        token_offset = 0
        for b in range(batch_size):
            seqlen = seqlen_list[b]
            # 切分当前batch的数据
            batch_data = tnd_array[token_offset:token_offset + seqlen, :, :]  # (seqlen, num_heads, head_size)
            # 转置为 (num_heads, seqlen, head_size)
            batch_data = np.transpose(batch_data, (1, 0, 2))
            # 填充到BNSD格式
            bnsd_array[b, :, :seqlen, :] = batch_data
            token_offset += seqlen
        
        # 如果输入是torch tensor，转换回torch tensor
        if is_torch:
            bnsd_tensor = torch.from_numpy(bnsd_array).to(device)
        else:
            bnsd_tensor = bnsd_array
        
        return bnsd_tensor

    @classmethod
    def online_softmax_attention(cls, q_block, kv_blocks, scale):
        """
        Online Softmax Attention (FlashAttention 风格)
        
        逐块更新 softmax 状态，避免合并所有 KV 块
        
        Args:
            q_block: (1, q_len, head_size) - 查询块
            kv_blocks: list of tuples (k_block, v_block)
                k_block: (1, head_size, k_len) - 键块  
                v_block: (1, k_len, head_size) - 值块
            scale: float - 缩放因子
            
        Returns:
            output: (1, q_len, head_size) - 注意力输出
            lse: (1, q_len, 1) - log-sum-exp值
            
        算法原理 (FlashAttention):
            维护三个状态量：
            - m_i: running max (当前最大logit)
            - l_i: running sum (归一化分母)
            - O_i: running output (累积输出)
            
            对每个 KV 块 i:
            1. 计算 S_i = Q @ K_i^T * scale
            2. 计算新的 max: m_new = max(m_old, max(S_i))
            3. 更新 sum: l_new = exp(m_old - m_new) * l_old + sum(exp(S_i - m_new))
            4. 更新输出: O_new = exp(m_old - m_new) * O_old + softmax(S_i, m_new) @ V_i
        """
        q_len = q_block.shape[1]
        head_size = q_block.shape[2]
        
        # 初始化状态量
        m_i = np.full((1, q_len, 1), -np.inf, dtype=np.float32)  # running max
        l_i = np.zeros((1, q_len, 1), dtype=np.float32)          # running sum
        O_i = np.zeros((1, q_len, head_size), dtype=np.float32)  # running output
        
        # 逐块处理
        for k_block, v_block in kv_blocks:
            # 1. 计算注意力分数 S_i = Q @ K_i^T * scale
            S_i = np.matmul(q_block.astype(np.float32), 
                           k_block.astype(np.float32))  # (1, q_len, k_len)
            S_i = S_i * scale
            
            # 2. 计算当前块的最大值
            m_block = np.max(S_i, axis=-1, keepdims=True)  # (1, q_len, 1)
            
            # 3. 计算新的全局最大值
            m_new = np.maximum(m_i, m_block)  # (1, q_len, 1)
            
            # 4. 计算修正因子
            # alpha: 旧输出的修正系数 = exp(m_old - m_new)
            # beta: 当前块的修正系数（用于 softmax）= exp(m_block - m_new)
            alpha = np.exp(m_i - m_new)  # (1, q_len, 1)
            
            # 5. 计算当前块的稳定 softmax 分子
            P_i = np.exp(S_i - m_new)  # (1, q_len, k_len)
            
            # 6. 更新 running sum
            l_i = alpha * l_i + np.sum(P_i, axis=-1, keepdims=True)  # (1, q_len, 1)
            
            # 7. 更新 running output
            # O_new = alpha * O_old + P_i @ V_i
            O_i = alpha * O_i + np.matmul(P_i, v_block.astype(np.float32))  # (1, q_len, head_size)
            
            # 8. 更新 running max
            m_i = m_new
        
        # 最终归一化
        O_final = O_i / l_i  # (1, q_len, head_size)
        
        # 计算LSE: log-sum-exp = m_i + log(l_i)
        lse = m_i + np.log(l_i)  # (1, q_len, 1)
        
        return O_final, lse

    def ref_select_idx_attention(self,
            query,  # (total_q_tokens, num_heads, head_size) - 所有batch拼接
            key,    # (total_kv_tokens, kv_heads, head_size) - 所有batch拼接
            value,  # (total_kv_tokens, kv_heads, head_size) - 所有batch拼接
            scale: float,
            select_idx_list: list,
            select_num_idx_list: list,
            s_block_x: int,
            s_block_y: int,
            total_q_blocks: int,
            max_kv_block_num: int,
            q_seqlen_list: list,
            kv_seqlen_list: list,
            batch: int,
            softmax_lse_flag: int = 0
    ):
        """新的selectIdx注意力参考实现，支持变长序列 - 修复多batch索引偏移问题"""
        print(f"ref_select_idx_attention input shapes:")
        print(f"  query: {query.shape}")
        print(f"  key: {key.shape}")
        print(f"  value: {value.shape}")
        print(f"  total_q_blocks: {total_q_blocks}")
        print(f"  max_kv_block_num: {max_kv_block_num}")
        print(f"  softmax_lse_flag: {softmax_lse_flag}")
        
        # 转置操作
        query = np.transpose(query, (1, 0, 2))  # (total_q_tokens, num_heads, head_size) -> (num_heads, total_q_tokens, head_size)
        key = np.transpose(key, (1, 2, 0))      # (total_kv_tokens, kv_heads, head_size) -> (kv_heads, head_size, total_kv_tokens)
        value = np.transpose(value, (1, 0, 2))   # (total_kv_tokens, kv_heads, head_size) -> (kv_heads, total_kv_tokens, head_size)
        
        print(f"After transpose:")
        print(f"  query: {query.shape}")
        print(f"  key: {key.shape}")
        print(f"  value: {value.shape}")
        
        num_heads = query.shape[0]
        kv_heads = key.shape[0]
        
        # 检查维度匹配
        if num_heads != kv_heads:
            print(f"Warning: num_heads ({num_heads}) != kv_heads ({kv_heads}), using group attention")
        
        # 初始化输出 - 注意这里应该是total_q_tokens而不是max_q_seqlen
        total_q_tokens = query.shape[1]
        head_size = query.shape[2]
        out_high = np.zeros((num_heads, total_q_tokens, head_size), dtype=np.float32)
        out = np.zeros((num_heads, total_q_tokens, head_size), dtype=query.dtype)
        
        # 初始化LSE输出（如果需要）
        lse_output = None
        if softmax_lse_flag == 1:
            lse_output = np.zeros((num_heads, total_q_tokens, 1), dtype=np.float32)
        
        # 【关键修复】：添加batch级别的累计偏移量
        q_token_offset = 0   # Q方向token累计偏移
        kv_token_offset = 0  # KV方向token累计偏移
        q_block_offset = 0   # Q块累计偏移（用于selectIdx索引）
        
        for batch_idx in range(batch):
            q_seqlen = q_seqlen_list[batch_idx]
            kv_seqlen = kv_seqlen_list[batch_idx]
            
            print(f"\n=== Processing Batch {batch_idx} ===")
            print(f"  q_seqlen: {q_seqlen}, kv_seqlen: {kv_seqlen}")
            print(f"  q_token_offset: {q_token_offset}, kv_token_offset: {kv_token_offset}")
            
            # 计算当前batch的分块数量
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
            
            for t_local in range(s_block_num_q):
                t_global = q_block_offset + t_local
                q_block_idx = t_local  # 当前batch内的Q块索引
                
                # 【关键修复】：batch内的相对位置
                q_start_local = q_block_idx * s_block_x
                q_end_local = min((q_block_idx + 1) * s_block_x, q_seqlen)
                
                # 【关键修复】：加上batch偏移得到全局位置
                q_start_global = q_token_offset + q_start_local
                q_end_global = q_token_offset + q_end_local
                
                for head in range(num_heads):
                    # 获取该头对应的selectIdx
                    select_idx_offset = t_global * num_heads * max_kv_block_num + head * max_kv_block_num
                    select_num_offset = t_global * num_heads + head
                    
                    select_num = select_num_idx_list[select_num_offset]
                    selected_kv_blocks = select_idx_list[select_idx_offset:select_idx_offset + max_kv_block_num]
                    
                    # 【GQA修复】：在循环之前提取 q_block 和计算 GQA 参数（只需要一次）
                    q_block = query[head:head+1, q_start_global:q_end_global, :]  # (1, q_block_size, head_size)
                    
                    # 处理 group attention 的情况
                    group_size = num_heads // kv_heads
                    kv_head_idx = head // group_size
                    
                    # 收集所有选中的KV块数据（作为 (K, V) 元组列表）
                    kv_blocks = []
                    
                    for kv_block_idx in selected_kv_blocks[:select_num]:
                        if kv_block_idx == -1:  # 跳过填充的-1
                            continue
                        
                        # 【关键修复】：batch内的相对位置
                        k_start_local = kv_block_idx * s_block_y
                        k_end_local = min((kv_block_idx + 1) * s_block_y, kv_seqlen)
                        
                        # 【关键修复】：加上batch偏移得到全局位置
                        k_start_global = kv_token_offset + k_start_local
                        k_end_global = kv_token_offset + k_end_local
                        
                        # 【修复】：使用全局位置访问key和value
                        k_block = key[kv_head_idx:kv_head_idx+1, :, k_start_global:k_end_global]    # (1, head_size, k_block_size)
                        v_block = value[kv_head_idx:kv_head_idx+1, k_start_global:k_end_global, :]  # (1, k_block_size, head_size)
                        # print(f"debug-ref_select_idx_attention q_start_local: {q_start_local}, q_end_global: {q_end_global}")
                        # print(f"debug-ref_select_idx_attention k_start_global: {k_start_global}, k_end_global: {k_end_global}")
                        kv_blocks.append((k_block, v_block))
                    
                    # 如果没有任何有效的KV块，跳过
                    if len(kv_blocks) == 0:
                        continue
                    
                    # 使用 Online Softmax 计算注意力（FlashAttention 风格）
                    out_block_high, lse_block = self.online_softmax_attention(q_block, kv_blocks, scale)  # (1, q_block_size, head_size), (1, q_block_size, 1)
                    out_block = out_block_high.astype(query.dtype)
                    
                    # 【修复】：输出到全局位置
                    out_high[head:head+1, q_start_global:q_end_global, :] = out_block_high
                    out[head:head+1, q_start_global:q_end_global, :] = out_block
                    
                    # 如果需要LSE输出，保存LSE值
                    if softmax_lse_flag == 1:
                        lse_output[head:head+1, q_start_global:q_end_global, :] = lse_block
            
            # 【关键修复】：更新累计偏移量，为下一个batch做准备
            q_token_offset += q_seqlen
            kv_token_offset += kv_seqlen
            q_block_offset += s_block_num_q
        
        # 转置回原始格式
        out_high = np.transpose(out_high, (1, 0, 2))  # (num_heads, total_q_tokens, head_size) -> (total_q_tokens, num_heads, head_size)
        out = np.transpose(out, (1, 0, 2))
        out = out.astype(query.dtype)
        
        # 如果需要LSE输出，也转置回原始格式
        if softmax_lse_flag == 1:
            lse_output = np.transpose(lse_output, (1, 0, 2))  # (num_heads, total_q_tokens, 1) -> (total_q_tokens, num_heads, 1)
            return out, out_high, lse_output
        else:
            return out, out_high

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, true_out) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.head_size
        head_size_vo = attention_inputs.shape_param.head_size
        block_size = attention_inputs.shape_param.block_size
        s_block_x = attention_inputs.shape_param.sparse_s_block_x
        s_block_y = attention_inputs.shape_param.sparse_s_block_y

        batch = len(attention_inputs.shape_param.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0
        
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            
            # 获取S矩阵稀疏块信息
            sparse_s_block_idx_list = attention_inputs.sparse_s_block_idx_list[i]
            
            keys = None
            values = None
            if attention_inputs.shape_param.kv_dtype == 1:
                keys = []
                values = []
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen):
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size

                    k = attention_inputs.key_cache[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)

                    v = attention_inputs.value_cache[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
                keys = np.stack(keys, axis=0)
                values = np.stack(values, axis=0)
            elif attention_inputs.shape_param.kv_dtype == 0:
                keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
            
            scale = 1.0 / (head_size_qk ** 0.5)
            
            # 使用S矩阵稀疏注意力
            out, out_high = self.ref_sparse_attention(q, keys, values, scale, 
                                                    sparse_s_block_idx_list, s_block_x, s_block_y)
            
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_high
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def calc_data(self, gen_data_params: GenDataParams):
        head_size_qk = gen_data_params.head_size
        head_size_vo = gen_data_params.head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        num_tokens = np.array(gen_data_params.q_seqlen_list).sum()
        num_kv_tokens = np.array(gen_data_params.k_seqlen_list).sum()
        batch_size = len(gen_data_params.q_seqlen_list)
        
        query = np.random.uniform(q_min_range, q_max_range,
            size=(num_tokens, gen_data_params.num_heads, head_size_qk)).astype(gen_data_params.dtype)
        max_k_seqlen = max(gen_data_params.k_seqlen_list)
        block_tables = []
        
        layout = 'TND'
        key_cache = None
        value_cache = None
        if gen_data_params.kv_dtype == 1:
            key_cache = np.random.uniform(kv_min_range, kv_max_range,
                size=(gen_data_params.num_blocks, gen_data_params.block_size,
                gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)

            value_cache = np.random.uniform(kv_min_range, kv_max_range,
                size=(gen_data_params.num_blocks, gen_data_params.block_size,
                gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
            max_num_blocks_per_seq = (max_k_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
            for i in range(batch_size):
                block_table = [
                    max_num_blocks_per_seq * i + j
                    for j in range(max_num_blocks_per_seq)
                ]
                block_tables.append(block_table)
        elif gen_data_params.kv_dtype == 0:
            if layout == 'TND':
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)
                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
        
        # 稀疏模式下不需要mask
        mask = None

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        ref_output = np.zeros(shape_out, dtype=gen_data_params.dtype)
        true_out = np.zeros(shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(query, key_cache, value_cache, block_tables,
            gen_data_params.q_seqlen_list, gen_data_params.k_seqlen_list, mask, gen_data_params.mask_type, gen_data_params,
            gen_data_params.select_idx_list, gen_data_params.select_num_idx_list)
        
        # 使用新的selectIdx注意力参考实现
        # 计算total_q_blocks和max_kv_block_num
        total_q_blocks = 0
        max_kv_block_num = 0
        
        for b in range(len(gen_data_params.q_seqlen_list)):
            q_seqlen = gen_data_params.q_seqlen_list[b]
            kv_seqlen = gen_data_params.k_seqlen_list[b]
            
            # Q方向分块计算
            s_block_num_q = (q_seqlen + gen_data_params.sparse_s_block_x - 1) // gen_data_params.sparse_s_block_x
            s_block_num_kv = (kv_seqlen + gen_data_params.sparse_s_block_y - 1) // gen_data_params.sparse_s_block_y
            
            total_q_blocks += s_block_num_q
            max_kv_block_num = max(max_kv_block_num, s_block_num_kv)
        
        # 【修复】：ref_select_idx_attention 返回顺序修复
        scale = 1.0 / np.sqrt(head_size_qk)
        
        # 根据softmax_lse_flag决定调用方式
        if gen_data_params.softmax_lse_flag == 1:
            ref_output, ref_output_high, lse_output = self.ref_select_idx_attention(
                query, key_cache, value_cache, scale,
                gen_data_params.select_idx_list, gen_data_params.select_num_idx_list,
                gen_data_params.sparse_s_block_x, gen_data_params.sparse_s_block_y,
                total_q_blocks, max_kv_block_num,
                gen_data_params.q_seqlen_list, gen_data_params.k_seqlen_list, len(gen_data_params.q_seqlen_list),
                gen_data_params.softmax_lse_flag
            )
        else:
            ref_output, ref_output_high = self.ref_select_idx_attention(
                query, key_cache, value_cache, scale,
                gen_data_params.select_idx_list, gen_data_params.select_num_idx_list,
                gen_data_params.sparse_s_block_x, gen_data_params.sparse_s_block_y,
                total_q_blocks, max_kv_block_num,
                gen_data_params.q_seqlen_list, gen_data_params.k_seqlen_list, len(gen_data_params.q_seqlen_list),
                gen_data_params.softmax_lse_flag
            )
        
        # 保存数据文件（同时保存bin和txt格式）
        def save_data(data, filename_prefix, description=""):
            """保存数据为bin和txt两种格式，txt文件按照原始数据排布格式输出"""

            bin_path = os.path.join(WORKSPACE, "data", f"{filename_prefix}.bin")
            print(f"Saved: {bin_path} ({data.nbytes} bytes)")
        # 保存各类数据
        print("\n开始保存数据文件...")
        save_data(num_tokens.astype(np.int32), "q_ntokens", "Query tokens总数")
        save_data(num_kv_tokens.astype(np.int32), "kv_ntokens", "KV tokens总数")
        save_data(np.array([total_q_blocks]).astype(np.int32), "total_q_blocks", "总Q块数量")
        save_data(np.array([max_kv_block_num]).astype(np.int32), "max_kv_block_num", "最大KV块数量")
        save_data(query, "q", f"Query矩阵 [{num_tokens}, {gen_data_params.num_heads}, {head_size_qk}]")
        save_data(key_cache, "k", f"Key cache矩阵 (TND格式)")
        save_data(value_cache, "v", f"Value cache矩阵 (TND格式)")
        save_data(np.array(block_tables).astype(np.int32), "block_table", f"Block tables [{batch_size}, max_blocks]")
        save_data(np.array(gen_data_params.q_seqlen_list).astype(np.int64), "q_seqlen", 
                  f"每个batch的Q序列长度列表 (batch={batch_size})")
        save_data(np.array(gen_data_params.k_seqlen_list).astype(np.int64), "kv_seqlen",
                  f"每个batch的KV序列长度列表 (batch={batch_size})")
        # select_idx以[total_q_blocks, num_heads, max_kv_block_num]三维数组形式保存
        select_idx_arr = np.array(gen_data_params.select_idx_list).astype(np.int32)
        num_heads = gen_data_params.num_heads
        select_idx_arr = select_idx_arr.reshape((total_q_blocks, num_heads, max_kv_block_num))
        save_data(select_idx_arr, "select_idx",
                  f"稀疏块选择索引 [total_q_blocks={total_q_blocks}, num_heads={num_heads}, max_kv_block_num={max_kv_block_num}]")
        # select_num_idx以[total_q_blocks, num_heads]二维数组形式保存到txt
        select_num_idx_arr = np.array(gen_data_params.select_num_idx_list).astype(np.int32)
        select_num_idx_arr = select_num_idx_arr.reshape((total_q_blocks, num_heads))
        save_data(select_num_idx_arr, "select_num_idx",
                  f"每个Q块每个头选择的KV块数量 [total_q_blocks * num_heads]")
        
        # 保存空的mask文件（保持接口一致性）
        empty_mask = np.zeros((1024, 1024), dtype=gen_data_params.dtype)
        save_data(empty_mask, "mask", "空mask矩阵（保持接口兼容性）")
        
        # 保存golden结果
        save_data(ref_output.astype(np.float32), "golden", 
                  f"参考输出（golden） [{num_tokens}, {gen_data_params.num_heads}, {head_size_vo}]")
        
        # 保存LSE结果（如果需要）
        if gen_data_params.softmax_lse_flag == 1:
            save_data(lse_output, "lse", 
                      f"LSE输出 [{num_tokens}, {gen_data_params.num_heads}, 1]")
            print(f"LSE output shape: {lse_output.shape}")
        
        print("\n所有数据文件保存完成！")


        # 准备调用接口的参数
        s_block_x = gen_data_params.sparse_s_block_x
        s_block_y = gen_data_params.sparse_s_block_y
        num_head = gen_data_params.num_heads
        kv_heads = gen_data_params.kv_heads
        q_seqlen_list = gen_data_params.q_seqlen_list
        kv_seqlen_list = gen_data_params.k_seqlen_list
        select_idx_list = gen_data_params.select_idx_list
        select_num_idx_list = gen_data_params.select_num_idx_list
        input_layout = gen_data_params.input_layout.upper()  # 转换为大写以便比较
        softmax_lse_flag = gen_data_params.softmax_lse_flag
        
        selectIdxTensor = torch.tensor(select_idx_list).view(total_q_blocks, num_head, max_kv_block_num).npu()
        selectNumIdxTensor = torch.tensor(select_num_idx_list).view(total_q_blocks, num_head).npu()
        blockShape = [s_block_x, s_block_y]
        
        print("scale value", scale)
        print("=============cpu out==================")
        print(ref_output)
        print("=============cpu lse out==================")
        if softmax_lse_flag:
            print(lse_output)
        else:
            print("lse flag false")
        print("res type", type(ref_output))
        
        # 根据输入格式选择不同的处理逻辑
        if input_layout == "TND":
            # TND格式：直接使用TND格式调用接口
            print(f"\n=== Using TND format (no conversion) ===")
            queryTensor = torch.from_numpy(query).npu()
            keyTensor = torch.from_numpy(key_cache).npu()
            valueTensor = torch.from_numpy(value_cache).npu()
            
            print(f"Query TND shape: {queryTensor.shape}")
            print(f"Key TND shape: {keyTensor.shape}")
            print(f"Value TND shape: {valueTensor.shape}")
            
            # 使用TND格式调用接口
            attention_out, lse_out = torch_npu.npu_rain_fusion_attention(
                query=queryTensor, 
                key=keyTensor, 
                value=valueTensor, 
                select_idx=selectIdxTensor, 
                select_num_idx=selectNumIdxTensor,
                block_shape=blockShape, 
                q_input_layout="TND", 
                kv_input_layout="TND", 
                num_key_value_heads=kv_heads, 
                mask_type=0, 
                scale_value=scale, 
                inner_precise=0, 
                block_size=128, 
                atten_mask=None, 
                actual_seq_qlen=q_seqlen_list, 
                actual_seq_kvlen=kv_seqlen_list, 
                block_table=None,
                softmax_lse_flag=softmax_lse_flag
            )
        elif input_layout == "BNSD":
            # BNSD格式：将TND格式转换为BNSD格式后调用接口
            print(f"\n=== Converting TND to BNSD format ===")
            
            # 首先创建TND格式的tensor
            queryTensorTND = torch.from_numpy(query).npu()
            keyTensorTND = torch.from_numpy(key_cache).npu()
            valueTensorTND = torch.from_numpy(value_cache).npu()
            
            print(f"Query TND shape: {queryTensorTND.shape}")
            print(f"Key TND shape: {keyTensorTND.shape}")
            print(f"Value TND shape: {valueTensorTND.shape}")
            print(f"Q seqlen list: {q_seqlen_list}")
            print(f"KV seqlen list: {kv_seqlen_list}")
            
            # 计算最大序列长度
            max_q_seqlen = max(q_seqlen_list)
            max_kv_seqlen = max(kv_seqlen_list)
            
            # 将TND格式转换为BNSD格式
            queryTensor = self.convert_tnd_to_bnsd(
                queryTensorTND, q_seqlen_list, num_head, head_size_qk, max_q_seqlen
            )
            keyTensor = self.convert_tnd_to_bnsd(
                keyTensorTND, kv_seqlen_list, kv_heads, head_size_qk, max_kv_seqlen
            )
            valueTensor = self.convert_tnd_to_bnsd(
                valueTensorTND, kv_seqlen_list, kv_heads, head_size_vo, max_kv_seqlen
            )
            
            print(f"Query BNSD shape: {queryTensor.shape}")
            print(f"Key BNSD shape: {keyTensor.shape}")
            print(f"Value BNSD shape: {valueTensor.shape}")
            
            # 使用BNSD格式调用接口
            attention_out, lse_out = torch_npu.npu_rain_fusion_attention(
                query=queryTensor, 
                key=keyTensor, 
                value=valueTensor, 
                select_idx=selectIdxTensor, 
                select_num_idx=selectNumIdxTensor,
                block_shape=blockShape, 
                q_input_layout="BNSD", 
                kv_input_layout="BNSD", 
                num_key_value_heads=kv_heads, 
                mask_type=0, 
                scale_value=scale, 
                inner_precise=0, 
                block_size=128, 
                atten_mask=None, 
                actual_seq_qlen=q_seqlen_list, 
                actual_seq_kvlen=kv_seqlen_list, 
                block_table=None,
                softmax_lse_flag=softmax_lse_flag
            )
        else:
            raise ValueError(f"Unsupported input_layout: {input_layout}. Must be 'TND' or 'BNSD'.")
        torch.npu.synchronize()
        print("=============npu out==================")
        print(attention_out)
        print("================npu lse out================")
        print(lse_out)

# 不输出LSE（默认）
# python case.py 1 128 128 4 4 128 256 128 half 0.3 TND

# 输出LSE
# python case.py 1 128 128 4 4 128 256 128 half 0.3 TND 1

# python case.py 1 1024 1024 4 4 128 256 128 half 0.3 TND 1
if __name__ == "__main__":
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)
    torch.npu.set_device(2)
    batch = int(sys.argv[1])
    q_seqlen = int(sys.argv[2])
    kv_seqlen = int(sys.argv[3])
    num_head = int(sys.argv[4])
    kv_heads = int(sys.argv[5])
    embedding_size = int(sys.argv[6])
    s_block_x = int(sys.argv[7])  # S矩阵Q方向分块大小
    s_block_y = int(sys.argv[8])  # S矩阵KV方向分块大小
    
    # 稀疏度比例（可选参数，默认0.3）
    sparsity_ratio = float(sys.argv[10]) if len(sys.argv) > 10 else 0.3
    
    # 输入格式（可选参数，默认TND）
    input_layout = str(sys.argv[11]).upper() if len(sys.argv) > 11 else "TND"
    if input_layout not in ["TND", "BNSD"]:
        logging.error(f"[ERROR] input_layout must be 'TND' or 'BNSD', got '{input_layout}'")
        sys.exit()
    
    # LSE输出标志（可选参数，默认0）
    softmax_lse_flag = int(sys.argv[12]) if len(sys.argv) > 12 else 0
    if softmax_lse_flag not in [0, 1]:
        logging.error(f"[ERROR] softmax_lse_flag must be 0 or 1, got '{softmax_lse_flag}'")
        sys.exit()
    
    # 支持泛化场景测试
    print(f"Generating data for sparse attention with:")
    print(f"  batch={batch}, q_seqlen={q_seqlen}, kv_seqlen={kv_seqlen}")
    print(f"  num_heads={num_head}, kv_heads={kv_heads}, embedding_size={embedding_size}")
    print(f"  s_block_x={s_block_x}, s_block_y={s_block_y}")
    print(f"  sparsity_ratio={sparsity_ratio} ({sparsity_ratio*100:.1f}% of KV blocks selected)")
    print(f"  input_layout={input_layout}")
    print(f"  softmax_lse_flag={softmax_lse_flag} ({'输出LSE' if softmax_lse_flag == 1 else '不输出LSE'})")
    
    # 泛化场景提示
    if s_block_y != 128 and s_block_y != 64 and s_block_y != 32:
        print(f"  Note: s_block_y={s_block_y} is a generalization case")
        print(f"  Each 128-block will contain CeilDiv(128, {s_block_y}) = {(128 + s_block_y - 1) // s_block_y} y-blocks")
        print(f"  Some y-blocks may be truncated across 128-block boundaries")

    block_size = 128
    mask_type = 3  # SPARSE_BLOCK
    kv_dtype = 0  # TND format (3D cache)

    str_dtype = str(sys.argv[9]) if len(sys.argv) > 9 else "half"
    if str_dtype == "half":
        dtype = np.float16
    elif str_dtype == "bf16":
        dtype = bfloat16
    else:
        logging.error("[ERROR] dtype must be half or bf16")
        sys.exit()

    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, batch)
    select_idx_list, select_num_idx_list, total_q_blocks, max_kv_block_num = gen_select_idx_data(
        q_seqlen_list, kv_seqlen_list, s_block_x, s_block_y, batch, num_head, sparsity_ratio)
    
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = batch * ((max_kv_seqlen + block_size - 1) // block_size)
    
    testObj = TestSparseAttentionInfer()
    testObj.check_attr(batch, q_seqlen, kv_seqlen, num_blocks, block_size, s_block_x, s_block_y)
    
    gen_data_params = testObj.GenDataParams(q_seqlen_list, kv_seqlen_list, num_head,
                                            kv_heads, embedding_size,
                                            num_blocks, block_size, mask_type, dtype, kv_dtype,
                                            s_block_x, s_block_y, select_idx_list, select_num_idx_list,
                                            input_layout=input_layout, softmax_lse_flag=softmax_lse_flag)
    testObj.calc_data(gen_data_params)

