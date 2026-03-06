import torch
import torch_npu
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import logging
import numpy as np
from array import array
import random
from ml_dtypes import bfloat16
from dataclasses import dataclass

def inspect_kwargs(kwargs):
    """
    遍历打印 kwargs 中的 key 和 value。
    如果是 Tensor/Array，只打印 shape 和 dtype。
    如果是标量或普通列表，打印具体值。
    """
    print(f"\n{'='*20} DEBUG: Inspect Kwargs {'='*20}")
    
    if not isinstance(kwargs, dict):
        print(f"Error: Expected dict, got {type(kwargs)}")
        return

    for key, value in kwargs.items():
        # 尝试获取类型名称
        type_name = type(value).__name__
        
        # 1. 处理 PyTorch Tensor
        if isinstance(value, torch.Tensor):
            print(f"[Tensor]  Key: {key:<25} | Shape: {list(value.shape)} | Dtype: {value.dtype} | Device: {value.device}")
            
        # 2. 处理 Numpy Array
        elif isinstance(value, np.ndarray):
            print(f"[Numpy]   Key: {key:<25} | Shape: {value.shape} | Dtype: {value.dtype}")
            
        # 3. 处理包含 Tensor 的列表 (常见于 List[Tensor])
        elif isinstance(value, (list, tuple)) and len(value) > 0 and (isinstance(value[0], torch.Tensor) or isinstance(value[0], np.ndarray)):
            shape_info = list(value[0].shape) if hasattr(value[0], 'shape') else "N/A"
            print(f"[List_T]  Key: {key:<25} | Len: {len(value)} | Item0_Shape: {shape_info}")
            
        # 4. 处理自定义对象 (如果有 shape 属性)
        elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
             print(f"[Custom]  Key: {key:<25} | Shape: {value.shape} | Dtype: {value.dtype}")

        # 5. 其他标量或普通对象，直接打印值
        else:
            # 如果值太长，截断一下
            str_val = str(value)
            if len(str_val) > 100:
                str_val = str_val[:100] + "..."
            print(f"[Scalar]  Key: {key:<25} | Value: {str_val}")

    print(f"{'='*62}\n")



dtypeMap = {
    torch.float16: np.float16,
    torch.bfloat16: bfloat16,
    torch.float32: np.float32
}

maskTypeMap = {
    ## sparseMode : golden maskType
    0: 0,
    3: 1,
    4: 2
}

class TestFIAV3SplitFuse():
    @dataclass
    class AuxAttrs:
        preTokens: int
        nextTokens: int
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int
        layout_dtype: int
        max_q_seqlen: int
        max_kv_seqlen: int
        inner_prec: int
        scale: float
    
    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        auxAttrs: any

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
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        # print(f"ljl-row_max:{row_max.shape},{row_max}")
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        # print(f"ljl-row_sum:{row_sum.shape},{row_sum}")
        soft_res = sim_sub / row_sum
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)

        return soft_res, lse, row_max

    def softmax1(
        self,
        qk_result,
        is_first,
        gm,
        interm_dtype = np.float16
    ):
        sim = qk_result
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub.astype(interm_dtype))
        # sim_sub = sim_sub.astype(np.float16)

        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        return sim_sub, row_sum, dm, gm


    def qkMM1(
        self,
        query,
        key
    ):
        result = None
        qk_k = key.shape[1]
        for qk_k_split in range(0, qk_k, 128):
            sub_k = 128
            if qk_k_split == 512:
                sub_k = 64
            query_k = query[:, :, qk_k_split: qk_k_split + sub_k]
            key_k = key[:, qk_k_split: qk_k_split + sub_k, :]
            # print(f"ljl-query.shape{query.shape},{key.shape},qk_k_split:{qk_k_split},qk_k:{qk_k},query_k{query_k.shape},key_k{key_k.shape}")
            result_split = self.group_matmul(query_k.shape[0], key_k.shape[0], query_k, key_k)
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def ref_flash_attention(
        self,
        query,
        key,
        value,
        scale,
        mask,
        attention_inputs: AttentionInputs
    ):
        data_type = attention_inputs.auxAttrs.dtype
        inner_prec = attention_inputs.auxAttrs.inner_prec
        # interm_dtype = np.float16 if inner_prec == 1 else np.float32
        interm_dtype = np.float32
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        value = np.transpose(value, (1, 0, 2))
        # scale = np.float16(scale) if inner_prec == 1 else np.float32(scale)
        scale = np.float32(scale)
        context_len = key.shape[2]
        context_size = 512
        group_num = query.shape[0] // key.shape[0]
        gl = None
        gl_high = None
        go = None
        go_high = None

        for kv_start in range(0, context_len, context_size):
            sub_len = context_size
            if kv_start + context_size > context_len:
                sub_len = context_len - kv_start
            # print(f"ljl-sub_len:{sub_len},{kv_start},{key.shape}")
            sub_key = key[:, :, kv_start: kv_start + sub_len]
            sub_mask = None
            if mask is not None:
                sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len].astype(interm_dtype)
            sub_value = value[:, kv_start: kv_start + sub_len, :]
            qk_result = self.qkMM1(query, sub_key).astype(interm_dtype)
            qk_result_high = self.qkMM1(query.astype(np.float32), sub_key.astype(np.float32))
            # print(f'LJL-qk_result:{qk_result.min()},{qk_result.max()}')
            qk_result = qk_result * scale
            qk_result_high = qk_result_high * scale

            if mask is not None:
                qk_result += sub_mask
                qk_result_high += sub_mask.astype(np.float32)
            if kv_start == 0:
                gm = None
            p_result, row_sum, dm, gm = self.softmax1(qk_result, kv_start == 0, gm, interm_dtype)
            p_result = p_result.astype(data_type)
            if kv_start == 0:
                gm_high = None
            p_result_high, row_sum_high, dm_high, gm_high = self.softmax1(qk_result_high, kv_start == 0, gm_high)
            lo = self.group_matmul(p_result.shape[0], sub_value.shape[0], p_result, sub_value)
            lo_high = self.group_matmul(p_result.shape[0], sub_value.shape[0], p_result.astype(np.float32), sub_value.astype(np.float32))
            if kv_start == 0:
                gl = row_sum
                gl_high = row_sum_high
                go = lo
                go_high = lo_high
            else:
                dm = np.exp(dm)
                dm_high = np.exp(dm_high)
                gl = gl * dm
                gl = gl + row_sum

                go = go * dm
                go = go + lo

                gl_high = gl_high * dm_high
                gl_high = gl_high + row_sum_high

                go_high = go_high * dm_high
                go_high = go_high + lo_high
        go = go / gl
        go_high = go_high / gl_high
        go = np.transpose(go, (1, 0, 2))
        go_high = np.transpose(go_high, (1, 0, 2))
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32) # lse仅支持fp32输出，无论采取何种精度运算，最终结果都是fp32
        lse_high = np.squeeze((np.log(gl_high) + gm_high), axis=-1)
        return go.astype(data_type), go_high, lse, lse_high

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask    # (q_seqlen, k_seqlen)
    ):
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        # print(f"ljl-sim_high:{sim_high},{sim_high.shape},{query},{key}")
        
        if mask is not None:
            # print(f"ljl-sim_high:{sim_high.shape}.{mask.shape}")
            sim_high = sim_high + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float32)
        # print(f"ljl-sim_high-2:{sim_high.shape}")
        # torch.save(sim_high, "1.bin") 
        p_high, lse_high, gm = self.softmax_numpy(sim_high)
        # print(f"ljl-p_high-2:{p_high},{p_high.shape}")
        # lse = lse_high.astype(query.dtype) # ljl
        lse_high = lse_high.astype(np.float32)
        lse_high = lse_high.astype(np.float32)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))
        
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        # print(f"ljl-out:{out},{value}")
        out_high = np.transpose(out_high, (1, 0, 2))
        out = np.transpose(out, (1, 0, 2))
        out = out.astype(query.dtype)
        # return out_high, lse_high
        return out, lse_high #1219

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output) -> None:
        num_heads = attention_inputs.auxAttrs.num_heads
        kv_heads = attention_inputs.auxAttrs.kv_heads
        head_size_qk = attention_inputs.auxAttrs.head_size
        head_size_vo = attention_inputs.auxAttrs.head_size
        block_size = attention_inputs.auxAttrs.block_size
        max_q_seqlen = attention_inputs.auxAttrs.max_q_seqlen
        inner_prec = attention_inputs.auxAttrs.inner_prec
        scale = attention_inputs.auxAttrs.scale
        
        batch = len(attention_inputs.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = None
            if attention_inputs.auxAttrs.layout_dtype == 1:
                q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            else:
                q = attention_inputs.query[i * max_q_seqlen:(i * max_q_seqlen + q_seqlen), :, :]
            keys = None
            values = None
            if attention_inputs.auxAttrs.kv_dtype == 1:
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
            elif attention_inputs.auxAttrs.kv_dtype == 0:
                if attention_inputs.auxAttrs.layout_dtype == 1:
                    keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                    values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                else:
                    keys = attention_inputs.key_cache[i, :, :, :]
                    values = attention_inputs.value_cache[i, :, :, :]
            # print(f"ljl-attention_inputs.auxAttrs.kv_dtype:{attention_inputs.auxAttrs.kv_dtype},{attention_inputs.auxAttrs.layout_dtype}")
            if attention_inputs.auxAttrs.mask_type == 1:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            elif attention_inputs.auxAttrs.mask_type == 2:
                # print(f"ljl-batch:{i},attention_inputs.global_mask:{attention_inputs.global_mask.shape},{cu_seqlen},{q_seqlen}")
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
                # print(f"ljl-batch:{i},attention_inputs.global_mask:{attention_inputs.global_mask.shape},{cu_seqlen},{q_seqlen}")
                # mask = attention_inputs.global_mask
            elif attention_inputs.auxAttrs.mask_type == 0:
                mask = None
            # # ljl
            preTokens = attention_inputs.auxAttrs.preTokens
            nextTokens = attention_inputs.auxAttrs.nextTokens
            
            preTokensChange = preTokens - k_seqlen + q_seqlen
            nextTokensChange = nextTokens + k_seqlen - q_seqlen
            nextTokensError = -nextTokensChange if nextTokensChange < 0 else 0
            preTokensError = (q_seqlen - k_seqlen - preTokensChange) if q_seqlen > k_seqlen + preTokensChange else 0
            actualSeq = q_seqlen
            # print(f"ljl-2 {i},:{preTokens},{nextTokens},{preTokensChange},{nextTokensChange},{preTokensError},{nextTokensError}")
            # nextTokensChange += nextTokensError
            # preTokensChange -= nextTokensError
            actualSeq -= nextTokensError
            actualSeq -= preTokensError
            if actualSeq != q_seqlen:
                if nextTokensError != 0:
                    # 前n行置0
                    actualSeq = q_seqlen - actualSeq
                elif preTokensError != 0:
                    # 后n行置0
                    actualSeq = actualSeq
            out_normal, lse = self.ref_masked_attention(q, keys, values, scale, mask)
            out_gpu, _, lse_gpu, _ = self.ref_flash_attention(q, keys, values, scale, mask, attention_inputs)
            # out_normal, lse = out_gpu.astype(np.float32), lse_gpu
            out_gpu_test = torch.from_numpy(out_gpu.astype(np.float32))
            nan_out_gpu = torch.isnan(out_gpu_test)
            nan_count = nan_out_gpu.sum().item()
            # if out_gpu.dtype == "float32":
            #     out_gpu = out_normal
            
            out = out_normal.reshape(-1, num_heads, head_size_vo)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_gpu = out_gpu.reshape(-1, num_heads, head_size_vo)
            # print(f"ljl-start-batch:{i},q_seqlen:{q_seqlen},actualSeq:{actualSeq},cu_seqlen:{cu_seqlen},preTokensError:{preTokensError},nextTokensError:{nextTokensError},{out.shape}")
            if attention_inputs.auxAttrs.layout_dtype == 1:
                output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse_gpu
                if actualSeq != q_seqlen :
                    if nextTokensError != 0:
                        output[cu_seqlen : cu_seqlen  + actualSeq, :, :] = 0  # 前n行置0
                        golden_gpu_output[cu_seqlen: cu_seqlen + actualSeq, :, :] = 0
                        golden_lse_output[:, cu_seqlen: cu_seqlen + actualSeq] = np.inf
                        golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + actualSeq] = np.inf
                    elif preTokensError != 0:
                        output[cu_seqlen + actualSeq: cu_seqlen  + q_seqlen, :, :] = 0  # 后n行置0
                        golden_gpu_output[cu_seqlen + actualSeq: cu_seqlen + q_seqlen, :, :] = 0
                        golden_lse_output[:, cu_seqlen + actualSeq: cu_seqlen  + q_seqlen] =  np.inf
                        golden_gpu_lse_output[:, cu_seqlen + actualSeq: cu_seqlen + q_seqlen] =  np.inf
            else:
                output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse_gpu
                if actualSeq != q_seqlen :
                    if nextTokensError != 0:
                        output[i * max_q_seqlen: i * max_q_seqlen + actualSeq, :, :] = 0
                        golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + actualSeq, :, :] = 0

                        golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + actualSeq] = np.inf
                        golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + actualSeq] = np.inf
                    elif preTokensError != 0:
                        output[i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen, :, :] = 0
                        golden_gpu_output[i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen, :, :] = 0

                        golden_lse_output[:, i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen] = np.inf
                        golden_gpu_lse_output[:, i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen] = np.inf
            
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen
    
    def calc_data(self, attention_inputs:AttentionInputs):
        num_tokens = attention_inputs.query.shape[0]
        shape_out = (num_tokens, attention_inputs.auxAttrs.num_heads, attention_inputs.auxAttrs.head_size)
        golden_output = np.zeros(shape_out, dtype=attention_inputs.auxAttrs.dtype)
        golden_gpu_output = np.zeros(shape_out, dtype=np.float32)

        lse_shape_out = (attention_inputs.auxAttrs.num_heads, num_tokens)
        golden_lse_output = np.zeros(lse_shape_out, dtype=np.float32)
        golden_gpu_lse_output = np.zeros(lse_shape_out, dtype=np.float32)

        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            golden_output,
            golden_gpu_output,
            golden_lse_output,
            golden_gpu_lse_output
        )

        golden_lse_output = np.transpose(golden_lse_output, (1, 0))
        golden_lse_output = np.expand_dims(golden_lse_output, axis=2)
        golden_gpu_lse_output = np.transpose(golden_gpu_lse_output, (1, 0))
        golden_gpu_lse_output = np.expand_dims(golden_gpu_lse_output, axis=2)
        # print(f"ljl-golden_output{golden_output},{golden_gpu_output}")
        return golden_output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output
        

def gen_list_from_cumSum(seqlenArray):
    seqlenList = []
    preSeqSum = 0
    for i in range(len(seqlenArray)):
        seqlenList.append(seqlenArray[i] - preSeqSum)
        preSeqSum = seqlenArray[i]
    return seqlenList

def gen_actual_seqlen_list_golden(actualseqlengths, actualseqlengthskv, inputLayout, pagedAttentionFlag):
    qSeqlenList = []
    kvSeqlenList = []
    if inputLayout == 'TND':
        qSeqlenList = gen_list_from_cumSum(actualseqlengths)
        if pagedAttentionFlag:
            kvSeqlenList = list(actualseqlengthskv)
        else:
            kvSeqlenList = gen_list_from_cumSum(actualseqlengthskv)
    else:
        qSeqlenList = list(actualseqlengths)
        kvSeqlenList = list(actualseqlengthskv)
    return qSeqlenList, kvSeqlenList

def create_binary_matrix(qSeqlen, kvSeqlen, preToken, nextToken):
    preToken = kvSeqlen - qSeqlen - preToken
    nextToken = kvSeqlen - qSeqlen + nextToken
    matrix = [[0 for _ in range(kvSeqlen)] for _ in range(qSeqlen)]
    for i in range(qSeqlen):
        for j in range(kvSeqlen):
            is_below_pretoken_line = (-i + j) < preToken
            is_above_nexttoken_line = (-i + j) > nextToken
            if is_below_pretoken_line or is_above_nexttoken_line:
                matrix[i][j] = 1
    
    return np.array(matrix)

def create_complex_mask(q, k,preToken, nextToken):
    """
    创建复杂的mask矩阵
    按照左上模式进行计算，其中 p=Px,n = Nx
    """
    p = k - q - preToken
    n = k - q + nextToken
    # 1. 创建全0的基础mask
    final_mask = np.zeros((q, k), dtype=int)
    f1 = np.zeros((q, k), dtype=int)
    f2 = np.zeros((q, k), dtype=int)
    # 2. 创建m1矩阵 (q x q) - 严格下三角
    m1 = np.ones((q, q), dtype=int)
    # import pdb;pdb.set_trace()
    m1 = np.tril(m1, -1)  # 严格下三角
    m1_end = min(q + p, k)
    # q:128,p:278,k:384,m1_end:384
    # print(f"q:{q},p:{p},k:{k},m1_end:{m1_end},m1:{m1}")
    # 3. 将m1放在最左边
    # import pdb;pdb.set_trace()
    if p < 0 and p > -q:
        f1[-p:, : q + p] = m1[ -p : , -p : ]
    elif p <= -q:
        f1 = np.zeros((q, k), dtype=int)
    elif p >= 0:
        if q + p > k: # 这里之前写错了。写成if q + n > k
            f1[:, p :m1_end] = m1[:,:k - p]
        else:
            f1[:, p:m1_end] = m1
    
    # print(f"f1:{f1}")
    # import pdb;pdb.set_trace()
    # 4. 创建m2矩阵并放置
    m2_end =  min(k, q + n)
    # print("m2_end",m2_end)
    if n >= 0 and n < k: 
        m2_size = min(q, k - n)
        # m2_size = q           =>    m2_end = q + n -1
        # m2_size = k - n + 1   =>    m2_end = k
        if m2_size > 0:
            # 创建m2矩阵：严格上三角
            m2 = np.ones((m2_size, m2_size), dtype=int)
            m2 = np.triu(m2, 1)  # 严格上三角
            # print("f2",m2,m2_size)
            # 将m2放在从n开始的位置
            f2[:m2_size, n:m2_end] = m2
    elif n >= k: 
        f2 = np.zeros((q, k), dtype=int)
    elif n < 0 and n > -q:
        m2 = np.ones((q, q), dtype=int)
        m2 = np.triu(m2, 1)  # 严格上三角
        f2[: , : q + n] = m2[:,-n :]
    # 5 相加
    if p < 0:
        start = 0
        f1 = f1[: , start : m2_end]
        f2 = f2[: , start : m2_end]
    else:
        start = p
        f1 = f1[: , start : m2_end]
        f2 = f2[: , start : m2_end]
    # import pdb;pdb.set_trace()
    f3 = f1 + f2
    final_mask[: , start :m2_end] = f3
    # 6 填充左边的
    if p > 0:
        m3 = np.ones((q, p), dtype=int)
        # import pdb;pdb.set_trace()
        final_mask[:, :p] = m3
        # print("填充左边的1")
        # 7 填充右边的
    if q + n < k:
        m4 = np.ones((q, k - m2_end), dtype=int)
        # import pdb;pdb.set_trace()
        final_mask[:, m2_end :] = m4
        # print(f"填充右边的1：{m4.shape}")
    # # 8 填充上边的
    # m5 = np.ones((-n, k), dtype=int)
    # # import pdb;pdb.set_trace()
    # final_mask[:-n, :] = m5
    # print("填充上边的0")
    
    return final_mask

def aclnn_op_func_fia_split_fuse_golden(input_data : InputDataset, name):
    input_data_dtype = input_data.kwargs["query"].dtype
    # input_data_dtype = torch.bfloat16
    query = input_data.kwargs["query"].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["query"].to(torch.float32).numpy().astype(bfloat16)
    key = input_data.kwargs["key"][0].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["key"][0].to(torch.float32).numpy().astype(bfloat16)
    value = input_data.kwargs["value"][0].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["value"][0].to(torch.float32).numpy().astype(bfloat16)
    # query = input_data.kwargs["query"].to(torch.float32).numpy().astype(bfloat16)
    # key = input_data.kwargs["key"][0].to(torch.float32).numpy().astype(bfloat16)
    # value = input_data.kwargs["value"][0].to(torch.float32).numpy().astype(bfloat16)
    # print(f"ljl-q-max:{query.max()},min:{query.min()}")
    # print(f"ljl-k-max:{key.max()},min:{key.min()}")
    
    blockTable = None
    pagedAttentionFlag = False
    if input_data.kwargs["blockTableOptional"] != None:
        blockTable = input_data.kwargs["blockTableOptional"].numpy()
        pagedAttentionFlag = True
    ## gen actual seqlen
    # print(f"ljl-v-max:{value.max()},min:{value.min()},{pagedAttentionFlag}")
    inputLayout = input_data.kwargs["inputLayout"]
    batch = len(input_data.kwargs["actualSeqLengthsOptional"])
    actualseqlengths = [0] * batch
    actualseqlengthsKv = [0] * batch
    for i in range(batch):
        actualseqlengths[i] = input_data.kwargs["actualSeqLengthsOptional"][i]
        actualseqlengthsKv[i] = input_data.kwargs["actualSeqLengthsKvOptional"][i]
    # 把actualseqlengths变成累加的list
    
    qSeqlenList, kvSeqlenList = gen_actual_seqlen_list_golden(actualseqlengths, actualseqlengthsKv, inputLayout, pagedAttentionFlag)
    # print(f"ljl-qSeqlenList:{qSeqlenList},kvSeqlenList:{kvSeqlenList}")
    maxKvSeqlen = max(kvSeqlenList)
    maxQSeqlen = max(qSeqlenList)
    totalQTokens = sum(qSeqlenList)
    # maxKvSeqlen = max([kvSeqlenList[0]] + [kvSeqlenList[i] - kvSeqlenList[i-1] for i in range(1, len(kvSeqlenList))])
    # maxQSeqlen = max([qSeqlenList[0]] + [qSeqlenList[i] - qSeqlenList[i-1] for i in range(1, len(qSeqlenList))])
    # totalQTokens = qSeqlenList[-1]
    preTokens = input_data.kwargs["preTokens"]
    nextTokens = input_data.kwargs["nextTokens"]
    # print(f"ljl-maxKvSeqlen:{maxKvSeqlen},maxQSeqlen:{maxQSeqlen},totalQTokens{totalQTokens}")
    ## gen mask
    fullMask = None
    pre_mask_factor = -3e38 if input_data_dtype == torch.bfloat16 else -6e4
    # pre_mask_factor = -6e4
    if input_data.kwargs["attenMaskOptional"] != None and input_data.kwargs["sparseMode"] == 3:
        maskDtype = dtypeMap[input_data_dtype]
        fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(maskDtype)
        prevQseqlen = 0
        for i in range(len(qSeqlenList)):
            qSeqlen = qSeqlenList[i]
            kSeqlen = kvSeqlenList[i]
            tri = np.ones((qSeqlen, qSeqlen))
            tri = np.triu(tri, 1)
            tri *= pre_mask_factor
            fullMask[prevQseqlen : (prevQseqlen + qSeqlen), kSeqlen - qSeqlen : kSeqlen] = tri
            prevQseqlen += qSeqlen
    # if input_data.kwargs["attenMaskOptional"] != None and input_data.kwargs["sparseMode"] == 4:
    if  input_data.kwargs["sparseMode"] == 4:
        maskDtype = dtypeMap[input_data_dtype]
        # fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(np.float16)
        fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(maskDtype)
        prevQseqlen = 0
        for i in range(len(qSeqlenList)):
            qSeqlen = qSeqlenList[i]
            kSeqlen = kvSeqlenList[i]
            # tri = create_complex_mask(qSeqlen, kSeqlen, preTokens, nextTokens)
            tri = create_binary_matrix(qSeqlen, kSeqlen, preTokens, nextTokens)
            # print(f"ljl-preTokens:{preTokens},nextTokens:{nextTokens},qSeqlen:{qSeqlen},kSeqlen:{kSeqlen},maxKvSeqlen:{maxKvSeqlen}")
            # print(f"ljl-i:{i},kSeqlen:{kSeqlen},qSeqlen{qSeqlen},prevQseqlen{prevQseqlen},fullMask:{fullMask.shape},maskDtype{maskDtype},{tri.dtype}")
            tri = tri.astype(maskDtype)
            # tri = tri.astype(np.float16)
            tri *= pre_mask_factor
            fullMask[prevQseqlen : (prevQseqlen + qSeqlen), :kSeqlen] = tri
            prevQseqlen += qSeqlen
    numHeads = input_data.kwargs["numHeads"]
    kvHeads = input_data.kwargs["numKeyValueHeads"]
    headSize = query.shape[2] if inputLayout == 'TND' else 0
    numBlocks = key.shape[0] if pagedAttentionFlag == True else 0
    blockSize = input_data.kwargs["blockSize"]
    maskType = maskTypeMap[input_data.kwargs["sparseMode"]]
    dtype = dtypeMap[input_data_dtype]
    kvOrgMode = 1 if pagedAttentionFlag == True else 0
    layoutMode = 1 if inputLayout == 'TND' else 0
    goldenGpuPrecision = input_data.kwargs["innerPrecise"]
    softmaxLseFlag = input_data.kwargs["softmaxLseFlag"]
    scale = float(input_data.kwargs["scaleValue"])
    

    if pagedAttentionFlag == True:
        key = key.reshape(key.shape[:-1] + (kvHeads, headSize))
        value = value.reshape(value.shape[:-1] + (kvHeads, headSize))
    testObj = TestFIAV3SplitFuse()
    auxAttrs = testObj.AuxAttrs(preTokens, nextTokens, numHeads, kvHeads, headSize, numBlocks, blockSize, maskType, dtype, kvOrgMode, layoutMode, maxQSeqlen, maxKvSeqlen, goldenGpuPrecision, scale)
    attentionInputs = testObj.AttentionInputs(query, key, value, blockTable, qSeqlenList, kvSeqlenList, fullMask, auxAttrs)

    golden_output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output = testObj.calc_data(attentionInputs)
    if golden_output.dtype == "bfloat16":
         golden_output = torch.from_numpy(golden_output.astype(np.float32))
    else:
        golden_output = torch.from_numpy(golden_output)
    # print(f"ljl-out:{golden_output.shape}")
    golden_gpu_output = torch.from_numpy(golden_gpu_output)
    golden_lse_output = torch.from_numpy(golden_lse_output)
    golden_gpu_lse_output = torch.from_numpy(golden_gpu_lse_output)
    if not softmaxLseFlag:
        golden_lse_output = torch.tensor([])
        golden_gpu_lse_output = torch.tensor([])

    if name == 'gpu':
        # print(f"ljl-golden_gpu_output:{golden_gpu_output}")
        return golden_gpu_output, golden_gpu_lse_output
    else:
        # print(f"ljl-golden_cpu_output:{golden_output}")
        return golden_output, golden_lse_output

@register("executor_fused_infer_attention_score_v4_swa")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)
    
    def init_by_input_data(self, input_data: InputDataset):
        # input_data.kwargs["actualSeqLengthsOptional"].sort()
        # import itertools
        # input_data.kwargs["actualSeqLengthsOptional"] = list(itertools.accumulate(input_data.kwargs["actualSeqLengthsOptional"]))
        # input_data.kwargs["actualSeqLengthsKvOptional"] = list(itertools.accumulate(input_data.kwargs["actualSeqLengthsKvOptional"]))
        np.random.seed(1)
        is_ifa_perf = False
        if is_ifa_perf:
            # 测性能的时 为了获取ifa/pfa的性能需要开启is_ifa_perf=True
            input_data.kwargs["preTokens"]=None
            input_data.kwargs["nextTokens"]=None
            input_data.kwargs["sparseMode"]=3
        # inspect_kwargs(input_data.kwargs)
        # print(input_data.kwargs["key"][0].dtype)
        pass
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.name == "perf" or "abnormal" in self.task_result.case_config.name:
            if input_data.kwargs["softmaxLseFlag"] == True:
                return torch.Tensor([1]), torch.Tensor([1])
            else:
                return torch.Tensor([1])
        output, output_lse = aclnn_op_func_fia_split_fuse_golden(input_data, name = self.name)
        if input_data.kwargs["softmaxLseFlag"] == True:
            return output, output_lse
        else:
            return output

@register("executor_aclnn_fused_infer_attention_score_v4_swa")
class aclnnFusedInferAttentionScoreApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnFusedInferAttentionScoreApi, self).__init__(task_result, backend)
    
    def gen_compressed_triU_mask(self, dim_num, mask_dtype):
        mask_shape_four_dims = (1, 1, 2048, 2048)
        mask_four_dims = torch.zeros(mask_shape_four_dims, dtype = mask_dtype)
        mask_triU = torch.triu(torch.ones(2048, 2048), diagonal=1)
        mask_four_dims[:] = mask_triU
        if dim_num == 2:
            return mask_four_dims[0][0]
        elif dim_num == 3:
            return mask_four_dims[0]
        elif dim_num == 4:
            return mask_four_dims
        else:
            print("invalid dim num, will provide a four dim mask anyway")
            return mask_four_dims
        return mask_four_dims[0][0]
    
    def init_by_input_data(self, input_data: InputDataset):
        np.random.seed(1)
        torch.npu.synchronize()
        # # print(f"ljl-input_data:{input_data}")
        if input_data.kwargs["attenMaskOptional"] != None:
            mask_dim_num = len(input_data.kwargs["attenMaskOptional"].shape)
            mask_dtype = input_data.kwargs["attenMaskOptional"].dtype
            # print(f"ljl-mask_dim_num:{mask_dim_num}")
            input_data.kwargs["attenMaskOptional"] = self.gen_compressed_triU_mask(mask_dim_num, mask_dtype).npu()
        # input_data.kwargs["actualSeqLengthsOptional"] = [129,129]
        # input_data.kwargs["actualSeqLengthsKvOptional"] = [1024,1024]
        input_args = []  # 算子的入参列表
        input_args, output_packages = super().init_by_input_data(input_data)
        # print(f"ljl-input_data:,{input_data}")
        output_packages = []  # 算子的出参数据包列表
        # perfix参数需要置为空
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
        AclIntArrayPtr=ctypes.POINTER(AclIntArray)
        input_args[23] = ctypes.cast(None, AclIntArrayPtr)
        for i, (name, kwarg) in enumerate(input_data.kwargs.items()):
            if kwarg is None and self.task_result.case_config.inputs[i].type == "tensor":
                from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
                input_args[i] = TensorPtr()
        input_args.pop()
        if input_data.kwargs["softmaxLseFlag"] == True:
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            output_packages.append(input_args[-2])
        return input_args, output_packages
    
    def __call__(self):
        torch.npu.synchronize()
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(dtype=torch.float)
            output.append(temp_output_pack)
        return output
