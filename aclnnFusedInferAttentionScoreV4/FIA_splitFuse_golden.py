import torch
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
np.random.seed(1)

import torch
import numpy as np

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

def overwrite_structured_mask(input_data):
    """
    根据 sparseMode 强制修改 attenMask 的数值结构：
    Mode 2/3: 下三角保留 (Causal)，上三角遮蔽。
    Mode 4:   Band 结构。
    """
    mode = input_data.kwargs.get('sparseMode', 0)
    mask_tensor = input_data.kwargs.get('attenMaskOptional', None)
    
    # 如果没有 mask 或者 mode 是 0/1 (Default/All)，通常保持随机或全零即可，不做强制修改
    # (或者根据需求，Mode 0 也可以重写，这里主要处理 2,3,4)
    if mask_tensor is None or mode not in [2, 3, 4]:
        return

    # 获取 Mask 的 Shape (预期是 2048x2048 或 [B, 1, 2048, 2048])
    shape = mask_tensor.shape
    # 获取最后两个维度
    S = shape[-2]
    KVS = shape[-1]
    
    # 构造标准的结构化 Mask (numpy)
    # PFA 定义: 1 代表遮蔽(Masked), 0 代表保留(Keep)
    new_mask = np.zeros((S, KVS), dtype=np.int8)
    
    if mode == 2 or mode == 3: 
        # Mode 2 (LeftUpCausal) / Mode 3 (RightDownCausal)
        # 构造上三角 Mask (k=1 表示对角线往上一格开始全是 1)
        new_mask = np.triu(np.ones((S, KVS), dtype=np.int8), k=1)
        
    elif mode == 4:
        # Mode 4 (Band)
        # 需要读取 preTokens 和 nextTokens
        # 注意：Generator生成的可能是 Tensor 或 int，要做兼容处理
        pre_t = input_data.kwargs.get('preTokens', 2147483647)
        next_t = input_data.kwargs.get('nextTokens', 2147483647)
        
        # 如果是 Tensor/List 取第一个值简化处理 (因为 Mode 4 Mask 是固定的 2048x2048)
        if hasattr(pre_t, 'item'): pre_t = pre_t.item()
        if isinstance(pre_t, (list, tuple)): pre_t = pre_t[0]
        if hasattr(next_t, 'item'): next_t = next_t.item()
        if isinstance(next_t, (list, tuple)): next_t = next_t[0]
        
        # 利用广播机制生成 Band
        rows = np.arange(S)[:, None]
        cols = np.arange(KVS)[None, :]
        # 遮蔽条件: j > i + next  OR  j < i - pre
        mask_condition = (cols > rows + next_t) | (cols < rows - pre_t)
        new_mask[mask_condition] = 1

    # --- 将构造好的 2D Mask 广播回原始 Shape ---
    
    # 1. 转回 Tensor
    # 保持和原 Mask 相同的 dtype (通常是 bool 或 int8)
    orig_dtype = torch.bool if mask_tensor.dtype == torch.float16 else torch.int8
    new_mask_tensor = torch.from_numpy(new_mask).to(orig_dtype)
    
    # 2. 恢复维度 (Broadcast)
    # 如果原 Mask 是 [B, 1, 2048, 2048]，需要把 2D 扩展回去
    if len(shape) == 4:
        # [2048, 2048] -> [1, 1, 2048, 2048] -> [B, 1, 2048, 2048]
        new_mask_tensor = new_mask_tensor.unsqueeze(0).unsqueeze(0)
        new_mask_tensor = new_mask_tensor.expand(shape[0], shape[1], -1, -1).contiguous()
    elif len(shape) == 3:
        new_mask_tensor = new_mask_tensor.unsqueeze(0)
        new_mask_tensor = new_mask_tensor.expand(shape[0], -1, -1).contiguous()
        
    # 3. 覆盖回 input_data
    input_data.kwargs['attenMaskOptional'] = new_mask_tensor
    return input_data

def safe_to_tensor(arr):
        if arr is None: 
            return None
        # 如果是 bfloat16，先转 float32 再转 torch
        if arr.dtype.name == 'bfloat16':
            return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
        # 其他类型直接转换
        return torch.from_numpy(arr)

dtypeMap = {
    torch.float16: np.float16,
    torch.bfloat16: bfloat16,
    torch.float32: np.float32
}

maskTypeMap = {
    ## sparseMode : golden maskType
    0: 0,
    3: 1
}

class TestFIAV3SplitFuse():
    @dataclass
    class AuxAttrs:
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
    def group_matmul(cls, head, kv_head, left, right, high_prec = 1):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            if high_prec == 0:
                group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :],
                                        right[i:(i + 1), :, :]).astype(np.float32)
            else:
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

        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)

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
        sim = qk_result.astype(interm_dtype)
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
        qk_k_split = 128
        qk_k_loop = (qk_k + 127) // 128
        for qk_k_loop_idx in range(qk_k_loop):
            sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
            partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
            partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
            result_split = self.group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result
    
    def pvMM2(
        self,
        p,
        value
    ):
        result = None
        pv_k = value.shape[1]
        pv_k_split = 128
        pv_k_loop = (pv_k + 127) // 128
        for pv_k_loop_idx in range(pv_k_loop):
            sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)

            partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
            # query_k = query[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
            partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :] 
            # key_k = key[:, qk_k_split: qk_k_split + sub_k, :]
            result_split = self.group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
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
        interm_dtype = np.float16 if inner_prec == 1 else np.float32
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        value = np.transpose(value, (1, 0, 2))
        scale = np.float16(scale) if inner_prec == 1 else np.float32(scale)
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
                # print(sub_len)
            sub_key = key[:, :, kv_start: kv_start + sub_len]
            sub_mask = None
            if mask is not None:
                sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len].astype(interm_dtype)
            sub_value = value[:, kv_start: kv_start + sub_len, :]
            qk_result = self.qkMM1(query, sub_key).astype(interm_dtype)
            # qk_result_high = self.qkMM1(query.astype(np.float32), sub_key.astype(np.float32))
            qk_result = qk_result * scale
            # qk_result_high = qk_result_high * scale

            if mask is not None:
                qk_result += sub_mask
                # qk_result_high += sub_mask.astype(np.float32)
            if kv_start == 0:
                gm = None
            p_result, row_sum, dm, gm = self.softmax1(qk_result, kv_start == 0, gm, interm_dtype)
            p_result = p_result.astype(data_type)
            if kv_start == 0:
                gm_high = None
            # p_result_high, row_sum_high, dm_high, gm_high = self.softmax1(qk_result_high, kv_start == 0, gm_high)
            lo = self.pvMM2(p_result, sub_value).astype(interm_dtype)
            # lo_high = self.pvMM2(p_result_high.astype(np.float32), sub_value.astype(np.float32))
            if kv_start == 0:
                gl = row_sum
                # gl_high = row_sum_high
                go = lo
                # go_high = lo_high
            else:
                dm = np.exp(dm)
                # dm_high = np.exp(dm_high)
                gl = gl * dm
                gl = gl + row_sum

                go = go * dm
                go = go + lo

                # gl_high = gl_high * dm_high
                # gl_high = gl_high + row_sum_high

                # go_high = go_high * dm_high
                # go_high = go_high + lo_high
        go = go / gl
        # go_high = go_high / gl_high
        go = np.transpose(go, (1, 0, 2))
        # go_high = np.transpose(go_high, (1, 0, 2))
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32) # lse仅支持fp32输出，无论采取何种精度运算，最终结果都是fp32
        # lse_high = np.squeeze((np.log(gl_high) + gm_high), axis=-1)
        return go.astype(data_type), lse

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask    # (q_seqlen, k_seqlen)
    ):
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key, 1)  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float32)
        p_high, lse_high, gm = self.softmax_numpy(sim_high)
        # lse = lse_high.astype(query.dtype)
        lse_high = lse_high.astype(np.float32)
        # lse_high = lse_high.astype(np.float32)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))
        
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value, 1)
        # out = self.group_matmul(query.shape[0], key.shape[0], p, value, 1)
        out_high = np.transpose(out_high, (1, 0, 2))
        # out = np.transpose(out, (1, 0, 2))
        # out = out.astype(query.dtype)
        return out_high, lse_high

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
            
            if attention_inputs.auxAttrs.mask_type == 1:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            elif attention_inputs.auxAttrs.mask_type == 0:
                mask = None
            out_normal, lse = self.ref_masked_attention(q, keys, values, scale, mask)
            out_gpu, lse_gpu = self.ref_flash_attention(q, keys, values, scale, mask, attention_inputs)

            out = out_normal.reshape(-1, num_heads, head_size_vo)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_gpu = out_gpu.reshape(-1, num_heads, head_size_vo)

            if attention_inputs.auxAttrs.layout_dtype == 1:
                output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse_gpu
            else:
                output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse_gpu
            
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


def aclnn_op_func_fia_split_fuse_golden(input_data : InputDataset, is_benchmark_task):
    input_data_dtype = input_data.kwargs["query"].dtype
    query = input_data.kwargs["query"].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["query"].to(torch.float32).numpy().astype(bfloat16)
    key = input_data.kwargs["key"][0].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["key"][0].to(torch.float32).numpy().astype(bfloat16)
    value = input_data.kwargs["value"][0].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["value"][0].to(torch.float32).numpy().astype(bfloat16)
    blockTable = None
    pagedAttentionFlag = False
    if input_data.kwargs["blockTableOptional"] != None:
        blockTable = input_data.kwargs["blockTableOptional"].numpy()
        pagedAttentionFlag = True
    ## gen actual seqlen
    inputLayout = input_data.kwargs["inputLayout"]
    # print(input_data.kwargs["actualSeqLengthsOptional"][0])
    batch = len(input_data.kwargs["actualSeqLengthsOptional"])
    actualseqlengths = [0] * batch
    actualseqlengthsKv = [0] * batch
    for i in range(batch):
        actualseqlengths[i] = input_data.kwargs["actualSeqLengthsOptional"][i]
        actualseqlengthsKv[i] = input_data.kwargs["actualSeqLengthsKvOptional"][i]
    qSeqlenList, kvSeqlenList = gen_actual_seqlen_list_golden(actualseqlengths, actualseqlengthsKv, inputLayout, pagedAttentionFlag)
    maxKvSeqlen = max(kvSeqlenList)
    maxQSeqlen = max(qSeqlenList)
    totalQTokens = sum(qSeqlenList)
    ## gen mask
    fullMask = None
    pre_mask_factor = -3e38 if input_data_dtype == torch.bfloat16 else -6e4
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
    # print(key.shape)
    testObj = TestFIAV3SplitFuse()
    auxAttrs = testObj.AuxAttrs(numHeads, kvHeads, headSize, numBlocks, blockSize, maskType, dtype, kvOrgMode, layoutMode, maxQSeqlen, maxKvSeqlen, goldenGpuPrecision, scale)
    attentionInputs = testObj.AttentionInputs(query, key, value, blockTable, qSeqlenList, kvSeqlenList, fullMask, auxAttrs)

    golden_output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output = testObj.calc_data(attentionInputs)
    # 使用辅助函数进行转换
    golden_output = safe_to_tensor(golden_output)
    golden_gpu_output = safe_to_tensor(golden_gpu_output)
    golden_lse_output = safe_to_tensor(golden_lse_output)
    golden_gpu_lse_output = safe_to_tensor(golden_gpu_lse_output)
    if not softmaxLseFlag:
        golden_lse_output = torch.tensor([])
        golden_gpu_lse_output = torch.tensor([])

    if not is_benchmark_task:
        # 标杆返回这个
        return golden_gpu_output, golden_gpu_lse_output
    else:
        # 真值返回下面的
        return golden_output, golden_lse_output

@register("executor_fused_infer_attention_score_v3")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)
    
    def init_by_input_data(self, input_data: InputDataset):
        
        input_data = overwrite_structured_mask(input_data)
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.name == "perf" or "abnormal" in self.task_result.case_config.name:
            if input_data.kwargs["softmaxLseFlag"]:
                return torch.Tensor([1]),torch.Tensor([1])
            else:
                return torch.Tensor([1])
        if self.name == "gpu":
            for k,v in input_data.kwargs.items():
                if isinstance(v, list):
                    # 2. 检查列表中的第一个元素（或任何元素）是否是 Tensor
                    # 假设列表中的元素类型是一致的，我们只检查第一个元素
                    if v and isinstance(v[0], torch.Tensor): 
                        # 使用列表推导式创建新列表，将每个 Tensor 移动到 CPU
                        input_data.kwargs[k] = [t.to("cpu") for t in v]
                # 3. 兼容单个 Tensor 的情况 (如果 v 本身不是列表，而是单个 Tensor)
                elif isinstance(v, torch.Tensor):
                    input_data.kwargs[k] = v.to("cpu")
        output, output_lse = aclnn_op_func_fia_split_fuse_golden(input_data, self.task_result.is_benchmark_task)
        softmaxLseFlag = input_data.kwargs["softmaxLseFlag"]
        if  softmaxLseFlag:
            return output,output_lse
        else:
            return output

@register("executor_aclnn_fused_infer_attention_score_v3")
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
        torch.npu.synchronize()
        # print("aclnn input_data is:", input_data)
        # for i in range(input_data.size):
        #     print("aclnn input_data is:", i, input_data.shape)
        if input_data.kwargs.get("attenMaskOptional") is not None:
            mask_dim_num = len(input_data.kwargs["attenMaskOptional"].shape)
            mask_dtype = input_data.kwargs["attenMaskOptional"].dtype
            input_data.kwargs["attenMaskOptional"] = self.gen_compressed_triU_mask(mask_dim_num, mask_dtype).npu()
        # block_table = input_data.kwargs["blockTableOptional"]
        # inspect_kwargs(input_data.kwargs)
        # print("========block_table=======:", block_table)
        input_args = []  # 算子的入参列表
        input_args, output_packages = super().init_by_input_data(input_data)
        # 将所有type是tensor values是None的输入 改为AclTensor类型的空指针
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
        for i, (name, kwarg) in enumerate(input_data.kwargs.items()):
            if kwarg is None and self.task_result.case_config.inputs[i].type == "tensor":
                from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
                input_args[i] = TensorPtr()
        # V4兼容V3 以下参数需要为空指针 dequantScaleQueryOptional learnableSinkOptional queryQuantMode
        input_args.insert(27,TensorPtr())
        input_args.insert(28,TensorPtr())
        input_args.insert(42, ctypes.c_long())
        output_packages = []  # 算子的出参数据包列表
        if  input_data.kwargs["softmaxLseFlag"]:
            input_args.pop()
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            input_args.pop()
            output_packages.append(input_args[-2])
        return input_args, output_packages
    
    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(dtype=torch.float)
            output.append(temp_output_pack)
        # print("aclnn_output is: ", output)
        return output