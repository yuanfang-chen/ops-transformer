# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from dataclasses import dataclass
import numpy as np
import math
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

import math
from typing import Optional
import random


def generate_tensor(shape, data_type, data_max):
    tensor = torch.rand(shape) * (data_max * 2) - data_max
    return tensor.to(data_type)

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def cdiv(a: torch.LongTensor
    , b : int):
    torch.empty
    return (a + b - 1) // b

def get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices):
    if cu_seqlens != None:
        seqIdx = chunk_indices[idx * 2]
        chunkIdx = chunk_indices[idx * 2 + 1]
        bos = cu_seqlens[seqIdx] + chunkIdx * chunk_size
        eos = bos + chunk_size
        if eos > cu_seqlens[seqIdx + 1]:
            eos = cu_seqlens[seqIdx + 1]
    else:
        bos = idx * chunk_size
        eos = bos + chunk_size
        if eos > T:
            eos = T
    # print(bos, eos)
    return bos,eos

def compute_dv_golden(
    A: torch.Tensor,      # [B, T, H, chunk_size] - 每个chunk的A值
    du: torch.Tensor,     # [B, T, H, D] - 上游梯度
    beta: torch.Tensor,   # [B, H, T] - beta参数
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    H: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, T, H, chunk_size]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    # 初始化dv，形状与du相同 [B, T, H, D]
    dv = torch.zeros_like(du)
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            # print(bos, eos, eos-bos)
        # 遍历所有batch
            for i_h in range(H):
                
            # 遍历所有head 
                # 获取当前chunk的A向量
                # A形状: [B, T, H, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,:eos - bos]  # [chunk_size, chunk_size]
                
                # 获取当前chunk的du
                du_chunk = du[i_b,i_h, bos:eos, :]  # [chunk_size, V]
                
                # 获取当前chunk的beta
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                
                # 计算 dv_chunk = A_chunk @ du_chunk * beta_chunk
                # 步骤1: b_dv_beta = A_chunk @ du_chunk
                b_dv_beta = torch.matmul(A_chunk.T.to(torch.float32), du_chunk.to(torch.float32))  # [chunk_size, V]
                
                # 步骤2: dv_chunk = b_dv_beta * beta_chunk.unsqueeze(-1)
                dv_chunk = b_dv_beta.to(torch.float32) * beta_chunk[:, None].to(torch.float32)  # [chunk_size, D]
                
                # 存储结果
                dv[i_b,i_h, bos:eos, :] = dv_chunk.to(dv.dtype)
    
    return dv


def compute_dk_golden(
    A: torch.Tensor,      # [B, T, H, chunk_size] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, H, D]
    g: torch.Tensor,     # [B, H, T]
    beta: torch.Tensor,   # [B, H, T] - beta参数
    dA: torch.Tensor,      # [B, T, H, chunk_size]
    k: torch.Tensor,     # [B, T, H, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    H: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, T, H, chunk_size]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dk = torch.zeros_like(k)
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
        # 遍历所有batch
            for i_h in range(H):
            # 遍历所有head 
                # 获取当前chunk的A向量
                # A形状: [B, T, H, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]
                
                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h, bos:eos, :]  # [chunk_size, D]
           
                # 获取当前chunk的beta,g
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunk_size]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))
                #   k________0
                k_chunk = k[i_b, i_h, bos:eos, : ]
                dA_chunk = dA[i_b,i_h, bos:eos,:eos-bos]  # [chunk_size, chunk_size]
                b_dk_beta = torch.matmul(dA_chunk.T.to(torch.float32), k_chunk.to(torch.float32))  # [chunk_size, D]
                #   k________1
                b_kt_beta = k_chunk.T.to(torch.float32) * beta_chunk.to(torch.float32)[None,: ]
                b_k_beta = k_chunk.to(torch.float32) * beta_chunk.to(torch.float32)[:, None]
                #   k________2
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T.to(torch.float32), dw_chunk.to(torch.float32))  # [chunk_size, D]
                
                # 步骤2: dk_chunk = b_dk_beta_g * beta_chunk[:, None] * g_exp_chunk[:, None] 
                dk_chunk = torch.matmul(dA_chunk.to(torch.float32), b_k_beta.to(k.dtype).to(torch.float32)).to(k.dtype).to(torch.float32)  # [chunk_size, D]
                dk_chunk = dk_chunk.to(k.dtype).to(torch.float32) + (b_dk_beta.to(k.dtype).to(torch.float32) * beta_chunk[:, None].to(torch.float32))
                dk_chunk = dk_chunk.to(k.dtype).to(torch.float32) + b_dk_beta_g.to(k.dtype).to(torch.float32) * (beta_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32))[:,None]  # [chunk_size, D]
                # 存储结果
                dk[i_b,i_h, bos:eos, :] = dk_chunk
    return dk

def compute_dg_golden(
    A: torch.Tensor,      # [B, T, H, chunk_size] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, H, D]
    g: torch.Tensor,     # [B, H, T]
    beta: torch.Tensor,   # [B, H, T] - beta参数
    dA: torch.Tensor,      # [B, T, H, chunk_size]
    k: torch.Tensor,     # [B, T, H, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    H: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, T, H, chunk_size]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dg = torch.zeros_like(g)
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
        # 遍历所有batch
            for i_h in range(H):

            # 遍历所有head 
                # 获取当前chunk的A向量
                # A形状: [B, T, H, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]
                
                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h, bos:eos, :]  # [chunk_size, D]
            
                # 获取当前chunk的beta,g
                # beta形状: [B, H, T]
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunk_size]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))
                
                #   g________0
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T.to(torch.float32), dw_chunk.to(torch.float32))  
 
                # 步骤2: b_dg += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None] * b_beta[:, None], 1)
                k_chunk = k[i_b, i_h, bos:eos, : ]
                b_kbg = k_chunk.to(torch.float32) * (beta_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32))[:,None]
                #   g________1 
                dA_chunk = dA[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]
                if k_chunk.size(0) == 1:
                    # 形状 [1, K] -> 计算外积等价于平方和
                    # 结果应为 [1, 1]
                    dot_val = torch.sum(k_chunk.to(torch.float32) * k_chunk.to(torch.float32), dim=1, keepdim=True)  # [1, 1]
                    b_A = dot_val
                else:
                    # 正常走 matmul 路径
                    k_f32 = k_chunk.to(torch.float32).contiguous()
                    b_A = torch.matmul(k_f32, k_f32.T.contiguous())
                # b_A = torch.matmul(k_chunk.to(torch.float32).contiguous(), k_chunk.to(torch.float32).T).to(k.dtype).to(torch.float32).contiguous()  # [chunk_size, chunk_size]
                b_A = b_A.to(torch.float32) * beta_chunk[:,None].to(torch.float32)
                b_dA_A = dA_chunk.to(torch.float32).T * b_A.to(torch.float32)

                # test
                dg_chunk = torch.sum(b_dk_beta_g.to(k.dtype).to(torch.float32) * b_kbg.to(torch.float32), dim = 1)# [chunk_size]
                dg_chunk = dg_chunk.to(dg.dtype).to(torch.float32) +  (torch.sum(b_dA_A, dim = 1) - torch.sum(b_dA_A, dim = 0))
                # 存储结果
                dg[i_b,i_h, bos:eos] = dg_chunk.to(k.dtype)
    return dg

def compute_dbeta_golden(
    A: torch.Tensor,      # [B, T, H, chunkSize] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, H, D]
    g: torch.Tensor,     # [B, H, T]
    beta: torch.Tensor,   # [B, H, T] - beta参数
    dA: torch.Tensor,      # [B, T, H, chunkSize]
    k: torch.Tensor,     # [B, T, H, D]
    v: torch.Tensor,      # [B, T, H, D]
    du: torch.Tensor,     # [B, T, H, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    H: int,
    T: int,
    D: int,
    chunkSize: int,  # chunkSize
    NT: int,  # T / chunkSize
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, T, H, chunkSize]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dbeta = torch.zeros_like(beta)
    for i_b in range(B):
        for idx in range(NT):
        # 遍历所有batch
            bos,eos = get_bos_eos(idx, T, chunkSize, cu_seqlens, chunk_indices)
            for i_h in range(H):
            # 遍历所有head 
                # 获取当前chunk的A向量
                # A形状: [B, T, H, chunkSize]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunkSize, chunkSize]
                v_chunk = v[i_b,i_h, bos:eos,:]  # [chunkSize, V]
                
                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h, bos:eos, :]  # [chunkSize, D]
                du_chunk = du[i_b,i_h, bos:eos, :]  # [chunkSize, D]


                # 获取当前chunk的beta,g
                # beta形状: [B, H, T]
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunkSize]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunkSize]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))
                k_chunk = k[i_b, i_h, bos:eos, : ]
                #   beta________0
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T.to(torch.float32), dw_chunk.to(torch.float32))  
                
                # 步骤2: b_dbeta += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None], 1)
                tmp = b_dk_beta_g * k_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32)[:, None] # [chunkSize, D]
                #   beta________1 
                
                b_dv_beta = torch.matmul(A_chunk.T.to(torch.float32), du_chunk.to(torch.float32))  # [chunkSize, V]
                #   beta________2 
                dA_chunk = dA[i_b,i_h, bos:eos,: eos - bos]  # [chunkSize, chunkSize]
                b_dk_beta = torch.matmul(dA_chunk.T.to(torch.float32), k_chunk.to(torch.float32))  # [chunkSize, V]
                tmp = b_dk_beta.to(k.dtype).to(torch.float32) * k_chunk
                # test
                dbeta_chunk = torch.sum(tmp, 1).to(torch.float16)
                tmp = b_dk_beta_g.to(k.dtype).to(torch.float32) * k_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32)[:, None] # [chunkSize, D]
                dbeta_chunk = dbeta_chunk.to(k.dtype).to(torch.float32) + torch.sum(tmp, dim = 1)# [chunkSize]
                dbeta_chunk = dbeta_chunk.to(k.dtype).to(torch.float32) + torch.sum(b_dv_beta.to(k.dtype).to(torch.float32) * v_chunk, 1)
                # 存储结果
                dbeta[i_b,i_h, bos:eos] = dbeta_chunk
    return dbeta

def prepare_cu_seqlens(T: int, L: int = 32, seed: int = 42):
    """
    直接生成一个长度为 L 的 cu_seqlens 列表 (list[int])：
      - 以 0 开头，以 T 结尾
      - 严格单调递增，无重复
      - 所有值在 [0, T] 范围内
      - 可复现（固定随机种子）
      
    此函数完全避开 torch.Tensor，直接返回 Python 原生 list，
    完美适配 npu 算子对 'Optional[list[int]]' 的类型要求。

    Args:
        T (int): 最大值（总 token 数）
        L (int): 输出列表的长度（必须满足 2 <= L <= T + 1）
        seed (int): 随机种子，默认 42

    Returns:
        list[int]: 例如 [0, 15, 32, ..., T]
    """
    if T < 1:
        raise ValueError("T must be at least 1.")
    if L < 2 or L > T + 1:
        raise ValueError(f"L must satisfy 2 <= L <= T + 1 (got L={L}, T={T}).")

    # 固定随机种子 (使用 Python 标准库)
    random.seed(seed)

    if L == 2:
        # 最简单情况：[0, T]
        return [0, T]

    # 需要在 (0, T) 开区间内选择 L - 2 个不重复的整数作为中间点
    # 候选集合：1, 2, ..., T-1
    # random.sample 直接返回不重复的列表，无需担心重复
    middle_points = random.sample(range(1, T), L - 2)
    
    # 必须排序以保证单调递增
    middle_points.sort()

    # 拼接：0 + 中间点 + T
    # 这里的 0, middle_points 中的元素, T 都是纯 Python int
    cu_seqlens = [0] + middle_points + [T]

    return cu_seqlens

def prepare_chunk_indices(
    cu_seqlens,
    chunk_size
): 
    """
    基于 cu_seqlens (list[int]) 生成 chunk 索引。
    
    注意：原 PyTorch 版本返回的是 shape [N, 2] 的 Tensor。
    为了保持纯 Python 兼容性，这里返回 list[tuple[start_seq_idx, chunk_idx_in_seq]]。
    如果算子需要扁平化的 list[int] (如 [s0, c0, s1, c1, ...])，请在调用前展开。
    
    逻辑复刻原代码：
    1. 计算每个序列的长度: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    2. 计算每个序列需要的 chunk 数: ceil(lens[i] / chunk_size)
    3. 生成对应的 (sequence_id, chunk_id) 对
    """
    indices = []
    
    # 遍历每个序列段
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        length = end - start
        
        if length <= 0:
            continue
            
        # 计算该序列需要多少个 chunk
        # 等价于 cdiv(length, chunk_size)
        num_chunks = (length + chunk_size - 1) // chunk_size
        
        for chunk_id in range(num_chunks):
            # 原逻辑: indices.eq(0).cumsum(0) - 1 对应的是序列索引 i
            # 原逻辑: indices 对应的是 chunk_id
            indices.append((i))
            indices.append((chunk_id))
            
    return indices




@register("executor_prepare_wy_repr_bwd_full")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.qkv_type = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        beta = input_data.kwargs["beta"]
        A = input_data.kwargs["A"]
        dA = input_data.kwargs["dA"]
        dw = input_data.kwargs["dw"]
        du = input_data.kwargs["du"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        B, H, T, K = input_data.kwargs["k"].shape
        V = input_data.kwargs["v"].shape[3]
        chunk_size = input_data.kwargs["chunk_size"]
        if chunk_indices!=None:
            NT = len(chunk_indices) // 2
        else:
            NT = (T + chunk_size - 1) // chunk_size
        print("zslllllll", B, H, T, K, V,chunk_size,NT)
        print(chunk_indices)
        # print("vvvvvvvvvvvvvvvvvvv", v)
        dk = compute_dk_golden(A, dw, g, beta, dA,k, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
        dv = compute_dv_golden(A, du, beta, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
        dbeta = compute_dbeta_golden(A, dw, g, beta, dA,k,v,du, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
        dg = compute_dg_golden(A, dw, g, beta, dA,k, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)

        dk = dk.to(k.dtype)
        dv = dv.to(k.dtype)
        dbeta = dbeta.to(k.dtype)
        dg = dg.to(k.dtype)
        print("zsl----------")

        return dk,dv,dbeta,dg

    def init_by_input_data(self, input_data: InputDataset):
        B, H, T, K = input_data.kwargs["k"].shape
        V = input_data.kwargs["v"].shape[3]

        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        beta = input_data.kwargs["beta"]
        A = input_data.kwargs["A"]
        dA = input_data.kwargs["dA"]
        dw = input_data.kwargs["dw"]
        du = input_data.kwargs["du"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]

        is_fix =  input_data.kwargs["is_fix"]
        self.qkv_type =  input_data.kwargs["qkv_type"]
        is_mix =  input_data.kwargs["is_mix"]

        print("zsl chunk_size = ",chunk_size)

        # k = generate_tensor((B, H, T, K), k.dtype, 5)
        # v = generate_tensor((B, H, T, V), k.dtype, 5)
        # A = generate_tensor((B, H, T, chunk_size), k.dtype, 5)
        # dA = generate_tensor((B, H, T, chunk_size), k.dtype, 5)
        # dw = generate_tensor((B, H, T, K), k.dtype, 5)
        # du = generate_tensor((B, H, T, V), k.dtype, 5)
        # # T = cu_seqlens[-1]
        # if is_mix:
        #     g = generate_tensor((B, H, T), torch.float32, 5)
        #     beta = generate_tensor((B, H, T), torch.float32, 5)
        # else:
        #     g = generate_tensor((B, H, T), k.dtype, 5)
        #     beta = generate_tensor((B, H, T), k.dtype, 5)
        print(is_fix)
        if is_fix:
            cu_seqlens = None
            chunk_indices = None
        else:
            # 构造cu_seqlens
            cu_seqlens = prepare_cu_seqlens(T, 3)
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        qkv_type = input_data.kwargs["k"].dtype
        print("qkv_type = ",qkv_type)
        g_type = input_data.kwargs["g"].dtype
        is_mix =  input_data.kwargs["is_mix"]


        if self.device == "pyaclnn":
            k = k.npu()
            v = v.npu()
            beta = beta.npu()
            A = A.npu()
            dA = dA.npu()
            dw = dw.npu()
            du = du.npu()
            g = g.npu()

        input_data.kwargs['k'] = k
        input_data.kwargs['v'] = v
        input_data.kwargs['beta'] = beta
        input_data.kwargs['A'] = A
        input_data.kwargs['dA'] = dA
        input_data.kwargs['dw'] = dw
        input_data.kwargs['du'] = du
        input_data.kwargs['g'] = g
        input_data.kwargs['cu_seqlens'] = cu_seqlens
        input_data.kwargs['chunk_indices'] = chunk_indices
        input_data.kwargs.pop("is_mix")
        input_data.kwargs.pop("is_fix")
        input_data.kwargs.pop("qkv_type")

        # print(input_args)
        # input_args[9] = ctypes.c_void_p(ctypes.addressof(input_args[9]))
        # input_args[10] = ctypes.c_void_p(ctypes.addressof(input_args[10]))

@register("executor_prepare_wy_repr_bwd_full")
class FunctionApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)

    # def init_by_input_data(self, input_data: InputDataset):
    #     import ctypes
    #     input_args, _ = super().init_by_input_data(input_data)
    #     print(input_args[9])

    # def get_cpp_func_signature_type(self):
    #     return "aclnnStatus aclnnPrepareWyReprBwdFullGetWorkspaceSize(\
    #         const aclTensor *k,\
    #         const aclTensor *v,\
    #         const aclTensor *beta,\
    #         const aclTensor *a,\
    #         const aclTensor *dA,\
    #         const aclTensor *dw,\
    #         const aclTensor *du,\
    #         const aclTensor *g,\
    #         const aclIntArray *cuSeqlensOptional,\
    #         const aclIntArray *chunkIndicesOptional,\
    #         int64_t chunkSize,\
    #         const aclTensor *dkOut,\
    #         const aclTensor *dvOut,\
    #         const aclTensor *dbetaOut,\
    #         const aclTensor *dgOut,\
    #         uint64_t *workspaceSize,\
    #         aclOpExecutor **executor);"
