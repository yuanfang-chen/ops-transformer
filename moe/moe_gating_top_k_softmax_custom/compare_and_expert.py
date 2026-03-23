import torch
import torch.nn as nn
import torch_npu
import torchair
import argparse
import os
import numpy as np

def cosine_similarity(src_np, dst_np):
    src_np = src_np.cpu().numpy().astype(np.float32)
    dst_np = dst_np.cpu().numpy().astype(np.float32)
    ab = np.sum(np.multiply(src_np, dst_np))
    aa = np.sqrt(np.sum(np.multiply(src_np, src_np)))
    bb = np.sqrt(np.sum(np.multiply(dst_np, dst_np)))
    cos = ab / (aa * bb)
    print("ab", ab)
    print("a*b", aa*bb)
    print(f"cosine similarity: {cos * 100} %")
    return cos

class MoeGatingTopkSoftmaxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, finish, k):
        res = torch_npu.npu_moe_gating_top_k_softmax(x, finish, k)
        return res

def load_input_data(input_path):
    data = torch.load(input_path).npu()
    return data

def softmax_func(x, axis=-1):
    """
    计算 Softmax 函数。
    
    Args:
        x: 输入数据（通常是张量或数组）。
        axis: 指定在哪个维度上进行 Softmax 操作。
    
    Returns:
        softmax: 计算得到的 Softmax 结果。
        exp_x: 指数计算结果（可选）。
        sum_exp_x: 指数求和结果（可选）。
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # 防止数值溢出
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    softmax = exp_x / sum_exp_x
    return softmax, exp_x, sum_exp_x

def cpu_op_exec(x, finished_optional, k):
    x_np = x.numpy()
    num_expert = x_np.shape[-1]
    softmax, _, _, = softmax_func(x_np, -1)
    expert_idx = np.argsort(-softmax, axis=-1, kind='stable')
    expert_idx = expert_idx[:, :k]
    y = np.take_along_axis(softmax, expert_idx, axis=-1)
    if finished_optional is not None:
        finished_optional = finished_optional.reshape(finished_optional.shape[0], 1)
        finished_optional = np.tile(finished_optional, (1, k))
        expert_idx = np.where(finished_optional, num_expert, expert_idx)
    row_idx = np.arange(y.shape[0] * y.shape[1]).reshape(y.shape[1], y.shape[0]).transpose(1, 0)
    if x_np.dtype == np.float16:
        y = y.astype(np.float16)
    return y, expert_idx.astype(np.int32), row_idx.astype(np.int32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MoeGatingTopkSoftmax Model Inference')
    parser.add_argument('--input', type=str, required=True, nargs='+', 
                       help='Paths to input data files (space separated)')
    args = parser.parse_args()

    moe_gating_topk_softmax_model = MoeGatingTopkSoftmaxModel().npu()

    x = load_input_data(args.input[0])
    golden_list = []

    print("$$$$ Begin Model Computing")
    k = 8
    print("k = ", k)
    res = moe_gating_topk_softmax_model(x, None, k)
    print("$$$$ End Model Computing")
    print(res[0].cpu())
    print(res[1].cpu())
    x_cpu = x.cpu()
    y, expert_idx, row_idx = cpu_op_exec(x_cpu, None, k)
    golden_list = [y, expert_idx]
    for i in range(len(golden_list)):
        expect = torch.from_numpy(golden_list[i])
        cosine_similarity(expect, res[i].cpu())

    
    print("save export air in ./export/")
    os.system("rm -rf ./export/*")
    torchair.dynamo_export(
        model=moe_gating_topk_softmax_model,
        export_path="./export",
        dynamic=False,
        x=x,
        finish=None,
        k=10
    )