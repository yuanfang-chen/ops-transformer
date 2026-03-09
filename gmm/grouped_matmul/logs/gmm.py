import math
import torch
import torch_npu
M = 576
K = 512
N = 7168
g = 4
x = torch.randint(-1, 1, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()

weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(1,2)

x2_scale = torch.randint(-1, 1, (g, N, math.ceil(K/64), 2), dtype=torch.int8).npu().transpose(1,2)
x1_scale = torch.randint(-1, 1, (M, math.ceil(K/64), 2), dtype=torch.int8).npu()

group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
split_item = 2
npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)

