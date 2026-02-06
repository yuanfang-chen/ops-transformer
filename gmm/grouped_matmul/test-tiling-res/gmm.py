import torch
import torch_npu

x_shape = (234,512)
weight_shape=(1,7168,512)
group_list=[234]
print(f"{x_shape=}")
print(f"{weight_shape=}")
print(f"{group_list=}")
x = torch.randint(-1, 1, x_shape, dtype=torch.int8).npu()

weight = torch.randint(-1, 1, weight_shape, dtype=torch.int8).npu().transpose(1,2)

x2_scale = torch.randint(-1, 1, (weight_shape[0], 1), dtype=torch.float32).npu()
x1_scale = torch.randint(-1, 1, (weight_shape[0],1), dtype=torch.float32).npu()

group_list = torch.Tensor(group_list).to(torch.int64).npu()
split_item = 2
npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch_npu.hifloat8, weight_dtype=torch_npu.hifloat8, group_list_type=0)
print(f"{npu_out=}")
