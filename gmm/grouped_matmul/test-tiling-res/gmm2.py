import math
import numpy as np
import torch
import torch_npu


# ============== HIF8 转换核心函数 ==============

def _get_hif8_fraction_bits_number(exponent):
    """根据指数返回 HIF8 格式的位分配"""
    if exponent < -22:
        return -1, 3, 0
    if -22 <= exponent < -15:
        return 0, 3, 0
    if exponent == 0:
        return 1, 0, 3
    if abs(exponent) == 1:
        return 2, 1, 3
    if 2 <= abs(exponent) <= 3:
        return 4, 2, 3
    if 4 <= abs(exponent) <= 7:
        return 8, 3, 2
    if 8 <= abs(exponent) <= 15:
        return 12, 4, 1
    if exponent > 15:
        return 12, 4, -1


def _fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    """TA 舍入"""
    if exponent == -23:
        return True, 0
    hif8_value_tmp = fraction32_int >> (23 - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    elif hif8_value_tmp == 0:
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1


def cvt_float32_to_hifuint8(x):
    """单个 float32 → HIF8 (uint8)"""
    sign_int_value = 0
    x_abs = math.fabs(x)
    
    if x < 0.0:
        sign_int_value = 128
    
    if np.isinf(x) or x_abs >= 40960.0:
        return 239 if x < 0 else 111
    
    if np.isnan(x):
        return 128
    
    if x_abs == 0.0:
        return 0
    
    exponent = math.floor(math.log2(x_abs))
    fraction_int = int(x_abs * pow(2, 23 - exponent) - pow(2, 23))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = _get_hif8_fraction_bits_number(exponent)
    
    carry_exp_status, hif8_frac_value = _fp32_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    
    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = _get_hif8_fraction_bits_number(exponent)
    
    if exponent < -23:
        return 0
    
    sig_exp = 1 if exponent < 0 else 0
    
    if dot_hif8_value <= 0:
        return 0 if exponent <= -23 else sign_int_value + exponent + 23
    elif dot_hif8_value == 1:
        return sign_int_value + (dot_hif8_value << 3) + hif8_frac_value
    else:
        abs_exp = abs(exponent) - pow(2, exponent_hif8_bits - 1)
        exp_int_value = int(abs_exp) << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        return int(sign_int_value + (dot_hif8_value << 3) + sig_exp + exp_int_value + hif8_frac_value)


def fp32_tensor_to_hif8_int8(fp32_tensor):
    """float32 tensor → HIF8 (int8 存储)"""
    fp32_np = fp32_tensor.cpu().numpy().astype(np.float32).flatten()
    hif8_uint8 = np.array([cvt_float32_to_hifuint8(v) for v in fp32_np], dtype=np.uint8)
    hif8_int8 = hif8_uint8.view(np.int8).reshape(fp32_tensor.shape)
    return torch.from_numpy(hif8_int8)


# ============== 测试脚本 ==============

x_shape = (4, 3)
weight_shape = (1, 3, 2)
group_list = [4]

# 1. 以 randint 创建原始数据，再转换为 float32
x_fp32 = torch.randint(-5, 5, x_shape).to(torch.float32)
weight_fp32 = torch.randint(-5, 5, weight_shape).to(torch.float32)

# 2. 转换为 HIF8 (int8 存储)
x_hif8 = fp32_tensor_to_hif8_int8(x_fp32).npu()
weight_hif8 = fp32_tensor_to_hif8_int8(weight_fp32).npu()

# 3. 准备参数并调用算子
x2_scale = torch.randint(1, 3, (weight_shape[0], 1)).to(torch.float32).npu()
x1_scale = torch.randint(1, 3, (weight_shape[0], 1)).to(torch.float32).npu()
group_list_tensor = torch.tensor(group_list, dtype=torch.int64).npu()

npu_out = torch_npu.npu_grouped_matmul(
    [x_hif8], 
    [weight_hif8], 
    scale=[x2_scale], 
    per_token_scale=[x1_scale], 
    group_list=group_list_tensor, 
    split_item=2, 
    group_type=0, 
    output_dtype=torch.bfloat16, 
    x_dtype=torch_npu.hifloat8, 
    weight_dtype=torch_npu.hifloat8, 
    group_list_type=0
)

print(f"{x_shape=}")
print(f"{weight_shape=}")
print(f"{group_list=}")

print("=" * 50)
print("原始 float32 输入 (由 randint 生成):")
print(f"x_fp32 =\n{x_fp32}")
print(f"weight_fp32 =\n{weight_fp32}")

print("=" * 50)
print("HIF8 (int8 存储):")
print(f"x_hif8 =\n{x_hif8}")
print(f"weight_hif8 =\n{weight_hif8}")

print("=" * 50)
print("Scale 参数:")
print(f"x1_scale (per_token_scale) = {x1_scale}")
print(f"x2_scale (scale) = {x2_scale}")

print("=" * 50)
print("NPU 输出:")
print(f"npu_out = {npu_out}")

# 4. 计算预期输出
# 预期: (x_fp32 @ weight_fp32) * x1_scale * x2_scale
weight_2d = weight_fp32.squeeze(0)  # (1, 5, 4) -> (5, 4)
matmul_result = x_fp32 @ weight_2d  # (3, 5) @ (5, 4) -> (3, 4)
expected_out = matmul_result * x1_scale.cpu() * x2_scale.cpu()

print("=" * 50)
print("预期输出 (CPU 计算):")
print(f"x_fp32 @ weight_fp32 =\n{matmul_result}")
print(f"expected_out (乘以 scale 后) =\n{expected_out}")
print(f"expected_out (bfloat16) =\n{expected_out.to(torch.bfloat16)}")

# 5. 对比差异
print("=" * 50)
print("对比:")
npu_out_cpu = npu_out[0].cpu().to(torch.float32)
expected_bf16 = expected_out.to(torch.bfloat16).to(torch.float32)
diff = torch.abs(npu_out_cpu - expected_bf16)
print(f"绝对误差 =\n{diff}")
print(f"最大误差 = {diff.max().item()}")

