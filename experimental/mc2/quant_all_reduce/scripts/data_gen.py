# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import logging
import argparse
from typing import Union, Dict, Type
import numpy as np
from ml_dtypes import float8_e5m2, float8_e8m0fnu, bfloat16

# 3. 日志配置（仅初始化一次，全局生效）
logging.basicConfig(level=logging.INFO, format='%(message)s')

class QuantAllReduceGoldenGenerator:
    """
    quant_all_reduce golden data grenerator
    计算公式: output = reduce(allgatherscales * allgatherdata)
    """
    # 类常量：数据类型范围映射（所有实例共享）
    DTYPE_RANGE: Dict[Type, tuple] = {
        np.int8: (-128, 127),
        np.int16: (-32768, 32767),
        np.float16: (-65504, 65504),
        np.float32: (-1e38, 1e38),
        bfloat16: (-1e38, 1e38), 
        float8_e5m2: (1e-9, 1e6),  
        float8_e8m0fnu: (2**-126, 2**127),  
    }

    # 类常量：数据类型映射表（命令行参数→实际类型）
    TYPE_MAP: Dict[str, Type] = {
        "int": np.int32,
        "int32_t": np.int32,
        "float16_t": np.float16,
        "float32_t": np.float32,
        "int8_t": np.int8,
        "fp8_e5m2_t": float8_e5m2,
        "fp8_e8m0_t": float8_e8m0fnu,
        "bfloat16_t": bfloat16
    }

    def __init__(self, args: argparse.Namespace):
        """
        初始化grenerator
        :param args: 命令行参数Namespace对象
        """
        # 保存命令行参数为实例属性
        self.case_name = args.case_name
        self.bs = args.bs
        self.hidden_size = args.hidden_size
        self.input_tensor_range = args.input_tensor_range
        self.input_tensor_type = self.TYPE_MAP.get(args.input_tensor_type, np.float16)
        self.scales_range = args.scales_range
        self.scales_type = self.TYPE_MAP.get(args.scales_type, np.float16)
        self.output_type = self.TYPE_MAP.get(args.output_type, np.float16)
        self.ranksize = args.ranksize
        self.reduce_op = args.reduce_op
        self.mxfp = args.mxfp
        self.seed = args.seed

        # 计算输入/scale长度（适配mxfp模式）
        self.input_len = self.bs * self.hidden_size
        self.scale_len = self._calc_scale_len()

        # 初始化Golden数据目录
        self.golden_dir = self._init_golden_dir()

        # 设置随机种子（保证结果可复现）
        np.random.seed(self.seed)

    def _calc_scale_len(self) -> int:
        """私有方法: 根据不同的量化类型计算scale长度"""
        if self.mxfp == 0:
            return self.bs * (self.hidden_size // 128)
        else:
            return self.bs * (self.hidden_size // 64) * 2

    def _init_golden_dir(self) -> str:
        """私有方法: 初始化golden的数据存放目录"""
        golden_dir = f"./golden/quantallreduce_{self.case_name}_{self.bs}_{self.hidden_size}"
        # 创建目录（不存在则创建，存在则不报错）
        os.makedirs(golden_dir, exist_ok=True)
        # 清空目录原有文件（避免残留旧数据）
        for file in os.listdir(golden_dir):
            file_path = os.path.join(golden_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return golden_dir

    def gen_random_data(self, size: int, dtype: Union[np.dtype, Type], drange: str) -> np.ndarray:
        """
        根据数据类型和数据范围生成随机数据
        :param size: 数据长度
        :param dtype: 目标数据类型
        :param drange: 数值范围
        :return: 符合要求的随机数组
        """
        # 解析数值范围
        try:
            low, high = map(float, drange.split())
            if low >= high:
                raise ValueError(f"无效的数值范围! low={low} >= high={high}")
        except Exception as e:
            raise ValueError(f"解析数值范围失败：{e}") from e

        # 校验数据类型并裁剪范围
        if dtype not in self.DTYPE_RANGE:
            clip_low, clip_high = low, high
            logging.info(f"未找到{dtype}的预设范围，使用输入范围：{drange}")
        else:
            dtype_low, dtype_high = self.DTYPE_RANGE[dtype]
            clip_low = max(low, dtype_low)
            clip_high = min(high, dtype_high)
            if clip_low >= clip_high:
                raise ValueError(f"{dtype}的预设范围{self.DTYPE_RANGE[dtype]}与输入范围{drange}无交集")

        # 生成随机数（先按float32生成，再做类型转换）
        random_data = np.random.uniform(low=clip_low, high=clip_high, size=size).astype(np.float32)

        # 特殊类型处理（适配FP8格式）
        if dtype == float8_e8m0fnu:
            log2_data = np.log2(np.abs(random_data) + 1e-10)  # 避免log2(0)报错
            round_log2 = np.round(log2_data)
            random_data = np.power(2, round_log2) * np.sign(random_data)
        elif "float8" in str(dtype):
            random_data = np.clip(random_data, dtype_low, dtype_high)

        # 转换为目标类型并检查NaN
        target_data = random_data.astype(dtype)
        nan_count = np.isnan(target_data.astype(np.float32)).sum()
        if nan_count > 0:
            raise RuntimeError(f"生成的数据包含{nan_count}个NaN值, 请检查参数!")

        return target_data

    def input_generate(self, data_name: str, data_len: int, data_type: Type, drange: str) -> np.ndarray:
        """
        生成输入数据并保存为bin文件
        :param data_name: 数据名称
        :param data_len: 数据长度
        :param data_type: 数据类型
        :param drange: 数值范围
        :return: 生成的输入数组
        """
        # 初始化输入数组
        input_gm = np.zeros((self.ranksize, data_len), dtype=data_type)

        # 为每个rank生成随机数据
        for i in range(self.ranksize):
            input_gm[i][:] = self.gen_random_data(data_len, dtype=data_type, drange=drange)

        # 保存为bin文件（每个rank一个文件）
        for i in range(self.ranksize):
            file_path = os.path.join(self.golden_dir, f"input_{data_name}_{i}.bin")
            input_gm[i].tofile(file_path)

        logging.info(f"{data_name}数据生成完成！保存路径：{self.golden_dir}")
        return input_gm

    def cpu_allgather(self, tensor: np.ndarray) -> np.ndarray:
        """
        CPU ALLGATHER
        :param tensor: 输入tensor (shape: [ranksize, data_len])
        :return: AllGather后的tensor (shape: [ranksize, bs, hidden_size])
        """
        # reshape为[ranksize, bs, hidden_size]，模拟分布式AllGather的汇聚结果
        tensor_reshaped = tensor.reshape(self.ranksize, self.bs, self.hidden_size)
        return tensor_reshaped

    def cpu_all_reduce(self, output: np.ndarray) -> np.ndarray:
        """
        CPU REDUCE (当前仅支持SUM)
        :param output: 输入tensor (shape: [ranksize, bs, hidden_size])
        :return: Reduce后的tensor (shape: [bs, hidden_size])
        """
        if self.reduce_op == "sum":
            # 按rank维度求和（归约）
            return np.sum(output, axis=0, keepdims=False)
        else:
            raise NotImplementedError(f"暂不支持Reduce操作: {self.reduce_op}, 仅支持sum!")

    def output_generate(self, x: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        output = reduce(allgatherscales * allgatherdata)
        :param x: 各rank的输入数据
        :param scales: 各rank的scale数据
        :return: CPU计算的Golden结果
        """
        # 步骤1：AllGather x → allgatherdata
        allgatherdata = self.cpu_allgather(x)  # shape: [ranksize, bs, hidden_size]
        
        # 步骤2：AllGather scales → allgatherscale（扩展到和x匹配的shape）
        group_size = 32 if self.mxfp else 128
        allgatherscale = []
        for i in range(self.ranksize):
            # 还原scale的原始shape → 按group扩展到hidden_size维度
            scale_rank = scales[i].reshape(self.bs, -1)  # shape: [bs, hidden_size/group_size]
            scale_expand = np.repeat(scale_rank, group_size, axis=-1)  # shape: [bs, hidden_size]
            allgatherscale.append(scale_expand)
        # 转换为数组，统一数据类型
        allgatherscale = np.array(allgatherscale, dtype=self.output_type)  # shape: [ranksize, bs, hidden_size]
        
        # 步骤3：逐元素相乘 → allgatherscales * allgatherdata
        allgatherdata = allgatherdata.astype(self.output_type)  # 统一类型避免精度偏差
        multiply_result = allgatherscale * allgatherdata  # shape: [ranksize, bs, hidden_size]
        
        # 步骤4：Reduce（sum）→ 最终output
        output_cpu = self.cpu_all_reduce(multiply_result)  # shape: [bs, hidden_size]
        output_cpu = output_cpu.flatten()  # 展平为一维，适配bin文件存储
        
        # 保存CPU Golden结果
        file_path = os.path.join(self.golden_dir, "output_cpu.bin")
        output_cpu.tofile(file_path)
        
        logging.info(f"CPU Golden数据生成完成! output shape: {output_cpu.shape}")
        return output_cpu

    def run(self):
        """
        RUN 执行流程
        """
        # 生成x输入数据
        input_x = self.input_generate(
            data_name="x",
            data_len=self.input_len,
            data_type=self.input_tensor_type,
            drange=self.input_tensor_range
        )

        # 生成scale输入数据
        input_scales = self.input_generate(
            data_name="scale",
            data_len=self.scale_len,
            data_type=self.scales_type,
            drange=self.scales_range
        )

        # 打印关键数据形状（便于调试）
        logging.info(f"input_x.shape: {input_x.shape}")
        logging.info(f"input_scales.shape: {input_scales.shape}")

        # 生成CPU输出Golden数据
        output_cpu = self.output_generate(input_x, input_scales)
        logging.info(f"output_cpu.shape: {output_cpu.shape}")

        # 最终提示
        logging.info(f"所有Golden数据生成完成! 保存目录: {self.golden_dir}")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    :return: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="quant_all_reduce golden grenerator")
    # 必选参数（按顺序）
    parser.add_argument('case_name', type=str, help="测试用例名称")
    parser.add_argument('bs', type=int, help="Batch Size")
    parser.add_argument('hidden_size', type=int, help="Hidden Size")
    parser.add_argument('input_tensor_range', type=str, help="input tensor范围")
    parser.add_argument('input_tensor_type', type=str, help="input tensor类型")
    parser.add_argument('scales_range', type=str, help="scale tensor范围")
    parser.add_argument('scales_type', type=str, help="scale tenso类型")
    parser.add_argument('output_type', type=str, help="输出类型")
    parser.add_argument('ranksize', type=int, help="Rank数量")
    parser.add_argument('reduce_op', type=str, help="Reduce操作 (仅支持sum)")
    parser.add_argument('mxfp', type=int, help="MXFP模式")
    parser.add_argument('seed', type=int, help="随机种子（保证结果可复现）")
    return parser.parse_args()


if __name__ == '__main__':
    """主函数：解析参数 → 实例化生成器 → 执行生成流程"""
    # 解析命令行参数
    args = parse_args()
    # 创建生成器实例
    generator = QuantAllReduceGoldenGenerator(args)
    # 执行全流程
    generator.run()