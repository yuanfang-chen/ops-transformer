# 基于sas算子的基本通路

cpu复现算子逻辑生成golden值
npu侧直接调用算子生成测试值
复用fuzz的精度对比方法作精度对比

# 根据excel 参数列表批量生成bin文件(cpu golden 和 npu input)


# 完全支持sas算子 从excel表格读取参数批量生成bin文件并读取bin 进行精度对比
