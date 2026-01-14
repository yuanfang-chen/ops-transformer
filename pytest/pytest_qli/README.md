## QLI算子测试框架
## 文件结构

pytest/
- test_qli.py                     # pytest测试用例运行主程序
- testcases.py                    # 测试用例入参配置
- check_result.py                 # cpu结果和npu结果精度对比
- qli_single.py                   # CPU侧算子逻辑实现获取golden，npu算子直调获取算子输出
- pytest.ini                      # 创建ci单算子和graph图模式的测试标记

## 功能说明

基于pytest测试框架，实现GQA算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据， 通过torchair入图
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

### 当前实现范围

✅**已实现**：基础GQA非量化PA场景
❌**未实现**：mask功能

### 参数限制

- **数据格式**：FP16 / BF16
- **in_layout**：BSND /BNSD
- **kvcache_layout**: BNBD / BBH
  - BSND格式仅支持BBH

## 环境配置

### 前置要求

1. 确认torch_npu为最新版本
2. source CANN包

### Custom包调用

支持custom包调用

## 使用方法

在pytest文件夹路径下执行：

### 方式一：运行泛化测试用例
单算子直调+图模式
```bash
python3 -m pytest -rA -s test_V1.py