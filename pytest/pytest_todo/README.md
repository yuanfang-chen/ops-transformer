# Pytest测试框架基线模板 
## operator为算子名称

## 文件结构

pytest/
- test_operator.py                # 测试用例运行主程序
- testcases_operator.py           # 测试用例入参配置
- check_valid_param.py            # 入参合法性校验
- operator_single.py              # CPU侧算子golden实现获取cpu结果与npu算子直调获得npu结果
- pytest.ini                      # 创建ci单算子直调的测试标记(后续增加图模式)
- check_result.py                 # npu的结果与cpu结果的两方精度对比

## 功能说明

基于pytest测试框架的基线模板
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能


## 环境配置

### 前置要求

1. 确认torch_npu为最新版本
2. source CANN包

### Custom包调用

支持custom包调用

### kernel侧打印

torch_npu直调算子可直接打印算子kernel侧加打印的内容，这是AscendC原生能力支持

## 使用方法
在pytest文件夹路径下执行：
单算子直调
```bash
python3 -m pytest -rA -s test_operator.py -v -m ci
```
指令执行逻辑：
使用当前 Python 环境的 pytest，以详细输出模式，展示所有打印内容，只执行 test_operator.py 文件中被 @pytest.mark.ci 标记的测试用例，并在执行结束后生成包含所有测试状态的完整结果报告。

指令参数说明：
python3 -m pytest :以Python模块的形式执行pytest，避免潜在的环境冲突问题
-rA: 展示所有类型的测试结果明细
-s:关闭 pytest 默认的「标准输出捕获机制」，让代码中的 print() 打印内容、日志输出、控制台信息实时显示在终端
-v:开启详细的日志输出，显示每个用例的名称、路径、结果
-m ci：根据 pytest 的「自定义标记 (marker)」，筛选出【符合标记条件】的测试用例执行，其余未标记的用例全部跳过不执行。pytest 只会运行 test_operator.py 中所有被 @pytest.mark.ci 装饰的测试用例，
