# pybind11注册自定义算子直调样例

## 概述

本样例展示了如何使用pybind11注册自定义算子，并通过`<<<>>>`内核调用符调用核函数，以简单的Add算子为例，实现两个向量的逐元素相加。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── pybind
│   ├── CMakeLists.txt        // 编译工程文件
│   ├── add_custom_test.py    // PyTorch调用脚本
│   └── add_custom.asc        // Ascend C算子实现 & pybind11注册
```

## 算子描述

- 算子功能：

  Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：

  ```
  z = x + y
  ```

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddCustom</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- 算子实现：

  Ascend C提供的矢量计算接口`Add`的操作元素都为`LocalTensor`，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：`CopyIn`，`Compute`，`CopyOut`。`CopyIn`任务负责将Global Memory上的输入Tensor `xGm`和`yGm`搬运到Local Memory，分别存储在`xLocal`、`yLocal`，`Compute`任务负责对`xLocal`、`yLocal`执行加法操作，计算结果存储在`zLocal`中，`CopyOut`任务负责将输出数据从`zLocal`搬运至Global Memory上的输出Tensor zGm中。

- 自定义算子注册：

  本样例在`add_custom.asc`中定义了一个名为`ascendc_ops`的命名空间，并在其中注册了`ascendc_add`函数。

  pybind11可以实现PyTorch框架调用算子Kernel程序，从而实现Ascend C算子在Pytorch框架的集成部署。

  `add_custom.asc`使用了`pybind11`库来将c++代码封装成python模块。该代码实现中定义了一个名为`m`的pybind11模块，其中包含一个名为`ascendc_add`的函数。该函数与`ascendc_ops::ascendc_add`函数相同，用于将c++函数转成python函数，例如：

  ```c++
  PYBIND11_MODULE(ascendc_ops, m)
  {
      m.doc() = "add_custom pybind11 interfaces";
      m.def("ascendc_add", &ascendc_ops::ascendc_add, "");
  }
  ```

  在`ascendc_add`函数中通过`c10_npu::getCurrentNPUStream()`函数获取当前NPU上的流，并通过内核调用符<<<>>>调用自定义的Kernel函数`add_custom`，在NPU上执行算子。

- Python测试脚本

  在`add_custom_test.py`调用脚本中，导入自定义模块`import ascendc_ops`，调用注册的`ascendc_add`函数，并通过对比NPU输出与CPU标准加法结果来验证自定义算子的数值正确性。

## 编译运行

- 安装PyTorch以及Ascend Extension for PyTorch插件

  请参考[pytorch: Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)开源代码仓或[Ascend Extension for PyTorch昇腾社区](https://hiascend.com/document/redirect/Pytorch-index)的安装说明，选取支持的`Python`版本配套发行版，完成`torch`和`torch-npu`的安装。

- 安装前置依赖

  ```bash
  pip3 install pybind11 expecttest
  ```

- 配置环境变量

  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包

    ```bash
    source ${install_path}/cann/set_env.sh
    ```
  
  - 云开发
    ```bash
    source /home/developer/Ascend/cann/set_env.sh
    ```

- 样例执行

  在本样例根目录下执行如下步骤，运行该样例。

  ```bash
  mkdir -p build; cd build
  cmake ..; make -j
  python3 ../add_custom_test.py
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  Ran 1 test in **s
  OK
  ```