## QuantAllReduce <<<>>>直调示例

# 文件目录
```
├── kernel                                  # 算子核函数目录
│    ├── quant_all_reduce_mte_one_shot.h    
│    ├── quant_all_reduce_tiling_data.h     
│    ├── mte_comm.h                         
│    ├── vec_comp.h                         
│    ├── utils.h                         
├── scripts                                 # 运行脚本工具
│   ├── data_gen.py                         # 用例数据生成脚本
│   ├── data_compare.py                     # 精度对比脚本
│   └── QuantAllReduce.csv                  # 算子用例输入相关信息
├── quant_all_reduce.cpp                    # <<<>>>核函数入口
├── test_quant_all_reduce.cpp               # <<<>>>直调示例代码
├── start.sh
├── run.sh                                  # 工程运行脚本
├── result.csv                              # 算子运行结果汇总
├── README.md                              
└── CMakeLists.txt                          
```

# 文件改动
新增接口：
quant_all_reduce_mte_one_shot.h __aicore__ inline void InitWithMc2Context(GM_ADDR x, GM_ADDR scales, GM_ADDR output, GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *pipe);
mte_comm.h __aicore__ inline void InitHcclContextByAddr(GM_ADDR mc2Context);
外部创建mc2Context，传入Device地址进行通信域结构体初始化

# 工程运行
设置环境变量
``` shell
source /usr/local/Ascend/cann/set_env.sh
```

``` shell
bash run.sh -v Ascend910_9599 -b Release -c ON -r OFF -n 2
```

```
-c cmake-rebuild --- 编译开关
-v soc-version --- 芯片版本
-i install-path --- cann包安装路径（source环境后不用指定）
-b build-type --- 编译选项 Release or Debug
-p install-prefix --- 编译文件输出目录
-r run-test --- 运行开关
-n target_line --- 执行用例行号
-f profing-type --- profing开关
```
`-c ON` 编译， `-r ON` 运行，可以组合。

常用组合
``` shell
bash run.sh -v Ascend910_9599 -b Release -c ON -r OFF --- 仅编译
bash run.sh -v Ascend910_9599 -b Release -c ON -r ON -n 2 --- 编译运行并指定行号
bash run.sh -v Ascend910_9599 -b Release -c OFF -r ON -n 2 --- 不编译直接运行
```