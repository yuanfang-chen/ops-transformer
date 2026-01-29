# Attention融合算子Experimental使用说明

## 前提条件

- 环境部署：调用算子之前，请先参考[环境部署](../../docs/zh/context/quick_install.md)完成基础环境搭建。

## 算子列表

项目提供的所有算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子名</th>
    <th rowspan="2">接口文档</th>
    <th colspan="2">算子实现</th>
    <th rowspan="2">torch调用</th>
    <th colspan="2">图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>Ascend IR</th>
    <th>aclgraph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>compressor</td>
    <td><a href="./compressor/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将每4或128个token的 KV cache 压缩成一个，然后每个token与这些压缩的 KV cache进行 DSA 计算。</td>
  </tr>
  <tr>
    <td>kv_quant_sparse_attn_sharedkv</td>
    <td><a href="./kv_quant_sparse_attn_sharedkv/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>支持量化模式的Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention计算。</td>
  </tr>
  <tr>
    <td>kv_quant_sparse_attn_sharedkv_metadata</td>
    <td>/</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Cpu</td>
    <td>该算子为kv_quant_sparse_attn_sharedkv算子提供分核结果。</td>
  </tr>
  <tr>
    <td>quant_lightning_indexer</td>
    <td><a href="./quant_lightning_indexer/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子是推理场景下，稀疏Attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。</td>
  </tr>
  <tr>
    <td>quant_lightning_indexer_metadata</td>
    <td>/</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Cpu</td>
    <td>该算子为quant_lightning_indexer算子提供分核结果。</td>
  </tr>
  <tr>
    <td>sparse_attn_sharedkv</td>
    <td><a href="./sparse_attn_sharedkv/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子支持非量化的Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention计算。</td>
  </tr>
  <tr>
    <td>sparse_attn_sharedkv_metadata</td>
    <td>/</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Cpu</td>
    <td>该算子为sparse_attn_sharedkv算子提供分核结果。</td>
  </tr>
</tbody>
</table>
    
## 自定义算子编译

1. 编译自定义算子包

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --experimental --soc=${soc_version} [--ops=${op_list}]
    # 执行命令示例：（在910c环境下编译sparse_attn_sharedkv和对应metadata算子）
    # bash build.sh --pkg --experimental --soc=ascend910_93 --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata
    # 执行命令示例：（在910d环境下编译sparse_attn_sharedkv和对应metadata算子）
    # bash build.sh --pkg --experimental --soc=ascend910_95 --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"，Atlas A5 训练系列产品/Atlas A5 推理系列产品使用"ascend910_95"。
    - --ops：自定义算子名称，多个自定义算子通过`,`分割。


    若提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-transformer-custom_linux.${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。

2. 安装自定义算子包
   
    ```bash
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run --install-path={install_path}
    ```

    自定义单算子包安装时若不指定参数`--install-path`，将默认安装于`${ASCEND_HOME_PATH}/opp/vendors`目录下。\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN ops包安装路径。如指定`--install-path={install_path}`，则自定义算子包将安装于`${install_path}/vendors`目录下。注意自定义算子包不支持卸载。安装完成后，需要激活自定义算子包环境变量或导出环境变量，具体方式如下：

    ```bash
    # experimental自定义单算子包自定义安装路径
    source ${install_path}/vendors/custom_transformer/bin/set_env.bash
    # experimental自定义单算子包默认安装路径${ASCEND_HOME_PATH}/opp/vendors
    # export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH
    ```

## torch_ops_extension包编译安装
torch_ops_extension包为自定义算子提供了torch.ops的拓展接口，具体安装方法请见[torch_ops_extension安装方法](https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85)。

## 自定义算子执行
项目中各算子通过pytest验证各算子的功能是否正常，各算子的pytest调用方法如下表。
<table><thead>
  <tr>
    <th rowspan="2">算子名</th>
    <th rowspan="2">pytest示例</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>compressor</td>
    <td>/</td>
  </tr>
  <tr>
    <td>kv_quant_sparse_attn_sharedkv</td>
    <td><a href="./kv_quant_sparse_attn_sharedkv/tests/pytest/README.md">示例</a></td>
  </tr>
  <tr>
    <td>quant_lightning_indexer</td>
    <td><a href="./quant_lightning_indexer/tests/pytest/README.md">示例</a></td>
  </tr>
  <tr>
    <td>sparse_attn_sharedkv</td>
    <td><a href="./sparse_attn_sharedkv/tests/pytest/README.md">示例</a></td>
  </tr>
</tbody>
</table>
    