1. pytorch仓库下载：
    git clone https://gitcode.com/ascend/pytorch.git -b v2.8.0 -depth 1

2. 更新./third_party/op-plugin：
    git submodule update --init --recursive

3. 将npu_chunk_gated_delta_rule和npu_chunk_gated_delta_rule_functional添加到./third_party/op-plugin/config/op_plugin_functions.yaml的custom:；

4. 将ChunkGatedDeltaRuleKernelNpuOpApi.cpp添加到./third_party/op-plugin/ops/opapi/中；

5. 直接编译:
    cd pytorch
    bash ci/build.sh --python=3.10