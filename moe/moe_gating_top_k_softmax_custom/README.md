# moe_gating_top_k_softmax_custom 自定义算子

## 前提条件

- 本算子需要提前安装cann8.5.0的ascend-toolkit

## moe_gating_top_k_softmax 自定义算子调用步骤

### 算子用例验证

- 执行算子样例，生成自定义算子包 custom_opp_ubuntu_aarch64.run

```bash
cd moe_gating_top_k_softmax_custom
bash mybuild.sh
```

- 执行算子测试

```bash
bash mybuild.sh run
```
若想要更改算子出入，需要更改以下文件：
gen_data.py:15 修改输入shape:[token_num, expert_num]
compare_and_expert.py:78 修改 k 值

若cpu与npu侧执行结果一致，看到输出余弦相似度接近100%，测试验证成功