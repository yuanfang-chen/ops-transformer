# FusedFloydAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |

## 功能说明

- 算子功能：


- 计算公式：


## 参数说明


## 约束说明


## 调用说明

| 调用方式           | 调用样例                                                                                                              | 说明                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_fused_floyd_attention_grad](./examples/test_fused_floyd_attention_grad.cpp)                     | 非TND场景，通过[aclnnFusedFloydAttentionGrad](./docs/aclnnFusedFloydAttentionGrad.md)接口方式调用FusedFloydAttentionGrad算子。                   |
