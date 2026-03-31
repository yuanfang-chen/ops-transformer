# all_gather_add

## 功能说明

`all_gather_add` 为 MC2 双输出算子。

- 输出1：`a_gathered = AllGather(a)`
- 输出2：`c = a_gathered + b`

当前版本冻结规格：

- `a=[1024,2048]`
- `b=[2048,2048]`
- `a_gathered=[2048,2048]`
- `c=[2048,2048]`
- `dtype=FP16`
- `format=ND`
- `rank_size=2`
- 轮次由 `comm_turn` 决定，当前版本仅支持 `comm_turn=2`

## 产品支持情况

当前目录已按 `references/skeleton/new_op` 完成 Stage 1/2 框架对齐。
最终产品支持情况以 Stage 5 编译验证和后续发布收敛结果为准。

## 调用说明

- 对外接口文档：`docs/aclnnAllGatherAdd.md`
- ACLNN 调用示例：`examples/test_aclnn_all_gather_add.cpp`
- 当前已完成 Stage 1~4 开发，最终以 Stage 5 编译验证结果为准。
