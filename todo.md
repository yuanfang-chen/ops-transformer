# 任务：为gmm适配低阶API mxfp4（nd/nz）

## 目标
先学习 ops-nn 仓 quantbatchmatmulv3 在低阶 API mxfp4 的 nd/nz 实现，再在本仓 gmm 的对应 tiling 与 kernel 位置完成等价能力适配。

## 待办事项
- [x] 拉取并阅读 ops-nn PR #2333（tiling）
- [x] 拉取并阅读 ops-nn PR #2751（kernel）
- [x] 映射到 gmm 对应文件与代码路径
- [x] 实现 gmm tiling 的 mxfp4 nd/nz 适配
- [x] 实现 gmm kernel 的 mxfp4 nd/nz 适配
- [x] 本地执行最小化检查（编译/静态检查）
- [ ] 汇总改动与后续建议

## 进度
6/7
