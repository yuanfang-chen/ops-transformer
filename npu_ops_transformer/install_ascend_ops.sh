#!/bin/bash

# 定义要删除的目录列表
dirs_to_remove=("build/" "dist/" "kernel_meta/" "ascend_ops.egg-info/")

# 遍历目录列表并删除每个目录
for dir in "${dirs_to_remove[@]}"
do
    echo "正在删除目录：$dir"
    rm -rf "$dir"
done

# 卸载ascend_ops包
echo "正在卸载ascend_ops包"
pip uninstall ascend_ops -y

# 使用python3构建wheel包
echo "正在构建wheel包"
python3 -m build -n --wheel

# 安装新构建的wheel包
echo "正在安装新构建的wheel包"
# pip install dist/*.whl
python3 -m pip install dist/*.whl --force-reinstall --no-deps

echo "操作完成"