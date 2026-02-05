#!/bin/bash

# 1. 获取脚本目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"

# 2. 清理旧目录
if [ -d "$BUILD_DIR" ]; then
    echo "[INFO] 删除旧的 build 目录..."
    rm -rf "$BUILD_DIR"
fi

# 3. 创建并进入 build 目录
mkdir -p "$BUILD_DIR"
# 使用 '|| exit' 防止 cd 失败后继续执行
cd "$BUILD_DIR" || { echo "[ERROR] 无法进入 build 目录"; exit 1; }

echo "[INFO] 当前工作目录: $(pwd)"

# 4. 编译
echo "[INFO] 开始 CMake 配置..."
cmake ..
# 检查 cmake 是否成功
if [ $? -ne 0 ]; then echo "[ERROR] CMake 失败"; exit 1; fi

echo "[INFO] 开始编译 (64 并行)..."
make -j64
# 检查 make 是否成功
if [ $? -ne 0 ]; then echo "[ERROR] 编译失败"; exit 1; fi

# 5. 设置环境变量并运行
export LD_LIBRARY_PATH=$BUILD_DIR:$BUILD_DIR/lib:$LD_LIBRARY_PATH

echo "[INFO] 运行测试程序..."
./test_moe_dispatch
