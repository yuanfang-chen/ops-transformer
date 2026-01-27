#!/bin/bash
# ====================== 配置区======================
# 需要读取的用例excel表格
PATH1="./excel/***"
# 用例pt的文件存放路径
PATH2="./pt_path/***"

# 脚本路径（如果和当前脚本同目录，直接写文件名）
QLI_PT_SAVE_SCRIPT="quant_lightning_indexer_pt_save.py"
TEST_QLI_BATCH_SCRIPT="test_quant_lightning_indexer_batch.py"
REPLACE_PATH_SCRIPT="replace_path.py"
# =======================================================================

echo "===== 第一步：执行quant_lightning_indexer_pt_save.py ====="
python3 $QLI_PT_SAVE_SCRIPT $PATH1 $PATH2
if [ $? -ne 0 ]; then
    echo "quant_lightning_indexer_pt_save.py 执行失败，退出"
    exit 1
fi

echo -e "\n===== 第二步：替换test_quant_lightning_indexer_batch.py中的路径 ====="
python3 $REPLACE_PATH_SCRIPT $TEST_QLI_BATCH_SCRIPT $PATH2
if [ $? -ne 0 ]; then
    echo "替换路径失败，退出"
    exit 1
fi

echo -e "\n===== 第三步：执行pytest命令 ====="
python3 -m pytest -rA -s $TEST_QLI_BATCH_SCRIPT -v -m ci
if [ $? -ne 0 ]; then
    echo " pytest执行失败"
    exit 1
fi
cp test_quant_lightning_indexer_batch.py.bak test_quant_lightning_indexer_batch.py

echo -e "\n 所有步骤执行完成！"