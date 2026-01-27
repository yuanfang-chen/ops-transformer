#!/bin/bash
chmod +x run_batch.sh
# ====================== 配置区======================
# 需要传入的两个路径
PATH1="./excel/qli_case_example2.xlsx"
PATH2="./pt_path/pt_path2"

# 脚本路径（如果和当前脚本同目录，直接写文件名）
QLI_PT_SAVE_SCRIPT="qli_pt_save.py"
TEST_QLI_BATCH_SCRIPT="test_qli_batch.py"
REPLACE_PATH_SCRIPT="replace_path.py"
# =======================================================================

echo "===== 第一步：执行qli_pt_save.py ====="
python3 $QLI_PT_SAVE_SCRIPT $PATH1 $PATH2
if [ $? -ne 0 ]; then
    echo "qli_pt_save.py 执行失败，退出"
    exit 1
fi

echo -e "\n===== 第二步：替换test_qli_batch.py中的路径 ====="
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
cp test_qli_batch.py.bak test_qli_batch.py

echo -e "\n 所有步骤执行完成！"