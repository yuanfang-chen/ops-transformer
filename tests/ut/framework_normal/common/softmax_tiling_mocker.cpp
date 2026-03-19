// 包含声明这两个函数的头文件，确保签名一致
#include "activation/softmax_tiling.h"
#include "softmax_tiling_mocker.h"

// 覆盖 tiling_api.so 中的 GetSoftMaxMinTmpSize
uint32_t GetSoftMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
{
    return 1;
}

// 覆盖 tiling_api.so 中的 GetSoftMaxFlashV2MinTmpSize
uint32_t GetSoftMaxFlashV2MinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock,
    const bool isFlashOutputBrc)
{
    return 1;
}