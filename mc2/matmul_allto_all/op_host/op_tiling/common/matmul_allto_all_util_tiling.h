/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_ALLTO_ALL_UTIL_TILING_H
#define MATMUL_ALLTO_ALL_UTIL_TILING_H

#include "tiling/mc2_tiling_utils.h"
#include "../../../../common/inc/hccl_stub.h"

namespace MC2Tiling {

using namespace std;
using namespace ge;
using namespace gert;

// 参数范围
const std::set<int> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32};
constexpr uint64_t K_MAX_VALUE = 65535UL;
// FOR NON_QUANT
const std::vector<uint32_t> NON_QUANT_X_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16};
// 维度范围
constexpr uint32_t TWO_DIMS = 2;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;

// input index
constexpr size_t INPUT_X1 = 0;
constexpr size_t INPUT_X2 = 1;
constexpr size_t INPUT_BIAS = 2;
constexpr size_t INPUT_X1_SCALE = 3;
constexpr size_t INPUT_X2_SCALE = 4;
constexpr size_t INPUT_COMM_SCALE = 5;
constexpr size_t INPUT_X1_OFFSET = 6;
constexpr size_t INPUT_X2_OFFSET = 7;

// output index
constexpr size_t OUTPUT_Y = 0;

// attr index
constexpr size_t ATTR_GROUP = 0;
constexpr size_t ATTR_WORLD_SIZE = 1;
constexpr size_t ATTR_ALLTO_ALL_AXES = 2;
constexpr size_t ATTR_Y_DTYPE = 3;
constexpr size_t ATTR_X1_QUANTMODE = 4;
constexpr size_t ATTR_X2_QUANTMODE = 5;
constexpr size_t ATTR_COMMON_QUANTMODE = 6;
constexpr size_t ATTR_COMMON_QUANTDTYPE = 7;
constexpr size_t ATTR_X1_TRANSPOSE = 8;
constexpr size_t ATTR_X2_TRANSPOSE = 9;
constexpr size_t ATTR_GROUP_SIZE = 10;

// 定义量化模式枚举，直接取量化组合
enum class QuantMode : uint8_t {
    NON_QUANT = 0, // 非量化模式
    KC_QUANT = 1,  // K量化模式,对应x1, x2分别为perToken量化与perchannel量化
    ERROR = 255    // 特殊设置，表示不支持的类型组合
};

// 封装输入参数，主要为输入获取得到的参数
struct TilingContextInfo {
    std::string group = "group"; // group属性
    QuantMode quantMode = QuantMode::NON_QUANT;
    mc2tiling::TilingArgs args_;
};

// 封装Tiling过程中推导得到的参数
struct TilingInferredInfo {
    uint64_t mmResultLen = 0UL; // 存储计算MM的地址大小
    uint64_t permuteLen = 0UL;  // 重排空间大小
    uint32_t biasLen = 0UL;     // 存储偏移的地址大小
    uint32_t tileM = 0UL;       // 头块大小
    uint32_t tileCnt = 0UL;     // 头块数量
    uint32_t tailM = 0UL;       // 尾块大小
    uint32_t tailCnt = 0UL;     // 尾块数量
};


class MatmulAlltoAllTilingUtil {
public:
    static QuantMode GetQuantMode(const gert::TilingContext *context, const char *opName);

    static ge::graphStatus CheckAttrsInfo(const gert::TilingContext *context, const char *opName);
    static ge::graphStatus CheckShapeInfo(const gert::TilingContext *context, const char *opName);
    static ge::graphStatus CheckNonQuantTensorDataType(const gert::TilingContext *context, const char *opName);

    static ge::graphStatus SetAttrsInfo(const gert::TilingContext *context, const char *opName,
                                        TilingContextInfo &contextInfo);
    static ge::graphStatus SetShapeInfo(const gert::TilingContext *context, TilingContextInfo &contextInfo);
    static ge::graphStatus SetDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                           TilingContextInfo &contextInfo);
};

// Builder模式成员函数通常使用小写字母开头
class Mc2CcTilingConfigBuilder {
private:
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig;
    static constexpr uint32_t SUCCESS = 0;
    std::set<std::string> errorSet{};

    uint32_t reduceType = 0;
    uint8_t dstDataType = 0;
    uint8_t srcDataType = 0;
    uint8_t stepSize = 0;
    uint8_t localRankDataToLocalDst = 0;
    // 0,1代表false 2代表true
    uint8_t srcDataFromWindow = 0;
    uint32_t debugMode = 0;
    uint16_t commBlockNum = 0;
    uint16_t queueNum = 0;
    uint8_t commEngine = 0;

    // 标识是否参与build操作,为false则跳过对应的set方法调用
    bool hasReduceType = false;
    bool hasStepSize = false;
    bool hasLocalRankDataToLocalDst = false;
    bool hasSrcDataFromWindow = false;
    bool hasDebugMode = false;
    bool hasCommBlockNum = false;
    bool hasQueueNum = false;
    bool hasCommEngine = false;

    bool hasGotInit = false;

public:
    struct ConfigFile {
        static constexpr const char *GROUP_NAME = "groupName";
        static constexpr const char *OP_TYPE = "opType";
        static constexpr const char *ALG_CONFIG = "algConfig";
        static constexpr const char *REDUCE_TYPE = "reduceType";
        static constexpr const char *STEP_SIZE = "stepSize";
        static constexpr const char *SKIP_LOCAL_RANK_COPY = "skipLocalRankCopy";
        static constexpr const char *SKIP_BUFFER_WINDOW_COPY = "skipBufferWindowCopy";
        static constexpr const char *DEBUG_MODE = "debugMode";
        static constexpr const char *COMM_BLOCK_NUM = "commBlockNum";
        static constexpr const char *QUEUE_NUM = "queueNum";
        static constexpr const char *COMM_ENGINE = "commEngine";
    };

    struct AlgConfigType {
        static constexpr const char *ALL_REDUCE = "AllReduce=level0:doublering";
        static constexpr const char *ALL_GATHER = "AllGather=level0:doublering";
        static constexpr const char *REDUCE_SCATTER = "ReduceScatter=level0:doublering";
        static constexpr const char *ALL_TO_ALL = "AlltoAll=level0:fullmesh;level1:pairwise";
        static constexpr const char *BATCH_WRITE = "BatchWrite=level0:fullmesh";
    };

    static constexpr const char *GROUP = "group";
    static constexpr const uint8_t CONSTANTS_ZERO = 0;
    static constexpr const uint8_t CONSTANTS_ONE = 1;
    static constexpr const uint8_t CONSTANTS_TWO = 2;

    // 构造函数
    Mc2CcTilingConfigBuilder(const std::string &groupName, uint32_t opType, const std::string &algConfig);

    // 链式配置方法（声明),通过change和with方法来修改hccl中的对应属性
    Mc2CcTilingConfigBuilder &withReduceType(const char *opName, AscendC::HcclReduceOp reduceType,
                                             ge::DataType dstDataType, ge::DataType srcDataType);
    Mc2CcTilingConfigBuilder &withStepSize(uint8_t stepSize);
    Mc2CcTilingConfigBuilder &isLocalRankDataToLocalDst(bool flag);
    Mc2CcTilingConfigBuilder &isSrcDataFromWindow(bool fromWindow);
    Mc2CcTilingConfigBuilder &withDebugMode(uint8_t debugMode);
    Mc2CcTilingConfigBuilder &withCommBlockNum(uint16_t commBlockNum);
    Mc2CcTilingConfigBuilder &withQueueNum(uint16_t queueNum);
    Mc2CcTilingConfigBuilder &withCommEngine(uint8_t commEngine);

    AscendC::Mc2CcTilingConfig build();
    bool isSuccess() const;
    std::string errorMsg() const;
    void setInitTilingData(::Mc2InitTiling initTiling);

    // 模版方法
    template <typename... Args>
    void setCcTilingData(Args &&...args)
    {
        (mc2CcTilingConfig.GetTiling(args), ...);
    }

    // 静态工厂
    static Mc2CcTilingConfigBuilder create(const std::string &groupName, mc2tiling::AicpuComType opType,
                                           const std::string &algConfig);
};

}; // namespace MC2Tiling

#endif // MATMUL_ALLTO_ALL_UTIL_TILING_H