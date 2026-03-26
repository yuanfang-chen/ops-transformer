/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
// #include "register/tilingdata_base.h"
// #include "fused_infer_attention_score_tiling_compile_info.h"
#include "fused_infer_attention_score_tiling_index.h"
#include "fused_infer_attention_score_tiling_constants.h"
// #include "../../incre_flash_attention/op_host/incre_flash_attention_tiling_struct.h"
// #include "../../incre_flash_attention/op_host/incre_flash_attention_tiling_base.h"
#include <string>
#include <stdio.h>
#include <limits>
#include <optional>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <cstdint>
#include <type_traits>

#ifdef ASCENDC_OP_TEST
#define FIA_EXTERN_C extern "C"
#else
#define FIA_EXTERN_C
#endif

namespace optiling {
using namespace arch35FIA;
const uint32_t FIA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐
struct FiaTiling {
    std::vector<int64_t> tShape;
    DataType tDataType = DT_UNDEFINED;
    bool tIsNull = true;
    DataType GetDataType() const {
        return tDataType;
    }
};

struct MDesc {
    DataType tDataType = DT_FLOAT16;
    DataType GetInputDesc() const {
        return tDataType;
    }
    void SetInputDesc(DataType type) {
        tDataType = type;
    }
};

struct StorageShape {
public:
    static constexpr size_t kMaxDimNum = 25;
    static constexpr int64_t kInvalidDimValue = std::numeric_limits<int64_t>::min();
    std::vector<int64_t> tShape;
    DataType tDataType = DT_FLOAT16;
    size_t GetShapeSize() const {
        return tShape.size();
    }
    size_t GetDim() const {
        return tShape.size();
    }
    size_t GetDim(int index) const {
        return tShape[index];
    }
    DataType GetDataType() const {
        return tDataType;
    }
    void SetDataType(DataType type) {
        tDataType = type;
    }
    bool isEmpty() const {
        return tShape.size() == 0;
    }
};

struct OptionalInputTensor {
    StorageShape storageShape;
    DataType tDataType = DT_UNDEFINED;
    bool tIsNull = false;
    StorageShape GetStorageShape() const {
        return storageShape;
    }
    void SetOptionalInputTensor(StorageShape shape) {
        storageShape = shape;
    }
};


struct OptionalInputShape {
    DataType tDataType = DT_UNDEFINED;
    DataType GetDataType() const {
        return tDataType;
    }
    void SetDataType(DataType type) {
        tDataType = type;
    }
};
// 枚举：属性类型
struct AttrType {
    enum Value : uint8_t {
        UINT32,
        FLOAT,
        INT64,
        BOOL,
        STRING
    };
    Value value;

    AttrType(Value v) : value(v) {}
    operator Value() const { return value; }
};
struct Attribute {
    AttrType type;
    union {
        uint32_t    u32;
        float       f;
        int64_t     i64;
        bool        b;
        std::string* s;
    } value;

    // 默认构造
    Attribute() : type(AttrType::UINT32) { value.u32 = 0; }

    // 修复：允许拷贝构造（map 必须要）
    Attribute(const Attribute& other) : type(other.type) {
        // 逐类型拷贝
        switch (type) {
            case AttrType::UINT32: value.u32 = other.value.u32; break;
            case AttrType::FLOAT:  value.f = other.value.f; break;
            case AttrType::INT64:  value.i64 = other.value.i64; break;
            case AttrType::BOOL:   value.b = other.value.b; break;
            case AttrType::STRING:
                // 字符串深拷贝，安全
                value.s = new std::string(*other.value.s);
                break;
        }
    }

    // 拷贝赋值运算符
    Attribute& operator=(const Attribute& other) {
        if (this == &other) return *this;

        // 先释放自己的字符串
        if (type == AttrType::STRING) {
            delete value.s;
        }

        type = other.type;
        // 逐类型拷贝
        switch (type) {
            case AttrType::UINT32: value.u32 = other.value.u32; break;
            case AttrType::FLOAT:  value.f = other.value.f; break;
            case AttrType::INT64:  value.i64 = other.value.i64; break;
            case AttrType::BOOL:   value.b = other.value.b; break;
            case AttrType::STRING:
                value.s = new std::string(*other.value.s);
                break;
        }
        return *this;
    }

    // 移动构造（可选，优化用）
    Attribute(Attribute&& other) noexcept : type(other.type) {
        value = other.value;
        other.type = AttrType::UINT32;
        other.value.u32 = 0;
    }

    // 移动赋值
    Attribute& operator=(Attribute&& other) noexcept {
        if (this == &other) return *this;
        if (type == AttrType::STRING) {
            delete value.s;
        }
        type = other.type;
        value = other.value;
        other.type = AttrType::UINT32;
        other.value.u32 = 0;
        return *this;
    }

    // 析构：释放字符串
    ~Attribute() {
        if (type == AttrType::STRING) {
            delete value.s;
        }
    }
};

// 主属性管理类
struct Attr {
    void SetAttr(uint32_t key, uint32_t val) {
        Attribute attr;
        attr.type = AttrType::UINT32;
        attr.value.u32 = val;
        map_[key] = attr; // 原始写法，最兼容
    }

    void SetAttr(uint32_t key, float val) {
        Attribute attr;
        attr.type = AttrType::FLOAT;
        attr.value.f = val;
        map_[key] = attr;
    }

    void SetAttr(uint32_t key, int64_t val) {
        Attribute attr;
        attr.type = AttrType::INT64;
        attr.value.i64 = val;
        map_[key] = attr;
    }

    void SetAttr(uint32_t key, bool val) {
        Attribute attr;
        attr.type = AttrType::BOOL;
        attr.value.b = val;
        map_[key] = attr;
    }

    void SetAttr(uint32_t key, const std::string& val) {
        Attribute attr;
        attr.type = AttrType::STRING;
        attr.value.s = new std::string(val);
        map_[key] = attr;
    }

    ~Attr() = default;

    template<typename T> struct AttributeAccessor;

    // ---------- Get指针 ----------
    template<typename T>
    T* GetAttrPointer(uint32_t key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        if (it->second.type != AttributeAccessor<T>::type) return nullptr;
        return AttributeAccessor<T>::get(it->second);
    }

    template<typename T>
    const T* GetAttrPointer(uint32_t key) const {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        if (it->second.type != AttributeAccessor<T>::type) return nullptr;
        return AttributeAccessor<T>::get(it->second);
    }

    // ---------- 便捷获取 ----------
    template<typename T>
    T GetAttr(uint32_t key, T default_val = {}) const {
        const T* ptr = GetAttrPointer<T>(key);
        return ptr ? *ptr : default_val;
    }

    bool isEmpty() const { return map_.empty(); }

private:
    std::unordered_map<uint32_t, Attribute> map_;
};

// template<> const AttrType Attr::AttributeAccessor<uint32_t>::type;
// template<> const AttrType Attr::AttributeAccessor<float>::type;
// template<> const AttrType Attr::AttributeAccessor<int64_t>::type;
// template<> const AttrType Attr::AttributeAccessor<bool>::type;
// template<> const AttrType Attr::AttributeAccessor<std::string>::type;

// 特化：uint32_t
template<>
struct Attr::AttributeAccessor<uint32_t> {
    inline static const AttrType type = AttrType::UINT32;
    static uint32_t* get(Attribute& attr) { return &attr.value.u32; }
    static const uint32_t* get(const Attribute& attr) { return &attr.value.u32; }
};

// 特化：float
template<>
struct Attr::AttributeAccessor<float> {
    inline static const AttrType type = AttrType::FLOAT;
    static float* get(Attribute& attr) { return &attr.value.f; }
    static const float* get(const Attribute& attr) { return &attr.value.f; }
};

// 特化：int64_t
template<>
struct Attr::AttributeAccessor<int64_t> {
    inline static const AttrType type = AttrType::INT64;
    static int64_t* get(Attribute& attr) { return &attr.value.i64; }
    static const int64_t* get(const Attribute& attr) { return &attr.value.i64; }
};

// 特化：bool
template<>
struct Attr::AttributeAccessor<bool> {
    inline static const AttrType type = AttrType::BOOL;
    static bool* get(Attribute& attr) { return &attr.value.b; }
    static const bool* get(const Attribute& attr) { return &attr.value.b; }
};

// 特化：std::string
template<>
struct Attr::AttributeAccessor<std::string> {
    inline static const AttrType type = AttrType::STRING;
    // 注意：value.s 本身是指针，直接返回
    static std::string* get(Attribute& attr) { return attr.value.s; }
    static const std::string* get(const Attribute& attr) { return attr.value.s; }
};

struct Single {
    public:
        MDesc desc;
        StorageShape shape;
        OptionalInputTensor optionalInputTensor;
        OptionalInputTensor &tensor = optionalInputTensor;
        OptionalInputShape optionalInputShape;
        Attr attr;
        Single() {
            optionalInputTensor.storageShape = shape;
        }
        // __gm__ uint8_t *tensor;
        std::string GetPlatformInfo() const {
            return "A5";
        }

        std::string GetNodeName() const {
            return "FIA";
        }

        const StorageShape& GetStorageShape() const {
            return optionalInputTensor.storageShape;
        }
        MDesc GetInputDesc() const {
            return desc;
        }

        MDesc GetOptionalInputDesc() const {
            return desc;
        }

        OptionalInputTensor GetOptionalInputTensor() const {
            return optionalInputTensor;
        }

        OptionalInputShape GetOptionalInputShape() const {
            return optionalInputShape;
        }

        void SetInputDesc(DataType type) {
            desc.SetInputDesc(type);
        }

        StorageShape SetShape(StorageShape s) {
            shape = s;
        }

        void SetOptionalInputTensor(OptionalInputTensor tensor) {
            optionalInputTensor = tensor;
        }

        void SetOptionalInputShape(OptionalInputShape shape) {
            optionalInputShape = shape;
        }

        Attr SetAttrs(Attr att) {
            attr = att;
        }
};

struct TilingContext {
    Single query;
    Single key;
    Single value;
    bool pseShift = false;
    bool deqScale1Shape = false;
    bool deqScale2Shape = false;
    bool scale1Shape = false;
    bool scale2Shape = false;
    bool offset2Shape = false;
    Single attenMask;
    Single actualSeqLengthsQ;
    Single actualSeqLengths;
    Single deqScale1;
    Single quantScale1;
    Single deqScale2;
    Single quantScale2;
    Single quantOffset2;
    Single antiquantScale;
    Single antiquantOffset;
    Single blockTable;
    Single queryPaddingSize;
    Single elseSingle;
    Attr attrs;
    std::map<uint32_t, Single> OBJECT_MAP = {
        {QUERY_INDEX, query}, 
        {KEY_INDEX, key},
        {VALUE_INDEX, value},
        {ATTEN_MASK_INDEX, attenMask},
        {ACTUAL_SEQ_Q_INDEX, actualSeqLengthsQ},
        {ACTUAL_SEQ_KV_INDEX, actualSeqLengths},
        {DEQUANT_SCALE1_INDEX, deqScale1},
        {QUANT_SCALE1_INDEX, quantScale1},
        {DEQUANT_SCALE2_INDEX, deqScale2},
        {QUANT_SCALE2_INDEX, quantScale2},
        {QUANT_OFFSET2_INDEX, quantOffset2},
        {ANTIQUANT_SCALE_INDEX, antiquantScale},
        {ANTIQUANT_OFFSET_INDEX, antiquantOffset},
        {BLOCK_TABLE_INDEX, blockTable},
        {QUERY_PADDING_SIZE_INDEX, queryPaddingSize},
        {KV_PADDING_SIZE_INDEX, elseSingle},
        {KEY_ANTIQUANT_SCALE_INDEX, elseSingle},
        {KEY_ANTIQUANT_OFFSET_INDEX, elseSingle},
        {VALUE_ANTIQUANT_SCALE_INDEX, elseSingle},
        {VALUE_ANTIQUANT_OFFSET_INDEX, elseSingle},
        {KEY_SHARED_PREFIX_INDEX, elseSingle},
        {VALUE_SHARED_PREFIX_INDEX, elseSingle},
        {ACTUAL_SHARED_PREFIX_LEN_INDEX, elseSingle},
        {QUERY_ROPE_INDEX, elseSingle},
        {KEY_ROPE_INDEX, elseSingle},
        {KEY_ROPE_ANTIQUANT_SCALE_INDEX, elseSingle},
        {DEQUANT_SCALE_QUERY_INDEX, elseSingle},
        {LEARNABLE_SINK_INDEX, elseSingle},
        {Q_START_IDX_INDEX, elseSingle},
        {KV_START_IDX_INDEX, elseSingle},
        {ATTENTION_OUT_INDEX, elseSingle}
    };

    MDesc GetOptionalInputDesc(uint32_t index) const {
        return OBJECT_MAP.at(index).GetInputDesc();
    }

    MDesc GetInputDesc(uint32_t index) const {
        return OBJECT_MAP.at(index).GetInputDesc();
    }

    MDesc GetOutputDesc(uint32_t index) const {
        return OBJECT_MAP.at(index).GetInputDesc();
    }    

    const StorageShape& GetStorageShape(uint32_t index) const {
        return OBJECT_MAP.at(index).GetStorageShape();
    }

    OptionalInputTensor GetInputShape(uint32_t index) const {
        return OBJECT_MAP.at(index).GetOptionalInputTensor();
    }

    OptionalInputTensor GetOutputShape(uint32_t index) const {
        return OBJECT_MAP.at(index).GetOptionalInputTensor();
    }

    OptionalInputTensor GetOptionalInputTensor(uint32_t index) const {
        return OBJECT_MAP.at(index).GetOptionalInputTensor();
    }

    OptionalInputShape GetOptionalInputShape(uint32_t index) const {
        return OBJECT_MAP.at(index).GetOptionalInputShape();
    }

    void SetInputDesc(uint32_t index, const DataType type) {
        OBJECT_MAP.at(index).SetInputDesc(type);
    }
    
    // void SetInputShape(uint32_t index, const std::vector<int64_t>& shape) {
    //     OBJECT_MAP.at(index).SetOptionalInputTensor(shape);
    // }
    
    void SetOutputShape(uint32_t index, const OptionalInputTensor& shape) {
        OBJECT_MAP.at(index).SetOptionalInputTensor(shape);
    }
    
    void SetOptionalInputTensor(uint32_t index, const OptionalInputTensor& tensor) {
        OBJECT_MAP.at(index).SetOptionalInputTensor(tensor);
    }
    
    void SetOptionalInputShape(uint32_t index, const OptionalInputShape& shape) {
        OBJECT_MAP.at(index).SetOptionalInputShape(shape);
    }

    void SetInputShapeFromVector(uint32_t index, const std::vector<int64_t>& shape, DataType tDataType) {
        OptionalInputTensor& tensor = OBJECT_MAP.at(index).optionalInputTensor;
        tensor.tDataType = tDataType;
        tensor.tIsNull = false;
        tensor.storageShape.tShape = shape;
    }
    
    void SetInputShapeNull(uint32_t index) {
        OptionalInputTensor& tensor = OBJECT_MAP.at(index).optionalInputTensor;
        tensor.storageShape.tShape.clear();
        tensor.tIsNull = true;
    }

    const Attr& GetAttrs() const {
        return attrs;
    }

    Attr& GetAttrs() {
        return attrs;
    }

    char* GetNodeName() {
        return "FIA";
    }

    char* GetPlatformInfo() {
        return "Platform";
    }
};

struct ContextParamsForPFATiling {
    DataType inputDataType;
    DataType kDataType;
    DataType vDataType;
    DataType pseShiftDataType;
    DataType maskDataType;
    DataType quantScale2Type;
    DataType quantOffset2Type;
    DataType blockTableType;
    DataType outputDataType;
    DataType KeyAntiquantScaleType;
    DataType valueAntiquantScaleType;
    DataType KeyAntiquantOffsetType;
    DataType valueAntiquantOffsetType;
    DataType blockTable;
};

struct IncreFlashAttentionContext {
    char* opName;
    char* platformInfo;
    Single query;
    Single key;
    Single value;
    Single attenOut;
    bool pseShift;
    Single attenMask;
    Single actualSeqLengthsQ;
    Single actualSeqLengths;
    Single deqScale1;
    Single quantScale1;
    Single deqScale2;
    Single quantScale2;
    Single quantOffset2;
    Single antiquantScale;
    Single antiquantOffset;
    Single blockTable;
    Single queryPaddingSize;
    Single kvPaddingSize;
    Single keyAntiquantScale;
    Single keyAntiquantOffset;
    Single valueAntiquantScale;
    Single valueAntiquantOffset;
    Single keySharedPrefix;
    Single valueSharedPrefix;
    Single actualSharedPrefixLen;
    Single queryRope;
    Single keyRope;
    Single keyRopeAntiquantScale;
    Single dequantScaleQuery;
    uint32_t workSpaces;
    uint32_t numHeads;
    float scaleValue;
    std::string layOut;
    uint32_t kvHeadNums;
    uint32_t blockSize;
    int64_t antiquantMode;
    bool softmaxLseFlag;
    int64_t keyAntiquantMode;
    int64_t valueAntiquantMode;
    uint32_t innerPrecise;
    uint32_t sparseMode;
    int64_t queryQuantMode;
    int64_t windowSize;
};

struct Tensor {
    FiaTiling tensor;
    FiaTiling GetOptionalInputTensor(uint32_t index) const {
        return tensor;
    }
};

// struct ContextParamsForPFATiling {
//     const Tensor *pseShift = nullptr;
//     const Tensor *attentionMask = nullptr;
//     const Tensor *sabi = nullptr;
//     const Tensor *actualSequenceLengthQ = nullptr;
//     const Tensor *actualSequenceLengthKV = nullptr;
//     const Tensor *antiquantScale = nullptr;
//     const Tensor *antiquantOffset = nullptr;
//     const Tensor *queryPaddingSize = nullptr;
//     const Tensor *kvPaddingSize = nullptr;
//     const Tensor *blockTable = nullptr;
//     const Tensor *keySharedPrefix = nullptr;
//     const Tensor *valueSharedPrefix = nullptr;
//     const Tensor *actualSharedPrefixLen = nullptr;
//     const Tensor *learnableSink = nullptr;

//     const Tensor *KeyAntiquantScale = nullptr;
//     const Tensor *valueAntiquantScale = nullptr;
//     const Tensor *KeyAntiquantOffset = nullptr;
//     const Tensor *valueAntiquantOffset = nullptr;

//     const Tensor *qStartIdx = nullptr;
//     const Tensor *kvStartIdx = nullptr;

//     DataType inputDataType = DT_FLOAT16;
//     DataType kDataType = DT_FLOAT16;
//     DataType vDataType = DT_FLOAT16;
//     DataType qRopeDataType = DT_FLOAT16;
//     DataType kRopeDataType = DT_FLOAT16;
//     DataType pseShiftDataType = DT_FLOAT16;
//     DataType maskDataType = DT_FLOAT16;
//     DataType sabiDataType = DT_FLOAT16;
//     DataType blockTableType = DT_FLOAT16;
//     DataType outputDataType = DT_FLOAT16;
//     const char *opName = nullptr;
//     const gert::StorageShape *queryInputShape = nullptr;
//     const gert::StorageShape *keyInputShape = nullptr;
//     const gert::StorageShape *queryRopeInputShape = nullptr;
//     const gert::StorageShape *keyRopeInputShape = nullptr;
//     const gert::StorageShape *valueInputShape = nullptr;
//     const gert::StorageShape *pseShiftShape = nullptr;
//     const gert::StorageShape *attentionMaskShape = nullptr;
//     const gert::StorageShape *sabiShape = nullptr;
//     const gert::StorageShape *deqScale1Shape = nullptr;
//     const gert::StorageShape *scale1Shape = nullptr;
//     const gert::StorageShape *deqScale2Shape = nullptr;
//     const gert::StorageShape *scale2Shape = nullptr;
//     const gert::StorageShape *offset2Shape = nullptr;
//     const gert::StorageShape *antiquantScaleShape = nullptr;
//     const gert::StorageShape *antiquantOffsetShape = nullptr;
//     const gert::StorageShape *blockTableShape = nullptr;
//     const gert::StorageShape *outputShape = nullptr;
//     const gert::StorageShape *lseoutputShape = nullptr;

//     const gert::StorageShape *KeyAntiquantScaleShape = nullptr;
//     const gert::StorageShape *valueAntiquantScaleShape = nullptr;
//     const gert::StorageShape *KeyAntiquantOffsetShape = nullptr;
//     const gert::StorageShape *valueAntiquantOffsetShape = nullptr;
//     const gert::StorageShape *queryRope = nullptr;
//     const gert::StorageShape *keyRope = nullptr;
//     const gert::StorageShape *learnableSinkShape = nullptr;
//     DataType KeyAntiquantScaleType = DataType::DT_FLOAT16;
//     DataType valueAntiquantScaleType = DataType::DT_FLOAT16;
//     DataType KeyAntiquantOffsetType = DataType::DT_FLOAT16;
//     DataType valueAntiquantOffsetType = DataType::DT_FLOAT16;

//     const int64_t *innerPrecisePtr = nullptr;
//     const int32_t *headsNumber = nullptr;
//     const int32_t *sparseMode = nullptr;
//     const int64_t *preToken = nullptr;
//     const int64_t *nextToken = nullptr;
//     const float *scaleValue = nullptr;
//     const int32_t *blockSize = nullptr;
//     const char *layout = nullptr;
//     const int32_t *numKeyValueHeads = nullptr;
//     size_t *workspaceSize = nullptr;
//     const int64_t *pseType = nullptr;
//     const BlitzSparseAttentionCompileInfo *compileInfoPtr = nullptr;
//     DataType deqScaleType = DataType::DT_FLOAT16;
//     DataType deqScale2Type = DataType::DT_FLOAT16;
//     DataType quantScale2Type = DataType::DT_FLOAT16;
//     DataType quantOffset2Type = DataType::DT_FLOAT16;
//     uint32_t isKvContinuous = 1;
//     std::vector<const gert::StorageShape *> kTensorList = {nullptr};
//     std::vector<const gert::StorageShape *> vTensorList = {nullptr};
//     uint32_t maxKVs = 0;
//     uint32_t fromFused = 0;
//     uint32_t emptyTensor = 0;
//     uint32_t isBSNDOut = 0;
//     const bool *softmaxLseFlag = nullptr;
//     bool isSoftMaxLseEnable = false;
//     uint32_t fromTilingSink = 0; // Flag indicating whether it is the step to enter the workspace calculation from tiling sinking
//     bool hasKeyAntiquantScale = 0;
//     bool hasValueAntiquantScale = 0;
//     uint32_t isMsd = 0;
//     const int64_t *keyAntiquantMode = nullptr;
//     const int64_t *valueAntiquantMode = nullptr;
//     bool hasKeyAntiquantOffset = 0;
//     bool hasLearnableSink = 0;
// };

// 基础参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionBaseParams)
// TILING_DATA_FIELD_DEF(uint32_t, bSize)
// TILING_DATA_FIELD_DEF(uint32_t, n2Size)
// TILING_DATA_FIELD_DEF(uint32_t, gSize)
// TILING_DATA_FIELD_DEF(uint32_t, s1Size)
// TILING_DATA_FIELD_DEF(uint32_t, s2Size)
// TILING_DATA_FIELD_DEF(uint32_t, headDim)
// TILING_DATA_FIELD_DEF(uint32_t, headDimRope)
// TILING_DATA_FIELD_DEF(uint32_t, actualSeqS1Dims)
// TILING_DATA_FIELD_DEF(uint32_t, actualSeqS2Dims)
// TILING_DATA_FIELD_DEF(uint32_t, accumQSeqFlag)
// TILING_DATA_FIELD_DEF(uint32_t, accumKVSeqFlag)
// TILING_DATA_FIELD_DEF(float, scaleValue)
// TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
// TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
// TILING_DATA_FIELD_DEF(uint32_t, batchContinuous)
// TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
// TILING_DATA_FIELD_DEF(uint32_t, needInit)
// TILING_DATA_FIELD_DEF(uint32_t, slidingFlag)
// TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
// TILING_DATA_FIELD_DEF(uint32_t, isLegacyIfa)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionBaseParamsOp, FusedInferAttentionBaseParams)

// // PageAttention 参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionPageAttentionParams)
// TILING_DATA_FIELD_DEF(uint32_t, blockSize)
// TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionPageAttentionParamsOp, FusedInferAttentionPageAttentionParams)

// // AttenMask 参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionMaskParams)
// TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
// TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatchStride)
// TILING_DATA_FIELD_DEF(uint32_t, attenMaskStride)
// TILING_DATA_FIELD_DEF(int32_t, preToken)
// TILING_DATA_FIELD_DEF(int32_t, nextToken)
// TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid)
// TILING_DATA_FIELD_DEF(uint32_t, isExistRowInvalid)
// TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionMaskParamsOp, FusedInferAttentionMaskParams)

// // 内切基本块参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionInnerSplitParams)
// TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
// TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionInnerSplitParamsOp, FusedInferAttentionInnerSplitParams)

// // workspace参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionWorkspaceParams)
// TILING_DATA_FIELD_DEF(uint32_t, mm1ResSize)
// TILING_DATA_FIELD_DEF(uint32_t, mm2ResSize)
// TILING_DATA_FIELD_DEF(uint32_t, fdAccumOutSize)
// TILING_DATA_FIELD_DEF(uint32_t, fdLogSumExpSize)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionWorkspaceParamsOp, FusedInferAttentionWorkspaceParams)

// // 外切分核参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionOuterSplitParams)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2End)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1End)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2End)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionOuterSplitParamsOp, FusedInferAttentionOuterSplitParams)

// // FlashDecode规约参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionFlashDecodeParams)
// TILING_DATA_FIELD_DEF(uint32_t, numOfFdHead)
// TILING_DATA_FIELD_DEF(uint32_t, reserved)
// TILING_DATA_FIELD_DEF(uint32_t, gS1BaseSizeOfFd)                                    // FD负载均衡中，每个FD任务按gS1切分的基本size
// TILING_DATA_FIELD_DEF(uint32_t, usedVecNumOfFd)                                     // FD负载均衡中，用到的vector数
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2IdxOfFdHead)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1IdxOfFdHead)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitNumOfFdHead)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitStartIdxOfCore)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1SplitNumOfFdHead)          // FD负载均衡中，每个FD任务按gS1基本size切分后的份数
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1LastPartSizeOfFdHead)      // FD负载均衡中，每个FD任务按gS1基本size切分后，最后一份的gS1大小，即尾块大小
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHead)        // FD负载均衡中，每个vector核处理的最后一个FD任务的序号
// TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHeadSplit)   // FD负载均衡中，每个vector核处理的最后一个FD任务的子划分的序号
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionFlashDecodeParamsOp, FusedInferAttentionFlashDecodeParams)

// // 公共前缀
// BEGIN_TILING_DATA_DEF(FusedInferAttentionPrefixParams)
// TILING_DATA_FIELD_DEF(uint64_t, prefixMaxLen)
// TILING_DATA_FIELD_DEF(uint64_t, prefixLen)
// TILING_DATA_FIELD_DEF(bool, prefixFlag)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionPrefixParamsOp, FusedInferAttentionPrefixParams)
// // Pse 注册参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionPseParams)
// TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
// TILING_DATA_FIELD_DEF(uint32_t, pseShiftByBatch)
// TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1)
// TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionPseParamsOp, FusedInferAttentionPseParams)

// // Left Padding 参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionLeftPaddingParams)
// TILING_DATA_FIELD_DEF(uint32_t, qPaddingFlag)
// TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionLeftPaddingParamsOp, FusedInferAttentionLeftPaddingParams)
// // 后量化 参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionPostQuantParams)
// TILING_DATA_FIELD_DEF(uint32_t, isPerChnOut)
// TILING_DATA_FIELD_DEF(uint32_t, isOutQuantTypeBf16)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionPostQuantParamsOp, FusedInferAttentionPostQuantParams)

// //MLA非量化模板TilingData
// BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreTilingData)
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionBaseParams, baseParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPageAttentionParams, pageAttenParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionMaskParams, maskParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionWorkspaceParams, workspaceParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionInnerSplitParams, innerSplitParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionOuterSplitParams, outerSplitParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionFlashDecodeParams, fdParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPrefixParams, prefixParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPseParams, pseParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionLeftPaddingParams, leftPaddingParams);
// TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPostQuantParams, postquantParams);
// END_TILING_DATA_DEF

// // empty tenmsor 模板TilingData
// BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreEmptyTensorTilingData)
// TILING_DATA_FIELD_DEF(uint64_t, totalOutputSize)
// TILING_DATA_FIELD_DEF(uint64_t, singleCoreSize)
// TILING_DATA_FIELD_DEF(uint64_t, totalLseSize)
// TILING_DATA_FIELD_DEF(uint64_t, singleCoreLseSize)
// TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
// TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
// TILING_DATA_FIELD_DEF(uint32_t, headDim)
// END_TILING_DATA_DEF

// // 全量化 参数 当前无
// BEGIN_TILING_DATA_DEF(FusedInferAttentionFullQuantParams)
// TILING_DATA_FIELD_DEF(uint32_t, placeHolder)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionFullQuantParamsOp, FusedInferAttentionFullQuantParams)

// // L2 Cache 参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionL2CacheParams)
// TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionL2CacheParamsOp, FusedInferAttentionL2CacheParams)

// // MSD 参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionMsdParams)
// TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionMsdParamsOp, FusedInferAttentionMsdParams)

// // 伪量化 参数
// BEGIN_TILING_DATA_DEF(FusedInferAttentionAntiqParams)
// TILING_DATA_FIELD_DEF(uint32_t, antiqSeqSize)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(FusedInferAttentionAntiqParamsOp, FusedInferAttentionAntiqParams)

// extern "C" {
// bool DeviceDoOpTilingIncreFlashAttention(optiling::TilingContext *context);
// bool DeviceDoOpTilingFusedInferAttentionScore(optiling::TilingContext *context);
// }
// bool TilingFusedInferAttentionScore(optiling::TilingContext *context);
// FIA_EXTERN_C bool DoOpTilingFusedInferAttentionScore(optiling::TilingContext *context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
