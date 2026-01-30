/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_PLATFORM_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_PLATFORM_H

#include "graph/ascend_string.h"
#include "platform/platform_info.h"
#include <cstdint>
#include <map>
#include <any>
#include <string>
#include <dlfcn.h>

#ifndef __SOC_SPEC_H__
#define __SOC_SPEC_H__

#include <cstdint>

enum class NpuArch : uint32_t {
    DAV_1001 = 1001,
    DAV_1002 = 1002,
    DAV_1003 = 1003,
    DAV_1004 = 1004,
    DAV_1999 = 1999,
    DAV_2002 = 2002,
    DAV_2003 = 2003,
    DAV_2004 = 2004,
    DAV_2006 = 2006,
    DAV_2102 = 2102,
    DAV_2103 = 2103,
    DAV_2104 = 2104,
    DAV_2201 = 2201,
    DAV_3002 = 3002,
    DAV_3003 = 3003,
    DAV_3004 = 3004,
    DAV_3102 = 3102,
    DAV_3103 = 3103,
    DAV_3113 = 3113,
    DAV_3502 = 3502,
    DAV_3505 = 3505,
    DAV_3510 = 3510,
    DAV_3801 = 3801,
    DAV_5101 = 5101,
    DAV_5102 = 5102,
    DAV_5161 = 5161,
    DAV_9201 = 9201,
    DAV_9202 = 9202,
    DAV_9301 = 9301,
    DAV_RESV = 0xFFFF
};

#endif

namespace op {

enum class SocVersion {
    ASCEND910 = 0,
    ASCEND910B,
    ASCEND910_93,
    ASCEND950,
    ASCEND910E,
    ASCEND310,
    ASCEND310P,
    ASCEND310B,
    ASCEND310C,
    ASCEND610LITE,
    KIRINX90,
    RESERVED_VERSION = 99999
};

enum class SocSpec { INST_MMAD = 0, RESERVED_SPEC = 99999 };

enum class SocSpecAbility {
    INST_MMAD_F162F16 = 0,
    INST_MMAD_F162F32,
    INST_MMAD_H322F32,
    INST_MMAD_F322F32,
    INST_MMAD_U32U8U8,
    INST_MMAD_S32S8S8,
    INST_MMAD_S32U8S8,
    INST_MMAD_F16F16F16,
    INST_MMAD_F32F16F16,
    INST_MMAD_F16F16U2,
    INST_MMAD_U8,
    INST_MMAD_S8,
    INST_MMAD_U8S8,
    INST_MMAD_F16U2,
};

class PlatformInfoImpl;
class PlatformThreadLockCtx;

class PlatformInfo {
    friend const PlatformInfo& GetCurrentPlatformInfo();
    friend class PlatformThreadLockCtx;

public:
    PlatformInfo() {};

    PlatformInfo(int32_t deviceId) : deviceId_(deviceId){};

    SocVersion GetSocVersion() const;

    const std::string& GetSocLongVersion() const;

    int32_t GetDeviceId() const;

    bool CheckSupport(SocSpec socSpec, SocSpecAbility ability) const;

    int64_t GetBlockSize() const;

    uint32_t GetCubeCoreNum() const;

    uint32_t GetVectorCoreNum() const;

    bool Valid() const;

    bool GetFftsPlusMode() const;

    NpuArch GetCurNpuArch() const;

    fe::PlatFormInfos *GetPlatformInfos() const;

private:
    PlatformInfo &operator=(const PlatformInfo &other) = delete;

    PlatformInfo &operator=(const PlatformInfo &&other) = delete;

    PlatformInfo(const PlatformInfo &other) = delete;

    PlatformInfo(const PlatformInfo &&other) = delete;

    void SetPlatformImpl(PlatformInfoImpl *impl);

    bool valid_ = false;
    int32_t deviceId_{-1};
    PlatformInfoImpl *impl_ = nullptr;

    ~PlatformInfo();
};

const PlatformInfo& GetCurrentPlatformInfo();

ge::AscendString ToString(SocVersion socVersion);

void SetPlatformSocVersion(SocVersion socVersion);

void SetPlatformNpuArch(NpuArch npuArch);

} // namespace op

namespace ops::adv::tests::utils {

class Platform {
public:
    enum class SocVersion {
        Ascend910B1,
        Ascend910B2,
        Ascend910B3,
        Ascend310P3,
        Ascend910_9591,
		SocVersionBottom,
    };
    class SocSpec {
    public:
        std::map<std::string, std::map<std::string, std::any>> spec;

        bool Get(const char *label, const char *key, std::any &value) const;
        bool Get(const char *label, const char *key, std::string &value) const;
        bool Get(const char *label, const char *key, uint32_t &value) const;
        bool Get(const char *label, const char *key, uint64_t &value) const;
    };
    SocSpec socSpec;

    static void SetGlobalPlatform(Platform *platform);
    static Platform *GetGlobalPlatform();

    bool InitArgsInfo(int argc, char **argv);
    const char *GetExeAbsPath();

    Platform &SetSocVersion(const SocVersion &socVersion);
    [[maybe_unused]] [[nodiscard]] uint32_t GetCoreNum() const;
    [[maybe_unused]] [[nodiscard]] int64_t GetBlockDim() const;

    [[maybe_unused]] static void *LoadSo(const char *absPath, int mode = RTLD_NOW | RTLD_GLOBAL);
    [[maybe_unused]] static bool UnLoadSo(void *hdl);
    [[maybe_unused]] [[nodiscard]] static void *LoadSoSym(void *hdl, const char *name);

    [[maybe_unused]] bool LoadOpTilingSo();
    [[maybe_unused]] bool UnLoadOpTilingSo();
    [[maybe_unused]] [[nodiscard]] void *LoadOpTilingSoSym(const char *name);

    [[maybe_unused]] bool LoadOpProtoSo();
    [[maybe_unused]] bool UnLoadOpProtoSo();

private:
    void *tilingSoHdl_ = nullptr;
    void *protoSoHdl_ = nullptr;
    std::string exeAbsPath_;
};

} // namespace ops::adv::tests::utils

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_PLATFORM_H