/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;

namespace {

string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    string exePath(path, len);
    auto pos = exePath.find_last_of("\\/");
    return pos == string::npos ? string(".\\") : exePath.substr(0, pos + 1);
#else
    char path[4096] = {0};
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return "./";
    }
    path[len] = '\0';
    string exePath(path);
    auto pos = exePath.find_last_of('/');
    return pos == string::npos ? string("./") : exePath.substr(0, pos + 1);
#endif
}

string GetCurrentFileDir()
{
    string filePath = __FILE__;
    auto pos = filePath.find_last_of("\\/");
    return pos == string::npos ? string(".") : filePath.substr(0, pos);
}

string JoinPath(const string &dir, const string &file)
{
    if (dir.empty()) {
        return file;
    }
    char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

bool FileExists(const string &path)
{
    ifstream in(path);
    return in.is_open();
}

string ResolveCsvPath(const string &csvName)
{
    const string exePath = JoinPath(GetExeDirPath(), csvName);
    if (FileExists(exePath)) {
        return exePath;
    }
    const string srcPath = JoinPath(GetCurrentFileDir(), csvName);
    if (FileExists(srcPath)) {
        return srcPath;
    }
    return exePath;
}

void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
{
    const auto delimiterLen = delimiter.size();
    string::size_type currPos = 0;
    string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }
    if (currPos <= input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

string Trim(string value)
{
    auto isNotSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), isNotSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(), value.end());
    return value;
}

bool ParseBool(const string &value)
{
    string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower == "true" || lower == "1" || lower == "yes";
}

vector<int64_t> ParseI64List(const string &value, const string &sep = "|")
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, sep, tokens);
    vector<int64_t> out;
    out.reserve(tokens.size());
    for (const auto &token : tokens) {
        out.emplace_back(stoll(Trim(token)));
    }
    return out;
}

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, ":", tokens);
    vector<int64_t> out;
    out.reserve(tokens.size());
    for (const auto &token : tokens) {
        out.emplace_back(stoll(Trim(token)));
    }
    return out;
}

vector<vector<int64_t>> ParseDimsList(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> items;
    SplitStr2Vec(trimmed, ";", items);
    vector<vector<int64_t>> out;
    out.reserve(items.size());
    for (const auto &item : items) {
        out.emplace_back(ParseDims(item));
    }
    return out;
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> kMap = {
        {"FLOAT", ACL_FLOAT},         {"FLOAT16", ACL_FLOAT16},   {"BF16", ACL_BF16},           {"INT8", ACL_INT8},
        {"INT4", ACL_INT4},           {"INT64", ACL_INT64},       {"UINT64", ACL_UINT64},       {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
    };
    auto it = kMap.find(dtype);
    return it == kMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> kMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
    };
    auto it = kMap.find(format);
    return it == kMap.end() ? ACL_FORMAT_ND : it->second;
}

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, op::SocVersion> socMap = {
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    auto it = socMap.find(socVersion);
    op::SetPlatformSocVersion(it == socMap.end() ? op::SocVersion::ASCEND910B : it->second);
}

TensorDesc MakeTensorDesc(const vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                          const vector<int64_t> &storageShape, const vector<int64_t> &stride)
{
    if (!storageShape.empty() || !stride.empty()) {
        return TensorDesc(shape, dtype, format, stride, 0, storageShape).ValueRange(-10, 10);
    }
    return TensorDesc(shape, dtype, format).ValueRange(-10, 10);
}

TensorListDesc MakeTensorListDesc(const vector<vector<int64_t>> &shapes, aclDataType dtype, aclFormat format,
                                  const vector<vector<int64_t>> &storageShapes, const vector<vector<int64_t>> &strides)
{
    vector<TensorDesc> descs;
    descs.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        const vector<int64_t> emptyVec;
        const auto &storage = i < storageShapes.size() ? storageShapes[i] : emptyVec;
        const auto &stride = i < strides.size() ? strides[i] : emptyVec;
        descs.emplace_back(MakeTensorDesc(shapes[i], dtype, format, storage, stride));
    }
    return TensorListDesc(descs);
}

struct SwigluOpApiCase {
    void Run() const
    {
        SetupPlatformForCase(socVersion);
        auto x = MakeTensorDesc(ParseDims(xShape), ParseDtype(xDtype), ParseFormat(xFormat), ParseDims(xStorageShape),
                                ParseDims(xStride));
        auto weight = MakeTensorListDesc(ParseDimsList(weightShapes), ParseDtype(weightDtype), ParseFormat(weightFormat),
                                         ParseDimsList(weightStorageShapes), ParseDimsList(weightStrides));
        auto weightScale =
            MakeTensorListDesc(ParseDimsList(weightScaleShapes), ParseDtype(weightScaleDtype), ParseFormat(weightScaleFormat),
                               ParseDimsList(weightScaleStorageShapes), ParseDimsList(weightScaleStrides));
        auto xScale = MakeTensorDesc(ParseDims(xScaleShape), ParseDtype(xScaleDtype), ParseFormat(xScaleFormat), {}, {});
        auto groupList = MakeTensorDesc(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat), {}, {});
        auto out1 = MakeTensorDesc(ParseDims(out1Shape), ParseDtype(out1Dtype), ParseFormat(out1Format), {}, {});
        auto out2 = MakeTensorDesc(ParseDims(out2Shape), ParseDtype(out2Dtype), ParseFormat(out2Format), {}, {});

        vector<int64_t> tuningValues = ParseI64List(tuningConfig, "|");
        aclIntArray *tuningConfigArr = tuningValues.empty() ? nullptr : aclCreateIntArray(tuningValues.data(), tuningValues.size());

        auto weightAssistShapesVec = ParseDimsList(weightAssistShapes);
        auto smoothScaleDims = ParseDims(smoothScaleShape);
        bool hasWeightAssist = !weightAssistShapesVec.empty();
        bool hasSmoothScale = !smoothScaleDims.empty();

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACL_SUCCESS;
        if (hasWeightAssist && hasSmoothScale) {
            auto weightAssist =
                MakeTensorListDesc(weightAssistShapesVec, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat),
                                   ParseDimsList(weightAssistStorageShapes), ParseDimsList(weightAssistStrides));
            auto smoothScale =
                MakeTensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat), {}, {});
            if (api == "WeightNzV2") {
                auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                    INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                          dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                    OUTPUT(out1, out2));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                    INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                          dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                    OUTPUT(out1, out2));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (hasWeightAssist) {
            auto weightAssist =
                MakeTensorListDesc(weightAssistShapesVec, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat),
                                   ParseDimsList(weightAssistStorageShapes), ParseDimsList(weightAssistStrides));
            auto smoothScale = nullptr;
            if (api == "WeightNzV2") {
                auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                    INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                          dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                    OUTPUT(out1, out2));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                    INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                          dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                    OUTPUT(out1, out2));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (hasSmoothScale) {
            auto weightAssist = nullptr;
            auto smoothScale =
                MakeTensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat), {}, {});
            if (api == "WeightNzV2") {
                auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                    INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                          dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                    OUTPUT(out1, out2));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                    INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                          dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                    OUTPUT(out1, out2));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else {
            auto weightAssist = nullptr;
            auto smoothScale = nullptr;
            if (api == "WeightNzV2") {
                auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                    INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                          dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                    OUTPUT(out1, out2));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                    INPUT(x, weight, weightScale, weightAssist, nullptr, xScale, smoothScale, groupList,
                                          dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                    OUTPUT(out1, out2));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        }

        if (tuningConfigArr != nullptr) {
            aclDestroyIntArray(tuningConfigArr);
        }
        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, static_cast<aclnnStatus>(expectRet));
        }
    }

    string socVersion;
    string caseName;
    string api;
    string checkRet;
    uint64_t expectRet;
    string xShape;
    string xDtype;
    string xFormat;
    string xStorageShape;
    string xStride;
    string weightShapes;
    string weightDtype;
    string weightFormat;
    string weightStorageShapes;
    string weightStrides;
    string weightScaleShapes;
    string weightScaleDtype;
    string weightScaleFormat;
    string weightScaleStorageShapes;
    string weightScaleStrides;
    string weightAssistShapes;
    string weightAssistDtype;
    string weightAssistFormat;
    string weightAssistStorageShapes;
    string weightAssistStrides;
    string xScaleShape;
    string xScaleDtype;
    string xScaleFormat;
    string smoothScaleShape;
    string smoothScaleDtype;
    string smoothScaleFormat;
    string groupListShape;
    string groupListDtype;
    string groupListFormat;
    int64_t dequantMode;
    int64_t dequantDtype;
    int64_t quantMode;
    int64_t quantDtype;
    string tuningConfig;
    string out1Shape;
    string out1Dtype;
    string out1Format;
    string out2Shape;
    string out2Dtype;
    string out2Format;
};

vector<SwigluOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<SwigluOpApiCase> cases;
    string line;
    bool headerSkipped = false;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != 45) {
            continue;
        }
        SwigluOpApiCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.api = Trim(cols[i++]);
        c.checkRet = Trim(cols[i++]);
        c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
        c.xShape = Trim(cols[i++]);
        c.xDtype = Trim(cols[i++]);
        c.xFormat = Trim(cols[i++]);
        c.xStorageShape = Trim(cols[i++]);
        c.xStride = Trim(cols[i++]);
        c.weightShapes = Trim(cols[i++]);
        c.weightDtype = Trim(cols[i++]);
        c.weightFormat = Trim(cols[i++]);
        c.weightStorageShapes = Trim(cols[i++]);
        c.weightStrides = Trim(cols[i++]);
        c.weightScaleShapes = Trim(cols[i++]);
        c.weightScaleDtype = Trim(cols[i++]);
        c.weightScaleFormat = Trim(cols[i++]);
        c.weightScaleStorageShapes = Trim(cols[i++]);
        c.weightScaleStrides = Trim(cols[i++]);
        c.weightAssistShapes = Trim(cols[i++]);
        c.weightAssistDtype = Trim(cols[i++]);
        c.weightAssistFormat = Trim(cols[i++]);
        c.weightAssistStorageShapes = Trim(cols[i++]);
        c.weightAssistStrides = Trim(cols[i++]);
        c.xScaleShape = Trim(cols[i++]);
        c.xScaleDtype = Trim(cols[i++]);
        c.xScaleFormat = Trim(cols[i++]);
        c.smoothScaleShape = Trim(cols[i++]);
        c.smoothScaleDtype = Trim(cols[i++]);
        c.smoothScaleFormat = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListDtype = Trim(cols[i++]);
        c.groupListFormat = Trim(cols[i++]);
        c.dequantMode = stoll(Trim(cols[i++]));
        c.dequantDtype = stoll(Trim(cols[i++]));
        c.quantMode = stoll(Trim(cols[i++]));
        c.quantDtype = stoll(Trim(cols[i++]));
        c.tuningConfig = Trim(cols[i++]);
        c.out1Shape = Trim(cols[i++]);
        c.out1Dtype = Trim(cols[i++]);
        c.out1Format = Trim(cols[i++]);
        c.out2Shape = Trim(cols[i++]);
        c.out2Dtype = Trim(cols[i++]);
        c.out2Format = Trim(cols[i++]);
        cases.emplace_back(c);
    }
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<SwigluOpApiCase> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class grouped_matmul_swiglu_quant_v2_opapi_csv_test : public testing::TestWithParam<SwigluOpApiCase> {};

TEST_P(grouped_matmul_swiglu_quant_v2_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_swiglu_quant_v2_opapi_csv,
    grouped_matmul_swiglu_quant_v2_opapi_csv_test,
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_v2_l2.csv"))),
    BuildCaseName);

}  // namespace
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_GroupedMatmulSwigluQuantV2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulSwigluQuantV2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_GroupedMatmulSwigluQuantV2_test TearDown" << endl;
    }
};

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_normal_nz_workspace_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_wrong_nd_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_wrong_nd_no_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a8_without_weight_assist_matrix_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_redundant_weight_assist_matrix_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc weight_assist_matrix = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_assist_matrix_desc = TensorListDesc({weight_assist_matrix});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_matrix_desc, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_smoothscale_1d_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_smoothscale_2d_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e, n / 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_smoothscale_invalid_dim_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e, n / 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_smoothscale_wrong_shape_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_wtrans_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {e, 1, k}, 0, {e, k / 64, n / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 =
        TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight1 =
        TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight2 =
        TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight3 =
        TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_nz_workspace_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_multi_weight_wrong_nd_no_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_normal_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_illegal_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_desc, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_normal_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_illegal_case_1_wscale_nullptr)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161001);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_illegal_case_2_shape_mismatch)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, n, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_illegal_case_3_dtype_mismatch)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_illegal_case_4_invalid_format)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_m0_case)
{
    int64_t m = 0;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                            nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_n0_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 0;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                            nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_k0_case)
{
    int64_t m = 2048;
    int64_t k = 0;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                            nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}