/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_grouped_matmul_v5.h"
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

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, ":", tokens);
    vector<int64_t> dims;
    for (const auto &token : tokens) {
        dims.emplace_back(stoll(Trim(token)));
    }
    return dims;
}

vector<int64_t> ParseI64List(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, "|", tokens);
    vector<int64_t> values;
    for (const auto &token : tokens) {
        values.emplace_back(stoll(Trim(token)));
    }
    return values;
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},   {"FLOAT16", ACL_FLOAT16}, {"BF16", ACL_BF16}, {"INT8", ACL_INT8},
        {"INT32", ACL_INT32},   {"INT64", ACL_INT64},     {"UINT64", ACL_UINT64},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> formatMap = {{"ND", ACL_FORMAT_ND}, {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ}};
    auto it = formatMap.find(format);
    return it == formatMap.end() ? ACL_FORMAT_ND : it->second;
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

struct GroupedMatmulOpApiCase {
    void Run() const
    {
        SetupPlatformForCase(socVersion);
        TensorListDesc x(1, TensorDesc(ParseDims(xShape), ParseDtype(xDtype), ParseFormat(xFormat)).ValueRange(-10, 10));
        TensorListDesc weight(
            1, TensorDesc(ParseDims(weightShape), ParseDtype(weightDtype), ParseFormat(weightFormat)).ValueRange(-10, 10));
        TensorListDesc out(1, TensorDesc(ParseDims(outShape), ParseDtype(outDtype), ParseFormat(outFormat)));
        TensorDesc groupList(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat));
        vector<int64_t> groupListVal = ParseI64List(groupListValues);
        if (!groupListVal.empty()) {
            groupList.Value(groupListVal);
        }
        TensorListDesc scale(1, TensorDesc(ParseDims(scaleShape), ParseDtype(scaleDtype), ParseFormat(scaleFormat))
                                    .ValueRange(-10, 10));
        TensorListDesc bias(1, TensorDesc(ParseDims(biasShape), ParseDtype(biasDtype), ParseFormat(biasFormat))
                                   .ValueRange(-10, 10));
        TensorListDesc perTokenScale(
            1, TensorDesc(ParseDims(perTokenScaleShape), ParseDtype(perTokenScaleDtype), ParseFormat(perTokenScaleFormat))
                   .ValueRange(-10, 10));

        auto offsetOptional = nullptr;
        auto antiquantScaleOptional = nullptr;
        auto antiquantOffsetOptional = nullptr;
        auto activationInputOptional = nullptr;
        auto activationQuantScaleOptional = nullptr;
        auto activationQuantOffsetOptional = nullptr;
        auto tuningConfigOptional = nullptr;
        auto activationFeatureOutOptional = nullptr;
        auto dynQuantScaleOutOptional = nullptr;

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACL_SUCCESS;
        if (ParseBool(hasBias) && ParseBool(hasPerTokenScale)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                INPUT(x, weight, bias, scale, offsetOptional, antiquantScaleOptional,
                                      antiquantOffsetOptional, perTokenScale, groupList, activationInputOptional,
                                      activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                      groupListType, actType, tuningConfigOptional),
                                OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasBias)) {
            auto perTokenScaleOptional = nullptr;
            auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                INPUT(x, weight, bias, scale, offsetOptional, antiquantScaleOptional,
                                      antiquantOffsetOptional, perTokenScaleOptional, groupList, activationInputOptional,
                                      activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                      groupListType, actType, tuningConfigOptional),
                                OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasPerTokenScale)) {
            auto biasOptional = nullptr;
            auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                INPUT(x, weight, biasOptional, scale, offsetOptional, antiquantScaleOptional,
                                      antiquantOffsetOptional, perTokenScale, groupList, activationInputOptional,
                                      activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                      groupListType, actType, tuningConfigOptional),
                                OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto biasOptional = nullptr;
            auto perTokenScaleOptional = nullptr;
            auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                INPUT(x, weight, biasOptional, scale, offsetOptional, antiquantScaleOptional,
                                      antiquantOffsetOptional, perTokenScaleOptional, groupList, activationInputOptional,
                                      activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                      groupListType, actType, tuningConfigOptional),
                                OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }
        EXPECT_EQ(ret, static_cast<aclnnStatus>(expectRet));
    }

    string socVersion;
    string caseName;
    string xShape;
    string xDtype;
    string xFormat;
    string weightShape;
    string weightDtype;
    string weightFormat;
    string scaleShape;
    string scaleDtype;
    string scaleFormat;
    string biasShape;
    string biasDtype;
    string biasFormat;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string perTokenScaleFormat;
    string groupListShape;
    string groupListValues;
    string groupListDtype;
    string groupListFormat;
    string outShape;
    string outDtype;
    string outFormat;
    int64_t splitItem;
    int64_t groupType;
    int64_t groupListType;
    int64_t actType;
    uint64_t expectRet;
    string hasBias;
    string hasPerTokenScale;
};

vector<GroupedMatmulOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulOpApiCase> cases;
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
        if (cols.size() != 32) {
            continue;
        }
        GroupedMatmulOpApiCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.xShape = Trim(cols[i++]);
        c.xDtype = Trim(cols[i++]);
        c.xFormat = Trim(cols[i++]);
        c.weightShape = Trim(cols[i++]);
        c.weightDtype = Trim(cols[i++]);
        c.weightFormat = Trim(cols[i++]);
        c.scaleShape = Trim(cols[i++]);
        c.scaleDtype = Trim(cols[i++]);
        c.scaleFormat = Trim(cols[i++]);
        c.biasShape = Trim(cols[i++]);
        c.biasDtype = Trim(cols[i++]);
        c.biasFormat = Trim(cols[i++]);
        c.perTokenScaleShape = Trim(cols[i++]);
        c.perTokenScaleDtype = Trim(cols[i++]);
        c.perTokenScaleFormat = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.groupListDtype = Trim(cols[i++]);
        c.groupListFormat = Trim(cols[i++]);
        c.outShape = Trim(cols[i++]);
        c.outDtype = Trim(cols[i++]);
        c.outFormat = Trim(cols[i++]);
        c.splitItem = stoll(Trim(cols[i++]));
        c.groupType = stoll(Trim(cols[i++]));
        c.groupListType = stoll(Trim(cols[i++]));
        c.actType = stoll(Trim(cols[i++]));
        c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
        c.hasBias = Trim(cols[i++]);
        c.hasPerTokenScale = Trim(cols[i++]);
        cases.emplace_back(c);
    }
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<GroupedMatmulOpApiCase> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class grouped_matmul_opapi_csv_test : public testing::TestWithParam<GroupedMatmulOpApiCase> {};

TEST_P(grouped_matmul_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_opapi_csv,
    grouped_matmul_opapi_csv_test,
    testing::ValuesIn(LoadCases(GetExeDirPath() + "test_aclnn_grouped_matmul.csv")),
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

/*!
 * \file test_aclnn_grouped_matmul.cpp
 * \brief CSV-driven opapi unit tests for grouped_matmul.
 */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_grouped_matmul_v5.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
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

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, ":", tokens);
    vector<int64_t> dims;
    dims.reserve(tokens.size());
    for (const auto &token : tokens) {
        dims.emplace_back(stoll(Trim(token)));
    }
    return dims;
}

vector<int64_t> ParseI64List(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, "|", tokens);
    vector<int64_t> values;
    values.reserve(tokens.size());
    for (const auto &token : tokens) {
        values.emplace_back(stoll(Trim(token)));
    }
    return values;
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},   {"FLOAT16", ACL_FLOAT16}, {"BF16", ACL_BF16},   {"INT8", ACL_INT8},
        {"INT32", ACL_INT32},   {"INT64", ACL_INT64},     {"UINT64", ACL_UINT64},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> formatMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
    };
    auto it = formatMap.find(format);
    return it == formatMap.end() ? ACL_FORMAT_ND : it->second;
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

struct GroupedMatmulOpApiCase {
    void Run() const
    {
        SetupPlatformForCase(socVersion);
        TensorListDesc x(1, TensorDesc(ParseDims(xShape), ParseDtype(xDtype), ParseFormat(xFormat)).ValueRange(-10, 10));
        TensorDesc weightTensor(ParseDims(weightShape), ParseDtype(weightDtype), ParseFormat(weightFormat));
        if (!ParseDims(weightStorageShape).empty()) {
            weightTensor = TensorDesc(ParseDims(weightShape), ParseDtype(weightDtype), ParseFormat(weightFormat), {}, 0,
                                      ParseDims(weightStorageShape));
        }
        TensorListDesc weight(1, weightTensor.ValueRange(-10, 10));
        TensorListDesc out(1, TensorDesc(ParseDims(outShape), ParseDtype(outDtype), ParseFormat(outFormat)));
        TensorDesc groupList(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat));
        vector<int64_t> groupListVal = ParseI64List(groupListValues);
        if (!groupListVal.empty()) {
            groupList.Value(groupListVal);
        } else {
            groupList.ValueRange(0, 64);
        }

        TensorListDesc scale(1, TensorDesc(ParseDims(scaleShape), ParseDtype(scaleDtype), ParseFormat(scaleFormat))
                                    .ValueRange(-10, 10));
        TensorListDesc bias(1, TensorDesc(ParseDims(biasShape), ParseDtype(biasDtype), ParseFormat(biasFormat))
                                   .ValueRange(-10, 10));
        TensorListDesc perTokenScale(
            1, TensorDesc(ParseDims(perTokenScaleShape), ParseDtype(perTokenScaleDtype), ParseFormat(perTokenScaleFormat))
                   .ValueRange(-10, 10));

        auto offsetOptional = nullptr;
        auto antiquantScaleOptional = nullptr;
        auto antiquantOffsetOptional = nullptr;
        auto activationInputOptional = nullptr;
        auto activationQuantScaleOptional = nullptr;
        auto activationQuantOffsetOptional = nullptr;
        auto tuningConfigOptional = nullptr;
        auto activationFeatureOutOptional = nullptr;
        auto dynQuantScaleOutOptional = nullptr;

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ACL_SUCCESS;
        if (ParseBool(hasBias) && ParseBool(hasPerTokenScale)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                INPUT(x, weight, bias, scale, offsetOptional, antiquantScaleOptional,
                                      antiquantOffsetOptional, perTokenScale, groupList, activationInputOptional,
                                      activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                      groupListType, actType, tuningConfigOptional),
                                OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
            getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasBias)) {
            auto perTokenScaleOptional = nullptr;
            auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                INPUT(x, weight, bias, scale, offsetOptional, antiquantScaleOptional,
                                      antiquantOffsetOptional, perTokenScaleOptional, groupList, activationInputOptional,
                                      activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                      groupListType, actType, tuningConfigOptional),
                                OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
            getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasPerTokenScale)) {
            auto biasOptional = nullptr;
            auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                INPUT(x, weight, biasOptional, scale, offsetOptional, antiquantScaleOptional,
                                      antiquantOffsetOptional, perTokenScale, groupList, activationInputOptional,
                                      activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                      groupListType, actType, tuningConfigOptional),
                                OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
            getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto biasOptional = nullptr;
            auto perTokenScaleOptional = nullptr;
            auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                INPUT(x, weight, biasOptional, scale, offsetOptional, antiquantScaleOptional,
                                      antiquantOffsetOptional, perTokenScaleOptional, groupList, activationInputOptional,
                                      activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                      groupListType, actType, tuningConfigOptional),
                                OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
            getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        }
        EXPECT_EQ(getWorkspaceResult, static_cast<aclnnStatus>(expectRet));
    }

    string socVersion;
    string caseName;
    string xShape;
    string xDtype;
    string xFormat;
    string weightShape;
    string weightStorageShape;
    string weightDtype;
    string weightFormat;
    string scaleShape;
    string scaleDtype;
    string scaleFormat;
    string biasShape;
    string biasDtype;
    string biasFormat;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string perTokenScaleFormat;
    string groupListShape;
    string groupListValues;
    string groupListDtype;
    string groupListFormat;
    string outShape;
    string outDtype;
    string outFormat;
    int64_t splitItem;
    int64_t groupType;
    int64_t groupListType;
    int64_t actType;
    uint64_t expectRet;
    string hasBias;
    string hasPerTokenScale;
};

vector<GroupedMatmulOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulOpApiCase> cases;
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
        if (cols.size() != 33) {
            continue;
        }
        GroupedMatmulOpApiCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.xShape = Trim(cols[i++]);
        c.xDtype = Trim(cols[i++]);
        c.xFormat = Trim(cols[i++]);
        c.weightShape = Trim(cols[i++]);
        c.weightStorageShape = Trim(cols[i++]);
        c.weightDtype = Trim(cols[i++]);
        c.weightFormat = Trim(cols[i++]);
        c.scaleShape = Trim(cols[i++]);
        c.scaleDtype = Trim(cols[i++]);
        c.scaleFormat = Trim(cols[i++]);
        c.biasShape = Trim(cols[i++]);
        c.biasDtype = Trim(cols[i++]);
        c.biasFormat = Trim(cols[i++]);
        c.perTokenScaleShape = Trim(cols[i++]);
        c.perTokenScaleDtype = Trim(cols[i++]);
        c.perTokenScaleFormat = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.groupListDtype = Trim(cols[i++]);
        c.groupListFormat = Trim(cols[i++]);
        c.outShape = Trim(cols[i++]);
        c.outDtype = Trim(cols[i++]);
        c.outFormat = Trim(cols[i++]);
        c.splitItem = stoll(Trim(cols[i++]));
        c.groupType = stoll(Trim(cols[i++]));
        c.groupListType = stoll(Trim(cols[i++]));
        c.actType = stoll(Trim(cols[i++]));
        c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
        c.hasBias = Trim(cols[i++]);
        c.hasPerTokenScale = Trim(cols[i++]);
        cases.emplace_back(c);
    }
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<GroupedMatmulOpApiCase> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class grouped_matmul_opapi_csv_test : public testing::TestWithParam<GroupedMatmulOpApiCase> {};

TEST_P(grouped_matmul_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_opapi_csv,
    grouped_matmul_opapi_csv_test,
    testing::ValuesIn(LoadCases(GetExeDirPath() + "test_aclnn_grouped_matmul.csv")),
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
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_v3.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_v4.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_v5.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_weight_nz.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_grouped_matmul_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_grouped_matmul_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_grouped_matmul_test TearDown" << endl;
    }
};

TEST_F(l2_grouped_matmul_test, Ascend910B2_grouped_matmul_fp16)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = nullptr;
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;
    //  auto sortedIndices = TensorListDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    //  auto routingMapOptional = TensorListDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType, tuningConfigOptional),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    /*此处校验161002是因为当前框架通过桩函数调用ut，导致无法正常infershape，正常现象*/
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8ofp16_weightNz_static)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, K, N}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(
        aclnnGroupedMatmulWeightNz,
        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional,
              perTokenScaleOptional, groupListOptional, activationInputOptional, activationQuantScaleOptional,
              activationQuantOffsetOptional, splitItem, groupType, groupListType, actType, tuningConfigOptional, 0),
        OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8ofp16_weightNz_pertoken)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(
        aclnnGroupedMatmulWeightNz,
        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional,
              perTokenScaleOptional, groupListOptional, activationInputOptional, activationQuantScaleOptional,
              activationQuantOffsetOptional, splitItem, groupType, groupListType, actType, tuningConfigOptional, 0),
        OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8ofp16_nd_staticTC)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    // auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 1;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8ofp16_nz_staticTC)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    // auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 2;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8ofp16_nd_dynamicKC)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8ofp16_nz_dynamicKC)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o_nz_dynamicKC_scale_bf16_y_bf16)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_BF16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8obf16_nz_dynamicKC_unsupport_scale_dtype)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_BF16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8obf16_nz_dynamicKC_unsupport_scale_shape)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, M}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_BF16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8obf16_nz_dynamicKC_unsupport_scale_dims)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E,}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_BF16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8obf16_nz_dynamicKC_unsupport_pertokenscale_dims)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_BF16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8obf16_nz_dynamicKC_unsupport_pertokenscale_shape)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({E,}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_BF16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
    

}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8obf16_nz_dynamicKC_unsupport_pertokenscale_dtype)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M,}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_BF16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_m0_empty_tensor)
{
    size_t M = 0;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_n0_empty_tensor)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 0;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_k0_empty_tensor)
{
    size_t M = 345;
    size_t K = 0;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_V5_a8w8o_nz_dynamicKC_scale_bf16_y_bf16)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_BF16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType, tuningConfigOptional),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_V5_a8w8ofp16_nz_dynamicKC)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 4;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                              antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                              activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                              splitItem, groupType, groupListType, actType, tuningConfigOptional),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_V5_k0_empty_tensor)
{
    size_t M = 345;
    size_t K = 0;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_FLOAT16, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType, tuningConfigOptional),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_v3_no_bias_case)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1, TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1, TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1, TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptionsl = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M, N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV3,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptionsl, antiquantScaleOptional,
                              antiquantOffsetOptional, groupListOptional, splitItem, groupType),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_v3_has_bias_case)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1, TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1, TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = TensorListDesc(1, TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto scaleOptional = TensorListDesc(1, TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptionsl = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M, N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV3,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptionsl, antiquantScaleOptional,
                              antiquantOffsetOptional, groupListOptional, splitItem, groupType),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o8_nd)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o32_nd)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT32, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o32_weightNz)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, K, N}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_INT32, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(
        aclnnGroupedMatmulWeightNz,
        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional,
            perTokenScaleOptional, groupListOptional, activationInputOptional, activationQuantScaleOptional,
            activationQuantOffsetOptional, splitItem, groupType, groupListType, actType, tuningConfigOptional, 0),
        OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o8_nd_v5)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType, tuningConfigOptional),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o32_nd_v5)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT32, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType, tuningConfigOptional),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o8_pertoken_not_null_error)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o32_pertoken_not_null_error)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = TensorListDesc(1,TensorDesc({M}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT32, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o8_bias_not_int32_error)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o32_bias_not_int32_error)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT32, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o8_scale_not_int64_error)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o8_scale_not_perchannel_error)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, N, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto biasOptional = TensorListDesc(1,TensorDesc({E, N}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1, TensorDesc({M,N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                            antiquantOffsetOptional, perTokenScaleOptional, groupListOptional,
                            activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                            splitItem, groupType, groupListType, actType),
                        OUTPUT(out,activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}

TEST_F(l2_grouped_matmul_test, Ascend950_grouped_matmul_a8w8o8_weightNz_error)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorListDesc(1,TensorDesc({M, K}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10));
    auto weight = TensorListDesc(1,TensorDesc({E, K, N}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10));
    auto biasOptional = nullptr;
    auto scaleOptional = TensorListDesc(1,TensorDesc({E, 1}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 10));
    auto offsetOptional = nullptr;
    auto antiquantScaleOptional = nullptr;
    auto antiquantOffsetOptional = nullptr;
    auto perTokenScaleOptional = nullptr;
    auto groupListOptional = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto activationInputOptional = nullptr;
    auto activationQuantScaleOptional = nullptr;
    auto activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    auto tuningConfigOptional = nullptr; //
    auto activationFeatureOutOptional = nullptr;
    auto dynQuantScaleOutOptional = nullptr;

    auto out = TensorListDesc(1,TensorDesc({M,N}, ACL_INT8, ACL_FORMAT_ND));
    int64_t split_item = 3;
    int64_t dtype = 0;
    bool paddedNum = true;
    auto ut = OP_API_UT(
        aclnnGroupedMatmulWeightNz,
        INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional,
            perTokenScaleOptional, groupListOptional, activationInputOptional, activationQuantScaleOptional,
            activationQuantOffsetOptional, splitItem, groupType, groupListType, actType, tuningConfigOptional, 0),
        OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 161002);
}