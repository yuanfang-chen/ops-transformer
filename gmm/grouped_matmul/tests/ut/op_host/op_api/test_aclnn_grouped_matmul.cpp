/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
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

string GetCurrentFileDir()
{
    string filePath = __FILE__;
    auto pos = filePath.find_last_of("\\/");
    return pos == string::npos ? string(".") : filePath.substr(0, pos);
}

string GetCwd()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetCurrentDirectoryA(MAX_PATH, path);
    return len == 0 ? string(".") : string(path, len);
#else
    char path[4096] = {0};
    if (getcwd(path, sizeof(path)) == nullptr) {
        return ".";
    }
    return string(path);
#endif
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
    const string cwd = GetCwd();
    const string codePath = []() {
        const char *env = std::getenv("CODE_PATH");
        return env == nullptr ? string() : string(env);
    }();

    vector<string> candidates;
    candidates.emplace_back(JoinPath(GetExeDirPath(), csvName));
    candidates.emplace_back(JoinPath(GetCurrentFileDir(), csvName));
    candidates.emplace_back(JoinPath(cwd, csvName));
    candidates.emplace_back(JoinPath(cwd, "../../../../../gmm/grouped_matmul/tests/ut/op_host/op_api/" + csvName));
    if (!codePath.empty()) {
        candidates.emplace_back(
            JoinPath(codePath, "gmm/grouped_matmul/tests/ut/op_host/op_api/" + csvName));
    }

    for (const auto &path : candidates) {
        if (FileExists(path)) {
            return path;
        }
    }
    return candidates.front();
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
        {"FLOAT", ACL_FLOAT}, {"FLOAT16", ACL_FLOAT16}, {"BF16", ACL_BF16}, {"INT8", ACL_INT8},
        {"INT32", ACL_INT32}, {"INT64", ACL_INT64},     {"UINT64", ACL_UINT64},
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
        if (cols.size() != 31) {
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
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
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
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_grouped_matmul.csv"))),
    BuildCaseName);

}  // namespace
