#include <string>
#include <functional>
#include <iomanip>
#include <sstream>
#include <cstring>

std::string hashStruct(std::string tilingDataStr) {
    // 使用标准库哈希
    std::hash<std::string> hasher;

    // 获取哈希值并转换为字符串
    auto hashValue = hasher(tilingDataStr);

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hashValue;
    return ss.str();
}

void CompareTilingData(const gert::TilingContextPara& tilingContextPara, const std::string expectTilingDataHashVal)
{
    if (expectResult == ge::GRAPH_FAILED) {
        return;
    }

    auto rawTilingData = tilingContext->GetRawTilingData();
    auto tilingDataReservedSize = tilingDataReservedLen * sizeof(uint64_t);
    auto tilingDataResult = to_string<int64_t>(rawTilingData->GetData() + tilingDataReservedSize,
                                                rawTilingData->GetDataSize() - tilingDataReservedSize);

    ASSERT_EQ(hashStruct(tilingDataResult), expectTilingDataHashVal);

}

static void TestOneParamCase(const MatmulAlltoAllTestParam &param)
{

    CompareTilingData
    expectTilingDataHashVal
}