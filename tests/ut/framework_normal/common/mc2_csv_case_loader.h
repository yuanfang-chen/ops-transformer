/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_CSV_CASE_LOADER_H
#define MC2_CSV_CASE_LOADER_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

using csv_map = std::unordered_map<std::string, std::string>;

template<typename T>
std::vector<T> GetParamsFromCsv(const std::string& csvPath)
{
    std::vector<T> params;
    std::ifstream csvFile(csvPath, std::ios::in);
    if (!csvFile.is_open()) {
        std::cout << "[ERROR] Cannot open case file \"" << csvPath << "\"!" << std::endl;
        return params;
    }

    std::vector<std::string> keys;
    std::string line;
    std::getline(csvFile, line);
    std::istringstream stream(line);
    for (std::string data; std::getline(stream, data, ','); ) {
        keys.emplace_back(data);
    }

    for (int lineNum = 1; std::getline(csvFile, line); ++lineNum) {
        std::vector<std::string> values;
        std::istringstream stream(line);
        for (std::string data; std::getline(stream, data, ','); ) {
            values.emplace_back(data);
        }
        if (keys.size() != values.size()) {
            std::cout << "[ERROR] CSV Line " << lineNum << ": Row data does not match CSV header columns. Row has "
                      << values.size() << " columns, but header has " << keys.size() << " columns." << std::endl;
            continue;
        }
        csv_map csvMap;
        for (size_t i = 0; i < keys.size(); ++i) {
            csvMap[keys[i]] = values[i];
        }
        params.emplace_back(csvMap);
    }
    return params;
}

inline std::string ReadCsvMap(const csv_map& csvMap, const std::string& key)
{
    return csvMap.counts(key) > 0 ? csvMap.at(key) : "";
}

#endif // MC2_CSV_CASE_LOADER_H
