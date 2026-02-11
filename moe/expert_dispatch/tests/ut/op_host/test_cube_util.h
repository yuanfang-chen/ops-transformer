/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_cube_util.h
 */
#include <iostream>
#include <map>
using namespace std;

void GetPlatFormInfos(const char *compile_info_str, map<string, string> &soc_infos, map<string, string> &aicore_spec,
                      map<string, string> &intrinsics);

std::string GetExeDirPath();
