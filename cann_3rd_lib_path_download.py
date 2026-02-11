#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import urllib.request
import os

def down_files_native(url_list):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    for url in url_list:
        try:
            file_name = url.split('/')[-1]

            if not file_name:
                file_name = "downloaded_file"

            file_path = os.path.join(current_dir,file_name)
            print(f"Start download {url}")

            urllib.request.urlretrieve(url, file_path)
            print(f"Successfully saved to {file_path}")
        except Exception as e:
            print(f"Download file from {url} failed: {e}")
    
if __name__ == "__main__":
    my_urls = [
        "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz",
        "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip",
        "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz",
        "https://gitcode.com/cann-src-third-party/pybind11/releases/download/v2.13.6/pybind11-2.13.6.tar.gz",
        "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0/eigen-5.0.0.tar.gz",
        "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz"
    ]

    down_files_native(my_urls)