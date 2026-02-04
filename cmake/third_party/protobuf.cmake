# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(_protobuf_url "")
if(CANN_PKG_SERVER)
  set(_protobuf_url "${CANN_PKG_SERVER}/libs/protobuf/v25.1.tar.gz")
endif()

# 检查本地是否存在protobuf包
set(PROTOBUF_VERSION_PKG protobuf-25.1.tar.gz)
if(EXISTS "${CANN_3RD_LIB_PATH}/pkg/${PROTOBUF_VERSION_PKG}")
    message(STATUS "Found protobuf archive in ${CANN_3RD_LIB_PATH}/pkg")
    set(_protobuf_url "file://${CANN_3RD_LIB_PATH}/pkg/${PROTOBUF_VERSION_PKG}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/${PROTOBUF_VERSION_PKG}")
    message(STATUS "Found protobuf archive in ${CANN_3RD_LIB_PATH}, moving to pkg")
    file(MAKE_DIRECTORY ${CANN_3RD_LIB_PATH}/pkg)
    file(RENAME "${CANN_3RD_LIB_PATH}/${PROTOBUF_VERSION_PKG}" "${CANN_3RD_LIB_PATH}/pkg/${PROTOBUF_VERSION_PKG}")
    set(_protobuf_url "file://${CANN_3RD_LIB_PATH}/pkg/${PROTOBUF_VERSION_PKG}")
endif()
    set(_protobuf_url "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz")
    message(STATUS "[ThirdPartyLib][protobuf] ${_protobuf_url} not found, need download.")

include(ExternalProject)
ExternalProject_Add(external_protobuf
  URL               ${_protobuf_url}
  DOWNLOAD_DIR      download/protobuf
  PREFIX            third_party
  SOURCE_SUBDIR     cmake
  CMAKE_CACHE_ARGS
      -DProtobuf_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -Dprotobuf_BUILD_TESTS:BOOL=OFF
      -Dprotobuf_BUILD_EXAMPLES:BOOL=OFF
      -Dprotobuf_BUILD_SHARED_LIBS:BOOL=OFF
      -DProtobuf_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(external_protobuf SOURCE_DIR)
ExternalProject_Get_Property(external_protobuf BINARY_DIR)

set(Protobuf_INCLUDE ${SOURCE_DIR}/src)
set(Protobuf_PATH ${BINARY_DIR})
set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_PATH}/protoc)

add_custom_command(
  OUTPUT ${Protobuf_PROTOC_EXECUTABLE}
  DEPENDS external_protobuf
)
add_custom_target(
  protoc ALL DEPENDS ${Protobuf_PROTOC_EXECUTABLE}
)
