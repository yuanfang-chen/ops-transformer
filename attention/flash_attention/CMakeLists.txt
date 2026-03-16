# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# FlashAttention unified operator CMakeLists.txt

cmake_minimum_required(VERSION 3.16)
project(flash_attention)

# op_api 层（aclnn两段式接口）
set(FLASH_ATTENTION_API_SRCS
    op_api/aclnn_flash_attention.cpp
    op_api/aclnn_flash_attention_inner.cpp
)

add_library(flash_attention_api SHARED ${FLASH_ATTENTION_API_SRCS})

target_include_directories(flash_attention_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api
    ${ASCEND_INCLUDEPATH}
    ${OPDEV_INCLUDEPATH}
    ${ACLNN_KERNELS_INCLUDEPATH}
)

target_link_libraries(flash_attention_api
    flash_attention_host
    ${ASCEND_LIBRARIES}
)

# op_host 子目录
add_subdirectory(op_host)

# op_kernel 由CANN编译框架单独处理，此处仅列出源文件供参考
# add_subdirectory(op_kernel)
