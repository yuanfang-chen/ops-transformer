# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# FlashAttention op_host CMakeLists.txt

set(FLASH_ATTENTION_HOST_SRCS
    flash_attention_def.cpp
    flash_attention_infershape.cpp
    flash_attention_tiling.cpp
)

add_library(flash_attention_host STATIC ${FLASH_ATTENTION_HOST_SRCS})

target_include_directories(flash_attention_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_INCLUDEPATH}
    ${OPDEV_INCLUDEPATH}
)

target_link_libraries(flash_attention_host
    ${ASCEND_LIBRARIES}
)
