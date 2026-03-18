message(STATUS "BUILD_TORCH_OPS ON in layernorm_stride")
# ROTARY_STRIDE operation sources
file(GLOB LAYERNORM_STRIDE_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(LAYERNORM_STRIDE_SOURCES ${LAYERNORM_STRIDE_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${LAYERNORM_STRIDE_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(layernorm_stride_objects OBJECT ${LAYERNORM_STRIDE_SOURCES})

target_compile_options(layernorm_stride_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(layernorm_stride_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()