add_library(attention_worker_combine SHARED
    ${CANN_ROOT}/ops/built-in/tbe/impl/ascendc/attention_worker_combine/attention_worker_combine.cpp
    ./test_attention_worker_combine.cpp
)

target_compile_options(attention_worker_combine PRIVATE -g -include ${CMAKE_CURRENT_SOURCE_DIR}/test_attention_worker_combine.h)

target_link_libraries(attention_worker_combine PRIVATE
                    $<BUILD_INTERFACE:intf_llt_pub_asan>
                    -Wl,--whole-archive
                    -Wl,--no-as-needed
                    ${PRIVATE_L}
                    -Wl,--as-needed
                    -Wl,--no-whole-archive
)
