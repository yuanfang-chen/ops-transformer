[2026-03-18 01:56:30] [2026-03-18 01:56:30] Info: cmake config -DBUILD_OPEN_PROJECT=ON -DASCEND_OP_NAME=moe_token_permute_with_routing_map -DENABLE_TILING_SINK=OFF -DENABLE_UT_EXEC=TRUE -DCANN_3RD_LIB_PATH=/data/kdy/cann/ops-transformer/third_party -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/data/kdy/cann/pkg/cann-9.0.0 -DCHECK_COMPATIBLE=true -DASCEND_COMPUTE_UNIT=ascend910b; -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON -DENABLE_BUILD_PKG=ON .
[2026-03-18 01:56:30] -- The C compiler identification is GNU 11.4.0
[2026-03-18 01:56:30] -- The CXX compiler identification is GNU 11.4.0
[2026-03-18 01:56:30] -- Detecting C compiler ABI info
[2026-03-18 01:56:31] -- Detecting C compiler ABI info - done
[2026-03-18 01:56:31] -- Check for working C compiler: /usr/bin/cc - skipped
[2026-03-18 01:56:31] -- Detecting C compile features
[2026-03-18 01:56:31] -- Detecting C compile features - done
[2026-03-18 01:56:31] -- Detecting CXX compiler ABI info
[2026-03-18 01:56:31] -- Detecting CXX compiler ABI info - done
[2026-03-18 01:56:31] -- Check for working CXX compiler: /usr/bin/c++ - skipped
[2026-03-18 01:56:31] -- Detecting CXX compile features
[2026-03-18 01:56:31] -- Detecting CXX compile features - done
[2026-03-18 01:56:31] -- ASCEND_CANN_PACKAGE_PATH=/data/kdy/cann/pkg/cann-9.0.0
[2026-03-18 01:56:31] -- compile project with library
[2026-03-18 01:56:31] -- Find opbase source dir: /data/kdy/cann/ops-transformer/third_party/opbase
[2026-03-18 01:56:31] -- Found Python3: /usr/bin/python3.10 (found version "3.10.12") found components: Interpreter 
[2026-03-18 01:56:31] -- ENABLE_CCACHE=ON, CUSTOM_CCACHE=
[2026-03-18 01:56:31] -- CCACHE_PROGRAM=CCACHE_PROGRAM-NOTFOUND
[2026-03-18 01:56:31] -- ASCEND_COMPUTE_UNIT=ascend910b;
[2026-03-18 01:56:31] -- ASCEND_OP_NAME=moe_token_permute_with_routing_map
[2026-03-18 01:56:31] -- TILING_KEY=
[2026-03-18 01:56:31] -- TESTS_UT_OPS_TEST=
[2026-03-18 01:56:31] -- TESTS_EXAMPLE_OPS_TEST=
[2026-03-18 01:56:31] -- CMAKE_BUILD_TYPE=Release
[2026-03-18 01:56:31] -- VERSION=
[2026-03-18 01:56:35] -- Found json: /data/kdy/cann/ops-transformer/third_party/json/include  
[2026-03-18 01:56:35] -- Start building custom package.
[2026-03-18 01:56:35] -- Search libs under install path /data/kdy/cann/pkg/cann-9.0.0
[2026-03-18 01:56:35] -- CMAKE_MODULE_PATH            :/data/kdy/cann/ops-transformer/cmake/modules;/data/kdy/cann/pkg/cann-9.0.0/include/ge/cmake
[2026-03-18 01:56:35] -- compile option:   -Wall -Wextra -Wshadow -Wformat=2 -fno-common -fPIC -Wformat-signedness -Wno-missing-include-dirs -Wno-write-strings
[2026-03-18 01:56:35] -- Found dlog: /data/kdy/cann/pkg/cann-9.0.0/pkg_inc/base  
[2026-03-18 01:56:35] -- Transformer Variables in dlog module:
[2026-03-18 01:56:35] -- dlog_TRANSFORMER_INCLUDE_DIR="/data/kdy/cann/pkg/cann-9.0.0/pkg_inc/base"
[2026-03-18 01:56:35] -- dlog_SHARED_LIBRARY="/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:35] -- 
[2026-03-18 01:56:35]  Properties for TARGET dlog:
[2026-03-18 01:56:35]    dlog.INTERFACE_COMPILE_DEFINITIONS = "LOG_CPP;PROCESS_LOG"
[2026-03-18 01:56:35]    dlog.INTERFACE_LINK_LIBRARIES = "dlog_headers"
[2026-03-18 01:56:35]    dlog.IMPORTED_LOCATION = "/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:35] 
[2026-03-18 01:56:35] -- 
[2026-03-18 01:56:35]  Properties for TARGET dlog_headers:
[2026-03-18 01:56:35]    dlog_headers.INTERFACE_INCLUDE_DIRECTORIES = "/data/kdy/cann/pkg/cann-9.0.0/pkg_inc/base"
[2026-03-18 01:56:35] 
[2026-03-18 01:56:35] -- Found securec: /data/kdy/cann/pkg/cann-9.0.0/include  
[2026-03-18 01:56:35] -- Found OPBASE: /data/kdy/cann/ops-transformer/third_party/opbase/pkg_inc  
[2026-03-18 01:56:35] -- Found OPABSE include:/data/kdy/cann/ops-transformer/third_party/opbase/pkg_inc
[2026-03-18 01:56:35] -- Found OPABSE lib:
[2026-03-18 01:56:35] -- Found platform: /data/kdy/cann/pkg/cann-9.0.0/include  
[2026-03-18 01:56:35] -- Found platform include:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include
[2026-03-18 01:56:35] -- Found register library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libregister.so
[2026-03-18 01:56:35] -- Found opp_registry library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libopp_registry.so
[2026-03-18 01:56:35] -- Found exe_graph library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libexe_graph.so
[2026-03-18 01:56:35] -- Found graph_base library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libgraph_base.so
[2026-03-18 01:56:35] -- Found error_manager library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/liberror_manager.so
[2026-03-18 01:56:35] -- Found metadef library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libmetadef.so
[2026-03-18 01:56:35] -- Found rt2_registry library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/librt2_registry.a
[2026-03-18 01:56:35] -- Found metadef: /data/kdy/cann/pkg/cann-9.0.0/include  
[2026-03-18 01:56:35] -- Found source metadef include dir:  /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/exe_graph
[2026-03-18 01:56:35] -- Found graph library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libgraph.so
[2026-03-18 01:56:35] -- Found source acl include dir:  /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include
[2026-03-18 01:56:35] -- [transformer]Found source npuruntime include dir: /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/runtime
[2026-03-18 01:56:35] -- Found aclnn include dir:  /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include
[2026-03-18 01:56:35] -- Found opdev include dir:  /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aclnn
[2026-03-18 01:56:35] -- Found tilingapi include:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include
[2026-03-18 01:56:35] -- TILINGAPI_INC_DIRS: /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api/lib;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api/impl;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api/tiling
[2026-03-18 01:56:35] -- Found tilingapi lib:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libtiling_api.a
[2026-03-18 01:56:35] -- Using AICPU include dirs: /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment/cce;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment/msprof;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu_common/context;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu_common/context/common;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu_common/context/cpu_proto;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu_common/context/utils;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aicpu_common/context;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aicpu_common/context/common;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aicpu_common/context/cpu_proto;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aicpu_common/context/utils;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment/datagw/aicpu/common;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment/datagw/aicpu/common;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu
[2026-03-18 01:56:35] -- Finding gen es pkg
[2026-03-18 01:56:35] -- [add_es_library] Module loaded from: /data/kdy/cann/pkg/cann-9.0.0/include/ge/cmake
[2026-03-18 01:56:35] -- Eigen path found in cache: /data/kdy/cann/ops-transformer/third_party/eigen
[2026-03-18 01:56:35] -- Eigen already exists at /data/kdy/cann/ops-transformer/third_party/eigen
[2026-03-18 01:56:35] -- Could NOT find abseil-cpp (missing: ABSL_SOURCE_DIR) 
[2026-03-18 01:56:35] -- [ThirdPartyLib][abseil-cpp] found in /data/kdy/cann/ops-transformer/third_party/pkg/abseil-cpp-20230802.1.tar.gz.
[2026-03-18 01:56:35] -- Could NOT find ascend_protobuf_build_transformer (missing: ASCEND_PROTOBUF_SHARED_INCLUDE) 
[2026-03-18 01:56:35] -- [ThirdPartyLib][ascend protobuf] ascend protobuf shared not found, finding binary file.
[2026-03-18 01:56:35] -- [ThirdPartyLib][ascend protobuf] found in file:///data/kdy/cann/ops-transformer/third_party/pkg/protobuf-25.1.tar.gz.
[2026-03-18 01:56:36] -- Found alog: /data/kdy/cann/pkg/cann-9.0.0/include  
[2026-03-18 01:56:36] -- Variables in alog module:
[2026-03-18 01:56:36] -- alog_a_INCLUDE_DIR="/data/kdy/cann/pkg/cann-9.0.0/include"
[2026-03-18 01:56:36] -- slog_a_SHARED_LIBRARY="/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:36] -- alog_a_SHARED_LIBRARY="/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:36] -- 
[2026-03-18 01:56:36]  Properties for TARGET slog_a:
[2026-03-18 01:56:36]    slog_a.INTERFACE_COMPILE_DEFINITIONS = "LOG_CPP;PROCESS_LOG"
[2026-03-18 01:56:36]    slog_a.INTERFACE_LINK_LIBRARIES = "alog_headers"
[2026-03-18 01:56:36]    slog_a.IMPORTED_LOCATION = "/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:36] 
[2026-03-18 01:56:36] -- 
[2026-03-18 01:56:36]  Properties for TARGET alog_a:
[2026-03-18 01:56:36]    alog_a.INTERFACE_COMPILE_DEFINITIONS = "LOG_CPP;PROCESS_LOG"
[2026-03-18 01:56:36]    alog_a.INTERFACE_LINK_LIBRARIES = "alog_headers"
[2026-03-18 01:56:36]    alog_a.IMPORTED_LOCATION = "/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:36] 
[2026-03-18 01:56:36] -- 
[2026-03-18 01:56:36]  Properties for TARGET alog_headers:
[2026-03-18 01:56:36]    alog_headers.INTERFACE_INCLUDE_DIRECTORIES = "/data/kdy/cann/pkg/cann-9.0.0/include;/data/kdy/cann/pkg/cann-9.0.0/include/base"
[2026-03-18 01:56:36] 
[2026-03-18 01:56:36] -- Could NOT find unified_dlog (missing: _INCLUDE_ROOT_DIR) 
[2026-03-18 01:56:36] -- Skip proto update: /data/kdy/cann/ops-transformer/moe/moe_token_permute_with_routing_map
[2026-03-18 01:56:36] -- Start Generating es transformer for custom pkg
[2026-03-18 01:56:36] -- Generated helper script: /data/kdy/cann/ops-transformer/build/cmake/run_gen_esb_with_lock.sh
[2026-03-18 01:56:36] -- _add_es_library_impl: EXCLUDE_OPS is not provided
[2026-03-18 01:56:36] -- add_es_package: Using default output directory: /data/kdy/cann/ops-transformer/build
[2026-03-18 01:56:36] -- add_es_package: OPP_PROTO_TARGET output directory: /data/kdy/cann/ops-transformer/build
[2026-03-18 01:56:36] -- add_es_package: Derived ASCEND_OPP_PATH: /data/kdy/cann/ops-transformer/build
[2026-03-18 01:56:36] --   - Created standard OPP path: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust
[2026-03-18 01:56:36] -- _add_es_library_impl: gen_esb target not found, searching in environment...
[2026-03-18 01:56:36] --     Found gen_esb in PATH (environment variables configured)
[2026-03-18 01:56:36] --     - gen_esb path: /data/kdy/cann/pkg/cann-9.0.0/bin/gen_esb
[2026-03-18 01:56:36] --     - library path: /data/kdy/cann/pkg/cann-9.0.0/lib64
[2026-03-18 01:56:36] -- Using proto library path (derived from OPP_PROTO_TARGET): /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust
[2026-03-18 01:56:36] -- add_es_package: Module name for gen_esb: transformer_cust
[2026-03-18 01:56:36] -- Trying to locate eager_style_graph_builder_base library from run package...
[2026-03-18 01:56:36] --   Found library in run package: /data/kdy/cann/pkg/cann-9.0.0/lib64/libeager_style_graph_builder_base.so
[2026-03-18 01:56:36] -- Configuring ES package: es_transformer_cust
[2026-03-18 01:56:36] --   - Exported target: es_transformer_cust (public interface)
[2026-03-18 01:56:36] --   - Internal .so target: es_transformer_cust_so
[2026-03-18 01:56:36] --   - Module name (for gen_esb): transformer_cust
[2026-03-18 01:56:36] --   - OPP proto target: proto_transformer_cust
[2026-03-18 01:56:36] --   - OPP proto path: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust
[2026-03-18 01:56:36] --   - Output path: /data/kdy/cann/ops-transformer/build/es_packages
[2026-03-18 01:56:36] --   - Library file: libes_transformer_cust.so
[2026-03-18 01:56:36] --   - Python package: es_transformer_cust
[2026-03-18 01:56:36] -- Found 1 source file(s) for package 'es_transformer_cust'
[2026-03-18 01:56:36] -- add_es_library: Using default compile options: -fPIC;-Wall;-std=c++17;-D_GLIBCXX_USE_CXX11_ABI=0;-O2;-Wno-free-nonheap-object
[2026-03-18 01:56:36] -- add_es_library: Set _GLIBCXX_USE_CXX11_ABI=0 for ABI compatibility
[2026-03-18 01:56:36] -- add_es_library: Deduced header base path from cmake file: /data/kdy/cann/pkg/cann-9.0.0
[2026-03-18 01:56:36] -- add_es_library: metadef_headers target not found, skipping
[2026-03-18 01:56:36] -- add_es_library: Found c_sec target
[2026-03-18 01:56:36] -- add_es_library: Linking with CANN libraries: c_sec
[2026-03-18 01:56:36] -- add_es_package: 'build_es_transformer_cust' is the first ES package (will be built first)
[2026-03-18 01:56:36] -- ES package 'es_transformer_cust' configured successfully:
[2026-03-18 01:56:36] --   - Exported target (for linking): es_transformer_cust
[2026-03-18 01:56:36] --   - Smart build target: build_es_transformer_cust
[2026-03-18 01:56:36] --   - Internal targets:
[2026-03-18 01:56:36] --     * Code generation: generate_es_transformer_cust_code
[2026-03-18 01:56:36] --     * Shared library: es_transformer_cust_so
[2026-03-18 01:56:36] --     * Imported library (linking): es_transformer_cust_imported
[2026-03-18 01:56:36] --     * Wheel generation: generate_es_transformer_cust_whl
[2026-03-18 01:56:36] --     * Install: install_es_transformer_cust
[2026-03-18 01:56:36] --   - Library file: libes_transformer_cust.so
[2026-03-18 01:56:36] --   - Python package: es_transformer_cust
[2026-03-18 01:56:36] --   - ES base library: /data/kdy/cann/pkg/cann-9.0.0/lib64/libeager_style_graph_builder_base.so (from library)
[2026-03-18 01:56:36] -- 
[2026-03-18 01:56:36] -- Quick start:
[2026-03-18 01:56:36] --   1. Link:    target_link_libraries(your_target PRIVATE es_transformer_cust)
[2026-03-18 01:56:36] --   2. Build:   make your_target  (recommended, triggers automatically)
[2026-03-18 01:56:36] -- 
[2026-03-18 01:56:36] -- custom cust proto to es
[2026-03-18 01:56:36] -- Detected architecture: ARM64
[2026-03-18 01:56:36] -- End building custom package.
[2026-03-18 01:56:36] -- create aclnn master header file: /data/kdy/cann/ops-transformer/build/aclnn_ops_transformer_custom.h
[2026-03-18 01:56:36] -- no operator info to generate
[2026-03-18 01:56:36] -- Start building custom package.
[2026-03-18 01:56:36] -- No aicpu json files to merge, skipping.
[2026-03-18 01:56:36] -- No aicpu cust obj targets found, skipping.
[2026-03-18 01:56:36] -- Configuring done
[2026-03-18 01:56:36] -- Generating done
[2026-03-18 01:56:36] -- Build files have been written to: /data/kdy/cann/ops-transformer/build
[2026-03-18 01:56:36] [  1%] Generating op_host_aclnn_inner_stub.cpp
[2026-03-18 01:56:36] [  2%] Generating op_host_aclnn_stub.cpp
[2026-03-18 01:56:36] [  4%] Building CXX object common/stub/op_api/CMakeFiles/opapi_math.dir/opapi_stub.cpp.o
[2026-03-18 01:56:36] [  4%] Generating version.ops-transformer.info
[2026-03-18 01:56:36] [  5%] Building CXX object CMakeFiles/op_host_aclnnExc.dir/moe/moe_token_permute_with_routing_map/op_host/moe_token_permute_with_routing_map_def.cpp.o
[2026-03-18 01:56:36] [  6%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/ophost_transformer_infer_obj.dir/proto_stub.cpp.o
[2026-03-18 01:56:36] [  7%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/ophost_transformer_infer_obj.dir/__/__/3rd/moe_gather_v2/op_host/proto_stub.cpp.o
[2026-03-18 01:56:36] [  8%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/ophost_transformer_tiling_obj.dir/moe_token_permute_with_routing_map_tiling.cpp.o
[2026-03-18 01:56:36] [  8%] Built target update_proto_target
[2026-03-18 01:56:36] [ 10%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_util_objs.dir/__/__/__/third_party/opbase/src/op_common/op_host/util/shape_util.cpp.o
[2026-03-18 01:56:36] [ 10%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_util_objs.dir/__/__/__/third_party/opbase/src/op_common/op_host/util/fp16.cpp.o
[2026-03-18 01:56:36] [ 15%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_infer_objs.dir/__/__/__/third_party/opbase/src/op_common/op_host/infershape_reduce_util.cpp.o
[2026-03-18 01:56:36] [ 15%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_util_objs.dir/__/__/__/third_party/opbase/src/op_common/op_host/util/math_util.cpp.o
[2026-03-18 01:56:36] [ 15%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_infer_objs.dir/__/__/__/third_party/opbase/src/op_common/op_host/infershape_elewise_util.cpp.o
[2026-03-18 01:56:36] [ 15%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_infer_objs.dir/__/__/__/third_party/opbase/src/op_common/op_host/infershape_broadcast_util.cpp.o
[2026-03-18 01:56:36] [ 15%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_util_objs.dir/__/__/__/third_party/opbase/src/op_common/log/log.cpp.o
[2026-03-18 01:56:36] [ 16%] Building CXX object common/stub/op_tiling/CMakeFiles/optiling.dir/runtime_kb_api.cpp.o
[2026-03-18 01:56:36] [ 17%] Generating binary/ascend910b/src/ascendc/common
[2026-03-18 01:56:36] [ 21%] Building CXX object common/CMakeFiles/common_transformer_obj.dir/src/fallback_comm_2stages.cpp.o
[2026-03-18 01:56:36] [ 21%] Building CXX object common/CMakeFiles/common_transformer_obj.dir/src/fallback_comm.cpp.o
[2026-03-18 01:56:36] [ 21%] Generating binary/ascend910b/src/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap.py
[2026-03-18 01:56:36] [ 25%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_tiling_objs.dir/__/__/__/third_party/opbase/src/op_common/atvoss/elewise/elewise_tiling.cpp.o
[2026-03-18 01:56:36] [ 25%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_tiling_objs.dir/__/__/__/third_party/opbase/src/op_common/atvoss/broadcast/broadcast_tiling.cpp.o
[2026-03-18 01:56:36] [ 25%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_tiling_objs.dir/__/__/__/third_party/opbase/src/op_common/atvoss/broadcast/broadcast_tiling_noncontiguous.cpp.o
[2026-03-18 01:56:36] [ 25%] Building CXX object common/stub/op_tiling/CMakeFiles/optiling.dir/tbe_tiling_api.cpp.o
[2026-03-18 01:56:36] [ 28%] Generating scripts/install.sh, scripts/upgrade.sh
[2026-03-18 01:56:36] [ 28%] Generating version.info
[2026-03-18 01:56:36] [ 28%] Creating directories for 'abseil_build_transformer'
[2026-03-18 01:56:36] [ 30%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_tiling_objs.dir/__/__/__/third_party/opbase/src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp.o
[2026-03-18 01:56:36] [ 30%] Building CXX object common/CMakeFiles/common_transformer_obj.dir/src/tiling_base/tiling_util.cpp.o
[2026-03-18 01:56:36] [ 31%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/opbase_tiling_objs.dir/__/__/__/third_party/opbase/src/op_common/atvoss/reduce/reduce_tiling.cpp.o
[2026-03-18 01:56:36] [ 31%] Generating binary/ascend910b/bin/moe_token_permute_with_routing_map
[2026-03-18 01:56:36] [ 32%] Building CXX object CMakeFiles/op_host_aclnnInner.dir/op_host_aclnn_inner_stub.cpp.o
[2026-03-18 01:56:36] [ 33%] Building CXX object CMakeFiles/op_host_aclnn.dir/op_host_aclnn_stub.cpp.o
[2026-03-18 01:56:36] [ 33%] Built target version_ops-transformer_info
[2026-03-18 01:56:36] [ 33%] Built target moe_token_permute_with_routing_map_ascend910b_mkdir
[2026-03-18 01:56:36] [ 33%] Built target moe_token_permute_with_routing_map_ascend910b_py_copy
[2026-03-18 01:56:36] [ 33%] Built target gen_version_info
[2026-03-18 01:56:36] [ 33%] Built target ophost_transformer_infer_obj
[2026-03-18 01:56:36] [ 33%] Built target ops_utils_inc_kernel_ascend910b
[2026-03-18 01:56:36] [ 34%] Linking CXX shared library libop_host_aclnn.so
[2026-03-18 01:56:36] [ 35%] Linking CXX shared library libop_host_aclnnInner.so
[2026-03-18 01:56:36] [ 35%] Built target modify_vendor
[2026-03-18 01:56:36] [ 36%] Generating binary/ascend910b/src/moe_gather_v2/moe_gather_v2_ascend910b_src_copy.done
[2026-03-18 01:56:36] [ 36%] Built target moe_gather_v2_ascend910b_src_copy
[2026-03-18 01:56:36] [ 36%] Built target op_host_aclnnInner
[2026-03-18 01:56:36] [ 36%] Built target op_host_aclnn
[2026-03-18 01:56:36] [ 37%] Generating binary/ascend910b/src/moe_token_permute_with_routing_map/moe_token_permute_with_routing_map_ascend910b_src_copy.done
[2026-03-18 01:56:36] [ 38%] Performing download step (verify and extract) for 'abseil_build_transformer'
[2026-03-18 01:56:36] [ 38%] Built target opbuild_gen_default
[2026-03-18 01:56:36] [ 38%] Built target opbuild_gen_inner
[2026-03-18 01:56:36] [ 38%] Built target moe_token_permute_with_routing_map_ascend910b_src_copy
[2026-03-18 01:56:36] [ 39%] Generating ops_aclnn_src_stub.cpp
[2026-03-18 01:56:36] -- extracting...
[2026-03-18 01:56:36]      src='/data/kdy/cann/ops-transformer/third_party/pkg/abseil-cpp-20230802.1.tar.gz'
[2026-03-18 01:56:36]      dst='/data/kdy/cann/ops-transformer/third_party/abseil-cpp'
[2026-03-18 01:56:36] -- extracting... [tar xfz]
[2026-03-18 01:56:36] [ 40%] Building CXX object CMakeFiles/ops_aclnn.dir/ops_aclnn_src_stub.cpp.o
[2026-03-18 01:56:36] [ 41%] Linking CXX static library libops_aclnn.a
[2026-03-18 01:56:37] -- extracting... [analysis]
[2026-03-18 01:56:37] -- extracting... [rename]
[2026-03-18 01:56:37] [ 41%] Built target ops_aclnn
[2026-03-18 01:56:37] [ 44%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/ophost_transformer_opapi_obj.dir/__/__/3rd/moe_gather_v2/op_host/op_api/moe_gather_v2.cpp.o
[2026-03-18 01:56:37] [ 44%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/ophost_transformer_opapi_obj.dir/op_api/aclnn_moe_token_permute_with_routing_map.cpp.o
[2026-03-18 01:56:37] [ 44%] Building CXX object moe/moe_token_permute_with_routing_map/op_host/CMakeFiles/ophost_transformer_opapi_obj.dir/op_api/moe_token_permute_with_routing_map.cpp.o
[2026-03-18 01:56:37] -- extracting... [clean up]
[2026-03-18 01:56:37] -- extracting... done
[2026-03-18 01:56:37] [ 45%] No update step for 'abseil_build_transformer'
[2026-03-18 01:56:37] [ 46%] Performing patch step for 'abseil_build_transformer'
[2026-03-18 01:56:37] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:37] patching file CMakeLists.txt
[2026-03-18 01:56:37] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:37] patching file absl/base/options.h
[2026-03-18 01:56:37] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:37] patching file absl/hash/internal/hash.cc
[2026-03-18 01:56:37] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:37] patching file absl/hash/internal/hash.h
[2026-03-18 01:56:37] patch unexpectedly ends in middle of line
[2026-03-18 01:56:37] Hunk #1 succeeded at 1159 with fuzz 1.
[2026-03-18 01:56:37] [ 47%] No configure step for 'abseil_build_transformer'
[2026-03-18 01:56:37] [ 48%] No build step for 'abseil_build_transformer'
[2026-03-18 01:56:37] [ 49%] No install step for 'abseil_build_transformer'
[2026-03-18 01:56:37] [ 49%] Completed 'abseil_build_transformer'
[2026-03-18 01:56:37] [ 49%] Built target abseil_build_transformer
[2026-03-18 01:56:37] [ 49%] Built target opbase_util_objs
[2026-03-18 01:56:37] [ 50%] Creating directories for 'ascend_protobuf_build_transformer'
[2026-03-18 01:56:37] [ 51%] Performing download step (verify and extract) for 'ascend_protobuf_build_transformer'
[2026-03-18 01:56:37] -- extracting...
[2026-03-18 01:56:37]      src='/data/kdy/cann/ops-transformer/third_party/pkg/protobuf-25.1.tar.gz'
[2026-03-18 01:56:37]      dst='/data/kdy/cann/ops-transformer/third_party/ascend_protobuf'
[2026-03-18 01:56:37] -- extracting... [tar xfz]
[2026-03-18 01:56:37] -- extracting... [analysis]
[2026-03-18 01:56:37] -- extracting... [rename]
[2026-03-18 01:56:37] [ 51%] Built target opbase_infer_objs
[2026-03-18 01:56:38] [ 52%] Linking CXX shared library ../../../libopapi_math.so
[2026-03-18 01:56:38] -- extracting... [clean up]
[2026-03-18 01:56:38] -- extracting... done
[2026-03-18 01:56:38] [ 52%] Built target common_transformer_obj
[2026-03-18 01:56:38] [ 53%] No update step for 'ascend_protobuf_build_transformer'
[2026-03-18 01:56:38] [ 54%] Performing patch step for 'ascend_protobuf_build_transformer'
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file CMakeLists.txt
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/arenastring.cc
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/arenastring.h
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/compiler/cpp/field_generators/string_field.cc
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/compiler/cpp/helpers.cc
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/compiler/cpp/helpers.h
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/compiler/csharp/csharp_helpers.cc
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/compiler/java/names.cc
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/extension_set.h
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/message_lite.h
[2026-03-18 01:56:38] (Stripping trailing CRs from patch; use --binary to disable.)
[2026-03-18 01:56:38] patching file src/google/protobuf/port_def.inc
[2026-03-18 01:56:38] patch unexpectedly ends in middle of line
[2026-03-18 01:56:38] Hunk #1 succeeded at 572 with fuzz 1.
[2026-03-18 01:56:38] [ 54%] Built target opapi_math
[2026-03-18 01:56:38] [ 55%] Performing configure step for 'ascend_protobuf_build_transformer'
[2026-03-18 01:56:38] [ 56%] Linking CXX shared library libop_host_aclnnExc.so
[2026-03-18 01:56:38] [ 56%] Built target op_host_aclnnExc
[2026-03-18 01:56:38] [ 57%] Built target opbuild_gen_exc
[2026-03-18 01:56:38] [ 60%] Building CXX object CMakeFiles/proto_transformer_cust.dir/autogen/exc/moe_token_permute_with_routing_map_proto.cpp.o
[2026-03-18 01:56:38] [ 60%] Generating autogen/aic-ascend910b-ops-info.json
[2026-03-18 01:56:38] [ 60%] Filtering proto srcs from update files
[2026-03-18 01:56:38] [ 61%] Built target generate_proto_filtered_target
[2026-03-18 01:56:38] 
[2026-03-18 01:56:38] 
[2026-03-18 01:56:38] ==============check valid for ops info start==============
[2026-03-18 01:56:38] ==============check valid for ops info end================
[2026-03-18 01:56:38] 
[2026-03-18 01:56:38] 
[2026-03-18 01:56:38] Compile op info cfg successfully.
[2026-03-18 01:56:38] INFO: /data/kdy/cann/ops-transformer/build/autogen/aic-ascend910b-ops-info.ini does not exists in this project, skip generating compile commands.
[2026-03-18 01:56:38] [ 61%] Built target generate_transformer_adapt_py
[2026-03-18 01:56:38] [ 61%] Built target generate_ops_info_ascend910b
[2026-03-18 01:56:38] [ 61%] Built target opbase_tiling_objs
[2026-03-18 01:56:38] [ 61%] Built target generate_ops_info
[2026-03-18 01:56:38] INFO: /data/kdy/cann/ops-transformer/build/autogen/inner/aic-ascend910b-ops-info.ini does not exists in this project, skip generating compile commands.
[2026-03-18 01:56:38] bisheng_flags is: 
[2026-03-18 01:56:38] bisheng_flags is: 
[2026-03-18 01:56:38] bisheng_flags is: 
[2026-03-18 01:56:38] bisheng_flags is: 
[2026-03-18 01:56:38] bisheng_flags is: 
[2026-03-18 01:56:38] bisheng_flags is: 
[2026-03-18 01:56:38] bisheng_flags is: 
[2026-03-18 01:56:38] bisheng_flags is: 
[2026-03-18 01:56:38] [ 61%] Built target generate_compile_cmd_ascend910b
[2026-03-18 01:56:38] [ 61%] Built target generate_compile_cmd
[2026-03-18 01:56:38] [ 61%] Built target prepare_build
[2026-03-18 01:56:39] [ 62%] Linking CXX shared library ../../../liboptiling.so
[2026-03-18 01:56:39] [ 62%] Built target optiling
[2026-03-18 01:56:39] -- Configuring done
[2026-03-18 01:56:39] -- Generating done
[2026-03-18 01:56:39] -- Build files have been written to: /data/kdy/cann/ops-transformer/build/ascend_protobuf_build_transformer-prefix/src/ascend_protobuf_build_transformer-build
[2026-03-18 01:56:39] [ 63%] Performing build step for 'ascend_protobuf_build_transformer'
[2026-03-18 01:56:39] [  0%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/spinlock_wait.dir/internal/spinlock_wait.cc.o
[2026-03-18 01:56:39] [  0%] Building CXX object third_party/abseil-cpp/absl/numeric/CMakeFiles/int128.dir/int128.cc.o
[2026-03-18 01:56:39] [  0%] Building CXX object third_party/abseil-cpp/absl/profiling/CMakeFiles/exponential_biased.dir/internal/exponential_biased.cc.o
[2026-03-18 01:56:39] [  0%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_commandlineflag_internal.dir/internal/commandlineflag.cc.o
[2026-03-18 01:56:39] [  0%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/time_zone_impl.cc.o
[2026-03-18 01:56:39] [  1%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/time_zone_if.cc.o
[2026-03-18 01:56:39] [  1%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/time_zone_info.cc.o
[2026-03-18 01:56:39] [  1%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/time_zone_format.cc.o
[2026-03-18 01:56:39] [  2%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/time_zone_lookup.cc.o
[2026-03-18 01:56:39] [  2%] Building C object third_party/utf8_range/CMakeFiles/utf8_range.dir/naive.c.o
[2026-03-18 01:56:39] [  4%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/time_zone_libc.cc.o
[2026-03-18 01:56:39] [  4%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_internal_nullguard.dir/internal/nullguard.cc.o
[2026-03-18 01:56:39] [  4%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_seed_gen_exception.dir/seed_gen_exception.cc.o
[2026-03-18 01:56:39] [  4%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/strerror.dir/internal/strerror.cc.o
[2026-03-18 01:56:39] [  4%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/zone_info_source.cc.o
[2026-03-18 01:56:39] [  4%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/time_zone_posix.cc.o
[2026-03-18 01:56:39] [  5%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_platform.dir/internal/randen_round_keys.cc.o
[2026-03-18 01:56:39] [  5%] Building C object third_party/utf8_range/CMakeFiles/utf8_range.dir/range2-sse.c.o
[2026-03-18 01:56:39] [  5%] Building C object third_party/utf8_range/CMakeFiles/utf8_range.dir/range2-neon.c.o
[2026-03-18 01:56:39] [  5%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time_zone.dir/internal/cctz/src/time_zone_fixed.cc.o
[2026-03-18 01:56:39] [  5%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/civil_time.dir/internal/cctz/src/civil_time_detail.cc.o
[2026-03-18 01:56:39] [  6%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/log_severity.dir/log_severity.cc.o
[2026-03-18 01:56:39] [  6%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/leak_check.dir/leak_check.cc.o
[2026-03-18 01:56:39] [ 64%] Linking CXX shared library libproto_transformer_cust.so
[2026-03-18 01:56:39] [  6%] Linking CXX static library libabsl_flags_commandlineflag_internal.a
[2026-03-18 01:56:39] [  7%] Linking CXX static library libabsl_random_internal_platform.a
[2026-03-18 01:56:40] [  7%] Linking CXX static library libabsl_leak_check.a
[2026-03-18 01:56:40] [  7%] Linking CXX static library libabsl_log_internal_nullguard.a
[2026-03-18 01:56:40] [  7%] Linking CXX static library libabsl_spinlock_wait.a
[2026-03-18 01:56:40] [  7%] Built target random_internal_platform
[2026-03-18 01:56:40] [  7%] Built target flags_commandlineflag_internal
[2026-03-18 01:56:40] [ 64%] Built target ophost_transformer_opapi_obj
[2026-03-18 01:56:40] [  7%] Built target leak_check
[2026-03-18 01:56:40] [ 66%] Generating cust_opapi_stub.cpp
[2026-03-18 01:56:40] [ 66%] Built target proto_transformer_cust
[2026-03-18 01:56:40] [  7%] Built target log_internal_nullguard
[2026-03-18 01:56:40] [  8%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_randen_slow.dir/internal/randen_slow.cc.o
[2026-03-18 01:56:40] [  8%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_randen_hwaes_impl.dir/internal/randen_hwaes.cc.o
[2026-03-18 01:56:40] [  8%] Built target spinlock_wait
[2026-03-18 01:56:40] [ 68%] Smart building ES package 'es_transformer_cust' (auto-handling reconfiguration)...
[2026-03-18 01:56:40] [  9%] Linking CXX static library libabsl_exponential_biased.a
[2026-03-18 01:56:40] [ 69%] Copying OPP proto library to standard path...
[2026-03-18 01:56:40] [ 69%] Building CXX object CMakeFiles/cust_opapi.dir/cust_opapi_stub.cpp.o
[2026-03-18 01:56:40] [  9%] Linking C static library libutf8_range.a
[2026-03-18 01:56:40] -- [Smart Build] Step 1: Building internal targets...
[2026-03-18 01:56:40] [ 70%] Linking CXX shared library libcust_opapi.so
[2026-03-18 01:56:40] [  9%] Built target exponential_biased
[2026-03-18 01:56:40] [  9%] Built target utf8_range
[2026-03-18 01:56:40] [ 71%] Generating ES API code for package 'es_transformer_cust' using run package gen_esb...
[2026-03-18 01:56:40] Generating ES code for package: es_transformer_cust
[2026-03-18 01:56:40] [  9%] Linking CXX static library libabsl_strerror.a
[2026-03-18 01:56:40] [  9%] Building CXX object third_party/abseil-cpp/absl/profiling/CMakeFiles/periodic_sampler.dir/internal/periodic_sampler.cc.o
[2026-03-18 01:56:40] Consolidate compiler generated dependencies of target op_host_aclnnExc
[2026-03-18 01:56:40] [  9%] Linking CXX static library libabsl_log_severity.a
[2026-03-18 01:56:40] [  9%] Linking CXX static library libabsl_random_seed_gen_exception.a
[2026-03-18 01:56:40] [ 11%] Built target op_host_aclnnExc
[2026-03-18 01:56:40] [  9%] Built target strerror
[2026-03-18 01:56:40] [ 27%] Built target cust_opapi
[2026-03-18 01:56:40] [ 33%] Built target opbuild_gen_exc
[2026-03-18 01:56:40] [ 10%] Linking CXX static library libabsl_periodic_sampler.a
[2026-03-18 01:56:40] [ 10%] Built target log_severity
[2026-03-18 01:56:40] [ 10%] Built target random_seed_gen_exception
[2026-03-18 01:56:40] Consolidate compiler generated dependencies of target op_host_aclnn
[2026-03-18 01:56:40] [ 10%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/raw_logging_internal.dir/internal/raw_logging.cc.o
[2026-03-18 01:56:40] [ 50%] Built target op_host_aclnn
[2026-03-18 01:56:40] [ 10%] Built target periodic_sampler
[2026-03-18 01:56:40] [ 50%] Built target opbuild_gen_default
[2026-03-18 01:56:40] Consolidate compiler generated dependencies of target op_host_aclnnInner
[2026-03-18 01:56:40] [ 66%] Built target op_host_aclnnInner
[2026-03-18 01:56:40] [ 72%] Built target ophost_transformer_tiling_obj
[2026-03-18 01:56:40] [ 72%] Built target opbuild_gen_inner
[2026-03-18 01:56:40] Consolidate compiler generated dependencies of target proto_transformer_cust
[2026-03-18 01:56:40] [ 77%] Linking CXX shared library libcust_opmaster_rt2.0.so
[2026-03-18 01:56:40] [ 94%] Built target proto_transformer_cust
[2026-03-18 01:56:40] [100%] Generating ES API code for package 'es_transformer_cust' using run package gen_esb...
[2026-03-18 01:56:40] Generating ES code for package: es_transformer_cust
[2026-03-18 01:56:40] [ 10%] Linking CXX static library libabsl_civil_time.a
[2026-03-18 01:56:40] [ 10%] Linking CXX static library libabsl_random_internal_randen_slow.a
[2026-03-18 01:56:40] [ 10%] Linking CXX static library libabsl_int128.a
[2026-03-18 01:56:40] ==========================================
[2026-03-18 01:56:40] ES Graph Builder Code Generator v1.0
[2026-03-18 01:56:40] Copyright (c) 2025 Huawei Technologies Co., Ltd.
[2026-03-18 01:56:40] ==========================================
[2026-03-18 01:56:40] Starting code generation...
[2026-03-18 01:56:40] Environment variable check: ASCEND_OPP_PATH = /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust
[2026-03-18 01:56:40] Environment variable check: LD_LIBRARY_PATH = /data/kdy/cann/pkg/cann-9.0.0/lib64:/data/kdy/cann/ops-transformer/build:/data/kdy/cann/pkg/cann-9.0.0/lib64:/data/kdy/cann/pkg/cann-9.0.0/lib64/plugin/opskernel:/data/kdy/cann/pkg/cann-9.0.0/lib64/plugin/nnengine:/data/kdy/cann/pkg/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/data/kdy/cann/pkg/cann-9.0.0/tools/aml/lib64:/data/kdy/cann/pkg/cann-9.0.0/tools/aml/lib64/plugin:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:/usr/local/Ascend/nnal/asdsip/latest/lib:/usr/local/Ascend/cann-8.5.0-beta.1/lib64:/usr/local/Ascend/cann-8.5.0-beta.1/lib64/plugin/opskernel:/usr/local/Ascend/cann-8.5.0-beta.1/lib64/plugin/nnengine:/usr/local/Ascend/cann-8.5.0-beta.1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/cann-8.5.0-beta.1/tools/aml/lib64:/usr/local/Ascend/cann-8.5.0-beta.1/tools/aml/lib64/plugin:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:/usr/local/Ascend/nnal/asdsip/latest/lib:
[2026-03-18 01:56:40] Output directory: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/
[2026-03-18 01:56:40] Module name: transformer_cust
[2026-03-18 01:56:40] ASCEND_OPP_PATH is: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust
[2026-03-18 01:56:40] ops proto path: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust/op_proto/custom/:/data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust/op_proto/built-in/
[2026-03-18 01:56:40] Get ops num: 1
[2026-03-18 01:56:40] Generate done, generated num 1, unsupported: 
[2026-03-18 01:56:40] Write to file: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops_c.h
[2026-03-18 01:56:40] Generated C generator aggregate file: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops_c.h
[2026-03-18 01:56:40] Write to file: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops.h
[2026-03-18 01:56:40] Generated C++ generator aggregate file: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops.h
[2026-03-18 01:56:40] Write to file: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops.py
[2026-03-18 01:56:40] Generated Python generator aggregate file: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops.py
[2026-03-18 01:56:40] Generated aggregate files:
[2026-03-18 01:56:40]   C generator: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops_c.h
[2026-03-18 01:56:40]   C++ generator: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops.h
[2026-03-18 01:56:40]   Python generator: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_transformer_cust_ops.py
[2026-03-18 01:56:40] Generated util files:
[2026-03-18 01:56:40]   Utils generator:
[2026-03-18 01:56:40]     /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/es_log.h
[2026-03-18 01:56:40] Generated files in: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/generated_code/
[2026-03-18 01:56:40] 
[2026-03-18 01:56:40] ==========================================
[2026-03-18 01:56:40] Code generation completed!
[2026-03-18 01:56:40] Module name: transformer_cust
[2026-03-18 01:56:40] ==========================================
[2026-03-18 01:56:40] [ 10%] Built target civil_time
[2026-03-18 01:56:40] [ 10%] Built target random_internal_randen_slow
[2026-03-18 01:56:40] [105%] Built target generate_es_transformer_cust_code
[2026-03-18 01:56:40] [ 10%] Built target int128
[2026-03-18 01:56:40] [116%] Skipping Python wheel package generation for 'es_transformer_cust'
[2026-03-18 01:56:40] [116%] Building CXX object CMakeFiles/es_transformer_cust_so.dir/es_transformer_cust_build/generated_code/placeholder.cpp.o
[2026-03-18 01:56:40] Skipping wheel generation
[2026-03-18 01:56:40] [ 10%] Linking CXX static library libabsl_raw_logging_internal.a
[2026-03-18 01:56:40] [ 10%] Linking CXX static library libabsl_random_internal_randen_hwaes_impl.a
[2026-03-18 01:56:40] [122%] Linking CXX shared library libes_transformer_cust.so
[2026-03-18 01:56:40] [122%] Built target generate_es_transformer_cust_whl
[2026-03-18 01:56:40] [ 10%] Built target random_internal_randen_hwaes_impl
[2026-03-18 01:56:40] [ 10%] Built target raw_logging_internal
[2026-03-18 01:56:40] [122%] Built target es_transformer_cust_so
[2026-03-18 01:56:40] [ 10%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_randen_hwaes.dir/internal/randen_detect.cc.o
[2026-03-18 01:56:40] [ 11%] Building CXX object third_party/abseil-cpp/absl/types/CMakeFiles/bad_variant_access.dir/bad_variant_access.cc.o
[2026-03-18 01:56:40] [ 11%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/debugging_internal.dir/internal/address_is_readable.cc.o
[2026-03-18 01:56:40] [ 11%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cordz_functions.dir/internal/cordz_functions.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/debugging_internal.dir/internal/elf_mem_image.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/types/CMakeFiles/bad_any_cast_impl.dir/bad_any_cast.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/debugging_internal.dir/internal/vdso_support.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/base.dir/internal/cycleclock.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/base.dir/internal/spinlock.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/throw_delegate.dir/internal/throw_delegate.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/scoped_set_env.dir/internal/scoped_set_env.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/base.dir/internal/unscaledcycleclock.cc.o
[2026-03-18 01:56:40] [ 12%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/base.dir/internal/sysinfo.cc.o
[2026-03-18 01:56:40] [ 13%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/base.dir/internal/thread_identity.cc.o
[2026-03-18 01:56:40] [ 13%] Building CXX object third_party/abseil-cpp/absl/types/CMakeFiles/bad_optional_access.dir/bad_optional_access.cc.o
[2026-03-18 01:56:41] [122%] Built target cust_opmaster
[2026-03-18 01:56:41] [ 13%] Linking CXX static library libabsl_random_internal_randen_hwaes.a
[2026-03-18 01:56:41] [127%] Generating compat/liboptiling.so
[2026-03-18 01:56:41] [ 13%] Built target random_internal_randen_hwaes
[2026-03-18 01:56:41] [127%] Built target optiling_compat
[2026-03-18 01:56:41] [ 13%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_randen.dir/internal/randen.cc.o
[2026-03-18 01:56:41] [150%] Generating binary/ascend910b/gen/moe_token_permute_with_routing_map_ascend910b_1.done
[2026-03-18 01:56:41] [150%] Generating binary/ascend910b/gen/moe_token_permute_with_routing_map_ascend910b_4.done
[2026-03-18 01:56:41] [150%] Generating binary/ascend910b/gen/moe_token_permute_with_routing_map_ascend910b_7.done
[2026-03-18 01:56:41] [150%] Generating binary/ascend910b/gen/moe_token_permute_with_routing_map_ascend910b_2.done
[2026-03-18 01:56:41] [155%] Generating binary/ascend910b/gen/moe_token_permute_with_routing_map_ascend910b_6.done
[2026-03-18 01:56:41] [172%] Generating binary/ascend910b/gen/moe_token_permute_with_routing_map_ascend910b_0.done
[2026-03-18 01:56:41] [172%] Generating binary/ascend910b/gen/moe_token_permute_with_routing_map_ascend910b_5.done
[2026-03-18 01:56:41] [172%] Generating binary/ascend910b/gen/moe_token_permute_with_routing_map_ascend910b_3.done
[2026-03-18 01:56:41] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_5226913cab1d55bc46a191277f25d0aa ...
[2026-03-18 01:56:41] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_f023cd8e1fdcb40f9bb20466aa054c11 ...
[2026-03-18 01:56:41] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_ce90d820a26ded18768878970d7ddd3f ...
[2026-03-18 01:56:41] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_7874a5d9ba91ea6ef7a06d577e840637 ...
[2026-03-18 01:56:41] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_73a356ee7e9b844f1cdeb0b21b426ed0 ...
[2026-03-18 01:56:41] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_70c842871515ad30301d19504b276dca ...
[2026-03-18 01:56:41] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_ec5c85f5a7829084c1a10c21f30ebf12 ...
[2026-03-18 01:56:41] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_73a7f4889ad86134a7af69582b8c3716 ...
[2026-03-18 01:56:41] [172%] Built target generate_es_transformer_cust_code
[2026-03-18 01:56:41] [172%] Built target generate_es_transformer_cust_whl
[2026-03-18 01:56:41] Consolidate compiler generated dependencies of target es_transformer_cust_so
[2026-03-18 01:56:41] [ 13%] Linking CXX static library libabsl_bad_variant_access.a
[2026-03-18 01:56:41] [172%] Built target es_transformer_cust_so
[2026-03-18 01:56:41] [ 13%] Linking CXX static library libabsl_bad_any_cast_impl.a
[2026-03-18 01:56:41] [177%] Installing ES package 'es_transformer_cust' to /data/kdy/cann/ops-transformer/build/es_packages
[2026-03-18 01:56:41] [ 13%] Built target bad_variant_access
[2026-03-18 01:56:41] [ 14%] Linking CXX static library libabsl_bad_optional_access.a
[2026-03-18 01:56:41] [ 14%] Linking CXX static library libabsl_scoped_set_env.a
[2026-03-18 01:56:41] Installing ES package: es_transformer_cust
[2026-03-18 01:56:41] -- Copied header: es_MoeTokenPermuteWithRoutingMap.h
[2026-03-18 01:56:41] -- Copied header: es_MoeTokenPermuteWithRoutingMap_c.h
[2026-03-18 01:56:41] -- Copied header: es_log.h
[2026-03-18 01:56:41] -- Copied header: es_transformer_cust_ops.h
[2026-03-18 01:56:41] -- Copied header: es_transformer_cust_ops_c.h
[2026-03-18 01:56:41] [ 15%] Linking CXX static library libabsl_throw_delegate.a
[2026-03-18 01:56:41] [ 15%] Built target bad_any_cast_impl
[2026-03-18 01:56:41] [ 15%] Built target scoped_set_env
[2026-03-18 01:56:41] [ 15%] Built target bad_optional_access
[2026-03-18 01:56:41] [177%] Built target install_es_transformer_cust
[2026-03-18 01:56:41] [ 15%] Built target throw_delegate
[2026-03-18 01:56:41] [ 15%] Linking CXX static library libabsl_cordz_functions.a
[2026-03-18 01:56:41] -- [Smart Build] Found 1 generated source file(s)
[2026-03-18 01:56:41] -- [Smart Build] Step 2: Reconfiguring to include generated sources...
[2026-03-18 01:56:41] [ 15%] Linking CXX static library libabsl_debugging_internal.a
[2026-03-18 01:56:41] -- ASCEND_CANN_PACKAGE_PATH=/data/kdy/cann/pkg/cann-9.0.0
[2026-03-18 01:56:41] [ 16%] Linking CXX static library libabsl_random_internal_randen.a
[2026-03-18 01:56:41] [ 16%] Built target cordz_functions
[2026-03-18 01:56:41] -- compile project with library
[2026-03-18 01:56:41] -- Find opbase source dir: /data/kdy/cann/ops-transformer/third_party/opbase
[2026-03-18 01:56:41] [ 16%] Linking CXX static library libabsl_base.a
[2026-03-18 01:56:41] [ 16%] Built target debugging_internal
[2026-03-18 01:56:41] [ 16%] Built target random_internal_randen
[2026-03-18 01:56:41] [ 16%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/stacktrace.dir/stacktrace.cc.o
[2026-03-18 01:56:41] [ 16%] Built target base
[2026-03-18 01:56:41] -- ENABLE_CCACHE=ON, CUSTOM_CCACHE=
[2026-03-18 01:56:41] -- CCACHE_PROGRAM=CCACHE_PROGRAM-NOTFOUND
[2026-03-18 01:56:41] -- ASCEND_COMPUTE_UNIT=ascend910b;
[2026-03-18 01:56:41] -- ASCEND_OP_NAME=moe_token_permute_with_routing_map
[2026-03-18 01:56:41] -- TILING_KEY=
[2026-03-18 01:56:41] -- TESTS_UT_OPS_TEST=
[2026-03-18 01:56:41] -- TESTS_EXAMPLE_OPS_TEST=
[2026-03-18 01:56:41] -- CMAKE_BUILD_TYPE=Release
[2026-03-18 01:56:41] -- VERSION=
[2026-03-18 01:56:41] [ 16%] Building CXX object third_party/abseil-cpp/absl/base/CMakeFiles/malloc_internal.dir/internal/low_level_alloc.cc.o
[2026-03-18 01:56:41] [ 16%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/demangle_internal.dir/internal/demangle.cc.o
[2026-03-18 01:56:41] [ 16%] Building CXX object third_party/abseil-cpp/absl/crc/CMakeFiles/crc_cpu_detect.dir/internal/cpu_detect.cc.o
[2026-03-18 01:56:41] [ 18%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/string_view.dir/string_view.cc.o
[2026-03-18 01:56:41] [ 18%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings_internal.dir/internal/ostringstream.cc.o
[2026-03-18 01:56:41] [ 18%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings_internal.dir/internal/escaping.cc.o
[2026-03-18 01:56:41] [ 19%] Building CXX object third_party/abseil-cpp/absl/hash/CMakeFiles/city.dir/internal/city.cc.o
[2026-03-18 01:56:41] [ 19%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings_internal.dir/internal/utf8.cc.o
[2026-03-18 01:56:41] [ 19%] Building CXX object third_party/abseil-cpp/absl/hash/CMakeFiles/low_level_hash.dir/internal/low_level_hash.cc.o
[2026-03-18 01:56:41] [ 19%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_internal_conditions.dir/internal/conditions.cc.o
[2026-03-18 01:56:41] [ 19%] Linking CXX static library libabsl_time_zone.a
[2026-03-18 01:56:41] [ 19%] Built target time_zone
[2026-03-18 01:56:41] [ 19%] Linking CXX static library libabsl_log_internal_conditions.a
[2026-03-18 01:56:41] [ 19%] Built target log_internal_conditions
[2026-03-18 01:56:42] [ 19%] Linking CXX static library libabsl_crc_cpu_detect.a
[2026-03-18 01:56:42] [ 19%] Built target crc_cpu_detect
[2026-03-18 01:56:42] [ 20%] Building CXX object third_party/abseil-cpp/absl/crc/CMakeFiles/crc_internal.dir/internal/crc_x86_arm_combined.cc.o
[2026-03-18 01:56:42] [ 20%] Building CXX object third_party/abseil-cpp/absl/crc/CMakeFiles/crc_internal.dir/internal/crc.cc.o
[2026-03-18 01:56:42] [ 22%] Linking CXX static library libabsl_city.a
[2026-03-18 01:56:42] [ 22%] Linking CXX static library libabsl_low_level_hash.a
[2026-03-18 01:56:42] [ 22%] Linking CXX static library libabsl_stacktrace.a
[2026-03-18 01:56:42] [ 22%] Built target low_level_hash
[2026-03-18 01:56:42] [ 22%] Built target city
[2026-03-18 01:56:42] [ 22%] Built target stacktrace
[2026-03-18 01:56:42] [ 22%] Linking CXX static library libabsl_strings_internal.a
[2026-03-18 01:56:42] [ 22%] Linking CXX static library libabsl_string_view.a
[2026-03-18 01:56:42] [ 22%] Built target strings_internal
[2026-03-18 01:56:42] [ 22%] Built target string_view
[2026-03-18 01:56:42] [ 23%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/ascii.cc.o
[2026-03-18 01:56:42] [ 23%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/charconv.cc.o
[2026-03-18 01:56:42] [ 23%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/escaping.cc.o
[2026-03-18 01:56:42] [ 23%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/match.cc.o
[2026-03-18 01:56:42] [ 24%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/str_cat.cc.o
[2026-03-18 01:56:42] [ 24%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/str_split.cc.o
[2026-03-18 01:56:42] [ 24%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/str_replace.cc.o
[2026-03-18 01:56:42] [ 25%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/internal/charconv_bigint.cc.o
[2026-03-18 01:56:42] [ 25%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/internal/damerau_levenshtein_distance.cc.o
[2026-03-18 01:56:42] [ 25%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/substitute.cc.o
[2026-03-18 01:56:42] [ 25%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/internal/memutil.cc.o
[2026-03-18 01:56:42] [ 25%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/numbers.cc.o
[2026-03-18 01:56:42] [ 25%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/internal/charconv_parse.cc.o
[2026-03-18 01:56:42] [ 25%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/strings.dir/internal/stringify_sink.cc.o
[2026-03-18 01:56:42] [ 25%] Linking CXX static library libabsl_malloc_internal.a
[2026-03-18 01:56:42] [ 25%] Built target malloc_internal
[2026-03-18 01:56:42] [ 25%] Linking CXX static library libabsl_demangle_internal.a
[2026-03-18 01:56:42] [ 25%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/graphcycles_internal.dir/internal/graphcycles.cc.o
[2026-03-18 01:56:42] [ 25%] Built target demangle_internal
[2026-03-18 01:56:42] [ 25%] Linking CXX static library libabsl_crc_internal.a
[2026-03-18 01:56:42] [ 25%] Built target crc_internal
[2026-03-18 01:56:43] [ 26%] Linking CXX static library libabsl_graphcycles_internal.a
[2026-03-18 01:56:43] [ 26%] Built target graphcycles_internal
[2026-03-18 01:56:43] [ 26%] Linking CXX static library libabsl_strings.a
[2026-03-18 01:56:43] [ 26%] Built target strings
[2026-03-18 01:56:43] [ 26%] Building CXX object third_party/utf8_range/CMakeFiles/utf8_validity.dir/utf8_validity.cc.o
[2026-03-18 01:56:43] [ 26%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time.dir/clock.cc.o
[2026-03-18 01:56:43] [ 26%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time.dir/civil_time.cc.o
[2026-03-18 01:56:43] [ 27%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_distributions.dir/gaussian_distribution.cc.o
[2026-03-18 01:56:43] [ 27%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_seed_material.dir/internal/seed_material.cc.o
[2026-03-18 01:56:43] [ 27%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time.dir/time.cc.o
[2026-03-18 01:56:43] [ 27%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time.dir/duration.cc.o
[2026-03-18 01:56:43] [ 27%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_distributions.dir/discrete_distribution.cc.o
[2026-03-18 01:56:43] [ 27%] Building CXX object third_party/abseil-cpp/absl/time/CMakeFiles/time.dir/format.cc.o
[2026-03-18 01:56:43] [ 28%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/str_format_internal.dir/internal/str_format/arg.cc.o
[2026-03-18 01:56:43] [ 28%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/symbolize.dir/symbolize.cc.o
[2026-03-18 01:56:43] [ 28%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/str_format_internal.dir/internal/str_format/extension.cc.o
[2026-03-18 01:56:43] [ 28%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/str_format_internal.dir/internal/str_format/bind.cc.o
[2026-03-18 01:56:43] [ 28%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/str_format_internal.dir/internal/str_format/float_conversion.cc.o
[2026-03-18 01:56:43] [ 28%] Building CXX object third_party/abseil-cpp/absl/hash/CMakeFiles/hash.dir/internal/hash.cc.o
[2026-03-18 01:56:43] [ 29%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_internal_proto.dir/internal/proto.cc.o
[2026-03-18 01:56:43] [ 29%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/str_format_internal.dir/internal/str_format/parser.cc.o
[2026-03-18 01:56:43] [ 29%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/str_format_internal.dir/internal/str_format/output.cc.o
[2026-03-18 01:56:43] [ 30%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_commandlineflag.dir/commandlineflag.cc.o
[2026-03-18 01:56:44] [ 31%] Linking CXX static library libutf8_validity.a
[2026-03-18 01:56:44] [ 31%] Linking CXX static library libabsl_flags_commandlineflag.a
[2026-03-18 01:56:44] [ 31%] Built target utf8_validity
[2026-03-18 01:56:44] [ 31%] Built target flags_commandlineflag
[2026-03-18 01:56:44] [ 31%] Linking CXX static library libabsl_log_internal_proto.a
[2026-03-18 01:56:44] [ 31%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_private_handle_accessor.dir/internal/private_handle_accessor.cc.o
[2026-03-18 01:56:44] [ 31%] Built target log_internal_proto
[2026-03-18 01:56:44] [ 32%] Linking CXX static library libabsl_random_distributions.a
[2026-03-18 01:56:44] [ 32%] Linking CXX static library libabsl_hash.a
[2026-03-18 01:56:44] [ 32%] Built target random_distributions
[2026-03-18 01:56:44] [ 32%] Linking CXX static library libabsl_random_internal_seed_material.a
[2026-03-18 01:56:44] [ 32%] Built target hash
[2026-03-18 01:56:44] [ 32%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_globals.dir/globals.cc.o
[2026-03-18 01:56:44] [ 32%] Built target random_internal_seed_material
[2026-03-18 01:56:44] [ 32%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_pool_urbg.dir/internal/pool_urbg.cc.o
[2026-03-18 01:56:44] [ 32%] Linking CXX static library libabsl_symbolize.a
[2026-03-18 01:56:44] [ 32%] Linking CXX static library libabsl_flags_private_handle_accessor.a
[2026-03-18 01:56:44] [ 32%] Built target flags_private_handle_accessor
[2026-03-18 01:56:44] [ 32%] Built target symbolize
[2026-03-18 01:56:44] [ 32%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/examine_stack.dir/internal/examine_stack.cc.o
[2026-03-18 01:56:45] [ 32%] Linking CXX static library libabsl_time.a
[2026-03-18 01:56:45] [ 32%] Built target time
[2026-03-18 01:56:45] [ 32%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/kernel_timeout_internal.dir/internal/kernel_timeout.cc.o
[2026-03-18 01:56:45] [ 32%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_entry.dir/log_entry.cc.o
[2026-03-18 01:56:45] [ 32%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_internal_globals.dir/internal/globals.cc.o
[2026-03-18 01:56:45] [ 32%] Linking CXX static library libabsl_str_format_internal.a
[2026-03-18 01:56:45] [ 32%] Linking CXX static library libabsl_examine_stack.a
[2026-03-18 01:56:45] [ 32%] Linking CXX static library libabsl_random_internal_pool_urbg.a
[2026-03-18 01:56:45] [ 32%] Built target str_format_internal
[2026-03-18 01:56:45] [ 32%] Built target examine_stack
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/crc/CMakeFiles/crc32c.dir/crc32c.cc.o
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/crc/CMakeFiles/crc32c.dir/internal/crc_memcpy_fallback.cc.o
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/crc/CMakeFiles/crc32c.dir/internal/crc_memcpy_x86_64.cc.o
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/crc/CMakeFiles/crc32c.dir/internal/crc_non_temporal_memcpy.cc.o
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_marshalling.dir/marshalling.cc.o
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_distribution_test_util.dir/internal/chi_square.cc.o
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_internal_distribution_test_util.dir/internal/distribution_test_util.cc.o
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/debugging/CMakeFiles/failure_signal_handler.dir/failure_signal_handler.cc.o
[2026-03-18 01:56:45] [ 33%] Built target random_internal_pool_urbg
[2026-03-18 01:56:45] [ 33%] Linking CXX static library libabsl_log_globals.a
[2026-03-18 01:56:45] [ 33%] Building CXX object third_party/abseil-cpp/absl/random/CMakeFiles/random_seed_sequences.dir/seed_sequences.cc.o
[2026-03-18 01:56:45] -- Start building custom package.
[2026-03-18 01:56:45] -- Search libs under install path /data/kdy/cann/pkg/cann-9.0.0
[2026-03-18 01:56:45] -- CMAKE_MODULE_PATH            :/data/kdy/cann/ops-transformer/cmake/modules;/data/kdy/cann/pkg/cann-9.0.0/include/ge/cmake
[2026-03-18 01:56:45] -- compile option:   -Wall -Wextra -Wshadow -Wformat=2 -fno-common -fPIC -Wformat-signedness -Wno-missing-include-dirs -Wno-write-strings
[2026-03-18 01:56:45] -- Transformer Variables in dlog module:
[2026-03-18 01:56:45] -- dlog_TRANSFORMER_INCLUDE_DIR="/data/kdy/cann/pkg/cann-9.0.0/pkg_inc/base"
[2026-03-18 01:56:45] -- dlog_SHARED_LIBRARY="/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:45] -- 
[2026-03-18 01:56:45]  Properties for TARGET dlog:
[2026-03-18 01:56:45]    dlog.INTERFACE_COMPILE_DEFINITIONS = "LOG_CPP;PROCESS_LOG"
[2026-03-18 01:56:45]    dlog.INTERFACE_LINK_LIBRARIES = "dlog_headers"
[2026-03-18 01:56:45]    dlog.IMPORTED_LOCATION = "/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:45] 
[2026-03-18 01:56:45] -- 
[2026-03-18 01:56:45]  Properties for TARGET dlog_headers:
[2026-03-18 01:56:45]    dlog_headers.INTERFACE_INCLUDE_DIRECTORIES = "/data/kdy/cann/pkg/cann-9.0.0/pkg_inc/base"
[2026-03-18 01:56:45] 
[2026-03-18 01:56:45] -- Found OPABSE include:/data/kdy/cann/ops-transformer/third_party/opbase/pkg_inc
[2026-03-18 01:56:45] -- Found OPABSE lib:
[2026-03-18 01:56:45] -- Found platform include:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include
[2026-03-18 01:56:45] -- Found register library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libregister.so
[2026-03-18 01:56:45] -- Found opp_registry library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libopp_registry.so
[2026-03-18 01:56:45] -- Found exe_graph library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libexe_graph.so
[2026-03-18 01:56:45] -- Found graph_base library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libgraph_base.so
[2026-03-18 01:56:45] -- Found error_manager library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/liberror_manager.so
[2026-03-18 01:56:45] -- Found metadef library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libmetadef.so
[2026-03-18 01:56:45] -- Found rt2_registry library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/librt2_registry.a
[2026-03-18 01:56:45] -- Found source metadef include dir:  /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/exe_graph
[2026-03-18 01:56:45] -- Found graph library:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libgraph.so
[2026-03-18 01:56:45] -- Found source acl include dir:  /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include
[2026-03-18 01:56:45] -- [transformer]Found source npuruntime include dir: /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/runtime
[2026-03-18 01:56:45] -- Found aclnn include dir:  /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include
[2026-03-18 01:56:45] -- Found opdev include dir:  /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aclnn
[2026-03-18 01:56:45] -- Found tilingapi include:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include
[2026-03-18 01:56:45] -- TILINGAPI_INC_DIRS: /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api/lib;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api/impl;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api/tiling
[2026-03-18 01:56:45] -- Found tilingapi lib:/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/lib64/libtiling_api.a
[2026-03-18 01:56:45] -- Using AICPU include dirs: /data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment/cce;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment/msprof;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu_common/context;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu_common/context/common;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu_common/context/cpu_proto;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu_common/context/utils;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aicpu_common/context;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aicpu_common/context/common;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aicpu_common/context/cpu_proto;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/aicpu_common/context/utils;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment/datagw/aicpu/common;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/include/experiment/datagw/aicpu/common;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc;/data/kdy/cann/pkg/cann-9.0.0/aarch64-linux/pkg_inc/aicpu
[2026-03-18 01:56:45] -- Finding gen es pkg
[2026-03-18 01:56:45] -- [add_es_library] Module loaded from: /data/kdy/cann/pkg/cann-9.0.0/include/ge/cmake
[2026-03-18 01:56:45] [ 33%] Built target log_globals
[2026-03-18 01:56:45] -- Eigen path found in cache: /data/kdy/cann/ops-transformer/third_party/eigen
[2026-03-18 01:56:45] -- Eigen already exists at /data/kdy/cann/ops-transformer/third_party/eigen
[2026-03-18 01:56:45] -- Could NOT find abseil-cpp (missing: ABSL_SOURCE_DIR) 
[2026-03-18 01:56:45] -- [ThirdPartyLib][abseil-cpp] found in /data/kdy/cann/ops-transformer/third_party/pkg/abseil-cpp-20230802.1.tar.gz.
[2026-03-18 01:56:45] -- Could NOT find ascend_protobuf_build_transformer (missing: ASCEND_PROTOBUF_SHARED_INCLUDE) 
[2026-03-18 01:56:45] -- [ThirdPartyLib][ascend protobuf] ascend protobuf shared not found, finding binary file.
[2026-03-18 01:56:45] -- [ThirdPartyLib][ascend protobuf] found in file:///data/kdy/cann/ops-transformer/third_party/pkg/protobuf-25.1.tar.gz.
[2026-03-18 01:56:45] -- Variables in alog module:
[2026-03-18 01:56:45] -- alog_a_INCLUDE_DIR="/data/kdy/cann/pkg/cann-9.0.0/include"
[2026-03-18 01:56:45] -- slog_a_SHARED_LIBRARY="/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:45] -- alog_a_SHARED_LIBRARY="/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:45] -- 
[2026-03-18 01:56:45]  Properties for TARGET slog_a:
[2026-03-18 01:56:45]    slog_a.INTERFACE_COMPILE_DEFINITIONS = "LOG_CPP;PROCESS_LOG"
[2026-03-18 01:56:45]    slog_a.INTERFACE_LINK_LIBRARIES = "alog_headers"
[2026-03-18 01:56:45]    slog_a.IMPORTED_LOCATION = "/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:45] 
[2026-03-18 01:56:45] -- 
[2026-03-18 01:56:45]  Properties for TARGET alog_a:
[2026-03-18 01:56:45]    alog_a.INTERFACE_COMPILE_DEFINITIONS = "LOG_CPP;PROCESS_LOG"
[2026-03-18 01:56:45]    alog_a.INTERFACE_LINK_LIBRARIES = "alog_headers"
[2026-03-18 01:56:45]    alog_a.IMPORTED_LOCATION = "/data/kdy/cann/pkg/cann-9.0.0/lib64/libascendalog.so"
[2026-03-18 01:56:45] 
[2026-03-18 01:56:45] -- 
[2026-03-18 01:56:45]  Properties for TARGET alog_headers:
[2026-03-18 01:56:45]    alog_headers.INTERFACE_INCLUDE_DIRECTORIES = "/data/kdy/cann/pkg/cann-9.0.0/include;/data/kdy/cann/pkg/cann-9.0.0/include/base"
[2026-03-18 01:56:45] 
[2026-03-18 01:56:45] -- Could NOT find unified_dlog (missing: _INCLUDE_ROOT_DIR) 
[2026-03-18 01:56:45] -- Skip proto update: /data/kdy/cann/ops-transformer/moe/moe_token_permute_with_routing_map
[2026-03-18 01:56:45] -- Start Generating es transformer for custom pkg
[2026-03-18 01:56:45] -- _add_es_library_impl: EXCLUDE_OPS is not provided
[2026-03-18 01:56:45] -- add_es_package: Using default output directory: /data/kdy/cann/ops-transformer/build
[2026-03-18 01:56:45] -- add_es_package: OPP_PROTO_TARGET output directory: /data/kdy/cann/ops-transformer/build
[2026-03-18 01:56:45] -- add_es_package: Derived ASCEND_OPP_PATH: /data/kdy/cann/ops-transformer/build
[2026-03-18 01:56:45] --   - Created standard OPP path: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust
[2026-03-18 01:56:45] -- _add_es_library_impl: gen_esb target not found, searching in environment...
[2026-03-18 01:56:45] --     Found gen_esb in PATH (environment variables configured)
[2026-03-18 01:56:45] --     - gen_esb path: /data/kdy/cann/pkg/cann-9.0.0/bin/gen_esb
[2026-03-18 01:56:45] --     - library path: /data/kdy/cann/pkg/cann-9.0.0/lib64
[2026-03-18 01:56:45] -- Using proto library path (derived from OPP_PROTO_TARGET): /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust
[2026-03-18 01:56:45] -- add_es_package: Module name for gen_esb: transformer_cust
[2026-03-18 01:56:45] -- Trying to locate eager_style_graph_builder_base library from run package...
[2026-03-18 01:56:45] --   Found library in run package: /data/kdy/cann/pkg/cann-9.0.0/lib64/libeager_style_graph_builder_base.so
[2026-03-18 01:56:45] -- Configuring ES package: es_transformer_cust
[2026-03-18 01:56:45] --   - Exported target: es_transformer_cust (public interface)
[2026-03-18 01:56:45] --   - Internal .so target: es_transformer_cust_so
[2026-03-18 01:56:45] --   - Module name (for gen_esb): transformer_cust
[2026-03-18 01:56:45] --   - OPP proto target: proto_transformer_cust
[2026-03-18 01:56:45] --   - OPP proto path: /data/kdy/cann/ops-transformer/build/es_transformer_cust_build/opp_standard_path_proto_transformer_cust
[2026-03-18 01:56:45] --   - Output path: /data/kdy/cann/ops-transformer/build/es_packages
[2026-03-18 01:56:45] --   - Library file: libes_transformer_cust.so
[2026-03-18 01:56:45] --   - Python package: es_transformer_cust
[2026-03-18 01:56:45] -- Found 2 source file(s) for package 'es_transformer_cust'
[2026-03-18 01:56:45] -- add_es_library: Using default compile options: -fPIC;-Wall;-std=c++17;-D_GLIBCXX_USE_CXX11_ABI=0;-O2;-Wno-free-nonheap-object
[2026-03-18 01:56:45] -- add_es_library: Set _GLIBCXX_USE_CXX11_ABI=0 for ABI compatibility
[2026-03-18 01:56:45] -- add_es_library: Deduced header base path from cmake file: /data/kdy/cann/pkg/cann-9.0.0
[2026-03-18 01:56:45] -- add_es_library: metadef_headers target not found, skipping
[2026-03-18 01:56:45] -- add_es_library: Found c_sec target
[2026-03-18 01:56:45] -- add_es_library: Linking with CANN libraries: c_sec
[2026-03-18 01:56:45] -- add_es_package: 'build_es_transformer_cust' depends on 'build_es_transformer_cust' to avoid concurrent builds of shared dependencies
[2026-03-18 01:56:45] -- ES package 'es_transformer_cust' configured successfully:
[2026-03-18 01:56:45] --   - Exported target (for linking): es_transformer_cust
[2026-03-18 01:56:45] --   - Smart build target: build_es_transformer_cust
[2026-03-18 01:56:45] --   - Internal targets:
[2026-03-18 01:56:45] --     * Code generation: generate_es_transformer_cust_code
[2026-03-18 01:56:45] --     * Shared library: es_transformer_cust_so
[2026-03-18 01:56:45] --     * Imported library (linking): es_transformer_cust_imported
[2026-03-18 01:56:45] --     * Wheel generation: generate_es_transformer_cust_whl
[2026-03-18 01:56:45] --     * Install: install_es_transformer_cust
[2026-03-18 01:56:45] --   - Library file: libes_transformer_cust.so
[2026-03-18 01:56:45] --   - Python package: es_transformer_cust
[2026-03-18 01:56:45] --   - ES base library: /data/kdy/cann/pkg/cann-9.0.0/lib64/libeager_style_graph_builder_base.so (from library)
[2026-03-18 01:56:45] -- 
[2026-03-18 01:56:45] -- Quick start:
[2026-03-18 01:56:45] --   1. Link:    target_link_libraries(your_target PRIVATE es_transformer_cust)
[2026-03-18 01:56:45] --   2. Build:   make your_target  (recommended, triggers automatically)
[2026-03-18 01:56:45] -- 
[2026-03-18 01:56:45] -- custom cust proto to es
[2026-03-18 01:56:45] -- Detected architecture: ARM64
[2026-03-18 01:56:45] [ 34%] Linking CXX static library libabsl_log_entry.a
[2026-03-18 01:56:45] -- End building custom package.
[2026-03-18 01:56:45] -- create aclnn master header file: /data/kdy/cann/ops-transformer/build/aclnn_ops_transformer_custom.h
[2026-03-18 01:56:45] -- no operator info to generate
[2026-03-18 01:56:45] -- Start building custom package.
[2026-03-18 01:56:45] -- No aicpu json files to merge, skipping.
[2026-03-18 01:56:45] -- No aicpu cust obj targets found, skipping.
[2026-03-18 01:56:45] -- Configuring done
[2026-03-18 01:56:45] [ 34%] Built target log_entry
[2026-03-18 01:56:45] [ 35%] Linking CXX static library libabsl_log_internal_globals.a
[2026-03-18 01:56:45] [ 35%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_sink.dir/log_sink.cc.o
[2026-03-18 01:56:45] -- Generating done
[2026-03-18 01:56:45] -- Build files have been written to: /data/kdy/cann/ops-transformer/build
[2026-03-18 01:56:45] -- [Smart Build] Step 3: Rebuilding with all generated sources...
[2026-03-18 01:56:45] [ 35%] Built target log_internal_globals
[2026-03-18 01:56:45] [ 36%] Linking CXX static library libabsl_kernel_timeout_internal.a
[2026-03-18 01:56:46] [ 36%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_internal_format.dir/internal/log_format.cc.o
[2026-03-18 01:56:46] [ 37%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_initialize.dir/initialize.cc.o
[2026-03-18 01:56:46] [ 37%] Linking CXX static library libabsl_failure_signal_handler.a
[2026-03-18 01:56:46] [ 37%] Built target kernel_timeout_internal
[2026-03-18 01:56:46] [ 38%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/barrier.cc.o
[2026-03-18 01:56:46] [ 38%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/internal/futex_waiter.cc.o
[2026-03-18 01:56:46] [ 39%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/internal/per_thread_sem.cc.o
[2026-03-18 01:56:46] [ 39%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/internal/waiter_base.cc.o
[2026-03-18 01:56:46] [ 39%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/internal/create_thread_identity.cc.o
[2026-03-18 01:56:46] [ 39%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/internal/sem_waiter.cc.o
[2026-03-18 01:56:46] [ 39%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/blocking_counter.cc.o
[2026-03-18 01:56:46] [ 39%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/internal/stdcpp_waiter.cc.o
[2026-03-18 01:56:46] [ 40%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/notification.cc.o
[2026-03-18 01:56:46] [ 40%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/internal/win32_waiter.cc.o
[2026-03-18 01:56:46] [ 40%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/mutex.cc.o
[2026-03-18 01:56:46] [ 40%] Building CXX object third_party/abseil-cpp/absl/synchronization/CMakeFiles/synchronization.dir/internal/pthread_waiter.cc.o
[2026-03-18 01:56:46] Consolidate compiler generated dependencies of target op_host_aclnnInner
[2026-03-18 01:56:46] Consolidate compiler generated dependencies of target op_host_aclnn
[2026-03-18 01:56:46] Consolidate compiler generated dependencies of target op_host_aclnnExc
[2026-03-18 01:56:46] [ 40%] Built target failure_signal_handler
[2026-03-18 01:56:46] [ 11%] Built target op_host_aclnn
[2026-03-18 01:56:46] [ 27%] Built target op_host_aclnnInner
[2026-03-18 01:56:46] [ 38%] Built target op_host_aclnnExc
[2026-03-18 01:56:46] [ 38%] Built target opbuild_gen_default
[2026-03-18 01:56:46] [ 44%] Built target opbuild_gen_inner
[2026-03-18 01:56:46] [ 44%] Built target opbuild_gen_exc
[2026-03-18 01:56:46] [ 40%] Linking CXX static library libabsl_random_seed_sequences.a
[2026-03-18 01:56:46] Consolidate compiler generated dependencies of target proto_transformer_cust
[2026-03-18 01:56:46] [ 61%] Built target proto_transformer_cust
[2026-03-18 01:56:46] [ 40%] Built target random_seed_sequences
[2026-03-18 01:56:46] [ 72%] Built target generate_es_transformer_cust_code
[2026-03-18 01:56:46] Consolidate compiler generated dependencies of target es_transformer_cust_so
[2026-03-18 01:56:46] [ 77%] Built target generate_es_transformer_cust_whl
[2026-03-18 01:56:46] [ 40%] Linking CXX static library libabsl_log_sink.a
[2026-03-18 01:56:46] [ 83%] Building CXX object CMakeFiles/es_transformer_cust_so.dir/es_transformer_cust_build/generated_code/es_MoeTokenPermuteWithRoutingMap.cpp.o
[2026-03-18 01:56:46] [ 40%] Linking CXX static library libabsl_log_initialize.a
[2026-03-18 01:56:46] [ 40%] Built target log_sink
[2026-03-18 01:56:46] [ 40%] Built target log_initialize
[2026-03-18 01:56:46] [ 40%] Linking CXX static library libabsl_random_internal_distribution_test_util.a
[2026-03-18 01:56:46] [ 40%] Built target random_internal_distribution_test_util
[2026-03-18 01:56:46] [ 40%] Linking CXX static library libabsl_crc32c.a
[2026-03-18 01:56:46] [ 40%] Built target crc32c
[2026-03-18 01:56:46] [ 41%] Building CXX object third_party/abseil-cpp/absl/crc/CMakeFiles/crc_cord_state.dir/internal/crc_cord_state.cc.o
[2026-03-18 01:56:47] [ 42%] Linking CXX static library libabsl_flags_marshalling.a
[2026-03-18 01:56:47] [ 42%] Linking CXX static library libabsl_log_internal_format.a
[2026-03-18 01:56:47] [ 42%] Built target log_internal_format
[2026-03-18 01:56:47] [ 42%] Built target flags_marshalling
[2026-03-18 01:56:47] [ 42%] Linking CXX static library libabsl_synchronization.a
[2026-03-18 01:56:47] [ 42%] Built target synchronization
[2026-03-18 01:56:47] [ 43%] Building CXX object third_party/abseil-cpp/absl/container/CMakeFiles/hashtablez_sampler.dir/internal/hashtablez_sampler.cc.o
[2026-03-18 01:56:47] [ 43%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_internal_log_sink_set.dir/internal/log_sink_set.cc.o
[2026-03-18 01:56:47] [ 43%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_program_name.dir/internal/program_name.cc.o
[2026-03-18 01:56:47] [ 43%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cordz_handle.dir/internal/cordz_handle.cc.o
[2026-03-18 01:56:47] [ 44%] Building CXX object third_party/abseil-cpp/absl/container/CMakeFiles/hashtablez_sampler.dir/internal/hashtablez_sampler_force_weak_definition.cc.o
[2026-03-18 01:56:48] [ 44%] Linking CXX static library libabsl_crc_cord_state.a
[2026-03-18 01:56:48] [ 44%] Built target crc_cord_state
[2026-03-18 01:56:48] [ 44%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord_internal.dir/internal/cord_rep_btree.cc.o
[2026-03-18 01:56:48] [ 44%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord_internal.dir/internal/cord_internal.cc.o
[2026-03-18 01:56:48] [ 44%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord_internal.dir/internal/cord_rep_crc.cc.o
[2026-03-18 01:56:48] [ 45%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord_internal.dir/internal/cord_rep_btree_reader.cc.o
[2026-03-18 01:56:48] [ 45%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord_internal.dir/internal/cord_rep_consume.cc.o
[2026-03-18 01:56:48] [ 45%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord_internal.dir/internal/cord_rep_ring.cc.o
[2026-03-18 01:56:48] [ 45%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord_internal.dir/internal/cord_rep_btree_navigator.cc.o
[2026-03-18 01:56:48] [ 45%] Linking CXX static library libabsl_flags_program_name.a
[2026-03-18 01:56:48] [ 45%] Built target flags_program_name
[2026-03-18 01:56:48] [ 46%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_config.dir/usage_config.cc.o
[2026-03-18 01:56:48] [ 46%] Linking CXX static library libabsl_cordz_handle.a
[2026-03-18 01:56:48] [ 46%] Linking CXX static library libabsl_hashtablez_sampler.a
[2026-03-18 01:56:48] [ 46%] Built target cordz_handle
[2026-03-18 01:56:48] [ 46%] Built target hashtablez_sampler
[2026-03-18 01:56:48] [ 46%] Linking CXX static library libabsl_log_internal_log_sink_set.a
[2026-03-18 01:56:48] [ 46%] Building CXX object third_party/abseil-cpp/absl/container/CMakeFiles/raw_hash_set.dir/internal/raw_hash_set.cc.o
[2026-03-18 01:56:48] [ 46%] Built target log_internal_log_sink_set
[2026-03-18 01:56:48] [ 88%] Linking CXX shared library libes_transformer_cust.so
[2026-03-18 01:56:48] [ 46%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_internal_message.dir/internal/log_message.cc.o
[2026-03-18 01:56:48] [ 94%] Built target es_transformer_cust_so
[2026-03-18 01:56:48] [100%] Installing ES package 'es_transformer_cust' to /data/kdy/cann/ops-transformer/build/es_packages
[2026-03-18 01:56:48] Installing ES package: es_transformer_cust
[2026-03-18 01:56:48] -- Copied header: es_MoeTokenPermuteWithRoutingMap.h
[2026-03-18 01:56:48] -- Copied header: es_MoeTokenPermuteWithRoutingMap_c.h
[2026-03-18 01:56:48] -- Copied header: es_log.h
[2026-03-18 01:56:48] -- Copied header: es_transformer_cust_ops.h
[2026-03-18 01:56:48] -- Copied header: es_transformer_cust_ops_c.h
[2026-03-18 01:56:49] [100%] Built target install_es_transformer_cust
[2026-03-18 01:56:49] -- [Smart Build]  Successfully built ES package 'es_transformer_cust' with 1 source file(s)
[2026-03-18 01:56:49] Built target build_es_transformer_cust
[2026-03-18 01:56:49] Building CXX object CMakeFiles/cust_proto.dir/autogen/exc/moe_token_permute_with_routing_map_proto.cpp.o
[2026-03-18 01:56:49] [ 46%] Linking CXX static library libabsl_flags_config.a
[2026-03-18 01:56:49] [ 46%] Built target flags_config
[2026-03-18 01:56:49] [ 46%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_internal.dir/internal/flag.cc.o
[2026-03-18 01:56:50] [ 46%] Linking CXX static library libabsl_log_internal_message.a
[2026-03-18 01:56:50] [ 46%] Built target log_internal_message
[2026-03-18 01:56:50] [ 46%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_internal_check_op.dir/internal/check_op.cc.o
[2026-03-18 01:56:50] [ 46%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/die_if_null.dir/die_if_null.cc.o
[2026-03-18 01:56:50] [ 47%] Linking CXX static library libabsl_raw_hash_set.a
[2026-03-18 01:56:50] [ 48%] Linking CXX static library libabsl_cord_internal.a
[2026-03-18 01:56:50] [ 48%] Built target raw_hash_set
[2026-03-18 01:56:50] [ 48%] Built target cord_internal
[2026-03-18 01:56:50] [ 49%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cordz_info.dir/internal/cordz_info.cc.o
[2026-03-18 01:56:50] Linking CXX shared library libcust_opsproto_rt2.0.so
[2026-03-18 01:56:50] [ 49%] Linking CXX static library libabsl_flags_internal.a
[2026-03-18 01:56:50] Built target cust_proto
[2026-03-18 01:56:50] [ 49%] Built target flags_internal
[2026-03-18 01:56:51] [ 50%] Linking CXX static library libabsl_die_if_null.a
[2026-03-18 01:56:51] [ 50%] Built target die_if_null
[2026-03-18 01:56:51] [ 50%] Linking CXX static library libabsl_log_internal_check_op.a
[2026-03-18 01:56:51] [ 50%] Built target log_internal_check_op
[2026-03-18 01:56:51] [ 50%] Linking CXX static library libabsl_cordz_info.a
[2026-03-18 01:56:52] [ 50%] Built target cordz_info
[2026-03-18 01:56:52] [ 51%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cordz_sample_token.dir/internal/cordz_sample_token.cc.o
[2026-03-18 01:56:52] [ 51%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord.dir/cord_analysis.cc.o
[2026-03-18 01:56:52] [ 51%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord.dir/cord_buffer.cc.o
[2026-03-18 01:56:52] [ 51%] Building CXX object third_party/abseil-cpp/absl/strings/CMakeFiles/cord.dir/cord.cc.o
[2026-03-18 01:56:52] [ 51%] Linking CXX static library libabsl_cordz_sample_token.a
[2026-03-18 01:56:52] [ 51%] Built target cordz_sample_token
[2026-03-18 01:56:55] [ 51%] Linking CXX static library libabsl_cord.a
[2026-03-18 01:56:56] [ 51%] Built target cord
[2026-03-18 01:56:56] [ 52%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_reflection.dir/reflection.cc.o
[2026-03-18 01:56:56] [ 52%] Building CXX object third_party/abseil-cpp/absl/status/CMakeFiles/status.dir/status.cc.o
[2026-03-18 01:56:56] [ 52%] Building CXX object third_party/abseil-cpp/absl/status/CMakeFiles/status.dir/status_payload_printer.cc.o
[2026-03-18 01:56:56] [INFO] RUNTIME(3819058,python3):2026-03-18-01:56:43.277.369 [task_fail_callback_data_manager.cc:93] 3819058 TaskFailCallBackManager: Constructor.
[2026-03-18 01:56:56] Opc tool start working now, please wait for a moment.
[2026-03-18 01:56:56] [INFO] RUNTIME(3819058,python3):2026-03-18-01:56:56.712.275 [task_fail_callback_data_manager.cc:98] 3819058 ~TaskFailCallBackManager: Destructor.
[2026-03-18 01:56:56] [INFO] ATRACE(3819058,python3):2026-03-18-01:56:56.845.824 [stacktrace_signal.c:194](tid:3819058) unregister all signal handlers, can not capture signal.
[2026-03-18 01:56:56] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_7874a5d9ba91ea6ef7a06d577e840637 Done
[2026-03-18 01:56:56] /usr/bin/gmake
[2026-03-18 01:56:57] Built target moe_token_permute_with_routing_map_ascend910b_3
[2026-03-18 01:56:57] [INFO] RUNTIME(3819057,python3):2026-03-18-01:56:43.277.397 [task_fail_callback_data_manager.cc:93] 3819057 TaskFailCallBackManager: Constructor.
[2026-03-18 01:56:57] Opc tool start working now, please wait for a moment.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819057,python3):2026-03-18-01:56:57.130.700 [task_fail_callback_data_manager.cc:98] 3819057 ~TaskFailCallBackManager: Destructor.
[2026-03-18 01:56:57] [INFO] ATRACE(3819057,python3):2026-03-18-01:56:57.285.142 [stacktrace_signal.c:194](tid:3819057) unregister all signal handlers, can not capture signal.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819068,python3):2026-03-18-01:56:43.277.381 [task_fail_callback_data_manager.cc:93] 3819068 TaskFailCallBackManager: Constructor.
[2026-03-18 01:56:57] Opc tool start working now, please wait for a moment.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819068,python3):2026-03-18-01:56:57.009.724 [task_fail_callback_data_manager.cc:98] 3819068 ~TaskFailCallBackManager: Destructor.
[2026-03-18 01:56:57] [INFO] ATRACE(3819068,python3):2026-03-18-01:56:57.180.440 [stacktrace_signal.c:194](tid:3819068) unregister all signal handlers, can not capture signal.
[2026-03-18 01:56:57] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_ce90d820a26ded18768878970d7ddd3f Done
[2026-03-18 01:56:57] /usr/bin/gmake
[2026-03-18 01:56:57] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_f023cd8e1fdcb40f9bb20466aa054c11 Done
[2026-03-18 01:56:57] /usr/bin/gmake
[2026-03-18 01:56:57] Built target moe_token_permute_with_routing_map_ascend910b_5
[2026-03-18 01:56:57] Built target moe_token_permute_with_routing_map_ascend910b_2
[2026-03-18 01:56:57] [INFO] RUNTIME(3819075,python3):2026-03-18-01:56:43.277.396 [task_fail_callback_data_manager.cc:93] 3819075 TaskFailCallBackManager: Constructor.
[2026-03-18 01:56:57] Opc tool start working now, please wait for a moment.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819075,python3):2026-03-18-01:56:57.270.882 [task_fail_callback_data_manager.cc:98] 3819075 ~TaskFailCallBackManager: Destructor.
[2026-03-18 01:56:57] [INFO] ATRACE(3819075,python3):2026-03-18-01:56:57.468.748 [stacktrace_signal.c:194](tid:3819075) unregister all signal handlers, can not capture signal.
[2026-03-18 01:56:57] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_70c842871515ad30301d19504b276dca Done
[2026-03-18 01:56:57] /usr/bin/gmake
[2026-03-18 01:56:57] Built target moe_token_permute_with_routing_map_ascend910b_6
[2026-03-18 01:56:57] [INFO] RUNTIME(3819073,python3):2026-03-18-01:56:43.277.419 [task_fail_callback_data_manager.cc:93] 3819073 TaskFailCallBackManager: Constructor.
[2026-03-18 01:56:57] Opc tool start working now, please wait for a moment.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819073,python3):2026-03-18-01:56:57.467.329 [task_fail_callback_data_manager.cc:98] 3819073 ~TaskFailCallBackManager: Destructor.
[2026-03-18 01:56:57] [INFO] ATRACE(3819073,python3):2026-03-18-01:56:57.611.036 [stacktrace_signal.c:194](tid:3819073) unregister all signal handlers, can not capture signal.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819077,python3):2026-03-18-01:56:43.277.408 [task_fail_callback_data_manager.cc:93] 3819077 TaskFailCallBackManager: Constructor.
[2026-03-18 01:56:57] Opc tool start working now, please wait for a moment.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819077,python3):2026-03-18-01:56:57.463.945 [task_fail_callback_data_manager.cc:98] 3819077 ~TaskFailCallBackManager: Destructor.
[2026-03-18 01:56:57] [INFO] ATRACE(3819077,python3):2026-03-18-01:56:57.633.889 [stacktrace_signal.c:194](tid:3819077) unregister all signal handlers, can not capture signal.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819078,python3):2026-03-18-01:56:43.277.408 [task_fail_callback_data_manager.cc:93] 3819078 TaskFailCallBackManager: Constructor.
[2026-03-18 01:56:57] Opc tool start working now, please wait for a moment.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819078,python3):2026-03-18-01:56:57.383.366 [task_fail_callback_data_manager.cc:98] 3819078 ~TaskFailCallBackManager: Destructor.
[2026-03-18 01:56:57] [INFO] ATRACE(3819078,python3):2026-03-18-01:56:57.568.580 [stacktrace_signal.c:194](tid:3819078) unregister all signal handlers, can not capture signal.
[2026-03-18 01:56:57] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_73a356ee7e9b844f1cdeb0b21b426ed0 Done
[2026-03-18 01:56:57] /usr/bin/gmake
[2026-03-18 01:56:57] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_ec5c85f5a7829084c1a10c21f30ebf12 Done
[2026-03-18 01:56:57] /usr/bin/gmake
[2026-03-18 01:56:57] Built target moe_token_permute_with_routing_map_ascend910b_4
[2026-03-18 01:56:57] Built target moe_token_permute_with_routing_map_ascend910b_7
[2026-03-18 01:56:57] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_73a7f4889ad86134a7af69582b8c3716 Done
[2026-03-18 01:56:57] /usr/bin/gmake
[2026-03-18 01:56:57] Built target moe_token_permute_with_routing_map_ascend910b_0
[2026-03-18 01:56:57] [INFO] RUNTIME(3819055,python3):2026-03-18-01:56:43.277.374 [task_fail_callback_data_manager.cc:93] 3819055 TaskFailCallBackManager: Constructor.
[2026-03-18 01:56:57] Opc tool start working now, please wait for a moment.
[2026-03-18 01:56:57] [INFO] RUNTIME(3819055,python3):2026-03-18-01:56:57.564.842 [task_fail_callback_data_manager.cc:98] 3819055 ~TaskFailCallBackManager: Destructor.
[2026-03-18 01:56:57] [INFO] ATRACE(3819055,python3):2026-03-18-01:56:57.709.628 [stacktrace_signal.c:194](tid:3819055) unregister all signal handlers, can not capture signal.
[2026-03-18 01:56:57] [ascend910b] Generating MoeTokenPermuteWithRoutingMap_5226913cab1d55bc46a191277f25d0aa Done
[2026-03-18 01:56:57] /usr/bin/gmake
[2026-03-18 01:56:57] Built target moe_token_permute_with_routing_map_ascend910b_1
[2026-03-18 01:56:58] Built target moe_token_permute_with_routing_map_ascend910b
[2026-03-18 01:56:58] Generating binary/ascend910b/bin/binary_info_config.json
[2026-03-18 01:56:58] Built target moe_token_permute_with_routing_map
[2026-03-18 01:56:58] Built target ops_config_ascend910b
[2026-03-18 01:56:58] Built target ops_transformer_config
[2026-03-18 01:56:58] [ 52%] Linking CXX static library libabsl_status.a
[2026-03-18 01:56:58] Built target ops_transformer_kernel
[2026-03-18 01:56:58] [ 52%] Built target status
[2026-03-18 01:56:58] [ 52%] Building CXX object third_party/abseil-cpp/absl/status/CMakeFiles/statusor.dir/statusor.cc.o
[2026-03-18 01:56:59] [ 52%] Linking CXX static library libabsl_flags_reflection.a
[2026-03-18 01:56:59] [ 52%] Built target flags_reflection
[2026-03-18 01:56:59] [ 52%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags.dir/flag.cc.o
[2026-03-18 01:56:59] [ 52%] Linking CXX static library libabsl_statusor.a
[2026-03-18 01:56:59] [ 52%] Built target statusor
[2026-03-18 01:57:00] [ 52%] Linking CXX static library libabsl_flags.a
[2026-03-18 01:57:00] [ 52%] Built target flags
[2026-03-18 01:57:00] [ 52%] Building CXX object third_party/abseil-cpp/absl/log/CMakeFiles/log_flags.dir/flags.cc.o
[2026-03-18 01:57:00] [ 52%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/any_lite.cc.o
[2026-03-18 01:57:00] [ 52%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/arena.cc.o
[2026-03-18 01:57:00] [ 52%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/message_lite.cc.o
[2026-03-18 01:57:00] [ 52%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/io/io_win32.cc.o
[2026-03-18 01:57:00] [ 52%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/arenaz_sampler.cc.o
[2026-03-18 01:57:00] [ 52%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/generated_message_util.cc.o
[2026-03-18 01:57:00] [ 54%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/io/coded_stream.cc.o
[2026-03-18 01:57:00] [ 54%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/any_lite.cc.o
[2026-03-18 01:57:00] [ 54%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/compiler/importer.cc.o
[2026-03-18 01:57:00] [ 54%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/arenastring.cc.o
[2026-03-18 01:57:00] [ 55%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/wrappers.pb.cc.o
[2026-03-18 01:57:00] [ 55%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/io/zero_copy_stream_impl.cc.o
[2026-03-18 01:57:00] [ 55%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/any.pb.cc.o
[2026-03-18 01:57:00] [ 56%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/io/zero_copy_stream_impl_lite.cc.o
[2026-03-18 01:57:00] [ 56%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/map.cc.o
[2026-03-18 01:57:00] [ 56%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/repeated_ptr_field.cc.o
[2026-03-18 01:57:00] [ 56%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/arena_align.cc.o
[2026-03-18 01:57:00] [ 56%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/any.cc.o
[2026-03-18 01:57:00] [ 57%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/arenastring.cc.o
[2026-03-18 01:57:00] [ 57%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/raw_ptr.cc.o
[2026-03-18 01:57:00] [ 57%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/implicit_weak_message.cc.o
[2026-03-18 01:57:00] [ 57%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/arena_align.cc.o
[2026-03-18 01:57:00] [ 57%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/repeated_field.cc.o
[2026-03-18 01:57:00] [ 57%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/dynamic_message.cc.o
[2026-03-18 01:57:00] [ 57%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/inlined_string_field.cc.o
[2026-03-18 01:57:00] [ 58%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/descriptor_database.cc.o
[2026-03-18 01:57:00] [ 58%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/cpp_features.pb.cc.o
[2026-03-18 01:57:00] [ 58%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/extension_set.cc.o
[2026-03-18 01:57:00] [ 58%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/field_mask.pb.cc.o
[2026-03-18 01:57:00] [ 58%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/generated_message_tctable_gen.cc.o
[2026-03-18 01:57:00] [ 59%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/generated_enum_util.cc.o
[2026-03-18 01:57:00] [ 59%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_usage_internal.dir/internal/usage.cc.o
[2026-03-18 01:57:00] [ 61%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/wire_format_lite.cc.o
[2026-03-18 01:57:00] [ 61%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/descriptor.pb.cc.o
[2026-03-18 01:57:00] [ 61%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/feature_resolver.cc.o
[2026-03-18 01:57:00] [ 61%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/generated_enum_util.cc.o
[2026-03-18 01:57:00] [ 62%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/struct.pb.cc.o
[2026-03-18 01:57:00] [ 62%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/source_context.pb.cc.o
[2026-03-18 01:57:00] [ 62%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/type.pb.cc.o
[2026-03-18 01:57:00] [ 62%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/timestamp.pb.cc.o
[2026-03-18 01:57:00] [ 62%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/arenaz_sampler.cc.o
[2026-03-18 01:57:00] [ 62%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/generated_message_tctable_lite.cc.o
[2026-03-18 01:57:00] [ 62%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/printer.cc.o
[2026-03-18 01:57:00] [ 63%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/generated_message_tctable_full.cc.o
[2026-03-18 01:57:00] [ 63%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/duration.pb.cc.o
[2026-03-18 01:57:00] [ 64%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/api.pb.cc.o
[2026-03-18 01:57:00] [ 64%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/zero_copy_stream_impl_lite.cc.o
[2026-03-18 01:57:00] [ 64%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/generated_message_tctable_lite.cc.o
[2026-03-18 01:57:00] [ 64%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/implicit_weak_message.cc.o
[2026-03-18 01:57:00] [ 64%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/generated_message_util.cc.o
[2026-03-18 01:57:00] [ 65%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/io_win32.cc.o
[2026-03-18 01:57:00] [ 65%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/strtod.cc.o
[2026-03-18 01:57:00] [ 65%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/tokenizer.cc.o
[2026-03-18 01:57:00] [ 66%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/inlined_string_field.cc.o
[2026-03-18 01:57:00] [ 66%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/gzip_stream.cc.o
[2026-03-18 01:57:00] [ 66%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/empty.pb.cc.o
[2026-03-18 01:57:00] [ 66%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/compiler/parser.cc.o
[2026-03-18 01:57:00] [ 66%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/stubs/common.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/zero_copy_stream.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/coded_stream.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/io/zero_copy_stream.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/generated_message_bases.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/descriptor.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/parse_context.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/json/internal/parser.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/json/internal/lexer.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/map_field.cc.o
[2026-03-18 01:57:00] [ 67%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/map.cc.o
[2026-03-18 01:57:00] [ 68%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/message_lite.cc.o
[2026-03-18 01:57:00] [ 68%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/arena.cc.o
[2026-03-18 01:57:00] [ 68%] Building CXX object CMakeFiles/libprotobuf-lite.dir/src/google/protobuf/port.cc.o
[2026-03-18 01:57:00] [ 69%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/json/internal/zero_copy_buffered_stream.cc.o
[2026-03-18 01:57:00] [ 69%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/repeated_ptr_field.cc.o
[2026-03-18 01:57:00] [ 69%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/parse_context.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/extension_set_heavy.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/extension_set.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/json/internal/unparser.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/generated_message_reflection.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/json/json.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/reflection_ops.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/raw_ptr.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/zero_copy_sink.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/internal_message_util.cc.o
[2026-03-18 01:57:00] [ 70%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/io/zero_copy_stream_impl.cc.o
[2026-03-18 01:57:00] [ 71%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/reflection_mode.cc.o
[2026-03-18 01:57:00] [ 71%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/text_format.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/json/internal/untyped_message.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/json/internal/writer.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/service.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/unknown_field_set.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/util/field_mask_util.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/message.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/util/message_differencer.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/util/time_util.cc.o
[2026-03-18 01:57:00] [ 72%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/port.cc.o
[2026-03-18 01:57:00] [ 73%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/json/internal/message_path.cc.o
[2026-03-18 01:57:00] [ 73%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/util/field_comparator.cc.o
[2026-03-18 01:57:00] [ 73%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/wire_format_lite.cc.o
[2026-03-18 01:57:00] [ 74%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/wire_format.cc.o
[2026-03-18 01:57:00] [ 74%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/util/type_resolver_util.cc.o
[2026-03-18 01:57:00] [ 74%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/repeated_field.cc.o
[2026-03-18 01:57:00] [ 74%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/util/delimited_message_util.cc.o
[2026-03-18 01:57:00] [ 75%] Building CXX object CMakeFiles/libprotobuf.dir/src/google/protobuf/stubs/common.cc.o
[2026-03-18 01:57:01] [ 75%] Linking CXX static library libabsl_log_flags.a
[2026-03-18 01:57:01] [ 75%] Built target log_flags
[2026-03-18 01:57:03] [ 75%] Linking CXX static library libabsl_flags_usage_internal.a
[2026-03-18 01:57:03] [ 75%] Built target flags_usage_internal
[2026-03-18 01:57:03] [ 76%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_usage.dir/usage.cc.o
[2026-03-18 01:57:03] [ 76%] Linking CXX static library libabsl_flags_usage.a
[2026-03-18 01:57:03] [ 76%] Built target flags_usage
[2026-03-18 01:57:04] [ 76%] Building CXX object third_party/abseil-cpp/absl/flags/CMakeFiles/flags_parse.dir/parse.cc.o
[2026-03-18 01:57:07] [ 76%] Linking CXX static library libabsl_flags_parse.a
[2026-03-18 01:57:07] [ 76%] Built target flags_parse
[2026-03-18 01:57:10] [ 76%] Linking CXX static library libascend_protobuf-lite.a
[2026-03-18 01:57:10] [ 76%] Built target libprotobuf-lite
[2026-03-18 01:57:24] [ 76%] Linking CXX static library libascend_protobuf.a
[2026-03-18 01:57:25] [ 76%] Built target libprotobuf
[2026-03-18 01:57:25] [ 76%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/allowlists/editions.cc.o
[2026-03-18 01:57:25] [ 76%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/allowlists/empty_package.cc.o
[2026-03-18 01:57:25] [ 77%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/allowlists/open_enum.cc.o
[2026-03-18 01:57:25] [ 77%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/allowlists/unused_imports.cc.o
[2026-03-18 01:57:25] [ 77%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/allowlists/weak_imports.cc.o
[2026-03-18 01:57:25] [ 77%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/command_line_interface.cc.o
[2026-03-18 01:57:25] [ 77%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/code_generator.cc.o
[2026-03-18 01:57:25] [ 77%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/enum.cc.o
[2026-03-18 01:57:25] [ 77%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/field.cc.o
[2026-03-18 01:57:25] [ 77%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/helpers.cc.o
[2026-03-18 01:57:25] [ 78%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/field_generators/map_field.cc.o
[2026-03-18 01:57:25] [ 79%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/extension.cc.o
[2026-03-18 01:57:25] [ 79%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/field_generators/primitive_field.cc.o
[2026-03-18 01:57:25] [ 79%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_enum.cc.o
[2026-03-18 01:57:25] [ 79%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/field_generators/cord_field.cc.o
[2026-03-18 01:57:25] [ 80%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/generator.cc.o
[2026-03-18 01:57:25] [ 80%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/field_generators/enum_field.cc.o
[2026-03-18 01:57:25] [ 80%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_message_field.cc.o
[2026-03-18 01:57:25] [ 80%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/field_generators/message_field.cc.o
[2026-03-18 01:57:25] [ 80%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_map_field.cc.o
[2026-03-18 01:57:25] [ 81%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_message.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/parse_function_generator.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/service.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/file.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/tracker.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_doc_comment.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/field_generators/string_field.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_primitive_field.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/doc_comment.cc.o
[2026-03-18 01:57:25] [ 82%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/enum.cc.o
[2026-03-18 01:57:25] [ 83%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/enum_field.cc.o
[2026-03-18 01:57:25] [ 83%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/generator.cc.o
[2026-03-18 01:57:25] [ 83%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/message.cc.o
[2026-03-18 01:57:25] [ 83%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_generator.cc.o
[2026-03-18 01:57:25] [ 83%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/cpp/padding_optimizer.cc.o
[2026-03-18 01:57:25] [ 83%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_source_generator_base.cc.o
[2026-03-18 01:57:25] [ 83%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/context.cc.o
[2026-03-18 01:57:25] [ 84%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/names.cc.o
[2026-03-18 01:57:25] [ 84%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/generator_factory.cc.o
[2026-03-18 01:57:25] [ 84%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/extension.cc.o
[2026-03-18 01:57:25] [ 85%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/java_features.pb.cc.o
[2026-03-18 01:57:25] [ 86%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/enum_field_lite.cc.o
[2026-03-18 01:57:25] [ 86%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/map_field_lite.cc.o
[2026-03-18 01:57:25] [ 86%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_helpers.cc.o
[2026-03-18 01:57:25] [ 86%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_field_base.cc.o
[2026-03-18 01:57:25] [ 86%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/extension_lite.cc.o
[2026-03-18 01:57:25] [ 86%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/message_builder.cc.o
[2026-03-18 01:57:25] [ 87%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/message.cc.o
[2026-03-18 01:57:25] [ 87%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_reflection_class.cc.o
[2026-03-18 01:57:25] [ 87%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/message_serialization.cc.o
[2026-03-18 01:57:25] [ 87%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_repeated_primitive_field.cc.o
[2026-03-18 01:57:25] [ 87%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/enum_lite.cc.o
[2026-03-18 01:57:25] [ 87%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_repeated_message_field.cc.o
[2026-03-18 01:57:25] [ 87%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/service.cc.o
[2026-03-18 01:57:25] [ 87%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_enum_field.cc.o
[2026-03-18 01:57:25] [ 88%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_wrapper_field.cc.o
[2026-03-18 01:57:25] [ 88%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/message_field.cc.o
[2026-03-18 01:57:25] [ 88%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/csharp/csharp_repeated_enum_field.cc.o
[2026-03-18 01:57:25] [ 88%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/kotlin_generator.cc.o
[2026-03-18 01:57:25] [ 88%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/file.cc.o
[2026-03-18 01:57:25] [ 88%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/primitive_field_lite.cc.o
[2026-03-18 01:57:25] [ 88%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/helpers.cc.o
[2026-03-18 01:57:25] [ 89%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/field.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/message_lite.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/shared_code_generator.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/names.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/message_field_lite.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/map_field.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/enum.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/file.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/helpers.cc.o
[2026-03-18 01:57:25] [ 90%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/generator.cc.o
[2026-03-18 01:57:25] [ 91%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/primitive_field.cc.o
[2026-03-18 01:57:25] [ 91%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/message_field.cc.o
[2026-03-18 01:57:25] [ 91%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/php/names.cc.o
[2026-03-18 01:57:25] [ 91%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/primitive_field.cc.o
[2026-03-18 01:57:25] [ 91%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/names.cc.o
[2026-03-18 01:57:25] [ 92%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/oneof.cc.o
[2026-03-18 01:57:25] [ 92%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/message_builder_lite.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/line_consumer.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/string_field.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/extension.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/enum_field.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/name_resolver.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/retention.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/map_field.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/plugin.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/text_format_decode_data.cc.o
[2026-03-18 01:57:25] [ 93%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/accessors/accessors.cc.o
[2026-03-18 01:57:25] [ 94%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/field.cc.o
[2026-03-18 01:57:25] [ 94%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/import_writer.cc.o
[2026-03-18 01:57:25] [ 95%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/objectivec/message.cc.o
[2026-03-18 01:57:25] [ 95%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/java/string_field_lite.cc.o
[2026-03-18 01:57:25] [ 96%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/php/php_generator.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/generator.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/accessors/unsupported_field.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/accessors/singular_scalar.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/oneof.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/python/generator.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/message.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/context.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/accessors/singular_string.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/ruby/ruby_generator.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/zip_writer.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/python/helpers.cc.o
[2026-03-18 01:57:25] [ 97%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/plugin.pb.cc.o
[2026-03-18 01:57:25] [ 98%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/python/pyi_generator.cc.o
[2026-03-18 01:57:25] [ 99%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/accessors/singular_message.cc.o
[2026-03-18 01:57:25] [100%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/subprocess.cc.o
[2026-03-18 01:57:25] [100%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/naming.cc.o
[2026-03-18 01:57:25] [100%] Building CXX object CMakeFiles/libprotoc.dir/src/google/protobuf/compiler/rust/relative_path.cc.o
[2026-03-18 01:57:43] [100%] Linking CXX static library libascend_protoc.a
[2026-03-18 01:57:43] [100%] Built target libprotoc
[2026-03-18 01:57:43] [100%] Building CXX object CMakeFiles/protoc.dir/src/google/protobuf/compiler/main.cc.o
[2026-03-18 01:57:47] [100%] Linking CXX executable protoc
[2026-03-18 01:57:47] [100%] Built target protoc
[2026-03-18 01:57:47] No install step for 'ascend_protobuf_build_transformer'
[2026-03-18 01:57:47] Completed 'ascend_protobuf_build_transformer'
[2026-03-18 01:57:47] Built target ascend_protobuf_build_transformer
[2026-03-18 01:57:47] Running C++ protocol buffer compiler on /data/kdy/cann/pkg/cann-9.0.0/include/proto/ge_onnx.proto
[2026-03-18 01:57:47] generate proto cpp_out onnx by /data/kdy/cann/pkg/cann-9.0.0/include/proto/ge_onnx.proto
[2026-03-18 01:57:47] Building CXX object common/src/framework/CMakeFiles/op_transformer_onnx_plugin_obj.dir/embedding_bag_onnx_plugin.cpp.o
[2026-03-18 01:57:47] Building CXX object common/src/framework/CMakeFiles/op_transformer_onnx_plugin_obj.dir/fillwindowcache_onnx_plugin.cpp.o
[2026-03-18 01:57:47] Building CXX object common/src/framework/CMakeFiles/op_transformer_onnx_plugin_obj.dir/npu_fused_attention_score_onnx_plugin.cpp.o
[2026-03-18 01:57:47] Building CXX object common/src/framework/CMakeFiles/op_transformer_onnx_plugin_obj.dir/npu_scaled_masked_softmax_onnx_plugin.cpp.o
[2026-03-18 01:57:47] Building CXX object common/src/framework/CMakeFiles/op_transformer_onnx_plugin_obj.dir/tfidf_vectorizer_onnx_plugin.cpp.o
[2026-03-18 01:57:47] Building CXX object common/src/framework/CMakeFiles/op_transformer_onnx_plugin_obj.dir/npu_masked_softmax_with_relposbias_onnx_plugin.cpp.o
[2026-03-18 01:57:47] Building CXX object common/src/framework/CMakeFiles/op_transformer_onnx_plugin_obj.dir/multi_head_attention_onnx_plugin.cpp.o
[2026-03-18 01:57:47] Building CXX object common/src/framework/CMakeFiles/op_transformer_onnx_plugin_obj.dir/npu_fused_attention_score_fwd_onnx_plugin.cpp.o
[2026-03-18 01:57:51] Built target op_transformer_onnx_plugin_obj
[2026-03-18 01:57:51] Run CPack packaging tool...
[2026-03-18 01:57:51] CPack: Create package using External
[2026-03-18 01:57:51] CPack: Install projects
[2026-03-18 01:57:51] CPack: - Run preinstall target for: cann_ops-transformer
[2026-03-18 01:58:00] CPack: - Install project: cann_ops-transformer []
[2026-03-18 01:58:00] CPack: Create package
[2026-03-18 01:58:00] 
[2026-03-18 01:58:00] About to compress 19164 KB of data...
[2026-03-18 01:58:00] Adding files to archive named "cann-ops-transformer-custom_linux-aarch64.run"...
[2026-03-18 01:58:01] ./help.info
[2026-03-18 01:58:01] ./install.sh
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_api/include/aclnnop/aclnn_moe_token_permute_with_routing_map.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_api/include/aclnnop/aclnn_ops_transformer_custom.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_api/lib/libcust_opapi.so
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/CMakeLists.txt
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/add_rms_norm/op_host/add_rms_norm_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/CMakeLists.txt
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/batch_mat_mul_v3_def.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/batch_mat_mul_v3_infershape.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/config/ascend310p/batch_mat_mul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/config/ascend310p/batch_mat_mul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/config/ascend910_93/batch_mat_mul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/config/ascend910_93/batch_mat_mul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/config/ascend910b/batch_mat_mul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/config/ascend910b/batch_mat_mul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/config/ascend950/batch_mat_mul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/config/ascend950/batch_mat_mul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_al1_full_load_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_al1_full_load_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_basic_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_basic_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_bl1_full_load_basic_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_bl1_full_load_basic_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_bl1_full_load_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_bl1_full_load_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_common_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_iterbatch_basicapi_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_iterbatch_basicapi_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_iterbatch_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_iterbatch_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_advanced.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_strategy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_base_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_base_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_asw_al1_full_load_kernel_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_asw_bl1_full_load_kernel_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_asw_block_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_asw_kernel_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_dasw_block_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_iterbatch_basicapi_act.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_iterbatch_basicapi_block_scheduler.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_iterbatch_block_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/arch35/batch_mat_mul_v3_iterbatch_kernel_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_nd2nz.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_apt.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_com_base_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_com_base_block_struct.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_data.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_key.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/CMakeLists.txt
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/math_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/matmul_common_infershape.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/matmul_common_infershape.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_api/cube_util.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_api/cube_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_api/matmul_util.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_api/matmul_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_tiling/debug_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_tiling/debug_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_tiling/hash.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_tiling/hash.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_tiling/lock.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_tiling/lock.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_tiling/tiling_cache.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/common/op_host/op_tiling/tiling_type.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_kernel/gqmm_tiling_key.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_kernel/grouped_matmul_tiling_data_apt.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_kernel/grouped_matmul_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_kernel/quant_block_sch.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_kernel/quant_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_tiling/grouped_matmul_host_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul/op_tiling/grouped_matmul_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/CMakeLists.txt
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend310p/Ascend310P3_8_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend310p/Ascend310P5_8_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend310p/Ascend310P7_8_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend310p/mat_mul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend310p/mat_mul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/Ascend910_9372_20_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/Ascend910_9381_24_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/Ascend910_9382_16_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/Ascend910_9382_24_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/Ascend910_9391_24_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/Ascend910_9392_16_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/Ascend910_9392_24_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/mat_mul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910_93/mat_mul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910b/Ascend910B1_24_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910b/Ascend910B2C_24_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910b/Ascend910B2_24_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910b/Ascend910B3_20_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910b/Ascend910B4_20_AiCore_Mc2MatMulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910b/mat_mul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend910b/mat_mul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend950/mat_mul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/config/ascend950/mat_mul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/mat_mul_v3_def.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_base_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_basic_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_basic_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_full_load_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_full_load_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_base_tiling_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_stream_k_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_stream_k_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_advanced.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_advanced.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_data.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_key.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_common.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_l2_cache.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_l2_cache.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_simplifiedkey.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_tuning.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/block_scheduler_aswt.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/block_scheduler_streamk.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_dasw_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_fixpipe_opti.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_fixpipe_opti_basic_act.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_full_load.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_pingpong_basic_act.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_stream_k_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_stream_k_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_streamk_basic_act.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_v3_full_load_kernel_helper.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mm_extension_interface/mm_copy_cube_out.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/arch35/mm_extension_interface/mm_custom_mm_policy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/batch_mat_mul_nd2nz.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_base_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_base_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_cvp_base_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_l1_full_load.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_multi_core_splitk_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_nd2nz.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_nd2nz_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_nd2nz_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_optimized_fixpipe_algorithm.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_sc_splitk_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel_gm_to_l1.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_unaligned_base_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_unaligned_deterministic_splitk_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_unaligned_sc_splitk_kernel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_unaligned_sc_splitk_kernel_gm_to_l1.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_v3.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_v3_apt.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_v3_common.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_v3_tiling_data.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_v3_tiling_key.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/norm_common/op_kernel/reduce_common.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/CMakeLists.txt
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/aclnn_kernels/cast.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/aclnn_kernels/common/op_error_check.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/aclnn_kernels/contiguous.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/aclnn_kernels/pad.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/aclnn_kernels/reshape.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/aclnn_kernels/slice.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/aclnn_kernels/transdata.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/aclnn_kernels/transpose.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/add.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/axpy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/broadcast_to.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/dot.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/fill.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/mul.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/muls.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/padv3.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/reduce_mean.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/reduce_sum_op.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/shape_op.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/squeeze.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_api/level0/unsqueeze.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/CMakeLists.txt
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/hcom_topo_info.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/legacy_common_manager.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/legacy_common_manager.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/op_cache_def_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/op_cache_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/op_cache_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/register/tuning_bank_key_registry.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/register/tuning_tiling_reflection_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/register/tuning_tiling_registry.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/runtime_kb_api.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/ops_legacy/op_tiling/runtime_kb_api.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend310p/Ascend310P3_8_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend310p/Ascend310P5_8_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend310p/Ascend310P7_8_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend310p/quant_batch_matmul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend310p/quant_batch_matmul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910_93/Ascend910_9381_24_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910_93/Ascend910_9382_24_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910_93/Ascend910_9391_24_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910_93/Ascend910_9391_25_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910_93/Ascend910_9392_24_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910_93/Ascend910_9392_25_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910_93/quant_batch_matmul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910_93/quant_batch_matmul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/Ascend910B1_24_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/Ascend910B1_25_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/Ascend910B2C_24_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/Ascend910B2_24_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/Ascend910B3_20_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/Ascend910B4-1_20_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/Ascend910B4_20_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/quant_batch_matmul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend910b/quant_batch_matmul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend950/quant_batch_matmul_v3_binary.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/config/ascend950/quant_batch_matmul_v3_simplified_key.ini
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_checker.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_checker.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_info_factory.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_compile_info.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_cache.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_cache.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/mm_extension_interface/qbmm_copy_cube_out.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/mm_extension_interface/qbmm_custom_mm_policy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_api_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_asw_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly_al1_full_load.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic_al1_full_load.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_perblock.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_perblock_api_param_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_perblock_api_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_apt.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16_basic.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16_opt.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_cube_basic.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_init_output.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_basic.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_opt.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_tiling_data.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_tiling_key.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_update.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/rms_norm/op_kernel/rms_norm_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/arch/arch.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/arch/cross_core_sync.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/arch/local_tensor_buffer.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/arch/resource.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/catlass.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/coord.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/detail/alignment.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/detail/callback.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/detail/dependent_false.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/detail/macros.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/detail/tag_to_layout.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_elemwise_one_source.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_fa_rescal_o.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_fa_softmax.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_gemm.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_gemv.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_mla_fd_rescal_o.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_mla_rescal_o.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_mla_softmax.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_mla_tp1_rescal_o.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_mla_tp1_softmax.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/block/block_epilogue_per_token_dequant.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/dispatch_policy.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/copy_gm_to_ub.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/copy_ub_to_gm.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_broadcast_inplace_by_column.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_broadcast_inplace_by_row.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_broadcast_mul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_broadcast_one_blk.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_cast.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_copy.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_elemwise_add.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_elemwise_mul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_elemwise_muls.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_swizzle.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_fa_pv.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_fa_qk.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_gemm.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_mla_pv.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_mla_pv_tp1_spec.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_mla_qk.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_mla_qk_tp1_spec.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_pingpong.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_pingpong_bias.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_pingpong_tla.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_preload.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_preload_async.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_preload_async_with_callback.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad_preload_tla.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_swizzle.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/dispatch_policy.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/gemm_type.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/helper.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/basic_matmul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/basic_matmul_tla.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/batched_matmul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/gemm.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/group_gemm.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/grouped_matmul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/grouped_matmul_slice_k.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/grouped_matmul_slice_k_per_token_dequant.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/grouped_matmul_slice_m.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/grouped_matmul_slice_m_per_token_dequant.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/matmul_epilogue.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/optimized_matmul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/optimized_matmul_tla.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/padding_matmul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/quant_matmul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/quant_matmul_multistage_workspace.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/kernel/splitk_matmul.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/copy_gm_to_l1.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/copy_gm_to_ub.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/copy_l0c_to_gm.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/copy_l1_to_bt.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/copy_l1_to_l0a.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/copy_l1_to_l0b.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/copy_ub_to_gm.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/tile_copy.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm/tile/tile_mmad.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemm_coord.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/block/block_gemv.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/block/block_gemv_aic.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/block/block_gemv_aiv.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/helper.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/kernel/kernel_gemv_aic.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/kernel/kernel_gemv_aiv.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/tile/matrix_copy_gm_to_ub.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/tile/tile_copy.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/tile/tile_vmad.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/tile/tile_vmuls.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/tile/vec_copy_gm_to_ub.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv/tile/vec_copy_ub_to_gm.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/gemv_coord.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/layout/layout.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/layout/matrix.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/layout/vector.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/matrix_coord.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/numeric_size.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/template_linear_algebra/status.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/tla/int_tuple.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/tla/layout.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/tla/numeric/integer_sequence.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/tla/numeric/integral_constant.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/tla/numeric/math.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/tla/tensor.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/tla/tuple.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra/include/tla/type_traits.hpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v3.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_api/weight_quant_batch_matmul_v2.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_api/weight_quant_batch_matmul_v2.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_split_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_split_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_basic_block_table.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_basic_block_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_basic_block_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_reg_base_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_reg_base_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_checker_for_mmads8s4.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_checker_for_mmads8s4.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_compute_matmul_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_compute_matmul_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_fixpipe.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_fixpipe.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_key.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_msd.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_msd.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_msd_group.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_msd_group.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_registry.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_splitk.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_splitk.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_tool.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_tool.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_weight_nz_tiling.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_weight_nz_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_white_list.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/anti_quant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/block/block_decl.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/block/block_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/block/constant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/block/david_ub_antiquant_scmc.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/block/david_ub_antiquant_scmc_load_in_advance_aic_tail_resplit.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/block/david_ub_antiquant_scmc_load_in_advance_aiv_tail_resplit.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/constant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/dispatch_policy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/iterator/continuous_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/iterator/tail_resplit_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/kernel/david_wqbmm_load_in_advance.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/pipeline/pipeline_stage_mixcore.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/pipeline/pipeline_stage_singlecore.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/pipeline/pipeline_stage_singlecore_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/pipeline/pipeline_stage_singlecore_copy_in_advance.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/pipeline/pipeline_state.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/pipeline/utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/scheduler/tile_scheduler_tail_resplit.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/simd/a16w4_pergroup_kn_nz.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/tile/copy_gm_to_l1.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/tile/copy_gm_to_ub.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/tile/copy_ub_to_l1.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/utils/constant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/utils/device_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass/utils/math_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/catlass_convertor.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/anti_quant_y_vf.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/basic_block_config.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/basic_block_vf_mx.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/basic_block_vf_nd.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/basic_block_vf_nz.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/weight_quant_batch_matmul_v2_basic_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/weight_quant_batch_matmul_v2_basic_block_controller.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/weight_quant_batch_matmul_v2_cube_compute.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/weight_quant_batch_matmul_v2_vcv_basic_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/weight_quant_batch_matmul_v2_vec_compute.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/tool_arch35.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_asw_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_vf.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/fixpipe/static_diag_constant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/fixpipe/weight_quant_batch_matmul_v2_fixpipe.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/fixpipe/weight_quant_batch_matmul_v2_fixpipe_stage1.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/fixpipe/weight_quant_batch_matmul_v2_fixpipe_stage2.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/tool.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_apt.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_common.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_constant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom_mix_splitk.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom_nz_splitk.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom_weight_nz.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_group.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_multicore.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_split_k.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_tiling_data.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_weight_nz_performance.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_weight_nz_performance_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/attention_ffn_schedule.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_base_struct.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_sch.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_sch_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_sch_nddma.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_sch_one_dim.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_sch_one_dim_advance.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_sch_ub_broadcast.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_sch_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_tiling_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/broadcast_tiling_noncontiguous.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/last_transpose/broadcast_sch_base_last_transpose.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/last_transpose/broadcast_sch_last_transpose.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/last_transpose/broadcast_sch_util_last_transpose.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/nlast_transpose/broadcast_sch_base_nlast_transpose_naddma.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/nlast_transpose/broadcast_sch_nlast_transpose_nddma.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/nlast_transpose/broadcast_sch_nlast_transpose_ub_broadcast.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/nlast_transpose/broadcast_sch_nlast_transpose_ub_broadcast_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/nlast_transpose/broadcast_sch_util_nlast_transpose_naddma.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/transpose/broadcast_sch_base_transpose.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/transpose/broadcast_sch_nddma_transpose.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/broadcast/noncontiguous/transpose/broadcast_utils_transpose.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/elewise/elewise_base_struct.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/elewise/elewise_sch.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/elewise/elewise_sch_16b.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/elewise/elewise_sch_with_scalar.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/elewise/elewise_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_operator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_sch.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_sch_aux.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_sch_aux_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_sch_aux_non_contiguous.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_sch_aux_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_tensor_empty.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_tensor_move.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_tiling_data.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_tiling_key_decl.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_tiling_key_decl_non_contiguous.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_tiling_key_sel.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_tiling_key_sel_non_contiguous.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/reduce/reduce_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/aux1.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/aux2.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/broadcast_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/buffer.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/dag.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/dfx.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/elems.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/log.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/loopbuf.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/node.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/placeholder.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/sync.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/atvoss/util/vec.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_grouped_matmul_builder.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad_builder.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad_multi_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad_multi_block_bias.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad_mx.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad_pertile.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad_pertile_param.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mmad_with_params.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_mx_mm_aic_to_aiv_builder.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_quant_matmul_builder.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_quant_mmad_multi_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_quant_with_tile_mmad_multi_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_scheduler_grouped_matmul_aswt.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_scheduler_policy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/block_scheduler_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/block/matmul_impl_traits.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/block_epilogue_dequant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/block_epilogue_dequant_swiglu.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/block_epilogue_empty.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/block_epilogue_finalize_routing.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/block_epilogue_pertile.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/block_epilogue_pertoken_quant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/block_epilogue_swiglu_mx_quant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/epilogue/fusion/default_fusion_op.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_gmm_finalize_routing.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_gmm_swiglu_pertoken_quant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_grouped_matmul.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_grouped_matmul_add.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_qgmm_inplace_add.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_qgmm_inplace_add_mix_online_dynamic.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/kernel_qgmm_pertile.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/kernel/semaphore.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/policy/dispatch_policy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/prologue/block_prologue_finalize_routing.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_in/copy_in_with_layout.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_in/copy_in_with_params.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_out/copy_out_enunitflag_with_layout.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_out/copy_out_split_m_with_params.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_out/copy_out_split_n_with_params.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_out/copy_out_to_gm_with_params_v310.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_out/copy_out_to_ub_with_params_v310.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_out/copy_out_with_layout.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/copy_out/copy_out_with_params_v220.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/load_data/copy_a1_to_a2_with_layout.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/load_data/copy_b1_to_b2_sparse_with_layout.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/load_data/copy_b1_to_b2_with_layout.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/load_data/load_to_l0_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/load_data/tbuf_pool_l0_default.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/tile_copy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/tile/tile_copy_policy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/arch.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/common_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/coord_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/device_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/fill_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/gemm_type.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/grouped_matmul_constant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/host_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/integral_constant.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/layout_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/matmul_layout_type.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/status_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/tensor_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/tuple_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/cgmct/utils/underscore.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/common.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/common_func.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/dropmask.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/gm_to_l1_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/gm_to_ub_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/hardware.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/all_gather_matmul_tiling.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/gm_ub_gm_copy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/mc2_common_def.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/mc2_kernel_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/mc2_matmul_block.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/mc2_matmul_block_l2cache.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/mc2_matmul_compute.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/mc2_moe_context.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/mc2_nd_to_nz.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/mc2_tiling_struct.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/moe_distribute_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/moe_distribute_comm_ctx.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/qbmm_asw_block_noncontiguous.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/qbmm_mix_perblock_noncontiguous.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/qbmm_perblock_api_utils_noncontiguous.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/reduce_sum.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/reduce_sum_cast_fp32.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/inc/kernel/reduce_sum_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/l0c_to_gm_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/l0c_to_l1_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/l0c_to_ub_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/l1_to_bt_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/l1_to_fb_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/l1_to_l0_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/l1_to_ub_iterator.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/layout.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/mem.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/mma.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/op_kernel/load_store_utils.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/op_kernel/math_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/op_kernel/platform_util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/pse.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/simd.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/util.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_gather_v2/
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/masked_select_v3.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_common.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_index_copy.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_index_copy_split_d.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_mrgsort.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_mrgsort_out.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_permute_prob.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_sort_base.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_sort_multi_core.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_sort_multi_core_last_dim.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_sort_one_core.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/moe_token_permute_with_routing_map/moe_token_permute_with_routing_map.cpp
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/moe_token_permute_with_routing_map.py
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_5226913cab1d55bc46a191277f25d0aa.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_5226913cab1d55bc46a191277f25d0aa.o
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_70c842871515ad30301d19504b276dca.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_70c842871515ad30301d19504b276dca.o
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_73a356ee7e9b844f1cdeb0b21b426ed0.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_73a356ee7e9b844f1cdeb0b21b426ed0.o
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_73a7f4889ad86134a7af69582b8c3716.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_73a7f4889ad86134a7af69582b8c3716.o
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_7874a5d9ba91ea6ef7a06d577e840637.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_7874a5d9ba91ea6ef7a06d577e840637.o
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_ce90d820a26ded18768878970d7ddd3f.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_ce90d820a26ded18768878970d7ddd3f.o
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_ec5c85f5a7829084c1a10c21f30ebf12.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_ec5c85f5a7829084c1a10c21f30ebf12.o
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_f023cd8e1fdcb40f9bb20466aa054c11.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/moe_token_permute_with_routing_map/MoeTokenPermuteWithRoutingMap_f023cd8e1fdcb40f9bb20466aa054c11.o
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910b/moe_token_permute_with_routing_map.json
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/liboptiling.so
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_proto/es/include/es_MoeTokenPermuteWithRoutingMap.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_proto/es/include/es_MoeTokenPermuteWithRoutingMap_c.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_proto/es/include/es_log.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_proto/es/include/es_transformer_cust_ops.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_proto/es/include/es_transformer_cust_ops_c.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_proto/es/lib/linux/aarch64/libes_transformer_cust.so
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_proto/inc/moe_token_permute_with_routing_map_proto.h
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
[2026-03-18 01:58:01] ./packages/vendors/custom_transformer/version.info
[2026-03-18 01:58:01] ./uninstall.sh
[2026-03-18 01:58:01] ./upgrade.sh
[2026-03-18 01:58:01] CRC: 1956391783
[2026-03-18 01:58:01] SHA256: 98f58667b895f015bbd19571872736f7c1cafcb713693c7e00a75e288f18e334
[2026-03-18 01:58:01] Skipping md5sum at user request
[2026-03-18 01:58:01] 
[2026-03-18 01:58:01] Self-extractable archive "cann-ops-transformer-custom_linux-aarch64.run" successfully created.
[2026-03-18 01:58:01] Copy /data/kdy/cann/ops-transformer/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/cann-ops-transformer-custom_linux-aarch64.run to /data/kdy/cann/ops-transformer/build/
[2026-03-18 01:58:01] Copy /data/kdy/cann/ops-transformer/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/cann-ops-transformer-custom_linux-aarch64.run to /data/kdy/cann/ops-transformer/build_out/
[2026-03-18 01:58:01] CPack: - package: /data/kdy/cann/ops-transformer/build/cann-ops-transformer-custom_linux-aarch64.run.json generated.
[2026-03-18 01:58:01] CPack: - package: /data/kdy/cann/ops-transformer/build/cann-ops-transformer-custom_linux-aarch64.run generated.
