# 备注

```
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.610 [api_impl.cc:7751]695120 GetResInCurrentThread:ctx is NULL!
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.631 [api_c_stream.cc:749]695120 rtsGetResInCurrentThread:ErrCode=107002, desc=[the context is a null pointer], InnerCode=0x7070001
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.634 [error_message_manage.cc:55]695120 FuncErrorReason:rtsGetResInCurrentThread execution failed, the context is a null pointer.
[2026-03-19 15:04:57] [ERROR] ASCENDCL(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.642 [device.cpp:528]695120 aclrtGetResInCurrentThreadImpl:call rtsGetResInCurrentThread failed, runtime result = 107002.
[2026-03-19 15:04:57] [       OK ] prompt_flash_attention_v3_opapi_ut.prompt_flash_attention_v3_aclnn_1 (0 ms)
```

lcov op_api里面报上面的日志内容属于正常现象，不必检查此错误，重点排查其他错误问题

项目里面我负责的主要模块是

-   attention/prompt_flash_attention
-   attention/lightning_indexer
-   attention/sparse_flash_attention

这是三个算子，我的任务是写算子目录下的tests/ut，例如attention/prompt_flash_attention/tests/ut，你只能动这些目录下的文件，不能改算子原来的代码

lcov看护覆盖率中：

ut下面的op_host/arch32看护的是算子目录下的op_host/*tiling.cpp文件，例如attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp

ut下面的op_api看护的是算子目录下op_host/op_api/里面的算子接口，例如attention/prompt_flash_attention/op_host/op_api/aclnn_prompt_flash_attention.cpp

./log/下面是运行覆盖率命令后的日志文件，例如log/li_opapi_ut.log就是lightning_indexer关注op_api下面算子接口的覆盖日志

你的所有回答全部写在这个文件，包括代码。

# 问题

1、读取这个tests/ut/framework_normal/common/mc2_hcom_topology_mocker.cpp，这个文件是对mc2/common/utils/mc2_hcom_topo_info.h的打桩，op_host ut运行时，不会编mc2/common/utils/mc2_hcom_topo_info.cpp，而是编打桩文件（tests/ut/framework_normal/common/mc2_hcom_topology_mocker.cpp）中的打桩函数，请问这是怎么实现的（读取tests/），以及我想自己实现一个打桩，针对attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp中6244和6545的GetSoftMaxMinTmpSize和GetSoftMaxFlashV2MinTmpSize实现一个自己的api接口，这两个api接口时/home/j60100428/Ascend_0320/cann-9.0.0/include/tiling/tiling_api.h头文件中的\#include "activation/softmax_tiling.h"引入的。
