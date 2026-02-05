#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "moe_distribute_dispatch_shmem_tiling.h"
#include "shmem.h"



__aicore__ inline int64_t GetShmemDataAddr(__gm__ uint8_t *shmemSpace, int32_t pe) {
    return (int64_t) aclshmem_ptr(shmemSpace, pe);
}

__aicore__ inline int64_t GetShmemSignalAddr(__gm__ uint8_t *shmemSpace, int32_t pe) {
    return (int64_t) aclshmem_ptr(shmemSpace, pe) + 1022 * 1024 * 1024;
}