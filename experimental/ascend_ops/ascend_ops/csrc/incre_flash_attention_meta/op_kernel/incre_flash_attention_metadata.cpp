#include "./acl/acl.h"
#include "split_core.h"
#include "aicpu_api.h"

using namespace AscendC;

extern "C" __global__ __aicpu__ uint32_t IncreFlashAttentionMetadataKernel(void *args)
{
    aicpu::kernels::IncreFlashAttentionMetadataArgs* arg_ptr = (aicpu::kernels::IncreFlashAttentionMetadataArgs *)args;
    AscendC::printf("Inside aicpu kernel!!\n");
    // AscendC::printf("batch_size %u!!\n", arg_ptr->batchSize);
    // AscendC::printf("layout %s!!\n", arg_ptr->layoutQuery);

    // int64_t *actSeqQLen = (int64_t*)arg_ptr->actSeqQLen;
    aicpu::kernels::SplitCore balancer;
    // balancer.Compute(arg_ptr);
    return 0;
}
