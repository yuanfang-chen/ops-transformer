#include <acl/acl.h>
#include <cstring>
#include <cmath>
#include "fa2_tiling.h"

extern "C" aclError aclrtlaunch_flash_attention2_do(
    uint32_t blockDim, aclrtStream stream,
    void* q, void* k, void* v, void* output,
    void* workspace, void* tilingGM);

static void* g_tilingDev = nullptr;
static void* g_workspcDev = nullptr;
static bool  g_inited = false;

static void ensureWorkspace() {
    if (g_inited) return;
    aclrtMalloc(&g_tilingDev, 512, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&g_workspcDev, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    g_inited = true;
}

extern "C" {

int fa2_forward(
    void* q, void* k, void* v, void* output,
    int batch, int heads, int seq_q, int seq_kv, int head_dim,
    float scale, int causal, void* stream_ptr)
{
    ensureWorkspace();

    int dAligned = ((head_dim + 7) / 8) * 8;
    int blockM = 32, blockN = 32;

    FA2TilingData tiling;
    memset(&tiling, 0, sizeof(tiling));
    tiling.batchSize     = batch;
    tiling.numHeads      = heads;
    tiling.seqLenQ       = seq_q;
    tiling.seqLenKV      = seq_kv;
    tiling.headDim       = head_dim;
    tiling.headDimAligned= dAligned;
    tiling.blockM        = blockM;
    tiling.blockN        = blockN;
    tiling.softmaxScale  = (scale > 0.f) ? scale : (1.f / sqrtf((float)head_dim));
    tiling.isCausal      = causal;
    tiling.numBlocksM    = (seq_q  + blockM - 1) / blockM;
    tiling.numBlocksN    = (seq_kv + blockN - 1) / blockN;
    tiling.totalCores    = batch * heads;

    aclrtStream st = (aclrtStream)stream_ptr;

    aclError ret = aclrtMemcpyAsync(
        g_tilingDev, sizeof(FA2TilingData),
        &tiling, sizeof(FA2TilingData),
        ACL_MEMCPY_HOST_TO_DEVICE, st);
    if (ret != 0) return (int)ret;

    ret = aclrtlaunch_flash_attention2_do(
        tiling.totalCores, st,
        q, k, v, output, g_workspcDev, g_tilingDev);
    return (int)ret;
}

void fa2_cleanup() {
    if (g_tilingDev)  { aclrtFree(g_tilingDev);  g_tilingDev  = nullptr; }
    if (g_workspcDev) { aclrtFree(g_workspcDev); g_workspcDev = nullptr; }
    g_inited = false;
}

} // extern "C"
