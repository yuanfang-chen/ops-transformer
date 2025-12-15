# 算子接口（aclnn）

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

算子接口列表如下：

|    接口名   |   说明     | 确定性说明（A2/A3） |   确定性说明（A5）   |
| ----------- | ------------------- | --------- | ------------- |
|[aclnnAllGatherMatmul](../../mc2/all_gather_matmul/docs/aclnnAllGatherMatmul.md)|完成AllGather通信与MatMul计算融合。|默认确定性实现||
|[aclnnAlltoAllAllGatherBatchMatMul](../../mc2/allto_all_all_gather_batch_mat_mul/docs/aclnnAlltoAllAllGatherBatchMatMul.md)|完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。|默认确定性实现||
|[aclnnAlltoAllvGroupedMatMul](../../mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMul.md)|完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合。|默认确定性实现||
|[aclnnAttentionUpdate](../../attention/attention_update/docs/aclnnAttentionUpdate.md)|将各SP域PA算子的输出的中间结果lse，localOut两个局部变量结果更新成全局结果。|默认确定性实现||
|[aclnnBatchMatMulReduceScatterAlltoAll](../../mc2/batch_mat_mul_reduce_scatter_allto_all/docs/aclnnBatchMatMulReduceScatterAlltoAll.md)|BatchMatMulReduceScatterAllToAll是通算融合算子，实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行的算子。|默认确定性实现||
|[aclnnDistributeBarrier](../../mc2/distribute_barrier/docs/aclnnDistributeBarrier.md)|完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。|默认确定性实现||
|[aclnnDistributeBarrierV2](../../mc2/distribute_barrier/docs/aclnnDistributeBarrierV2.md)|完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。|默认确定性实现||
|[aclnnElasticReceivableInfoCollect](../../mc2/elastic_receivable_info_collect/docs/aclnnElasticReceivableInfoCollect.md)|在快恢检测流程中，收集并整理检测通信域内Test结果发送的状态位，并将结果输出，供下一步检测流程判断全局联通情况。|默认确定性实现||
|[aclnnElasticReceivableTest](../../mc2/elastic_receivable_test/docs/aclnnElasticReceivableTest.md)|对一个通信域内的所有卡发送数据并写状态位，以检测通信链路是否正常。|默认确定性实现||
|[aclnnFFN](../../ffn/ffn/docs/aclnnFFN.md)|该FFN算子提供MoeFFN和FFN的计算功能。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionScore](../../attention/flash_attention_score/docs/aclnnFlashAttentionScore.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFlashAttentionScoreGrad](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGrad.md)|训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionUnpaddingScoreGradV4](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV4.md)|训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionVarLenScoreV4](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV4.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFusedInferAttentionScoreV4](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md)|适配Decode & Prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。|默认确定性实现||
|[aclnnGatherPaKvCache](../attention/gather_pa_kv_cache/docs/aclnnGatherPaKvCache.md)|根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。|默认确定性实现||
|[aclnnGroupedMatMulAllReduce](../../mc2/grouped_mat_mul_all_reduce/docs/aclnnGroupedMatMulAllReduce.md)|在GroupedMatMul的基础上实现多卡并行AllReduce功能，实现分组矩阵乘计算， 每组矩阵乘的维度大小可以不同。|默认确定性实现||
|[aclnnGroupedMatMulAlltoAllv](../../mc2/grouped_mat_mul_allto_allv/docs/aclnnGroupedMatMulAlltoAllv.md)|完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合。|默认确定性实现||
|[aclnnGroupedMatmulFinalizeRouting](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRouting.md)|GroupedMatMul和MoeFinalizeRouting的融合算子。|默认非确定性实现，支持配置开启||
|[aclnnGroupedMatmulFinalizeRoutingWeightNz](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingWeightNz.md)|GroupedMatMul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持输入Weight为AI处理器亲和数据排布格式(NZ)。|默认非确定性实现，支持配置开启||
|[aclnnGroupedMatmulSwigluQuant](../../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md)|融合GroupedMatMul、Dequant、Swiglu和Quant。|默认确定性实现||
|[aclnnGroupedMatmulSwigluQuantWeightNZ](../../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md)|融合GroupedMatMul、Dequant、Swiglu和Quant，输入权重Weight会被强制视为NZ格式。|默认确定性实现||
|[aclnnGroupedMatmulV5](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现||
|[aclnnGroupedMatmulWeightNz](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同，输入权重Weight会被强制视为NZ格式。|默认确定性实现||
|[aclnnIncreFlashAttentionV4](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)|在全量推理场景的FlashAttention算子的基础上实现增量推理。|默认确定性实现||
|[aclnnInplaceMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnInplaceQuantMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduce.md)|完成MatMul计算与AllReduce通信融合。|默认非确定性实现，支持配置开启||
|[aclnnMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnMatmulReduceScatter](../../mc2/matmul_reduce_scatter/docs/aclnnMatmulReduceScatter.md)|完成mm + reduce_scatter_base计算。|默认非确定性实现，支持配置开启||
|[aclnnMlaPreprocess](../../attention/mla_preprocess/docs/aclnnMlaPreprocess.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现||
|[aclnnMlaProlog](../../attention/mla_prolog/docs/aclnnMlaProlog.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现||
|[aclnnMlaPrologV2WeightNz](../../attention/mla_prolog_v2/docs/aclnnMlaPrologV2WeightNz.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现||
|[aclnnMlaPrologV3WeightNz](../../attention/mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz.md)|Multi-Head Latent Attention前处理的计算 。|默认非确定性实现，支持配置开启||
|[aclnnMoeComputeExpertTokens](../../moe/moe_compute_expert_tokens/docs/aclnnMoeComputeExpertTokens.md)|MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。|默认确定性实现||
|[aclnnMoeDistributeBufferReset](../../mc2/moe_distribute_buffer_reset/docs/aclnnMoeDistributeBufferReset.md)|故障检测流程中，对EP通信域做数据区与状态区的清理。|默认确定性实现||
|[aclnnMoeDistributeCombine](../../mc2/moe_distribute_combine/docs/aclnnMoeDistributeCombine.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。|默认确定性实现||
|[aclnnMoeDistributeCombineAddRmsNormV2](../../mc2/moe_distribute_combine_add_rms_norm/docs/aclnnMoeDistributeCombineAddRmsNormV2.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加），之后完成Add + RmsNorm融合。|默认确定性实现||
|[aclnnMoeDistributeCombineV3](../../mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV3.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。|默认确定性实现||
|[aclnnMoeDistributeDispatchV3](../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)|对Token数据进行量化（可选）。|默认确定性实现||
|[aclnnMoeFinalizeRouting](../../moe/moe_finalize_routing/docs/aclnnMoeFinalizeRouting.md)|MoE计算中，最后处理合并MoE FFN的输出结果。|默认确定性实现||
|[aclnnMoeFinalizeRoutingV2](../../moe/moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV2.md)|MoE计算中，最后处理合并MoE FFN的输出结果，支持配置dropPadMode。|默认确定性实现||
|[aclnnMoeFinalizeRoutingV2Grad](../../moe/moe_finalize_routing_v2_grad/docs/aclnnMoeFinalizeRoutingV2Grad.md)|aclnnMoeFinalizeRoutingV2的反向传播。|默认确定性实现||
|[aclnnMoeGatingTopK](../../moe/moe_gating_top_k/docs/aclnnMoeGatingTopK.md)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|默认确定性实现||
|[aclnnMoeGatingTopKSoftmax](../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)|MoE计算中，对x的输出做Softmax计算，取TopK操作。|默认确定性实现||
|[aclnnMoeGatingTopKSoftmaxV2](../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)|MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取TopK操作；如果renorm=1，先对x的输出做TopK操作，再进行Softmax操作。|默认确定性实现||
|[aclnnMoeInitRouting](../../moe/moe_init_routing/docs/aclnnMoeInitRouting.md)|MoE的routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做Routing处理。|默认确定性实现||
|[aclnnMoeInitRoutingQuant](../../moe/moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant.md)|MoE的Routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做Routing处理，并对结果进行量化。|默认确定性实现||
|[aclnnMoeInitRoutingQuantV2](../../moe/moe_init_routing_quant_v2/docs/aclnnMoeInitRoutingQuantV2.md)|MoE的Routing计算，根据aclnnMoeGatingTopKSoftmaxV2的计算结果做Routing处理。|默认确定性实现||
|[aclnnMoeInitRoutingV2](../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)|该算子对应MoE中的Routing计算，以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。|默认确定性实现||
|[aclnnMoeInitRoutingV2Grad](../../moe/moe_init_routing_v2_grad/docs/aclnnMoeInitRoutingV2Grad.md)|aclnnMoeInitRoutingV2的反向传播，完成Tokens的加权求和。|默认确定性实现||
|[aclnnMoeUpdateExpert](../../mc2/moe_update_expert/docs/aclnnMoeUpdateExpert.md)|本API支持负载均衡和专家剪枝功能。经过映射后的专家表和Mask可传入MoE层进行数据分发和处理。|默认确定性实现||
|[aclnnNsaCompress](../../attention/nsa_compress/docs/aclnnNsaCompress.md)|训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。|默认确定性实现||
|[aclnnNsaCompressAttention](../../attention/nsa_compress_attention/docs/aclnnNsaCompressAttention.md)|NSA中compress attention以及select topk索引计算。|默认确定性实现||
|[aclnnNsaCompressAttentionInfer](../../attention/nsa_compress_attention_infer/docs/aclnnNsaCompressAttentionInfer.md)|Native Sparse Attention推理过程中，Compress Attention的计算。|默认确定性实现||
|[aclnnNsaCompressWithCache](../../attention/nsa_compress_with_cache/docs/aclnnNsaCompressWithCache.md)|实现Native-Sparse-Attention推理阶段的KV压缩。|默认确定性实现||
|[aclnnNsaSelectedAttention](../../attention/nsa_selected_attention/docs/aclnnNsaSelectedAttention.md)|训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。|默认确定性实现||
|[aclnnNsaSelectedAttentionGrad](../../attention/nsa_selected_attention_grad/docs/aclnnNsaSelectedAttentionGrad.md)|根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启||
|[aclnnNsaSelectedAttentionInfer](../../attention/nsa_selected_attention_infer/docs/aclnnNsaSelectedAttentionInfer.md)|Native Sparse Attention推理过程中，Selected Attention的计算。|默认确定性实现||
|[aclnnPromptFlashAttentionV3](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV3.md)|全量推理场景的FlashAttention算子。|默认确定性实现||
|[aclnnQuantGroupedMatmulInplaceAdd](../../gmm/quant_grouped_matmul_inplace_add/docs/aclnnQuantGroupedMatmulInplaceAdd.md)|实现分组矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合。|默认确定性实现||
|[aclnnQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduce.md)|对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant计算，接着与x3进行Add操作，最后做AllReduce计算。|默认非确定性实现，支持配置开启||
|[aclnnQuantMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnRingAttentionUpdate](../../attention/ring_attention_update/docs/aclnnRingAttentionUpdate.md)|将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|默认确定性实现||
|[aclnnRingAttentionUpdateV2](../../attention/ring_attention_update/docs/aclnnRingAttentionUpdateV2.md)|指定softmax的输入排布，将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|默认确定性实现||
|[aclnnWeightQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnWeightQuantMatmulAllReduce.md)|对入参x2进行伪量化计算后，完成MatMul和AllReduce计算。|默认非确定性实现，支持配置开启||
|[aclnnWeightQuantMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnWeightQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||