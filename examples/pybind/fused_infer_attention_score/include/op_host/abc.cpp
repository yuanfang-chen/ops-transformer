#include "abc.h"
#include <cstdio>
#include "fused_infer_attention_score_tiling_constants.h"

namespace optiling {    
    static bool CheckQKV(optiling::TilingContext &context) {
        auto tempQ = context.GetInputShape(QUERY_INDEX);
        auto tempK = context.GetInputShape(KEY_INDEX);
        auto tempV = context.GetInputShape(VALUE_INDEX);
        auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);
        OP_CHECK_IF((tempQ.tIsNull), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query input is null pointer!"),
                return GRAPH_FAILED);
        OP_CHECK_IF((tempK.tIsNull), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Key input is null pointer!"),
                return GRAPH_FAILED);
        OP_CHECK_IF((tempV.tIsNull), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Value input is null pointer!"),
                return GRAPH_FAILED);
        OP_CHECK_IF((tempOut.tIsNull),
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Attention_Out is null pointer!"),
                return GRAPH_FAILED);
        OP_CHECK_IF((tempQ.GetStorageShape().GetShapeSize() == 0) && (tempOut.GetStorageShape().GetShapeSize() != 0),
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                "Query head should not be 0, or when attentionOut is not empty tensor, query input shoud not be empty tensor!"),
                return GRAPH_FAILED);
        OP_CHECK_IF((tempQ.GetStorageShape().GetShapeSize() == StorageShape::kInvalidDimValue),
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query input dims are invalid!"),
                return GRAPH_FAILED);
        OP_CHECK_IF((tempK.GetStorageShape().GetShapeSize() == StorageShape::kInvalidDimValue),
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Key input dims are invalid!"),
                return GRAPH_FAILED);
        OP_CHECK_IF((tempV.GetStorageShape().GetShapeSize() == StorageShape::kInvalidDimValue),
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Value input dims are invalid!"),
                return GRAPH_FAILED);

        return GRAPH_SUCCESS;
    }

    bool DoOpTilingFusedInferAttentionScore(optiling::TilingContext *context) {
        if (context == nullptr) {
            OP_LOGE("FusedInferAttentionScore", "tiling context is nullptr!");
            return GRAPH_FAILED;
        }
        
        OP_CHECK_IF(CheckQKV(*context) != GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check query/key/value failed"), return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
}