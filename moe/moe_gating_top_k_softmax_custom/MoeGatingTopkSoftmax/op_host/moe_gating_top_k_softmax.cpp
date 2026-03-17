
#include "moe_gating_top_k_softmax_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

#include <climits>

using namespace AscendC;

namespace {


    inline int64_t Align(int64_t x, int64_t y) {
        return (x + y - 1) / y * y;
    }
    inline int64_t CeilDiv(int64_t x, int64_t y) {
        return (x + y - 1) / y;
    }

    const constexpr uint32_t FP16_PER_REPEAT = 16;

    const constexpr uint32_t FP16_SIZE = 2;

    static const bool IS_SOFTMAX_REUSE_SOURCE = false;

} // anonymous namespace



namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  MoeGatingTopKSoftmaxTilingData tiling;
  const gert::StorageShape* xShapePtr = context->GetInputShape(0);
  const gert::Shape& x_shape = xShapePtr->GetStorageShape();
  const size_t x_dim_num = x_shape.GetDimNum();


  const gert::StorageShape* yShapePtr = context->GetOutputShape(0);
  const gert::Shape& y_shape = yShapePtr->GetStorageShape();
  const size_t y_dim_num = y_shape.GetDimNum();


  const gert::StorageShape* expertIdxShapePtr = context->GetOutputShape(1);
  const gert::Shape& expertIdx_shape = expertIdxShapePtr->GetStorageShape();
  const size_t expertIdx_dim_num = expertIdx_shape.GetDimNum();

  //const gert::StorageShape* rowIdxShapePtr = context->GetOutputShape(2);
  //const gert::Shape& rowIdx_shape = rowIdxShapePtr->GetStorageShape();
  //const size_t rowIdx_dim_num = rowIdx_shape.GetDimNum();

  const gert::StorageShape *finishedShapePtr = context->GetInputShape(1);
    int32_t hasFinished = 1;
    if (finishedShapePtr == nullptr) {
        hasFinished = 0;
    }


    // Set row
    int64_t row = 1;
    for (size_t i = 0; i < x_dim_num - 1; ++i) {
        const int64_t shape_dim = x_shape.GetDim(i);
            row *= x_shape.GetDim(i);
    }

    tiling.set_row(static_cast<int32_t>(row));

    // Set col
    const int64_t col = x_shape.GetDim(x_dim_num - 1);
        tiling.set_col(static_cast<int32_t>(col));


    // Set k
    const int64_t k = expertIdx_shape.GetDim(expertIdx_dim_num - 1);
        tiling.set_k(static_cast<int32_t>(k));
        
    const int64_t kAlign = Align(k, 16);
        tiling.set_kAlign(static_cast<int32_t>(kAlign));




    // Set blockNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t blockNum = 0;
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        blockNum = ascendcPlatform.GetCoreNum();
        tiling.set_blockNum(blockNum);
    } else {
        return ge::GRAPH_FAILED; // SocVersion is invalid
    }
    
    context->SetBlockDim(blockNum);


    tiling.set_hasFinished(hasFinished);

    // Set FormerRow
    const int64_t oneCoreRow = Align(row, 16 * blockNum) / blockNum;
    const int64_t activateCore = CeilDiv(row, oneCoreRow);
    const int64_t tailRow = row - (activateCore - 1) * oneCoreRow;

    tiling.set_oneCoreRow(oneCoreRow);
    tiling.set_activateCore(activateCore);
    tiling.set_tailRow(tailRow);

    // Set tilingKey
    uint64_t tilingKey = 0;
    context->SetTilingKey(tilingKey);

    // Save tiling data
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // softmax tiling 
    int32_t dataTypeSize = FP16_SIZE;
    auto FormersoftmaxShape = ge::Shape({tiling.get_oneCoreRow(), tiling.get_col()}); // ?
    SoftMaxTilingFunc(
        FormersoftmaxShape, dataTypeSize, GetSoftMaxMaxTmpSize(FormersoftmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
        tiling.FormerSoftmaxTilingData);

    auto TailsoftmaxShape = ge::Shape({tiling.get_tailRow(), tiling.get_col()}); // ?
    SoftMaxTilingFunc(
        TailsoftmaxShape, dataTypeSize, GetSoftMaxMaxTmpSize(TailsoftmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
        tiling.TailSoftmaxTilingData);




    // topk tiling
    auto FormertopkShape = ge::Shape({tiling.get_oneCoreRow(), tiling.get_col()});
    bool FormerTopkTilingSuccess = TopKTilingFunc(
        ascendcPlatform, 
        tiling.get_col() ,          //inner
        tiling.get_oneCoreRow(),    //outter
        tiling.get_kAlign(),        //k
        dataTypeSize,               // 
        true, TopKMode::TOPK_NORMAL, true,
        tiling.FormerTopkTilingData);

    auto TailtopkShape = ge::Shape({tiling.get_tailRow(), tiling.get_col()});
    bool TailTopkTilingSuccess = TopKTilingFunc(
        ascendcPlatform,  
        tiling.get_col() , 
        tiling.get_tailRow(), 
        tiling.get_kAlign(), 
        dataTypeSize, 
        true, TopKMode::TOPK_NORMAL, true,
        tiling.TailTopkTilingData);


    if(!(FormerTopkTilingSuccess && TailTopkTilingSuccess))
        return ge::GRAPH_FAILED;



    uint32_t maxsize = 0;
    uint32_t minsize = 0;
    AscendC::GetTopKMaxMinTmpSize(
        ascendcPlatform, 
        tiling.get_col() , 
        tiling.get_oneCoreRow(), 
        false, true, AscendC::TopKMode::TOPK_NORMAL, true, 
        dataTypeSize, maxsize, minsize);
    tiling.set_FormerTmpMinsize(minsize);

    AscendC::GetTopKMaxMinTmpSize(
        ascendcPlatform, 
        tiling.get_col(), 
        tiling.get_tailRow(), 
        false, true, AscendC::TopKMode::TOPK_NORMAL, true, 
        dataTypeSize, maxsize, minsize);
    tiling.set_TailTmpMinsize(minsize);


    // Set workspace size
    size_t userWorkspaceSize = 1024 + kAlign * row * activateCore *  ( sizeof(int16_t) + sizeof(int32_t) ) ;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    tiling.set_workspaceSize(currentWorkspace[0]);


        

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    
    gert::Shape* y_shape = context->GetOutputShape(0);
    gert::Shape* expertIdx_shape = context->GetOutputShape(1);
    //gert::Shape* rowIdx_shape = context->GetOutputShape(2);
    
    auto attrs = context->GetAttrs();
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t k = *kPtr;
    
    const int64_t xDimNum = x_shape->GetDimNum();

    y_shape->SetDimNum(xDimNum);
    expertIdx_shape->SetDimNum(xDimNum);

    y_shape->SetDim(0, x_shape->GetDim(0));
    expertIdx_shape->SetDim(0, x_shape->GetDim(0));

    y_shape->SetDim(xDimNum - 1, k);
    expertIdx_shape->SetDim(xDimNum - 1, k);
    
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class MoeGatingTopKSoftmax : public OpDef {
public:
    explicit MoeGatingTopKSoftmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("finished")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("expertIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rowIdx")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
            

        this->Attr("k").Int();
        

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");

    }
};

OP_ADD(MoeGatingTopKSoftmax);
}
