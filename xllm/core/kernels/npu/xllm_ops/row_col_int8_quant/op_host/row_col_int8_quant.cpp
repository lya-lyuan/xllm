
#include <algorithm>

#include "../op_kernel/row_col_int8_quant_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  RowColInt8QuantTilingData* tiling =
      context->GetTilingData<RowColInt8QuantTilingData>();
  const auto& shape = context->GetInputShape(0)->GetStorageShape();
  if (shape.GetDimNum() != 2 || shape.GetDim(0) < 1 || shape.GetDim(0) > 4096 ||
      shape.GetDim(1) < 2 || shape.GetDim(1) > 4096 ||
      shape.GetDim(1) % 2 != 0) {
    return ge::GRAPH_FAILED;
  }
  tiling->m = shape.GetDim(0);
  tiling->n = shape.GetDim(1);
  tiling->rowTiles = (tiling->m + 63) / 64;
  tiling->colTiles = (tiling->n + 255) / 256;
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  tiling->blocks = std::min<uint32_t>(48, platform.GetCoreNumAiv());
  tiling->blocks =
      std::min(tiling->blocks, tiling->rowTiles * tiling->colTiles);
  context->SetBlockDim(tiling->blocks);
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] =
      16 * 1024 * 1024 + 4 * tiling->rowTiles * tiling->colTiles * 320;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  if (x1_shape->GetDimNum() != 2) return GRAPH_FAILED;
  *context->GetOutputShape(0) = *x1_shape;
  *context->GetOutputShape(1) = *x1_shape;
  context->GetOutputShape(2)->SetDimNum(1);
  context->GetOutputShape(2)->SetDim(0, x1_shape->GetDim(0));
  context->GetOutputShape(3)->SetDimNum(1);
  context->GetOutputShape(3)->SetDim(0, x1_shape->GetDim(1));
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  if (context->GetInputDataType(0) != DT_FLOAT16) return GRAPH_FAILED;
  context->SetOutputDataType(0, DT_INT8);
  context->SetOutputDataType(1, DT_INT8);
  context->SetOutputDataType(2, DT_FLOAT);
  context->SetOutputDataType(3, DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RowColInt8Quant : public OpDef {
 public:
  explicit RowColInt8Quant(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("qRow")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("qCol")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("rowAbsmax")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("colAbsmax")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(RowColInt8Quant);
}  // namespace ops
