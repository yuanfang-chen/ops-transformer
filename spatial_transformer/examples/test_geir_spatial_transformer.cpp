/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in root of software repository for full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/spatial_transformer_proto.h"

#define FAILED -1
#define SUCCESS 0

#include "graph/operator.h"
#include "graph/operator_reg.h"
namespace ge {

REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                          \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                           \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0); \
    TensorDesc placeholder##inputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_NCHW, inputDtype);     \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_NCHW);                                    \
    Tensor tensor_placeholder##inputIndex;                                                  \
    ret = GenOnesData(placeholder##inputIndex##_shape,                                      \
        tensor_placeholder##inputIndex,                                                     \
        placeholder##inputIndex##_desc,                                                     \
        inputDtype,                                                                         \
        2);                                                                                  \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);           \
    input.push_back(tensor_placeholder##inputIndex);                                        \
    graph.AddOp(placeholder##inputIndex);                                                   \
    add1.set_input_##inputName(placeholder##inputIndex);                                   \
    inputs.push_back(placeholder##inputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                        \
    TensorDesc outputName##outputIndex##_desc =                                              \
        TensorDesc(ge::Shape(outputShape), FORMAT_NCHW, outputDtype);                          \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ADD_INPUT_ATTR(attrName, attrValue)                                                  \
    add1.set_attr_##attrName(attrValue)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    }
    return dilation;
}

int32_t GenOnesData(
    vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type, int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    
    if (data_type == ge::DT_FLOAT || data_type == ge::DT_FLOAT16 || data_type == ge::DT_DOUBLE) {
        float* pData = new (std::nothrow) float[size];
        for (uint32_t i = 0; i < size; ++i) {
            *(pData + i) = static_cast<float>(value);
        }
        input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype1, DataType inDtype2, std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
    std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto add1 = op::SpatialTransformer("SpatialTransformer");
    std::vector<std::vector<int64_t>> shapes = {{2, 2, 2, 2}, {2, 3}, {2, 2, 2, 2}};

    ADD_INPUT(1, x, inDtype1, shapes[0]);
    ADD_INPUT(2, theta, inDtype2, shapes[1]);
    ADD_OUTPUT(3, y, inDtype1, shapes[2]);

    // 添加属性
    vector<int64_t> output_size = {-1, -1};
    vector<float> default_theta = {};
    bool align_corners = false;
    vector<int64_t> use_default_theta = {};

    ADD_INPUT_ATTR(output_size, output_size);
    ADD_INPUT_ATTR(default_theta, default_theta);
    ADD_INPUT_ATTR(align_corners, align_corners);
    ADD_INPUT_ATTR(use_default_theta, use_default_theta);

    outputs.push_back(add1);
    // 添加完毕
    return SUCCESS;
}

bool InitEnv() {
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return false;
    }
    return true;
}

bool CreateAndConfigGraph(Graph& graph, std::vector<ge::Tensor>& input) {
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype1 = DT_FLOAT;
    DataType inDtype2 = DT_FLOAT;
    std::cout << inDtype1 << std::endl;
    std::cout << inDtype2 << std::endl;
    Status ret = CreateOppInGraph(inDtype1, inDtype2, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return false;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }
    return true;
}

bool AddGraphToSession(ge::Session* session, Graph& graph, uint32_t graph_id) {
    std::map<AscendString, AscendString> graph_options = {
    };
    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }

    return true;
}

bool DumpAndRunGraph(
    ge::Session* session, Graph& graph, std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output,
    uint32_t graph_id)
{
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    Status ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    return true;
}

void ProcessInputData(std::vector<ge::Tensor>& input) {
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        int64_t shape_size = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout <<"input: "<<i<<" dtype: "<<input[i].GetTensorDesc().GetDataType()<<" shape_size: "<<shape_size<<std::endl;
        float* input_data_i = (float*)input[i].GetData();
        for (int64_t j = 0; j < shape_size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, input_data_i[j]);
        }
    }
}

void ProcessOutputData(std::vector<ge::Tensor>& output) {
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        int64_t shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout <<"output: "<<i<<" dtype: "<<output[i].GetTensorDesc().GetDataType()<<" shape_size: "<<shape_size<<std::endl;
        float* output_data_i = (float*)output[i].GetData();
        for (int64_t j = 0; j < shape_size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, output_data_i[j]);
        }
    }
}

int FinalizeRes() {
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    LOG_PRINT("=== SpatialTransformer GEIR Test Start ===\n");
    // 初始化环境
    if (!InitEnv()) {
        return FAILED;
    }

    // 创建计算图
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    if (!CreateAndConfigGraph(graph, input)) {
        LOG_PRINT("ERROR: CreateAndConfigGraph failed\n");
        return FAILED;
    }

    // 创建会话并添加图
    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        LOG_PRINT("ERROR: Failed to create session\n");
        ge::GEFinalize();
        return FAILED;
    }

    uint32_t graph_id = 0;
    if (!AddGraphToSession(session, graph, graph_id)) {
        LOG_PRINT("ERROR: AddGraphToSession failed\n");
        return FAILED;
    }

    // 执行图
    std::vector<ge::Tensor> output;
    if (!DumpAndRunGraph(session, graph, input, output, graph_id)) {
        LOG_PRINT("ERROR: DumpAndRunGraph failed\n");
        return FAILED;
    }

    // 处理输入输出数据
    ProcessInputData(input);
    ProcessOutputData(output);

    // 清理资源
    LOG_PRINT("=== Test completed successfully ===\n");
    return FinalizeRes();
}