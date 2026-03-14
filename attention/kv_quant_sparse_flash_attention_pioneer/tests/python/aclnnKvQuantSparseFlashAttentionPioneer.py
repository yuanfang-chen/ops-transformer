同样的，以下脚本该如何修改？
# -*- coding: utf-8 -*-

import os
import json
import argparse
import sys

import xlrd
from ast import literal_eval
import pandas as pd
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/..")  # 加入上一层目录
import libs.tools as tools

cfg_fk = tools.ConfigFmk()
framework = cfg_fk.test_fmk
opname = "aclnn_name" if framework == "aclnn" else "opname"

ATTR_TYPE = ["int_array", "float_array", "bool_array", "scalar", "data_type", "buildins", "list_array"]
TENSOR_TYPE = ["bool", "bf16", "fp16", "fp32", "fp64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
               "uint64", "string", "complex64", "complex128"]
SCALAR_TYPE = ["bool", "bf16", "fp16", "fp32", "fp64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
               "uint64", "string", "complex64", "complex128"]
BUILDINS_TYPE = ["int", "bool", "float16", "bfloat16", "float", "double", "int8_t", "int16_t", "int32_t", "int64_t",
                 "uint8_t", "uint16_t", "uint32_t", "uint64_t", "string"]
DATA_TYPE = ["bool", "bf16", "fp16", "fp32", "fp64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
             "uint64", "string", "complex64", "complex128"]


def check_input_shape(tensor_shape, tensor_type):
    for i in range(len(tensor_type)):
        _type = tensor_type[i]
        _shape = tensor_shape[i]
        if _type == 'tensor':
            if not (isinstance(_shape, list) and all(map(lambda x: isinstance(x, int), _shape))):
                raise ValueError(f"input {_shape} is not a list!")
        if _type == 'tensor_list':
            for item in _shape:
                if not (isinstance(item, list) and all(map(lambda x: isinstance(x, int), item))):
                    raise ValueError(f"input {item} is not a list!")


def check_attr_dtype(attr_type, attr_dtype):
    if "tensor" in attr_type:
        if not attr_dtype in TENSOR_TYPE:
            raise ValueError(f"attr_dtype: {attr_dtype} is not match attr_type: {attr_type}!")
    if attr_type == 'scalar':
        if not attr_dtype in SCALAR_TYPE:
            raise ValueError(f"attr_dtype: {attr_dtype} is not match attr_type: {attr_type}!")
    if attr_type == 'data_type':
        if not attr_dtype in DATA_TYPE:
            raise ValueError(f"attr_dtype: {attr_dtype} is not match attr_type: {attr_type}!")
    if attr_type == 'buildins':
        if not attr_dtype in BUILDINS_TYPE:
            raise ValueError(f"attr_dtype: {attr_dtype} is not match attr_type: {attr_type}!")


def check_attr_value(attr_type, attr_value):
    if "array" in attr_type:
        if not isinstance(attr_value, list):
            raise ValueError(f"attr_value of attr_type: {attr_type} must be a list!")


def create_tensor_range(tensor_shape, tensor_type):
    tensor_range = []
    for i in range(len(tensor_type)):
        _type = tensor_type[i]
        _shape = tensor_shape[i]
        if _type == 'tensor':
            tensor_range.append([-10, 10])
        if _type == 'tensor_list':
            tensor_range.append([[-10, 10]] * len(_shape))
    return tensor_range


def create_tensor_dtype(tensor_shape, tensor_type):
    tensor_dtype = []
    for i in range(len(tensor_type)):
        _type = tensor_type[i]
        _shape = tensor_shape[i]
        if _type == 'tensor':
            tensor_dtype.append('fp32')
        if _type == 'tensor_list':
            tensor_dtype.append(['fp32'] * len(_shape))
    return tensor_dtype


def create_tensor_format(tensor_shape, tensor_type):
    tensor_format = []
    for i in range(len(tensor_type)):
        _type = tensor_type[i]
        _shape = tensor_shape[i]
        if _type == 'tensor':
            tensor_format.append('nd')
        if _type == 'tensor_list':
            tensor_format.append(['nd'] * len(_shape))
    return tensor_format


def create_tensor_input(input_info: list):
    input_tensor_dic = {}
    if input_info[0]:
        tensor_shape = literal_eval(input_info[0])
    else:
        raise ValueError("input tensor shape is missing!")

    if input_info[4]:
        tensor_type = literal_eval(input_info[4])
    else:
        raise ValueError("input tensor type is missing!")

    if len(tensor_shape) != len(tensor_type):
        raise ValueError("num of input_tensor_shape is not euqal to num of input_tensor_type!")

    if input_info[1]:
        tensor_range = literal_eval(input_info[1])
        if len(tensor_range) != len(tensor_shape):
            raise ValueError("num of input_tensor_range is not euqal to num of input_tensor_shape!")
    else:
        tensor_range = create_tensor_range(tensor_shape, tensor_type)
    print('tensor_shape: ', tensor_shape)
    print('tensor_type: ', tensor_type)
    check_input_shape(tensor_shape, tensor_type)

    if input_info[2]:
        tensor_dtype = literal_eval(input_info[2])
        if len(tensor_dtype) != len(tensor_shape):
            raise ValueError("num of input_tensor_dtype is not euqal to num of input_tensor_shape!")
    else:
        tensor_dtype = create_tensor_dtype(tensor_shape, tensor_type)
    print('tensor_dtype: ', tensor_dtype)
    if input_info[3]:
        tensor_format = literal_eval(input_info[3])
        if len(tensor_format) != len(tensor_shape):
            raise ValueError("num of input_tensor_format is not euqal to num of input_tensor_shape!")
    else:
        tensor_format = create_tensor_format(tensor_shape, tensor_type)

    if input_info[5]:
        tensor_index = literal_eval(input_info[5])
    else:
        tensor_index = [i for i in range(len(tensor_shape))]

    input_tensor_dic['shape'] = tensor_shape
    input_tensor_dic['range'] = tensor_range
    input_tensor_dic['dtype'] = tensor_dtype
    input_tensor_dic['format'] = tensor_format
    input_tensor_dic['type'] = tensor_type
    input_tensor_dic['index'] = tensor_index

    return input_tensor_dic


def create_tensor_output(input_info: list):
    output_tensor_dic = {}
    if input_info[0]:
        tensor_shape = literal_eval(input_info[0])
        output_tensor_dic['shape'] = tensor_shape
    if input_info[2]:
        tensor_dtype = literal_eval(input_info[2])
        output_tensor_dic['dtype'] = tensor_dtype

    if input_info[3]:
        tensor_format = literal_eval(input_info[3])
        output_tensor_dic['format'] = tensor_format

    if input_info[4]:
        tensor_type = literal_eval(input_info[4])
        output_tensor_dic['type'] = tensor_type
    else:
        output_tensor_dic['type'] = []

    return output_tensor_dic


def create_name_input(name: str):
    return name


def create_type_input(_type: str):
    if _type not in ATTR_TYPE:
        raise ValueError(f"attr type is not in {ATTR_TYPE}, please check!")
    return _type


def create_dtype_input(dtype: str):
    return dtype


def create_value_input(value: str):
    if value:
        try:
            value = literal_eval(value)
        except Exception as e:
            pass
    return value


def read_data_from_excel(excel_path, sheet_name, case_name):
    def _is_valid_data(sub_list):
        return not all(not x for x in sub_list)

    workbook = xlrd.open_workbook(excel_path)
    print(workbook.sheet_names())
    worksheet = workbook.sheet_by_name(sheet_name)
    num_rows = worksheet.nrows
    num_cols = worksheet.ncols

    input_aclnn_data = []
    for row_index in range(1, num_rows):
        row_data = []
        info_dic = {}

        if case_name and worksheet.cell_value(row_index, 1) != case_name:
            continue
        for col_index in range(num_cols):
            cell_value = worksheet.cell_value(row_index, col_index)
            row_data.append(cell_value)

        info_dic[f'{opname}'] = row_data[0]
        info_dic['case_name'] = row_data[1]
        print('row_data: ', row_data)
        print('row_data[2:8] ', row_data[2:8])
        input_tensor = create_tensor_input(row_data[2:8])
        info_dic['input_tensor'] = input_tensor

        output_tensor = create_tensor_output(row_data[8:13])
        info_dic['output_tensor'] = output_tensor

        attr_name, attr_type, attr_dtype, attr_value = [], [], [], []
        valid_data = _is_valid_data(row_data[13:17])

        i = 1
        while valid_data:
            tensor_info = row_data[13 + 4 * (i - 1):13 + 4 * i]
            attr_name.append(create_name_input(tensor_info[0]).lower())
            attr_type.append(create_type_input(tensor_info[1]))
            attr_dtype.append(create_dtype_input(tensor_info[2]).lower())
            attr_value.append(create_value_input(tensor_info[3]))
            i += 1
            if len(row_data) >= 4 + 4 * i - 1:
                valid_data = _is_valid_data(row_data[13 + 4 * (i - 1):13 + 4 * i])
            else:
                break

        for i in range(len(attr_name)):
            attr_dic = {}
            if attr_name[i]:
                attr_dic['name'] = attr_name[i]
            if attr_type[i]:
                attr_dic['type'] = attr_type[i]
            if attr_dtype[i]:
                attr_dic['dtype'] = attr_dtype[i]

            if attr_type[i] and attr_dtype[i]:
                check_attr_dtype(attr_type[i], attr_dtype[i])

            if attr_value[i] is not "":
                attr_dic['value'] = int(attr_value[i]) if "int" in attr_dtype[i] else attr_value[i]

            if attr_type[i] and attr_value[i] is not "":
                check_attr_value(attr_type[i], attr_value[i])
            key = 'attr_' + str(i + 1)
            info_dic[key] = attr_dic

        # print(info_dic)
        input_aclnn_data.append(info_dic)

    return input_aclnn_data


def save_file_to_cs(case_dict: dict, case_path: str):
    cs_info = dict()
    tensor_shapes = case_dict['input_tensor']['shape']
    tensor_ranges = case_dict['input_tensor']['range']
    tensor_dtypes = case_dict['input_tensor']['dtype']
    tensor_formats = case_dict['input_tensor']['format']
    tensor_types = case_dict['input_tensor']['type']
    cs_info['shape_input'] = []
    cs_info['range_input'] = []
    cs_info['dtype_input'] = []
    cs_info['format_input'] = []
    for i in range(len(tensor_types)):
        if tensor_types[i] == "tensor":
            cs_info['shape_input'].append(tensor_shapes[i])
            cs_info['range_input'].append(tensor_ranges[i])
            cs_info['dtype_input'].append(tensor_dtypes[i])
            cs_info['format_input'].append(tensor_formats[i])

        elif tensor_types[i] == "tensor_list":
            cs_info['shape_input'] += tensor_shapes[i]
            cs_info['range_input'] += tensor_ranges[i]
            cs_info['dtype_input'] += tensor_dtypes[i]
            cs_info['format_input'] += tensor_formats[i]
    output_tensor_shapes = case_dict['output_tensor']['shape']
    output_tensor_dtypes = case_dict['output_tensor']['dtype']
    output_tensor_types = case_dict['output_tensor']['type']
    cs_info['dtype_output'] = []
    cs_info['shape_output'] = []
    if output_tensor_types:
        for i in range(len(output_tensor_types)):
            if output_tensor_types[i] == "tensor":
                cs_info['dtype_output'].append(output_tensor_dtypes[i])
                cs_info['shape_output'].append(output_tensor_shapes[i])

            elif output_tensor_types[i] == "tensor_list":
                cs_info['dtype_output'] += output_tensor_dtypes[i]
                cs_info['shape_output'] += output_tensor_shapes[i]
    else:
        cs_info['dtype_output'] = case_dict['output_tensor']['dtype']
        cs_info['shape_output'] = case_dict['output_tensor']['shape']
    cs_info['type_input'] = ['tensor'] * len(cs_info['shape_input'])

    cur_path = os.path.realpath(os.path.dirname(__file__))
    main_path = os.path.join(cur_path, "../")
    json_name = "aclnn_op" if framework == "aclnn" else framework
    op_report_file = os.path.join(main_path, "configs/op_report.json")
    aclnn_op_dtype_file = os.path.join(main_path, f"configs/{json_name}_dtype.json")
    aclnn_op_bm_std_file = os.path.join(main_path, f"configs/{json_name}_bm_cmp_std.json")
    aclnn_op_red_range_file = os.path.join(main_path, f"configs/{json_name}_red_range.json")
    op_name = case_dict[f"{opname}"]

    with open(aclnn_op_dtype_file, "r") as dtype_info_file:
        dtype_info_dict = json.load(dtype_info_file)
        op_dict = dtype_info_dict['pytorch_op_dict']
    try:
        config_dtype = op_dict[op_name][0]
        config_dtype_list = config_dtype.split("/")
        set_op_dtype = True
    except KeyError:
        config_dtype = ""
        config_dtype_list = []
        set_op_dtype = False

    if cs_info['dtype_output'][0] == "int8":
        diff_thd = 0.005
        pct_thd = 0
        max_diff_thd = 10
        rtol = 0
        atol = 1
    elif cs_info['dtype_output'][0] == "bf16":
        diff_thd = 0.005
        pct_thd = 0.005
        max_diff_thd = 10
        rtol = 0.005
        atol = 0.0001
    else:
        diff_thd = pct_thd = 0.005
        max_diff_thd = 10
        rtol = 0.005
        atol = 0.000025

    with open(aclnn_op_bm_std_file, "r") as bm_cmp_std_file:
        bm_cmp_std_info_dict = json.load(bm_cmp_std_file)
        op_dict = bm_cmp_std_info_dict['pytorch_op_dict']
    try:
        bm_cmp_std = op_dict[op_name]
    except KeyError:
        bm_cmp_std = {"fp32": {"max_re_rtol": 10.0,
                               "avg_re_rtol": 2.0,
                               "rmse_rtol": 2.0,
                               "small_value": 0.000001,
                               "small_value_atol": 0.000000
                               },
                      "fp16": {"max_re_rtol": 10.0,
                               "avg_re_rtol": 2.0,
                               "rmse_rtol": 2.0,
                               "small_value": 0.0009765625,
                               "small_value_atol": 0.0000152587890625
                               },
                      "bf16": {"max_re_rtol": 10.0,
                               "avg_re_rtol": 2.0,
                               "rmse_rtol": 2.0,
                               "small_value": 0.0009765625,
                               "small_value_atol": 0.0000152587890625
                               }
                      }
    with open(aclnn_op_red_range_file, "r") as red_range_file:
        red_range_info_dict = json.load(red_range_file)
        op_dict = red_range_info_dict['pytorch_op_dict']
    try:
        red_range = op_dict[op_name]
    except KeyError:
        red_range = {
            "fp32": "0.000001/0.00001/0.0001/0.0005",
            "fp16": "0.001/0.002/0.005/0.01",
            "bf16": "0.001/0.002/0.005/0.01",
            "hf32": "0.001/0.002/0.005/0.01"
        }

    cs_info['diff_thd'] = float(diff_thd)
    cs_info['pct_thd'] = float(pct_thd)
    cs_info['max_diff_thd'] = float(max_diff_thd)
    cs_info['rtol'] = float(rtol)
    cs_info['atol'] = float(atol)
    cs_info['bm_cmp_std'] = bm_cmp_std
    cs_info['red_range'] = red_range
    cs_info["distribution_input"] = "uniform"

    # add precision_method
    cs_info['precision_method'] = 1
    cs_info['diff_thd_1'] = 0.005
    cs_info['pct_thd_1'] = 0.005
    cs_info['max_diff_thd_1'] = 10
    cs_info['atol_1'] = 0.000025
    cs_info['rtol_1'] = 0.005

    # 配置多输出精度标准
    precision_param_list = []
    for output_dtype in cs_info['dtype_output']:
        if output_dtype == "int8":
            diff_thd = 0.005
            pct_thd = 0
            max_diff_thd = 10
            rtol = 0
            atol = 1
        elif output_dtype == "bf16":
            diff_thd = 0.005
            pct_thd = 0.005
            max_diff_thd = 10
            rtol = 0.005
            atol = 0.0001
        else:
            diff_thd = pct_thd = 0.005
            max_diff_thd = 10
            rtol = 0.005
            atol = 0.000025
        param = {"diff_thd": float(diff_thd),
                 "pct_thd": float(pct_thd),
                 "max_diff_thd": float(max_diff_thd),
                 "rtol": float(rtol),
                 "atol": float(atol)
                 }
        precision_param_list.append(param)
    cs_info["precision_param_list"] = precision_param_list

    for params_key, params_value in case_dict.items():
        if "attr" in params_key:
            cs_info[params_key] = params_value['name']
            cs_info[params_value['name']] = params_value['value']
            cs_info[f"required_{params_value['name']}"] = 1

    case_name = case_dict[f'{opname}'] + "_" + case_dict['case_name']
    cs_info['case_name'] = case_name

    encode_cs = json.dumps(cs_info, indent=4)
    with open(
            os.path.join(case_path, case_name + ".cs"), "w"
    ) as json_file:
        json_file.write(encode_cs)
    return cs_info


def save_data_to_json(case_dict: dict,
                      cs_info: dict,
                      case_path: str):
    json_info = dict()

    json_info['acl_name'] = case_dict[f'{opname}']
    case_name = case_dict[f'{opname}'] + "_" + case_dict['case_name']
    json_info['case_name'] = case_name
    input_count = len(case_dict['input_tensor']['shape'])
    tensor_num = len(cs_info['shape_input'])
    json_info['input'], json_info['output'] = [], []

    # input_tensor
    tensor_num = 0
    for i in range(input_count):
        sub_input = dict()
        sub_input["name"] = f"input{case_dict['input_tensor']['index'][i]}"
        sub_input["type"] = case_dict['input_tensor']['type'][i]
        sub_input["data_range"] = case_dict['input_tensor']['range'][i]

        if sub_input["type"] == "tensor":
            sub_input["shape"] = case_dict['input_tensor']['shape'][i]
            sub_input["dtype"] = case_dict['input_tensor']['dtype'][i]
            sub_input["acl_format"] = case_dict['input_tensor']['format'][i]
            sub_input["bin_path"] = case_name + '_input_%s.bin' % tensor_num
            tensor_num += 1
        elif sub_input["type"] == "tensor_list":
            sub_input["shapes"] = case_dict['input_tensor']['shape'][i]
            sub_input["dtype"] = case_dict['input_tensor']['dtype'][i][0]
            sub_input["acl_format"] = case_dict['input_tensor']['format'][i][0]
            sub_input["bin_paths"] = []
            for _ in range(len(sub_input["shapes"])):
                sub_input["bin_paths"].append(case_name + '_input_%s.bin' % tensor_num)
                tensor_num += 1

        sub_input["data_distribution"] = 'uniform'
        json_info['input'].append(sub_input)

    # input_tensor_list

    # attr
    for params_key, params_value in case_dict.items():
        sub_input = dict()
        if "attr" in params_key:
            sub_input["name"] = params_value["name"] if "name" in params_value else ""
            sub_input["type"] = params_value["type"] if "type" in params_value else ""
            sub_input["dtype"] = params_value["dtype"] if "dtype" in params_value else ""
            sub_input["value"] = params_value["value"] if "value" in params_value else ""
            json_info['input'].append(sub_input)

    # output_tensor
    output_shape = case_dict['output_tensor']['shape']
    output_tensor_types = case_dict['output_tensor']['type']
    output_count = len(case_dict['output_tensor']['dtype'])
    output_num = 0
    output_num_name = 0
    if output_tensor_types and "tensor_list" in output_tensor_types:
        for i in range(len(output_tensor_types)):
            sub_output = dict()
            sub_output["name"] = f"output{output_num_name}"
            sub_output["type"] = output_tensor_types[i]
            output_num_name += 1
            if sub_output["type"] == "tensor":
                sub_output["dtype"] = case_dict['output_tensor']['dtype'][i]
                sub_output["acl_format"] = case_dict['output_tensor']['format'][i]
                sub_output["golden_path"] = case_name + '_cpu_output_%s.bin' % output_num
                sub_output["bin_path"] = case_name + '_output_%s.bin' % output_num
                sub_output["data_range"] = ""
                output_num += 1
            elif sub_output["type"] == "tensor_list":
                sub_output["dtype"] = case_dict['output_tensor']['dtype'][i][0]
                sub_output["acl_format"] = case_dict['output_tensor']['format'][i][0]
                sub_output["bin_paths"] = []
                sub_output["golden_paths"] = []
                sub_output["data_range"] = ""
                for _ in range(len(case_dict['output_tensor']['dtype'][i])):
                    sub_output["bin_paths"].append(case_name + '_output_%s.bin' % output_num)
                    sub_output["golden_paths"].append(case_name + '_cpu_output_%s.bin' % output_num)
                    output_num += 1
            json_info['output'].append(sub_output)

    else:
        for i in range(output_count):
            sub_output = dict()
            sub_output["name"] = f"output{output_num}"
            sub_output["type"] = "tensor"
            sub_output["dtype"] = case_dict['output_tensor']['dtype'][i]
            sub_output["shape"] = case_dict['output_tensor']['shape'][i]
            sub_output["acl_format"] = case_dict['output_tensor']['format'][i]
            sub_output["data_range"] = ""
            sub_output["golden_path"] = case_name + '_cpu_output_%s.bin' % i
            sub_output["bin_path"] = case_name + '_output_%s.bin' % i
            output_num += 1
            json_info['output'].append(sub_output)

    encode_npu_json = json.dumps(json_info, indent=4)
    with open(
            os.path.join(case_path, case_name + ".json"), "w"
    ) as json_file:
        json_file.write(encode_npu_json)


def save_data_to_file(case_list, case_path):
    # 里面有output.shape
    # case list: k ,v output_tensor {'shape': [[20, 16, 128], [20, 1, 16]], 'dtype': ['fp16', 'int32'], 'format': ['ND', 'ND'], 'type': []}
    for sub_case in case_list:
        cs_info = save_file_to_cs(sub_case, case_path)
        save_data_to_json(sub_case, cs_info, case_path)


def main():
    if len(sys.argv) == 2:
        op = sys.argv[1]
        sheet_name = "Sheet1"
        case_name = None
    elif len(sys.argv) == 3:
        op = sys.argv[1]
        sheet_name = sys.argv[2]
        case_name = None
    else:
        op = sys.argv[1]
        sheet_name = sys.argv[2]
        case_name = sys.argv[3]

    base_path = os.path.dirname(os.path.abspath(__file__))
    design_path = os.path.join(base_path, f"../design/design_file/{framework}/")
    case_path = os.path.join(base_path, f"../testcase/{framework}_case/{op}")
    if not os.path.exists(case_path):
        os.makedirs(case_path)
    design_file = os.path.join(design_path, f"design2_{framework}_{op}.xlsx")
    print('base_path: ', base_path)
    print('design_path: ', design_path)
    print('case_path: ', case_path)
    print('design_file: ', design_file)
    # 按行读取excel的数据
    case_list = read_data_from_excel(design_file, sheet_name, case_name)

    save_data_to_file(case_list, case_path)


if __name__ == '__main__':
    main()
