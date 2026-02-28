# -*- coding: utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd

from common.op_testcase import OpTestCase


class OP_aclnnRainFusionAttention_Accu_FILTER(OpTestCase):

    TEST_MODE = "Accu"
    ENABLE_FILTER = True
    FILTER_PATH = "testcase_filter.yml"
    FILTER_KEY = "CANN_ACCU_ALL_aclnnRainFusionAttention"
    GEN_BY_YAML = False
    JSON_PATH = 'aclnnRainFusionAttention_Accu.json'
    NODE_YAML_PATH = 'nodes_aclnnRainFusionAttention_Accu.yaml'
    TEAM = "ascend"

class OP_aclnnRainFusionAttention_Perf_FILTER(OpTestCase):

    TEST_MODE = "Perf"
    ENABLE_FILTER = True
    FILTER_PATH = "testcase_filter.yml"
    FILTER_KEY = "CANN_PERF_ALL_aclnnRainFusionAttention"
    GEN_BY_YAML = False
    JSON_PATH = 'aclnnRainFusionAttention_Perf.json'
    NODE_YAML_PATH = 'nodes_aclnnRainFusionAttention_Perf.yaml'
    TEAM = "ascend"

class OP_aclnnRainFusionAttention_Sanitizer_FILTER(OpTestCase):

    TEST_MODE = "Sanitizer"
    ENABLE_FILTER = True
    FILTER_PATH = "testcase_filter.yml"
    FILTER_KEY = "CANN_MEMO_ALL_aclnnRainFusionAttention"
    GEN_BY_YAML = False
    JSON_PATH = 'aclnnRainFusionAttention_Perf.json'
    NODE_YAML_PATH = 'nodes_aclnnRainFusionAttention_Sanitizer.yaml'
    TEAM = "ascend"

class OP_aclnnRainFusionAttention_Special_FILTER(OpTestCase):

    TEST_MODE = "Special"
    ENABLE_FILTER = True
    FILTER_PATH = "testcase_filter.yml"
    FILTER_KEY = "CANN_SPEC_OTH_aclnnRainFusionAttention"
    GEN_BY_YAML = False
    JSON_PATH = 'aclnnRainFusionAttention_Special.json'
    NODE_YAML_PATH = 'nodes_aclnnRainFusionAttention_Accu.yaml'
    TEAM = "ascend"

class OP_aclnnRainFusionAttention_Abnormal_FILTER(OpTestCase):

    TEST_MODE = "Abnormal"
    ENABLE_FILTER = True
    FILTER_PATH = "testcase_filter.yml"
    FILTER_KEY = "CANN_OUTL_OTH_aclnnRainFusionAttention"
    GEN_BY_YAML = False
    JSON_PATH = 'aclnnRainFusionAttention_Abnormal.json'
    NODE_YAML_PATH = 'nodes_aclnnRainFusionAttention_Accu.yaml'
    TEAM = "ascend"

class OP_aclnnRainFusionAttention_Dete_FILTER(OpTestCase):

    TEST_MODE = "Dete"
    ENABLE_FILTER = True
    FILTER_PATH = "testcase_filter.yml"
    FILTER_KEY = "CANN_Dete_aclnnRainFusionAttention"
    GEN_BY_YAML = False
    JSON_PATH = 'aclnnRainFusionAttention_Perf.json'
    NODE_YAML_PATH = 'nodes_aclnnRainFusionAttention_Dete.yaml'
    TEAM = "ascend"

class OP_aclnnRainFusionAttention_Dete_Acc_FILTER(OpTestCase):

    TEST_MODE = "Dete_Acc"
    ENABLE_FILTER = True
    FILTER_PATH = "testcase_filter.yml"
    FILTER_KEY = "CANN_Dete_Acc_aclnnRainFusionAttention"
    GEN_BY_YAML = False
    JSON_PATH = 'aclnnRainFusionAttention_Perf.json'
    NODE_YAML_PATH = 'nodes_aclnnRainFusionAttention_Dete_Acc.yaml'
    TEAM = "ascend"

class OP_aclnnRainFusionAttention_Dete_Perf_FILTER(OpTestCase):

    TEST_MODE = "Dete_Perf"
    ENABLE_FILTER = True
    FILTER_PATH = "testcase_filter.yml"
    FILTER_KEY = "CANN_Dete_Perf_aclnnRainFusionAttention"
    GEN_BY_YAML = False
    JSON_PATH = 'aclnnRainFusionAttention_Perf.json'
    NODE_YAML_PATH = 'nodes_aclnnRainFusionAttention_Dete_Perf.yaml'
    TEAM = "ascend"
