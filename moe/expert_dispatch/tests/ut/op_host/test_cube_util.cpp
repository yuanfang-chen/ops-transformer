#include <nlohmann/json.hpp>
#include <unistd.h>
#include <limits.h>
#include "test_cube_util.h"

void GetPlatFormInfos(const char *compile_info_str, map<string, string> &soc_infos, map<string, string> &aicore_spec,
                      map<string, string> &intrinsics) {
  string default_hardward_info = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32}})";
  nlohmann::json compile_info_json = nlohmann::json::parse(compile_info_str);
  if (compile_info_json.type() != nlohmann::json::value_t::object) {
    compile_info_json = nlohmann::json::parse(default_hardward_info.c_str());
  }

  map<string, string> soc_info_keys = {{"ai_core_cnt", "CORE_NUM"}, {"l2_size", "L2_SIZE"}, {"cube_core_cnt", "cube_core_cnt"}, {"vector_core_cnt", "vector_core_cnt"}, {"core_type_list", "core_type_list"}};
  soc_infos["core_type_list"] = "AICore";

  for (auto &t : soc_info_keys) {
    if (compile_info_json.contains("hardware_info") && compile_info_json["hardware_info"].contains(t.second)) {
      auto &obj_json = compile_info_json["hardware_info"][t.second];
      if (obj_json.is_number_integer()) {
        soc_infos[t.first] = to_string(compile_info_json["hardware_info"][t.second].get<uint32_t>());
      } else if (obj_json.is_string()) {
        soc_infos[t.first] = obj_json;
      }
    }
  }
  map<string, string> aicore_spec_keys = {{"ub_size", "UB_SIZE"},
                                          {"l0_a_size", "L0A_SIZE"},
                                          {"l0_b_size", "L0B_SIZE"},
                                          {"l0_c_size", "L0C_SIZE"},
                                          {"l1_size", "L1_SIZE"},
                                          {"bt_size", "BT_SIZE"},
                                          {"load3d_constraints", "load3d_constraints"}};
  aicore_spec["cube_freq"] = "cube_freq";
  for (auto &t : aicore_spec_keys) {
    if (compile_info_json.contains("hardware_info") && compile_info_json["hardware_info"].contains(t.second)) {
      if (t.second == "load3d_constraints") {
        aicore_spec[t.first] = compile_info_json["hardware_info"][t.second].get<string>();
      } else {
        aicore_spec[t.first] = to_string(compile_info_json["hardware_info"][t.second].get<uint32_t>());
      }
    }
  }

  std::string intrinsics_keys[] = {"Intrinsic_data_move_l12ub", "Intrinsic_data_move_l0c2ub",
                                   "Intrinsic_fix_pipe_l0c2out", "Intrinsic_data_move_out2l1_nd2nz",
                                   "Intrinsic_matmul_ub_to_ub", "Intrinsic_conv_ub_to_ub",
                                   "Intrinsic_data_move_l12bt"};
  for (string key : intrinsics_keys) {
    if (compile_info_json.contains("hardware_info") && compile_info_json["hardware_info"].contains(key) &&
        compile_info_json["hardware_info"][key].get<bool>()) {
      intrinsics[key] = "float16";
      if (key.find("Intrinsic_data_move_l12bt") != string::npos) {
        intrinsics[key] = "bf16";
      }
    }
  }
}

std::string GetExeDirPath() {
  std::string exe_path("./");
  char path[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", path, sizeof(path));
  if (n > 0) {
    path[n] = '\0';
    exe_path.assign(path);
    auto pos = exe_path.find_last_of('/');
    if (pos != std::string::npos) {
      exe_path.erase(pos + 1);
    } else {
      exe_path.assign("./");
    }
  }

  return exe_path;
}
