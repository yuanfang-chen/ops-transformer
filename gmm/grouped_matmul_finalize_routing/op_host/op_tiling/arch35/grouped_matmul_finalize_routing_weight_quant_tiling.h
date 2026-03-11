/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing_weight_quant_tiling.h
 * \brief
 */
#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILING_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILING_H

#include <graph/utils/type_utils.h>
#include <sstream>
#include <map>
#include <unordered_set>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

#include "./grouped_matmul_finalize_routing_quant_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/grouped_matmul_finalize_routing_weight_quant_tiling_data.h"
#include "../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h"
#include "../../grouped_matmul_finalize_routing_tiling.h"
#include "../../../../grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.h"
#include "../../../../grouped_matmul/op_host/grouped_matmul_host_util.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace GroupedMatmulFinalizeRoutingArch35WeightQuantTiling {
constexpr static int64_t GMMFR_WEIGHT_QUANT_TILING_VEC_ANTIQUANT = 2;
namespace param_validator {
enum class ParamType {
    INPUT_TENSOR,    // tensor类型参数
    OUTPUT_TENSOR,    // tensor类型参数
    ATTR       // attr类型参数
};

struct ParamMeta {
    std::string name;           // 参数名称
    ParamType type;             // 参数类型
    bool required;              // 是否必选
    std::string description;    // 参数描述
};

enum class ScenarioType {
    NONE = 0,
    MX_A8W4_WEIGHT_NZ = 1,
};

struct ScenarioCondition {
    enum class ConditionType {
        PARAM_EXISTS,
        PARAM_NOT_EXISTS,
        PARAM_EQUALS_VALUE,
        PARAM_NOT_EQUALS_VALUE,
        TENSOR_EQUALS_SHAPE,
        TENSOR_NOT_EQUALS_SHAPE,
        TENSOR_SHAPE_DIM_EQUALS,
        TENSOR_RANK_EQUALS,
        LOGICAL_AND,
        LOGICAL_OR,
        LOGICAL_NOT,
        CUSTOM
    } type;
    
    std::string paramName;
    std::any value;
    std::vector<int64_t> shapeValue;
    int dimIndex;
    int64_t dimValue;
    
    LogicalOperator logicalOp;
    std::vector<std::shared_ptr<ScenarioCondition>> subConditions;
    
    std::function<bool(const ParamAccessor&)> customCondition;
    
    bool Evaluate(const ParamAccessor& accessor) const;
};

struct ScenarioDefinition {
    ScenarioType type;
    int priority;
    std::shared_ptr<ScenarioCondition> condition;
    
    bool Matches(const ParamAccessor& accessor) const {
        return condition && condition->Evaluate(accessor);
    }
};

class ParamRegistry {
public:
    void RegisterInputTensor(const std::string& name, bool required);

    void RegisterOutputTensor(const std::string& name, bool required);
    
    void RegisterAttr(const std::string& name, bool required);
    
    const ParamMeta& GetMeta(const std::string& name) const;
    
    bool IsRequired(const std::string& name) const;
    
    bool HasParam(const std::string& name) const;
    
private:
    std::map<std::string, ParamMeta> params_;
};




    

    class ParamAccessor {
        public:
            ParamAccessor(void* params) : params_(params) {}
            
            template<typename T>
            T GetAttrValue(const std::string& name) const {
                void* ptr = GetPointer(name);
                if (ptr == nullptr) {
                    throw std::runtime_error("Parameter " + name + " is null");
                }
                return *static_cast<T*>(ptr);
            }
            
            bool IsTensorNull(const std::string& name) const {
                void* tensor = GetTensor(name);
                return tensor == nullptr;
            }
            
            std::vector<int64_t> GetTensorShape(const std::string& name) const {
                void* tensor = GetTensor(name);
                if (tensor == nullptr) {
                    return {};
                }
                return GetShape(tensor);
            }
            
        protected:
            void* params_;
            
            virtual void* GetTensor(const std::string& name) const = 0;
            virtual void* GetPointer(const std::string& name) const = 0;
            virtual std::vector<int64_t> GetShape(void* tensor) const = 0;
        };

    class IValidationRule {
        public:
            virtual ~IValidationRule() = default;
            virtual bool Validate(const ParamAccessor& accessor) const = 0;
            virtual std::string GetErrorMessage() const = 0;
        };
    
    class RequiredParamRule : public IValidationRule {
    public:
        RequiredParamRule(const std::string paramName) 
            : paramName_(paramName) {}
        
        bool Validate(const ParamAccessor& accessor) const override {
            return !accessor.IsTensorNull(paramName_);
        }
        
        std::string GetErrorMessage() const override {
            return "Required parameter '" + paramName_ + "' is null";
        }
        
    private:
        std::string paramName_;
    };
    
    enum class BindingType {
        EXISTENCE,     // 存在性绑定
        SHAPE,         // Shape绑定
        VALUE          // 值绑定
    };
            
    class ExistenceBindingRule : public IValidationRule {
    public:
        ExistenceBindingRule(const std::string sourceParam, 
                                const std::string targetParam)
            : sourceParam_(sourceParam), targetParam_(targetParam) {}
        
        bool Validate(const ParamAccessor& accessor) const override {
            bool sourceExists = !accessor.IsTensorNull(sourceParam_);
            bool targetExists = !accessor.IsTensorNull(targetParam_);
            
            if (sourceExists && !targetExists) {
                return false;
            }
            return true;
        }
        
        std::string GetErrorMessage() const override {
            return "When '" + sourceParam_ + "' exists, '" + targetParam_ + "' must also exist";
        }
        
    private:
        std::string sourceParam_;
        std::string targetParam_;
    };
            
    class ShapeBindingRule : public IValidationRule {
    public:
        using ShapeConstraint = std::function<bool(
            const std::vector<int64_t>&, 
            const std::vector<int64_t>&)>;
        
        ShapeBindingRule(const std::string param1, 
                        const std::string param2,
                        ShapeConstraint constraint)
            : param1_(param1), param2_(param2), constraint_(constraint) {}
        
        bool Validate(const ParamAccessor& accessor) const override {
            auto shape1 = accessor.GetTensorShape(param1_);
            auto shape2 = accessor.GetTensorShape(param2_);
            return constraint_(shape1, shape2);
        }
        
        std::string GetErrorMessage() const override {
            return "Shape constraint failed between '" + param1_ + "' and '" + param2_ + "'";
        }
        
    private:
        std::string param1_;
        std::string param2_;
        ShapeConstraint constraint_;
    };

    
    enum class LogicalOperator {
        AND,
        OR,
        NOT
    };
    

    
    namespace ScenarioConditions {
        std::shared_ptr<ScenarioCondition> Exists(const std::string& paramName);
        std::shared_ptr<ScenarioCondition> NotExists(const std::string& paramName);
        std::shared_ptr<ScenarioCondition> EqualsValue(const std::string& paramName, 
                                                       const std::any& value);
        std::shared_ptr<ScenarioCondition> EqualsShape(const std::string& paramName, 
                                                       const std::vector<int64_t>& shape);
        std::shared_ptr<ScenarioCondition> EqualsShapeDim(const std::string& paramName,
                                                           int dimIndex,
                                                           int64_t dimValue);
        std::shared_ptr<ScenarioCondition> EqualsRank(const std::string& paramName,
                                                      int rank);
        std::shared_ptr<ScenarioCondition> And(std::shared_ptr<ScenarioCondition> cond1,
                                               std::shared_ptr<ScenarioCondition> cond2);
        std::shared_ptr<ScenarioCondition> Or(std::shared_ptr<ScenarioCondition> cond1,
                                              std::shared_ptr<ScenarioCondition> cond2);
        std::shared_ptr<ScenarioCondition> Not(std::shared_ptr<ScenarioCondition> cond);
    }

    
    class ScenarioInference {
    public:
        void RegisterScenario(const ScenarioDefinition& definition) {
            scenarios_[definition.type] = definition;
        }
        
        ScenarioType InferScenario(const ParamAccessor& accessor) const {
            ScenarioType bestScenario = ScenarioType::UNKNOWN;
            int bestPriority = -1;
            
            for (const auto& [type, definition] : scenarios_) {
                if (definition.Matches(accessor) && definition.priority > bestPriority) {
                    bestScenario = type;
                    bestPriority = definition.priority;
                }
            }
            
            return bestScenario;
        }
        
        std::string GetScenarioName(ScenarioType type) const {
            auto it = scenarios_.find(type);
            if (it != scenarios_.end()) {
                return it->second.name;
            }
            return "Unknown";
        }
        
        bool MatchesScenario(const ParamAccessor& accessor, ScenarioType type) const {
            auto it = scenarios_.find(type);
            return it != scenarios_.end() && it->second.Matches(accessor);
        }
        
    private:
        std::map<ScenarioType, ScenarioDefinition> scenarios_;
    };


    struct BindingDefinition {
        std::string id;
        BindingType type;
        std::string sourceParam;
        std::string targetParam;
        std::function<bool(const ParamAccessor&)> validator;
        std::string description;
        std::set<ScenarioType> applicableScenarios;
    };
    
    class ScenarioAwareBindingManager {
    public:
        void RegisterGlobalBinding(const BindingDefinition& binding) {
            globalBindings_.push_back(binding);
        }
        
        void RegisterScenarioBinding(const BindingDefinition& binding) {
            for (auto scenario : binding.applicableScenarios) {
                scenarioBindings_[scenario].push_back(binding);
            }
        }
        
        std::vector<BindingDefinition> GetActiveBindings(ScenarioType scenario) const {
            std::vector<BindingDefinition> result = globalBindings_;
            
            auto it = scenarioBindings_.find(scenario);
            if (it != scenarioBindings_.end()) {
                result.insert(result.end(), it->second.begin(), it->second.end());
            }
            
            return result;
        }
        
        bool ValidateBindings(ScenarioType scenario,
                             const ParamAccessor& accessor,
                             std::vector<std::string>& errors) const {
            bool success = true;
            auto bindings = GetActiveBindings(scenario);
            
            for (const auto& binding : bindings) {
                if (binding.validator && !binding.validator(accessor)) {
                    errors.push_back(binding.description);
                    success = false;
                }
            }
            
            return success;
        }
        
    private:
        std::vector<BindingDefinition> globalBindings_;
        std::map<ScenarioType, std::vector<BindingDefinition>> scenarioBindings_;
    };


    struct ScenarioConstraint {
        ScenarioType scenario;
        std::string paramName;
        enum class ConstraintType {
            VALUE_RANGE,
            VALUE_FORBIDDEN,
            VALUE_REQUIRED,
            SHAPE_CONSTRAINT,
            CUSTOM
        } type;
        
        std::any constraintValue;
        std::function<bool(const ParamAccessor&)> customValidator;
        std::string errorMessage;
    };
    
    class ScenarioConstraintManager {
    public:
        void RegisterConstraint(const ScenarioConstraint& constraint) {
            constraints_[constraint.scenario].push_back(constraint);
        }
        
        bool ValidateConstraints(ScenarioType scenario, 
                                const ParamAccessor& accessor,
                                std::vector<std::string>& errors) const {
            bool success = true;
            auto it = constraints_.find(scenario);
            
            if (it != constraints_.end()) {
                for (const auto& constraint : it->second) {
                    if (constraint.customValidator && !constraint.customValidator(accessor)) {
                        errors.push_back(constraint.errorMessage);
                        success = false;
                    }
                }
            }
            
            return success;
        }
        
        std::vector<ScenarioConstraint> GetConstraints(ScenarioType scenario) const {
            auto it = constraints_.find(scenario);
            if (it != constraints_.end()) {
                return it->second;
            }
            return {};
        }
        
    private:
        std::map<ScenarioType, std::vector<ScenarioConstraint>> constraints_;
    };



enum class ExtractionType {
    DIRECT_VALUE,
    PARAM_POINTER,
    TENSOR_SHAPE,
    TENSOR_SHAPE_DIM,
    TENSOR_RANK,
    COMPUTED_VALUE,
    PARAM_COLLECTION
};

struct ExtractionRule {
    ScenarioType scenario;
    std::string outputKey;
    ExtractionType type;
    std::string sourceParam;
    int dimIndex;
    std::function<std::any(const ParamAccessor&)> extractor;
    std::string description;
};

namespace ExtractionRules {
    ExtractionRule ExtractAttrValue(const std::string& outputKey,
                                   const std::string& paramName);
    
    ExtractionRule ExtractTensorShape(const std::string& outputKey,
                                      const std::string& paramName);
    
    ExtractionRule ExtractTensorShapeDim(const std::string& outputKey,
                                         const std::string& paramName,
                                         int dimIndex);
    
    ExtractionRule ExtractTensorRank(const std::string& outputKey,
                                     const std::string& paramName);
    
    ExtractionRule CustomExtract(const std::string& outputKey,
                                 std::function<std::any(const ParamAccessor&)> extractor);
}

struct ExtractionResult {
    ScenarioType scenario;
    std::string scenarioName;
    std::map<std::string, std::any> keyParameters;
};

class KeyParameterExtractor {
public:
    void RegisterExtractionRule(const ExtractionRule& rule) {
        extractionRules_[rule.scenario].push_back(rule);
    }
    
    ExtractionResult ExtractKeyParameters(ScenarioType scenario,
                                         const ParamAccessor& accessor) const {
        ExtractionResult result;
        result.scenario = scenario;
        result.scenarioName = scenarioInference_->GetScenarioName(scenario);
        
        auto rules = GetRulesForScenario(scenario);
        for (const auto& rule : rules) {
            result.keyParameters[rule.outputKey] = ExtractByRule(rule, accessor);
        }
        
        return result;
    }
    
    void SetScenarioInference(ScenarioInference* inference) {
        scenarioInference_ = inference;
    }
    
private:
    std::any ExtractByRule(const ExtractionRule& rule, 
                          const ParamAccessor& accessor) const {
        switch (rule.type) {
            case ExtractionType::DIRECT_VALUE:
                return accessor.GetAttrValue<std::string>(rule.sourceParam);
                
            case ExtractionType::TENSOR_SHAPE:
                return accessor.GetTensorShape(rule.sourceParam);
                
            case ExtractionType::TENSOR_SHAPE_DIM: {
                auto shape = accessor.GetTensorShape(rule.sourceParam);
                if (rule.dimIndex >= 0 && rule.dimIndex < shape.size()) {
                    return shape[rule.dimIndex];
                }
                return int64_t(-1);
            }
                
            case ExtractionType::TENSOR_RANK: {
                auto shape = accessor.GetTensorShape(rule.sourceParam);
                return static_cast<int>(shape.size());
            }
                
            case ExtractionType::COMPUTED_VALUE:
                if (rule.extractor) {
                    return rule.extractor(accessor);
                }
                break;
                
            default:
                break;
        }
        return std::any();
    }
    
    std::vector<ExtractionRule> GetRulesForScenario(ScenarioType scenario) const {
        auto it = extractionRules_.find(scenario);
        if (it != extractionRules_.end()) {
            return it->second;
        }
        return {};
    }
    
    std::map<ScenarioType, std::vector<ExtractionRule>> extractionRules_;
    ScenarioInference* scenarioInference_ = nullptr;
};




struct ValidationResult {
    bool success;
    std::vector<std::string> errors;
    
    std::string GetFullMessage() const {
        if (success) {
            return "Validation passed";
        }
        std::string msg = "Validation failed:\n";
        for (const auto& error : errors) {
            msg += "  - " + error + "\n";
        }
        return msg;
    }
};

class ReorderedValidationEngine {
public:
    ReorderedValidationEngine(std::shared_ptr<ParamRegistry> registry,
                             std::shared_ptr<ScenarioInference> scenarioInference,
                             std::shared_ptr<ScenarioAwareBindingManager> bindingManager,
                             std::shared_ptr<ScenarioConstraintManager> constraintManager)
        : registry_(registry),
          scenarioInference_(scenarioInference),
          bindingManager_(bindingManager),
          constraintManager_(constraintManager),
          currentScenario_(ScenarioType::UNKNOWN) {}
    
    ValidationResult Validate(const ParamAccessor& accessor) {
        ValidationResult result;
        result.success = true;
        
        if (!ValidateBasicExistence(accessor)) {
            result.success = false;
            result.errors.push_back("Basic existence validation failed");
            return result;
        }
        
        currentScenario_ = InferScenario(accessor);
        
        if (!ValidateBindings(currentScenario_, accessor)) {
            result.success = false;
            result.errors.push_back("Binding validation failed");
        }
        
        if (!ValidateScenarioConstraints(currentScenario_, accessor)) {
            result.success = false;
            result.errors.push_back("Scenario constraint validation failed");
        }
        
        return result;
    }
    
    ScenarioType GetInferredScenario() const {
        return currentScenario_;
    }
    
private:
    bool ValidateBasicExistence(const ParamAccessor& accessor) {
        return true;
    }
    
    ScenarioType InferScenario(const ParamAccessor& accessor) {
        return scenarioInference_->InferScenario(accessor);
    }
    
    bool ValidateBindings(ScenarioType scenario, const ParamAccessor& accessor) {
        std::vector<std::string> errors;
        return bindingManager_->ValidateBindings(scenario, accessor, errors);
    }
    
    bool ValidateScenarioConstraints(ScenarioType scenario, const ParamAccessor& accessor) {
        std::vector<std::string> errors;
        return constraintManager_->ValidateConstraints(scenario, accessor, errors);
    }
    
    std::shared_ptr<ParamRegistry> registry_;
    std::shared_ptr<ScenarioInference> scenarioInference_;
    std::shared_ptr<ScenarioAwareBindingManager> bindingManager_;
    std::shared_ptr<ScenarioConstraintManager> constraintManager_;
    ScenarioType currentScenario_;
};







class MyOpValidator {
    public:
        MyOpValidator() {
            // RegisterParameters();
            // RegisterScenarios();
            1111111111111111111111
            RegisterGlobalBindings();
            RegisterScenarioBindings();
            RegisterScenarioConstraints();
            RegisterExtractionRules();
            
            extractor_.SetScenarioInference(&scenarioInference_);
        }
        
        ValidationResult Validate(void* params) {
            MyParamAccessor accessor(params);
            ReorderedValidationEngine engine(
                std::make_shared<ParamRegistry>(registry_),
                std::make_shared<ScenarioInference>(scenarioInference_),
                std::make_shared<ScenarioAwareBindingManager>(bindingManager_),
                std::make_shared<ScenarioConstraintManager>(constraintManager_)
            );
            
            auto result = engine.Validate(accessor);
            if (result.success) {
                auto scenario = engine.GetInferredScenario();
                extractionResult_ = extractor_.ExtractKeyParameters(scenario, accessor);
                UseExtractedParameters();
            }
            return result;
        }
        
        const ExtractionResult& GetExtractionResult() const {
            return extractionResult_;
        }
        
    private:


        
        void RegisterGlobalBindings() {
            BindingDefinition binding1;
            binding1.id = "d_j_existence";
            binding1.type = BindingType::EXISTENCE;
            binding1.sourceParam = "d";
            binding1.targetParam = "j";
            binding1.description = "When d exists, j must also exist";
            binding1.validator = [](const ParamAccessor& accessor) {
                bool dExists = !accessor.IsTensorNull("d");
                bool jExists = !accessor.IsTensorNull("j");
                return !dExists || jExists;
            };
            bindingManager_.RegisterGlobalBinding(binding1);
        }
        
        void RegisterScenarioBindings() {
            BindingDefinition binding1;
            binding1.id = "d_j_shape_binding";
            binding1.type = BindingType::SHAPE;
            binding1.sourceParam = "d";
            binding1.targetParam = "j";
            binding1.description = "d and j must have compatible shapes";
            binding1.applicableScenarios = {ScenarioType::SCENARIO_1};
            binding1.validator = [](const ParamAccessor& accessor) {
                auto shape1 = accessor.GetTensorShape("d");
                auto shape2 = accessor.GetTensorShape("j");
                return shape1.size() == shape2.size();
            };
            bindingManager_.RegisterScenarioBinding(binding1);
        }
        
        void RegisterScenarioConstraints() {
            ScenarioConstraint constraint;
            constraint.scenario = ScenarioType::SCENARIO_1;
            constraint.paramName = "k";
            constraint.type = ScenarioConstraint::ConstraintType::VALUE_FORBIDDEN;
            constraint.errorMessage = "In scenario 1, k cannot be 1, 2, or 3";
            constraint.customValidator = [](const ParamAccessor& accessor) {
                return true;
            };
            constraintManager_.RegisterConstraint(constraint);
        }
        
        void RegisterExtractionRules() {
            extractor_.RegisterExtractionRule(ExtractionRules::ExtractTensorShapeDim(
                "batch_size", "a", 0
            ));
            
            extractor_.RegisterExtractionRule(ExtractionRules::ExtractTensorShapeDim(
                "hidden_size", "b", 1
            ));
            
            extractor_.RegisterExtractionRule(ExtractionRules::ExtractAttrValue(
                "mode", "j"
            ));
            
            ExtractionRule customRule;
            customRule.scenario = ScenarioType::SCENARIO_1;
            customRule.outputKey = "total_elements";
            customRule.type = ExtractionType::COMPUTED_VALUE;
            customRule.extractor = [](const ParamAccessor& accessor) {
                auto shape_a = accessor.GetTensorShape("a");
                auto shape_b = accessor.GetTensorShape("b");
                if (!shape_a.empty() && !shape_b.empty()) {
                    return shape_a[0] * shape_b[0];
                }
                return int64_t(0);
            };
            customRule.description = "Calculate product of first dimensions of a and b";
            extractor_.RegisterExtractionRule(customRule);
        }
        
        void UseExtractedParameters() {
            switch (extractionResult_.scenario) {
                case ScenarioType::SCENARIO_1: {
                    auto batch_size = std::any_cast<int64_t>(
                        extractionResult_.keyParameters["batch_size"]);
                    auto hidden_size = std::any_cast<int64_t>(
                        extractionResult_.keyParameters["hidden_size"]);
                    auto mode = std::any_cast<std::string>(
                        extractionResult_.keyParameters["mode"]);
                    break;
                }
                default:
                    break;
            }
        }
        

        ScenarioInference scenarioInference_;
        ScenarioAwareBindingManager bindingManager_;
        ScenarioConstraintManager constraintManager_;
        KeyParameterExtractor extractor_;
        ExtractionResult extractionResult_;
    };
} // namespace param_validator

struct GMMFRWeightQuantInputParams {
    ge::DataType xDtype = ge::DT_INT8;
    ge::DataType wDtype = ge::DT_INT8;
    ge::Format xFormat = ge::FORMAT_ND;
    ge::Format wFormat = ge::FORMAT_ND;
    bool xTrans = False;
    bool wTrans = False;

    float shareInputWeight = 0.0;
    int64_t shareInputOffset = 0;

    int64_t mSize = -1,
    int64_t kSize = -1,
    int64_t nSize = -1,
    int64_t groupListType = -1,
    int64_t groupSize = -1,
    int64_t groupNum = -1,
    int64_t coreNum = -1,
    int64_t groupType = -1,
    int64_t splitItem = -1,
};

class TilingKeyConfigure {
public:
    uint8_t socVersionType = 0;
    uint8_t quantizationScenario = 0;
    uint8_t algorithm = 0;
    uint8_t transposeSituation = 0;
    uint8_t antiquantType = 0;
    uint8_t quantType = 0;
    uint8_t optionInputSituation = 0;
    uint8_t weightFormat = 0;
    uint8_t templateCustom = 0;
    uint8_t apiConstexpr = 0;
    void PrintTilingKeyLog() const
    {
        std::stringstream ss;
        ss << "socVersionType: " << static_cast<uint32_t>(this->socVersionType)
           << " quantizationScenario: " << static_cast<uint32_t>(this->quantizationScenario)
           << " algorithm: " << static_cast<uint32_t>(this->algorithm)
           << " transposeSituation: " << static_cast<uint32_t>(this->transposeSituation)
           << " antiquantType: " << static_cast<uint32_t>(this->antiquantType)
           << " quantType: " << static_cast<uint32_t>(this->quantType)
           << " optionInputSituation: " << static_cast<uint32_t>(this->optionInputSituation)
           << " weightFormat: " << static_cast<uint32_t>(this->weightFormat)
           << " templateCustom: " << static_cast<uint32_t>(this->templateCustom)
           << " apiConstexpr: " << static_cast<uint32_t>(this->apiConstexpr);
        OP_LOGI("GMMWeightQuantBatchMatmul", "tilingKeyConfigure: %s", ss.str().c_str());
        return;
    }
    uint64_t GenTilingKey() const;
};

constexpr uint32_t MAX_TENSOR_CONT = 128;
constexpr uint32_t GROUPLIST_INDEX = 7;
constexpr uint32_t PER_TOKEN_SCALE_INDEX = 8;
constexpr uint32_t MIN_ND_DIM = 2;

template <typename T>
auto CeilDiv(T a, T b) -> T {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
auto CeilAlign(T a, T b) -> T {
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

enum DataSize {
    B4_DATA_SIZE = 0,
    B8_DATA_SIZE = 1,
    RESERVED = 999,
};

enum DataSize GetSizeByDataType(ge::DataType dType);


class GMMFRWeightQuantTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
    public:
        explicit GMMFRWeightQuantTiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
        {
            Reset();
        }
        ~GMMFRWeightQuantTiling() override = default;
    
        void Reset(gert::TilingContext *context) override
        {
            Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
            Reset();
        }
    
    protected:
        bool IsCapable() override;
        // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
        ge::graphStatus GetPlatformInfo() override;
        // 2、获取INPUT/OUTPUT/ATTR信息
        ge::graphStatus GetShapeAttrsInfo() override;
        // 3、计算数据切分TilingData
        ge::graphStatus DoOpTiling() override;
        // 4、计算高阶API的TilingData
        ge::graphStatus DoLibApiTiling() override;
        // 5、计算TilingKey
        uint64_t GetTilingKey() const override;
        // 6、计算Workspace 大小
        ge::graphStatus GetWorkspaceSize() override;
        // 7、保存Tiling数据
        ge::graphStatus PostTiling() override;
        void Reset();

        GMMFRWeightQuantInputParams inputParams_;
    
    private:
    const GroupedMatmulFinalizeRoutingCompileInfo* compileInfoPtr_;
        bool CheckRequiredParams();
        ScenarioType scenarioType_ = ScenarioType::NONE;
        vector<std::function<bool(gert::TilingContext *context)>> checkConditionFuncs_;
        vector<std::function<void(gert::TilingContext *context, GMMFRWeightQuantInputParams& inputParams_)>> SetInputParamsFuncs_;
        GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData tilingData_;
    };
} // namespace GroupedMatmulFinalizeRoutingArch35WeightQuantTiling
} // namespace optiling

#endif // GROUPED_QUANT_MATMUL_TILING_H