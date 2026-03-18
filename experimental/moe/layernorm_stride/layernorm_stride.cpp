#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace npu_ops_transformer_ext
{
    namespace LayernormStride
    {
        #include <iostream>
        #include <stdio.h>
        #include "kernel_operator.h"
        #include "tiling/platform/platform_ascendc.h"
        #include "dtype_convert.h"

        using namespace AscendC;

        constexpr int64_t UB_MAX_BYTES = 184*1024;
        constexpr int64_t BUFFER_NUM = 1;
        constexpr int64_t WORK_LOCAL_SIZE = 64;

        template <typename T>
        class layernorm_stride_kernel {
        public:
            __aicore__ inline layernorm_stride_kernel() {}

            __aicore__ inline void Init(const int64_t gbH, const int64_t gbW, 
                    const int64_t stride, const float epsilon, GM_ADDR in, GM_ADDR gamma, GM_ADDR beta, GM_ADDR out)
            {
                blockNum_ = GetBlockNum();
                blockIdx_ = GetBlockIdx();

                gbH_ = gbH;
                gbW_ = gbW;
                stride_ = stride;
                epsilon_ = epsilon;
                invert_ = (float)(1.0) / gbW;

                bkH_ = 1;
                bkW_ = gbW_;
                bkAlignW_ = AlignUp(bkW_, 64);
                bkLoop_ = (int64_t)(gbH_ / blockNum_);
                if(gbH_ % blockNum_ != 0){ bkLoop_ += 1; }

                inGm_.SetGlobalBuffer((__gm__ T*)in, gbH_ * gbW_);
                gammaGm_.SetGlobalBuffer((__gm__ T*)gamma, gbW_);
                betaGm_.SetGlobalBuffer((__gm__ T*)beta, gbW_);
                outGm_.SetGlobalBuffer((__gm__ T*)out, gbH_ * gbW_);

                pipe_.InitBuffer(inQueIn_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T));
                pipe_.InitBuffer(inQueGamma_, BUFFER_NUM, bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(inQueBeta_, BUFFER_NUM, bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(outQueOut_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T) * 2);

                pipe_.InitBuffer(bufQueWork_, WORK_LOCAL_SIZE * sizeof(float));
                pipe_.InitBuffer(bufQueSum_, 16*sizeof(float));


            }

            __aicore__ inline void Process()
            {
                LocalTensor<T> gamma_local = inQueGamma_.AllocTensor<T>();
                LocalTensor<T> beta_local = inQueBeta_.AllocTensor<T>();

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(gamma_local[bkAlignW_], gammaGm_, copy_pas, pad_pas);
                DataCopyPad(beta_local[bkAlignW_], betaGm_, copy_pas, pad_pas);

                inQueGamma_.EnQue(gamma_local);
                inQueBeta_.EnQue(beta_local);
                gammaLm_ = inQueGamma_.DeQue<T>();
                betaLm_ = inQueBeta_.DeQue<T>();

                gammaFloatLm_ = gammaLm_.template ReinterpretCast<float>();
                betaFloatLm_ = betaLm_.template ReinterpretCast<float>();
                
                Cast(gammaFloatLm_, gammaLm_[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);
                Cast(betaFloatLm_, betaLm_[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);

                for(int64_t i = 0; i < bkLoop_; i++){
                    if(i * blockNum_ + blockIdx_ < gbH_){
                        CopyIn(i);
                        Compute(i);
                        CopyOut(i);
                    }
                }

                inQueGamma_.FreeTensor(gamma_local);
                inQueBeta_.FreeTensor(beta_local);
            }

        private:
            __aicore__ inline void CopyIn(int64_t process)
            {
                LocalTensor<T> in_local = inQueIn_.AllocTensor<T>();

                int64_t offset = (process * blockNum_ + blockIdx_) * stride_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(in_local, inGm_[offset], copy_pas, pad_pas);
                inQueIn_.EnQue(in_local);
            }

            __aicore__ inline void Compute(int64_t progress)
            {
                LocalTensor<T> in_local = inQueIn_.DeQue<T>();
                
                if(bkAlignW_-AlignUp(bkW_,16) != 0){
                    Duplicate(in_local[AlignUp(bkW_,16)], (T)0.0, bkAlignW_-AlignUp(bkW_,16));
                }
                
                LocalTensor<T> out_local = outQueOut_.AllocTensor<T>();
                LocalTensor<float> infloat_local = out_local.template ReinterpretCast<float>();

                LocalTensor<float> sum_local     = bufQueSum_.Get<float>();
                LocalTensor<float> work_local    = bufQueWork_.Get<float>();
                
                Duplicate<float>(infloat_local, (float)0.0, bkAlignW_);
                Cast(infloat_local, in_local, RoundMode::CAST_NONE, bkAlignW_);
                {
                    Duplicate<float>(work_local, (float)0.0, 64);
                    BinaryRepeatParams byrt_pas = {1,1,1,0,8,0};
                    Add(work_local, infloat_local, work_local, 64, bkAlignW_/64, byrt_pas);
                    WholeReduceSum(sum_local, work_local, 64, 1, 1, 1, 8);
                }
                
                Muls(sum_local, sum_local, (float)invert_, 1);
                Muls(sum_local, sum_local, (float)(-1.0), 1);
                float cur_mean = sum_local.GetValue(0);

                Cast(infloat_local, in_local, RoundMode::CAST_NONE, bkAlignW_);
                Mul(infloat_local, infloat_local, infloat_local, bkW_);
                {
                    Duplicate<float>(work_local, (float)0.0, 64);
                    BinaryRepeatParams byrt_pas = {1,1,1,0,8,0};
                    Add(work_local, infloat_local, work_local, 64, bkAlignW_/64, byrt_pas);
                    WholeReduceSum(sum_local, work_local, 64, 1, 1, 1, 8);
                }

                Muls(sum_local, sum_local, (float)invert_, 1);
                Adds(sum_local, sum_local, (float)epsilon_, 1);
                Rsqrt(sum_local, sum_local, 1);
                float cur_var = sum_local.GetValue(0);

                Duplicate<float>(infloat_local, (float)0.0, bkAlignW_);
                Cast(infloat_local, in_local, RoundMode::CAST_NONE, bkW_);
                Muls(infloat_local, infloat_local, cur_var, bkW_);
                Mul(infloat_local, infloat_local, gammaFloatLm_, bkW_);
                Add(infloat_local, infloat_local, betaFloatLm_, bkW_);
                Cast(out_local, infloat_local, RoundMode::CAST_ROUND, bkW_);

                inQueIn_.FreeTensor(in_local);

                outQueOut_.EnQue(out_local);
            }

            __aicore__ inline void CopyOut(int64_t progress)
            {
                LocalTensor<T> out_local = outQueOut_.DeQue<T>();

                int64_t offset = (progress * blockNum_ + blockIdx_) * stride_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                DataCopyPad(outGm_[offset], out_local, copy_pas);

                outQueOut_.FreeTensor(out_local);
            }

        private:
            TPipe pipe_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueIn_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueGamma_, inQueBeta_;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOut_;

            GlobalTensor<T> inGm_, gammaGm_, betaGm_;
            GlobalTensor<T> outGm_;

            LocalTensor<T> gammaLm_, betaLm_;
            LocalTensor<float> gammaFloatLm_, betaFloatLm_;

            TBuf<QuePosition::VECCALC> bufQueWork_;
            TBuf<QuePosition::VECCALC> bufQueSum_;

            int64_t blockNum_, blockIdx_;
            int64_t gbH_, gbW_;
            
            int64_t bkH_;
            int64_t bkW_, bkAlignW_;
            int64_t bkLoop_;

            float epsilon_ = (float)0.0f;
            float invert_ = (float)0.0f;

            int64_t stride_;
        };

        extern "C" __global__ __aicore__ void compute_layernorm_stride(const int64_t gbH, const int64_t gbW, 
                const int64_t stride, const float epsilon, GM_ADDR in, GM_ADDR gamma, GM_ADDR beta, GM_ADDR out, int32_t dtype)
        {
            
                TYPE_SWITCH(dtype, T, {
                layernorm_stride_kernel<T> op;
                op.Init(gbH, gbW, stride, epsilon, in, gamma, beta, out);
                op.Process();
                });

        }


        void layernorm_stride_kernel_lanuch(int64_t blockDim, void* stream,
                const int64_t gbH, const int64_t gbW, const int64_t stride, const float epsilon, 
                uint8_t* in, uint8_t* gamma, uint8_t* beta, uint8_t* out, int32_t dtype)
        {   
            if(gbH < blockDim){ blockDim = gbH; }

            compute_layernorm_stride<<<blockDim, nullptr, stream>>>(gbH, gbW, stride, epsilon, in, gamma, beta, out, dtype);
        }

        inline int64_t align_up(const int64_t number, const int64_t alignSize)
        {
            if(number % alignSize == 0){
                return number;
            }
            
            return ((number / alignSize + 1) * alignSize);
        }        
        
        int judge_layernorm_stride_lanuch(const int64_t gbW,
                const int64_t ubSize)
        {
            constexpr int64_t BUFFER_NUM = 1;
            constexpr int64_t WORK_LOCAL_SIZE = 64;
            
            int64_t gbW_;
            
            int64_t bkH_;
            int64_t bkW_, bkAlignW_;
            
            gbW_ = gbW;

            bkH_ = 1;
            bkW_ = gbW_;
            bkAlignW_ = align_up(bkW_, 64);
            
            float use_byte = BUFFER_NUM * bkH_ * bkAlignW_ * sizeof(half) * 2;
            use_byte += BUFFER_NUM * bkAlignW_ * sizeof(half) * 2;
            use_byte += (WORK_LOCAL_SIZE+ 16 + 2 * bkAlignW_) * sizeof(float);  // bf16 增加 2* bkAlignW_ * sizeof(bfloat16_t) tbuf
            
            if(use_byte > ubSize){     
                std::cout << __FUNCTION__ << ": " << "bkW_,bkAlignW_,UB = "<< bkW_ << "," << bkAlignW_ << "," << use_byte/1024 << " KB" << std::endl;
                return 1; 
            }

            return 0;
        }

        int layernorm_stride_lanuch(int64_t blockDim, void* stream,
                const int64_t gbH, const int64_t gbW, const int64_t stride, const float epsilon, 
                uint8_t* in, uint8_t* gamma, uint8_t* beta, uint8_t* out, int32_t dtype)
        {
            int64_t ubSize = 184 * 1024;
            
            int ret = judge_layernorm_stride_lanuch(gbW, ubSize);
            if(ret == 0){
                layernorm_stride_kernel_lanuch(blockDim, stream, gbH, gbW, stride, epsilon, in, gamma, beta, out, dtype);
                return 0;
            }

            std::cout << __FUNCTION__ << ": " << "UB size is limited, please check!" << std::endl;
            return 1;
        }

        int64_t layernorm_stride_npu(int64_t blockDim, const int64_t hiddenDim, const double epsilon, torch::Tensor &in, torch::Tensor &gamma, torch::Tensor &beta, torch::Tensor &out) {
            TORCH_CHECK(torch_npu::utils::is_npu(in), "input tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(gamma), "gama tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(beta), "beta tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(out), "output tensor must be on NPU device");
            TORCH_CHECK(in.scalar_type() == at::kBFloat16 || in.scalar_type() == at::kHalf, "dtype of input tensor is invalid, only BF16 or FP16 is supported.");
            TORCH_CHECK(gamma.scalar_type() == at::kBFloat16 || gamma.scalar_type() == at::kHalf, "dtype of gamma tensor is invalid, only BF16 or FP16 is supported.");
            TORCH_CHECK(beta.scalar_type() == at::kBFloat16 || beta.scalar_type() == at::kHalf, "dtype of beta tensor is invalid, only BF16 or FP16 is supported.");
            TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf, "dtype of output tensor is invalid, only BF16 or FP16 is supported.");

            auto stream = c10_npu::getCurrentNPUStream().stream(false);
            int launchStatus = 0;
            auto acl_call = [=, &launchStatus]() -> int
            {
                launchStatus = layernorm_stride_lanuch(blockDim, stream, in.size(0) * in.size(1), hiddenDim, in.size(2), epsilon, (uint8_t *)in.data_ptr(), (uint8_t *)gamma.data_ptr(), (uint8_t *)beta.data_ptr(), (uint8_t *)out.data_ptr(), in.scalar_type() == at::kHalf ? 1 : 27);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("layernormStride", acl_call);

            return launchStatus;
        }

        TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
        {
            m.impl("layernorm_stride", layernorm_stride_npu);
        }
    }
}