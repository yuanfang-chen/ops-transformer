#idndef GROUPED_MATMUL_SCHEDULER_N_RESPLIT_H
#define GROUPED_MATMUL_SCHEDULER_N_RESPLIT_H

namespace Block {
    class GroupedMatmulSchedulerNResplit {
        public:
        struct Params {
            uint64_t mainBlockCount;
            uint64_t mainBlockSize;
            uint64_t firstTailBlockCount;
            uint64_t firstTailBlockSize;
            uint64_t secondTailBlockCount;
            uint64_t secondTailBlockSize;
            uint64_t coreNum;
            uint64_t cubeNumBlocksN;
            uint64_t baseM;
            uint64_t nSize;
        }

        GroupedMatmulSchedulerNResplit(const Params& params) : params_(params) {
        }

        template<typename CallbackM, typename CallbackMN>
        __aicore__ inline void operator()(uint64_t &startBasicBlockId, uint64_t mSize, CallbackM &&callbackM, CallbackMN &&callbackMN) {
            uint64_t mBlkNum = CeilDivide(mSize, static_cast<uint64_t>(params_.baseM));
            uint64_t mL1Step = CeilDivide(mSize, mBlkNum);
            uint64_t curBasicBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + params_.params_.coreNum;
            uint64_t basicBlockLimit = startBasicBlockId;
            for (uint64_t mOffset = 0; mOffset < mSize; mOffset += mL1Step) {
                uint64_t mL1Size = mOffset + mL1Step > mSize ? mSize - mOffset : mL1Step;
                callbackM(mOffset, mL1Size);

                uint64_t nBaseOffset = 0;
                RunSegment(curBasicBlockId, basicBlockLimit, nBaseOffset, mOffset, mL1Size, callbackMN, params_.mainBlockCount, params_.mainBlockSize);
                RunSegment(curBasicBlockId, basicBlockLimit, nBaseOffset, mOffset, mL1Size, callbackMN, params_.firstTailBlockCount, params_.firstTailBlockSize);
                RunSegment(curBasicBlockId, basicBlockLimit, nBaseOffset, mOffset, mL1Size, callbackMN, params_.secondTailBlockCount, params_.secondTailBlockSize);
            }
            startBasicBlockId = basicBlockLimit % params_.coreNum;
        }

        private:
        template<typename CallbackMN>
        __aicore__ inline void RunSegment(uint64_t &curBasicBlockId, uint64_t &basicBlockLimit, uint64_t &nBaseOffset, uint64_t mOffset, uint64_t mL1Size, CallbackMN &&callbackMN, uint64_t blockCount, uint64_t blockSize) {
            if (blockCount == 0) {
                return;
            }
            for (; curBasicBlockId < basicBlockLimit + blockCount; curBasicBlockId += params_.coreNum) {
                uint64_t nOffset =
                    nBaseOffset + ((curBasicBlockId - basicBlockLimit) % blockCount) * blockSize;
                uint64_t nL1Size = nOffset + blockSize > params_.nSize ? params_.nSize - nOffset : blockSize;

                callbackMN(mOffset, mL1Size, nOffset, nL1Size);
            }
            nBaseOffset += blockSize * blockCount;
            basicBlockLimit += blockCount;
        }

        Params params_;
    }
}

#endif