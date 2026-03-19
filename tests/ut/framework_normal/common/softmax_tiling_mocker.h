#pragma once
#include <cstdint>

// Mocker 单例，测试用例通过它控制 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize 的返回值
class SoftmaxTilingMocker {
public:
    static SoftmaxTilingMocker& GetInstance() {
        static SoftmaxTilingMocker instance;
        return instance;
    }

    void SetSoftMaxMinTmpSize(uint32_t val) { softMaxMinTmpSize_ = val; }
    void SetSoftMaxFlashV2MinTmpSize(uint32_t val) { softMaxFlashV2MinTmpSize_ = val; }
    void Reset() { softMaxMinTmpSize_ = 0; softMaxFlashV2MinTmpSize_ = 0; }

    uint32_t GetSoftMaxMinTmpSize() const { return softMaxMinTmpSize_; }
    uint32_t GetSoftMaxFlashV2MinTmpSize() const { return softMaxFlashV2MinTmpSize_; }

private:
    SoftmaxTilingMocker() = default;
    uint32_t softMaxMinTmpSize_{0};
    uint32_t softMaxFlashV2MinTmpSize_{0};
};