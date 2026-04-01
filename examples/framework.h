#include <type_traits>
#include <tuple>
#include <cstring>

// ========== 1.1 AOT Holder 基类（C++17 兼容） ==========

// 使用指针指向全局 constexpr 变量
template <typename T, const uint8_t* ValuePtr>
struct AOTHolder {
    using value_type = T;
    static constexpr const uint8_t* value_ptr = ValuePtr;
    static constexpr const T value = convert_from_bytes<T>(ValuePtr);

    static constexpr const uint8_t* bytes() {
        return ValuePtr;
    }
};

// ========== 1.3 运行时 Holder（保持兼容） ==========

struct TilingRuntimeHolder {
    static constexpr uint8_t value[] = {0};
};

// ========== 1.4 AOT 注册表 ==========

// 注册表基类
template <typename... Holders>
struct AOTRegistry {
    using holder_tuple = std::tuple<Holders...>;
    static constexpr size_t size = sizeof...(Holders);

    // 获取第 N 个 Holder
    template <size_t N>
    using get = std::tuple_element_t<N, holder_tuple>;
};

// ========== 1.5 分发器 ==========

// 运行时分发器：单入口，自动检测并分发
template <typename T, typename Registry>
struct AOTDispatcher {

    // 主入口：接收运行时值，自动分发
    // template <int BM, int BN, typename Func>
    template <typename Func>
    static void dispatch(const uint8_t* value, Func&& kernel_func) {
        dispatch_impl<0>(value, std::forward<Func>(kernel_func));
    }

private:
    // 编译时遍历所有注册的 AOT Holder
    template <size_t N, typename Func>
    static void dispatch_impl(const uint8_t* value, Func&& kernel_func) {
        if constexpr (N < Registry::size) {
            using Holder = typename Registry::template get<N>;

            // 运行时比较：检查是否匹配此 AOT Holder
            if (compare_equal(value, Holder::value_ptr)) {
                // 匹配：调用 AOT 版本（Holder 作为模板参数）
                kernel_func(Holder{}, *reinterpret_cast<const T*>(value));
            } else {
                // 不匹配：检查下一个 Holder
                dispatch_impl<N + 1>(value, std::forward<Func>(kernel_func));
            }
        } else {
            // 所有 Holder 都不匹配：调用运行时版本
            kernel_func(TilingRuntimeHolder{}, *reinterpret_cast<const T*>(value));
        }
    }

    // 结构体相等比较（使用 memcmp）
    static bool compare_equal(const uint8_t* a, const uint8_t* b) {
        return std::memcmp(a, b, sizeof(T)) == 0;
    }
};

template <typename T, typename S>
struct TilingHelper {
    static inline const T value = convert_from_bytes<T>(S::value_ptr);
};

template <typename T, typename HT>
inline __aicore__ const T *GetTiling(const T *rt_tiling)
{
    if constexpr (std::is_same_v<HT, TilingRuntimeHolder>) {
        return rt_tiling;
    } else {
        return &TilingHelper<T, HT>::value;
    }
}

