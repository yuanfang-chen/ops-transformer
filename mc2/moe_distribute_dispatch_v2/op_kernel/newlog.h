namespace MoeDispatchLog_NEW {

template<typename T>
constexpr bool is_decimal_integer_v =
    std::is_same_v<T, int8_t> ||
    std::is_same_v<T, int16_t> ||
    std::is_same_v<T, int32_t> ||
    std::is_same_v<T, int64_t>;


template<typename T>
constexpr bool is_unsigned_integer_v =
    std::is_same_v<T, bool> ||
    std::is_same_v<T, uint8_t> ||
    std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, uint64_t>;


template<typename T>
constexpr bool is_fp_point_v =
    std::is_same_v<T, float> ||
    std::is_same_v<T, half> ||
    std::is_same_v<T, bfloat16_t>;

using namespace AscendC;
class Log {
public:

    __aicore__ inline void Init(TPipe * tpipe_,int32_t rankId);

    template<typename T>
    __aicore__ inline void print_value(const T& val);
    __aicore__ inline void LogInfo(const uint32_t line, const __gm__ char *msg, const GM_ADDR addr);
    __aicore__ inline void LogInfo(const uint32_t line, const __gm__ char *msg);
    template<typename T>
    __aicore__ inline void LogInfo(const uint32_t line, const __gm__ char *msg,   const T num);

    template<typename T>
    __aicore__ inline uint32_t GetElementWidth(T val);

    template<typename T, size_t N>
    __aicore__  inline void PrintLevel(const LocalTensor<T>& tensor, size_t actual_size,
                       const uint32_t (&shape)[N], const int (&strides)[N],
                       uint32_t max_element_width);

    template<typename T, size_t N>
    __aicore__ inline  void PrintValue(const LocalTensor<T>& tensor, const uint32_t (&shape)[N]);

    template<typename T, size_t N>
    __aicore__  inline void PrintLevel(const GlobalTensor<T>& tensor, size_t actual_size,
                        const uint32_t (&shape)[N], const int (&strides)[N],
                        uint32_t max_element_width);
    

    template<typename T, size_t N>
    __aicore__ inline void PrintValue(const GlobalTensor<T>& tensor, const uint32_t (&shape)[N]);
    
    template <typename T,size_t N>
    __aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg, LocalTensor<T>& tensor,const uint32_t (&shape)[N]); //按指定shap打

    template <typename T,size_t N>
    __aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg, GlobalTensor<T>& tensor,const uint32_t (&shape)[N]); //按指定shap打

private:
    int32_t rankId_ = -1;
    uint32_t aivId_ = GetBlockIdx();
    TPipe * tpipe_ = nullptr;
};

template<typename T>
__aicore__ inline void Log::print_value(const T& val) {
    if constexpr (is_decimal_integer_v<T>) {
        printf("%d", val);
    }
    else if constexpr (is_unsigned_integer_v<T>) {
        printf("%u",val);
    }
    else if constexpr (is_fp_point_v<T>) {
        printf("%f", val);
    }
}

__aicore__ inline void Log::Init(TPipe * tpipe,int32_t rankId) {
    tpipe_ = tpipe;
    rankId_ =rankId;
}

__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg, const GM_ADDR addr){
    const __gm__ void * addrtemp = static_cast<const __gm__ void*>(addr);
    printf("[rankId: %d][aivId: %d][line: %d]", rankId_, aivId_, line);
    printf("%s: %p\n", msg, addrtemp);
}
__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg){
    printf("[rankId: %d][aivId: %d][line: %d]%s\n", rankId_, aivId_, line, msg);
}

template<typename T>
__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg,   const T num){
    printf("[rankId: %d][aivId: %d][line: %d]%s: ", rankId_, aivId_, line, msg);
    print_value(num);
    printf("\n");
}

template<typename T>
__aicore__ inline uint32_t Log::GetElementWidth(T val) {
    float32_t d_val{0};
    if constexpr ( is_fp_point_v<T> && std::is_same_v<T, bfloat16_t>){
        d_val = AscendC::ToFloat(val);
    }
    else{
       d_val = static_cast<float32_t>(val);
    }
     
    uint32_t length = 0;
    if(d_val < 0) {
        length++;
        d_val = -d_val;
    }
    long long int_part = static_cast<long long>(d_val);
    if(int_part == 0){
        length++;
    } else {
        uint64_t temp = int_part;
        while(temp > 0){
            temp = temp / 10;
            length++;
        }
    }

    if(std::is_floating_point<T>::value) {
        length +=7;
    }

    return length;
}

template<typename T, size_t N>
__aicore__  inline void Log::PrintLevel(const LocalTensor<T>& tensor, size_t actual_size,
                       const uint32_t (&shape)[N], const int (&strides)[N],
                       uint32_t max_element_width) {
    struct State {
        int dim;
        int pos;
        size_t offset;
    };

    State stack[8];
    int top = 0;
    stack[0] = {0, 0, 0};
    printf("[");

    while (top >= 0) {
        State& s = stack[top];

        if (s.dim == N - 1) {
            size_t elem_offset = s.offset + s.pos * strides[s.dim];
            bool is_last = (s.pos == shape[s.dim] - 1);

            int current_printed = 0;
            if (elem_offset < actual_size) {
                // 打印数值
                T val = tensor(elem_offset);
                current_printed = GetElementWidth(val);
                print_value(val);
            } else {
                // 打印占位符，并记录其宽度为 1
                printf("-");
                current_printed = 1;
            }

            // 打印逗号
            if (!is_last) {
                printf(",");
                current_printed += 1;
            }

            // 计算补白：确保 "-" 和数字占据相同的总宽度
            // 补白 = (最大宽度 + 逗号位) - 当前已打印长度 + 额外间距
            int padding = (max_element_width + (is_last ? 0 : 1)) - current_printed + 1;
            for (int i = 0; i < padding; ++i){ // 增加 +1，保持间隔
                printf(" ");
            }

            if (is_last) {
                printf("]");
                top--;
                if (top >= 0) {
                    stack[top].pos++;
                    if (stack[top].pos < shape[stack[top].dim]) {
                        printf("\n");
                        for (int i = 0; i <= top; ++i) printf(" ");
                    }
                }
            } else {
                s.pos++;
            }
        } else {
            if (s.pos < shape[s.dim]) {
                size_t next_offset = s.offset + s.pos * strides[s.dim];
                top++;
                stack[top] = {(int)s.dim + 1, 0, next_offset};
                printf("[");
            } else {
                printf("]");
                top--;
                if (top >= 0) {
                    stack[top].pos++;
                    if (stack[top].pos < shape[stack[top].dim]) {
                        printf(",\n");
                        for (int i = 0; i <= top; ++i){
                            printf(" ");
                        }
                    }
                }
            }
        }
    }
    printf("\n");
}


template<typename T, size_t N>
__aicore__ inline void Log::PrintValue(const LocalTensor<T>& tensor, const uint32_t (&shape)[N]) {
    PipeBarrier<PIPE_ALL>();
    size_t actual_size = tensor.GetSize();

    uint32_t max_w = 0;
    for (uint32_t index = 0; index < actual_size; index++) {
        T val = tensor(index);
        uint32_t len = GetElementWidth(val);
        max_w = max_w > len? max_w : len;
    }

    int strides[N];
    strides[N-1] = 1;
    for (int i = (int)N - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    } 

    PrintLevel(tensor, actual_size, shape, strides, max_w);
    printf("\n");
}


template <typename T,size_t N>
__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg, LocalTensor<T>& tensor,const uint32_t (&shape)[N]) {//按指定shap打
    printf("[rankId: %d][aivId: %d][line: %d]", rankId_, aivId_, line);
    printf("%s\n", msg);
    if(tensor.GetSize() == 0){
        printf("tensor is empty\n");
        return;
    }
    bool cast = !is_decimal_integer_v<T> && !is_unsigned_integer_v<T> && !is_fp_point_v<T>;
    if (cast) {
        printf("[ERROR][printf only support float, int, uint, bool .. plese read document and cast data type]\n");
        return ;
    }
    PrintValue(tensor, shape);
}

template<typename T, size_t N>
__aicore__  inline void Log::PrintLevel(const GlobalTensor<T>& tensor, size_t actual_size,
                       const uint32_t (&shape)[N], const int (&strides)[N],
                       uint32_t max_element_width) {
    struct State {
        int dim;
        int pos;
        size_t offset;
    };

    State stack[8];
    int top = 0;
    stack[0] = {0, 0, 0};
    printf("[");

    while (top >= 0) {
        State& s = stack[top];

        if (s.dim == N - 1) {
            size_t elem_offset = s.offset + s.pos * strides[s.dim];
            bool is_last = (s.pos == shape[s.dim] - 1);

            int current_printed = 0;
            if (elem_offset < actual_size) {
                // 打印数值
                T val = tensor(elem_offset);
                current_printed = GetElementWidth(val);
                print_value(val);
            } else {
                // 打印占位符，并记录其宽度为 1
                printf("-");
                current_printed = 1;
            }

            // 打印逗号
            if (!is_last) {
                printf(",");
                current_printed += 1;
            }

            // 计算补白：确保 "-" 和数字占据相同的总宽度
            // 补白 = (最大宽度 + 逗号位) - 当前已打印长度 + 额外间距
            int padding = (max_element_width + (is_last ? 0 : 1)) - current_printed + 1;
            for (int i = 0; i < padding; ++i){ // 保留 +1，间距隔开
                printf(" ");
            }

            if (is_last) {
                printf("]");
                top--;
                if (top >= 0) {
                    stack[top].pos++;
                    if (stack[top].pos < shape[stack[top].dim]) {
                        printf("\n");
                        for (int i = 0; i <= top; ++i) printf(" ");
                    }
                }
            } else {
                s.pos++;
            }
        } else {
            if (s.pos < shape[s.dim]) {
                size_t next_offset = s.offset + s.pos * strides[s.dim];
                top++;
                stack[top] = {(int)s.dim + 1, 0, next_offset};
                printf("[");
            } else {
                printf("]");
                top--;
                if (top >= 0) {
                    stack[top].pos++;
                    if (stack[top].pos < shape[stack[top].dim]) {
                        printf(",\n");
                        for (int i = 0; i <= top; ++i){
                            printf(" ");
                        }
                    }
                }
            }
        }
    }
    printf("\n");
}

template<typename T, size_t N>
__aicore__  inline void Log::PrintValue(const GlobalTensor<T>& tensor, const uint32_t (&shape)[N]) {
    PipeBarrier<PIPE_ALL>();
    size_t actual_size = tensor.GetSize();

    // 预扫描获取最大元素宽度
    uint32_t max_w = 0;
    for (uint32_t index = 0; index < actual_size; index++) {
        T val = tensor(index);
        uint32_t len = GetElementWidth(val);
        max_w = max_w > len? max_w : len;
    }

    int strides[N];
    strides[N-1] = 1;
    for (int i = (int)N - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    } 

    PrintLevel(tensor, actual_size, shape, strides, max_w);
    printf("\n");
}

template <typename T,size_t N>
__aicore__ inline void Log::LogInfo(const uint32_t line, const __gm__ char *msg, GlobalTensor<T>& tensor,const uint32_t (&shape)[N]) {//按指定shap打
    printf("[rankId: %d][aivId: %d][line: %d]", rankId_, aivId_, line);
    printf("%s\n", msg);
    bool cast = !is_decimal_integer_v<T> && !is_unsigned_integer_v<T> && !is_fp_point_v<T>;
    if (cast) {
        printf("[ERROR][printf only support float, int, uint, bool.. plese read document and cast data type]\n");
        return ; 
    }
    uint64_t elemNum = 1;
    for (uint32_t i = 0; i < N; i++) {
        elemNum = elemNum * shape[i];
    }
    if(elemNum == 0) {
        printf("tensor is empty\n");
        return;
    }
    GlobalTensor<T> tempTensor;
    tempTensor.SetGlobalBuffer(const_cast<__gm__ T*>(tensor.GetPhyAddr()),elemNum);
    PrintValue(tempTensor, shape);
}
//用法，在类中定义logger，然后在需要打印的地方调用LOG_INFO(...)
//在最开始定义LOG_INIT(...)，
//#define DEBUG 1
//#if DEBUG
//#define LOG_INFO(...) this->logger.LogInfo(__LINE__,__VA_ARGS__)
//#define LOG_INIT(...) this->logger.Init(__VA_ARGS__)
//#else
//#define LOG_INFO(...) 
//#define LOG_INIT(...) 
//#endif

}
