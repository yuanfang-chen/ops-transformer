# MlaPrologV2

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

说明：根据 `attention/mla_prolog_v2/op_host/op_api/*` 当前实现，`aclnnMlaPrologV2WeightNz*` 直调接口在 Ascend950 上会返回不支持；如需直调，请迁移到 `aclnnMlaPrologV3WeightNz*`。

## 功能说明

-  **算子功能**：用于推理场景下的 Multi-Head Latent Attention 前处理计算。主要计算过程分为四路：首先对输入$x$乘以$W^{DQ}$进行下采样并执行 RmsNorm，然后分成两路；第一路依次乘以$W^{UQ}$和$W^{UK}$，经过两次上采样后得到$q^N$；第二路乘以$W^{QR}$后再做旋转位置编码（ROPE），得到$q^R$。第三路对输入$x$乘以$W^{DKV}$并执行 RmsNorm 后，将结果写入 Cache，得到$k^C$；第四路对输入$x$乘以$W^{KR}$后执行旋转位置编码，再写入另一组 Cache，得到$k^R$。
-  **计算公式**：

    RmsNorm公式

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query计算公式，包括下采样，RmsNorm和两次上采样

    $$
    c^Q = RmsNorm(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    对Query的进行ROPE旋转位置编码

    $$
    q^R = ROPE(c^Q \cdot W^{QR})
    $$

    Key计算公式，包括下采样和RmsNorm，将计算结果存入cache

    $$
    c^{KV} = RmsNorm(x \cdot W^{DKV})
    $$

    $$
    k^C = Cache(c^{KV})
    $$

    对Key进行ROPE旋转位置编码，并将结果存入cache

    $$
    k^R = Cache(ROPE(x \cdot W^{KR}))
    $$

## 参数说明
  
| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| token_x                     | 输入      | 公式中计算 Query 和 Key 的输入 tensor | INT8, BF16 | ND         |
| weight_dq                   | 输入      | 公式中计算 Query 的下采样权重矩阵$W^{DQ}$ | INT8, BF16 | FRACTAL_NZ |
| weight_uq_qr                 | 输入      | 公式中计算 Query 上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$ | INT8, BF16 | FRACTAL_NZ |
| weight_uk                   | 输入      | 公式中计算 Query 第二次上采样的权重矩阵$W^{UK}$ | BF16       | ND         |
| weight_dkv_kr                | 输入      | 公式中计算 KV 下采样权重矩阵$W^{DKV}$和 Key 位置编码权重矩阵$W^{KR}$ | INT8, BF16| FRACTAL_NZ |
| rmsnorm_gamma_cq             | 输入      | 计算$c^Q$的 RmsNorm 公式中$\gamma$参数 | BF16       | ND         |
| rmsnorm_gamma_ckv            | 输入      | 计算$c^{KV}$的 RmsNorm 公式中$\gamma$参数 | BF16       | ND         |
| rope_sin                    | 输入      | 旋转位置编码的正弦参数矩阵 | BF16       | ND         |
| rope_cos                    | 输入      | 旋转位置编码的余弦参数矩阵 | BF16       | ND         |
| cache_index                 | 输入      | 存储kvCache和krCache的索引 | INT64          | ND         |
| kv_cache                 | 输入/ 输出| KV Cache，对应$k^C$，计算结果原地更新 | BF16, INT8 | ND         |
| kr_cache                 | 输入/ 输出| KR Cache，对应$k^R$，计算结果原地更新 | BF16, INT8 | ND         |
| dequant_scale_x      | 输入      | `token_x` 为 INT8 的全量化场景下使用的反量化参数 | FLOAT          | ND         |
| dequant_scale_w_dq    | 输入      | `weight_dq` 为 INT8 的全量化场景下使用的反量化参数 | FLOAT          | ND         |
| dequant_scale_w_uq_qr  | 输入      | `weight_uq_qr` 为 INT8 时用于 MatmulQcQr 后反量化的参数 | FLOAT          | ND         |
| dequant_scale_w_dkv_kr | 输入      | `weight_dkv_kr` 为 INT8 的全量化场景下使用的反量化参数 | FLOAT          | ND         |
| quant_scale_ckv      | 输入      | `kv_cache` 为 INT8 时使用的量化参数 | FLOAT          | ND         |
| quant_scale_ckr      | 输入      | `kr_cache` 为 INT8 时使用的量化参数 | FLOAT          | ND         |
| smooth_scales_cq     | 输入      | `weight_uq_qr` 为 INT8 时可选的动态量化平滑参数 | FLOAT          | ND         |
| rmsnorm_epsilon_cq           | 输入      | 计算$c^Q$的RmsNorm公式中$\epsilon$参数 | DOUBLE         | -          |
| rmsnorm_epsilon_ckv          | 输入      | 计算$c^{KV}$的RmsNorm公式中$\epsilon$参数 | DOUBLE         | -          |
| cache_mode          | 输入      | Cache 模式，当前仅支持 `PA_BSND` 和 `PA_NZ` | CHAR*          | -          |
| query                   | 输出      | 公式中 Query 的输出 tensor（对应$q^N$） | BF16, INT8 | ND         |
| query_rope               | 输出      | 公式中 Query 位置编码的输出 tensor（对应$q^R$） | BF16       | ND |
| dequant_scale_q_nope | 输出     | Query 输出为 INT8 且 `kv_cache` 为 INT8 时，对应的反量化参数 | FLOAT             | ND         |
                   
## 约束说明

-   shape约束
    -   若token_x的维度采用BS合轴，即(T, He)
        - rope_sin和rope_cos的shape为(T, Dr)
        - cache_index的shape为(T,)
        - dequant_scale_x的shape为(T, 1)
        - query的shape为(T, N, Hckv)
        - query_rope的shape为(T, N, Dr)
        - `token_x` 和 `kv_cache` 同为 INT8 的场景下，dequant_scale_q_nope 的 shape 为(T, N, 1)
    - 若token_x的维度不采用BS合轴，即(B, S, He)
        - rope_sin和rope_cos的shape为(B, S, Dr)
        - cache_index的shape为(B, S)
        - dequant_scale_x的shape为(B*S, 1)
        - query的shape为(B, S, N, Hckv)
        - query_rope的shape为(B, S, N, Dr)
        - `token_x` 和 `kv_cache` 同为 INT8 的场景下，dequant_scale_q_nope 的 shape 为(B*S, N, 1)
    -   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        - 如果B、S、T取值为0，则query、query_rope输出空Tensor，kv_cache、kr_cache不做更新。
        - 如果Skv取值为0，则query、query_rope、dequant_scale_q_nope正常计算，kv_cache、kr_cache不做更新，即输出空Tensor。
-   在 aclnn 直调场景下，非 `token_x` 和 `kv_cache` 同为 INT8 的场景，`dequant_scale_q_nope` 对应输出为空Tensor；若传入空指针，接口内部会创建占位Tensor。
- weight_dq，weight_uq_qr，weight_dkv_kr在不转置的情况下各个维度的表示：（k，n）。
-  aclnnMlaPrologV2WeightNz接口支持场景：
    <table style="table-layout: auto;" border="1">
      <tr>
        <th colspan="2">场景</th>
        <th>含义</th>
      </tr>
      <tr>
        <td colspan="2">非量化</td>
        <td>
            入参：所有入参皆为非量化数据 <br> 
            出参：所有出参皆为非量化数据
        </td>
      </tr>
      <tr>
        <td rowspan="2">部分量化</td>
        <td>kv_cache非量化 </td>
        <td>
            入参：weight_uq_qr传入pertoken量化数据，其余入参皆为非量化数据 <br> 
            出参：所有出参返回非量化数据 
        </td>
      </tr>
      <tr>
        <td>kv_cache量化 </td>
        <td> 
            入参：weight_uq_qr传入pertoken量化数据，kv_cache、kr_cache传入perchannel量化数据，其余入参皆为非量化数据 <br> 
            出参：kv_cache、kr_cache返回perchannel量化数据，其余出参返回非量化数据 
        </td>
      </tr>
      <tr>
        <td rowspan="2">全量化</td>
        <td> kv_cache非量化</td>
        <td> 
            入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，其余入参皆为非量化数据 <br> 
            出参：所有出参皆为非量化数据
        </td>
      </tr>
      </tr>
      <tr>
        <td> kv_cache量化 </td>
        <td>
            入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，kv_cache传入pertensor量化数据，其余入参皆为非量化数据 <br>
            出参：query返回pertoken_head量化数据，kv_cache出参返回pertensor量化数据，其余出参范围非量化数据 
        </td>
      </tr>
    </table>

-  在不同量化场景下，参数的dtype和shape组合需要满足如下条件：
    <div style="overflow-x: auto; width: 100%;">
    <table style="table-layout: auto;" border="1">
      <tr>
        <th rowspan="3">参数名</th>
        <th rowspan="2" colspan="2">非量化场景</th>
        <th colspan="4">部分量化场景</th>
        <th colspan="4">全量化场景</th>
      </tr>
      <tr>
        <th colspan="2">kv_cache非量化</th>
        <th colspan="2">kv_cache量化</th>
        <th colspan="2">kv_cache非量化</th>
        <th colspan="2">kv_cache量化</th>
      </tr>
      <tr>
        <th>dtype</th>
        <th>shape</th>
        <th>dtype</th>
        <th>shape</th>
        <th>dtype</th>
        <th>shape</th>
        <th>dtype</th>
        <th>shape</th>
        <th>dtype</th>
        <th>shape</th>
      </tr>
      <tr>
        <td>token_x</td>
        <td>BFLOAT16</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>BFLOAT16</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>BFLOAT16</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>INT8</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>INT8</td>
        <td>· (B,S,He) <br> · (T, He)</td>
      </tr>
      <tr>
        <td>weight_dq</td>
        <td>BFLOAT16</td>
        <td> (He, Hcq)</td>
        <td>BFLOAT16</td>
        <td> (He, Hcq)</td>
        <td>BFLOAT16</td>
        <td> (He, Hcq)</td>
        <td>INT8</td>
        <td> (He, Hcq)</td>
        <td>INT8</td>
        <td> (He, Hcq)</td>
      </tr>
      <tr>
        <td>weight_uq_qr</td>
        <td>BFLOAT16</td>
        <td> (Hcq, N*(D+Dr))</td>
        <td>INT8</td>
        <td> (Hcq, N*(D+Dr))</td>
        <td>INT8</td>
        <td> (Hcq, N*(D+Dr))</td>
        <td>INT8</td>
        <td> (Hcq, N*(D+Dr))</td>
        <td>INT8</td>
        <td> (Hcq, N*(D+Dr))</td>
      </tr>
      <tr>
        <td>weight_uk</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
      </tr>
      <tr>
        <td>weight_dkv_kr</td>
        <td>BFLOAT16</td>
        <td> (He, Hckv+Dr)</td>
        <td>BFLOAT16</td>
        <td> (He, Hckv+Dr)</td>
        <td>BFLOAT16</td>
        <td> (He, Hckv+Dr)</td>
        <td>INT8</td>
        <td> (He, Hckv+Dr)</td>
        <td>INT8</td>
        <td> (He, Hckv+Dr)</td>
      </tr>
      <tr>
        <td> rmsnorm_gamma_cq </td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
      </tr>
      <tr>
        <td> rmsnorm_gamma_ckv </td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
      </tr>
      <tr>
        <td> rope_sin </td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
      </tr>
      <tr>
        <td> rope_cos </td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
      </tr>
      <tr>
        <td> cache_index </td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
      </tr>
      <tr>
        <td> kv_cache </td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
        <td>INT8</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
        <td>INT8</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
      </tr>
      <tr>
        <td> kr_cache </td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>INT8</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      </tr>
      <tr>
        <td> dequant_scale_x </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> · (B*S, 1) <br> · (T, 1)</td>
        <td>FLOAT</td>
        <td> · (B*S, 1) <br> · (T, 1)</td>
      </tr>
      <tr>
        <td> dequant_scale_w_dq </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hcq)</td>
        <td>FLOAT</td>
        <td> (1, Hcq)</td>
      </tr>
      <tr>
        <td> dequant_scale_w_uq_qr </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
      </tr>
      <tr>
        <td> dequant_scale_w_dkv_kr </td>
        <td> 无需赋值 </td>
        <td> / </td>
        <td> 无需赋值 </td>
        <td> / </td>
        <td> 无需赋值 </td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hckv+Dr) </td>
        <td>FLOAT</td>
        <td> (1, Hckv+Dr) </td>
      </tr>
      <tr>
        <td> quant_scale_ckv </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hckv) </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hckv) </td>
      </tr>
      <tr>
        <td> quant_scale_ckr </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Dr) </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
      </tr>
      <tr>
        <td> smooth_scales_cq </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hcq) </td>
        <td>FLOAT</td>
        <td> (1, Hcq) </td>
        <td>FLOAT</td>
        <td> (1, Hcq) </td>
        <td>FLOAT</td>
        <td> (1, Hcq) </td>
      </tr>
      <tr>
        <td> query </td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
        <td>INT8</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
      </tr>
      <tr>
        <td> query_rope </td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      </tr>
      <tr>
        <td> dequant_scale_q_nope </td>
        <td>无需赋值</td>
        <td>/</td>
        <td>无需赋值</td>
        <td>/</td>
        <td>无需赋值</td>
        <td>/</td>
        <td>无需赋值</td>
        <td>/</td>
        <td>FLOAT</td>
        <td> · (B*S, N, 1) <br> · (T, N, 1)</td>
      </tr>
    </table>
    </div>

## 调用说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0lax">调用方式</th>
    <th class="tg-0lax">样例代码</th>
    <th class="tg-0lax">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_v2.cpp">aclnnMlaPrologV2WeightNz接口通用示例代码
    </a>
    </td>
    <td class="tg-lboi" rowspan="2">通过
    <a href="./docs/aclnnMlaPrologV2WeightNz.md">aclnnMlaPrologV2WeightNz
    </a>
    接口方式调用算子。当前 `examples` 目录中保留了通用示例和非量化（BSH）示例；Ascend950 直调限制请参见接口文档说明。
  </td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_v2_nq_bsh.cpp">MlaPrologV2非量化（BSH）接口示例代码
    </a>
    </td>
  </tr>
</tbody></table>

<!-- ## 参考资源
[MlaProlog算子设计原理](../mla_prolog/docs/MlaProlog算子设计介绍.md) -->
