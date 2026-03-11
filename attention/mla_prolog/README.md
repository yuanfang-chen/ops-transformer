# MlaProlog
## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

-  **算子功能**：推理场景下，完成 Multi-Head Latent Attention 前处理计算。主要计算过程分为四路：首先对输入$x$乘以$W^{DQ}$进行下采样并执行 RmsNorm，随后分为两路；第一路依次乘以$W^{UQ}$和$W^{UK}$，经过两次上采样后得到$q^N$；第二路乘以$W^{QR}$后经过旋转位置编码（RoPE）得到$q^R$；第三路是输入$x$乘以$W^{DKV}$进行下采样和 RmsNorm 后写入 Cache，得到$k^C$；第四路是输入$x$乘以$W^{KR}$后经过旋转位置编码，再写入另一个 Cache，得到$k^R$。
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

    对Query进行RoPE旋转位置编码

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
| token_x                    | 输入      | 公式中用于计算 Query 和 Key 的输入 tensor | BF16, INT8 | ND         |
| weight_dq                  | 输入      | 公式中用于计算 Query 下采样的权重矩阵$W^{DQ}$ | BF16, INT8 | FRACTAL_NZ |
| weight_uq_qr               | 输入      | 公式中用于计算 Query 上采样和 Query RoPE 投影的拼接权重矩阵$[W^{UQ}|W^{QR}]$ | BF16, INT8 | FRACTAL_NZ |
| weight_uk                  | 输入      | 公式中用于计算最终 Query 输出的上采样权重矩阵$W^{UK}$ | BF16 | ND         |
| weight_dkv_kr              | 输入      | 公式中用于计算 Key 下采样和 Key RoPE 投影的拼接权重矩阵$[W^{DKV}|W^{KR}]$ | BF16, INT8 | FRACTAL_NZ |
| rmsnorm_gamma_cq           | 输入      | 计算$c^Q$的 RmsNorm 公式中的$\gamma$参数 | BF16 | ND         |
| rmsnorm_gamma_ckv          | 输入      | 计算$c^{KV}$的 RmsNorm 公式中的$\gamma$参数 | BF16 | ND         |
| rope_sin                   | 输入      | 旋转位置编码的正弦参数矩阵 | BF16 | ND         |
| rope_cos                   | 输入      | 旋转位置编码的余弦参数矩阵 | BF16 | ND         |
| cache_index                | 输入      | 存储 kv_cache 和 kr_cache 的更新索引 | INT64          | ND         |
| kv_cache                   | 输入/输出 | Cache tensor，计算结果原地更新（对应$k^C$） | BF16, INT8 | ND         |
| kr_cache                   | 输入/输出 | Key 位置编码 Cache，计算结果原地更新（对应$k^R$） | BF16, INT8 | ND         |
| dequant_scale_x            | 输入      | token_x 为 INT8 时的反量化参数；其他场景传入空指针 | FLOAT          | ND         |
| dequant_scale_w_dq         | 输入      | weight_dq 为 INT8 时的反量化参数；其他场景传入空指针 | FLOAT          | ND         |
| dequant_scale_w_uq_qr      | 输入      | weight_uq_qr 为 INT8 时的反量化参数 | FLOAT          | ND         |
| dequant_scale_w_dkv_kr     | 输入      | weight_dkv_kr 为 INT8 时的反量化参数；其他场景传入空指针 | FLOAT          | ND         |
| quant_scale_ckv            | 输入      | kv_cache 输出为 INT8 时的量化参数 | FLOAT          | ND         |
| quant_scale_ckr            | 输入      | kr_cache 输出为 INT8 时的量化参数 | FLOAT          | ND         |
| smooth_scales_cq           | 输入      | weight_uq_qr 为 INT8 时可选的动态量化平滑参数 | FLOAT          | ND         |
| rmsnorm_epsilon_cq         | 属性      | 计算$c^Q$的 RmsNorm 公式中的$\epsilon$参数 | DOUBLE         | -          |
| rmsnorm_epsilon_ckv        | 属性      | 计算$c^{KV}$的 RmsNorm 公式中的$\epsilon$参数 | DOUBLE         | -          |
| cache_mode                 | 属性      | Cache 模式，仅支持`PA_BSND`和`PA_NZ` | CHAR*          | -          |
| query                      | 输出      | 公式中 Query 的输出 tensor（对应$q^N$），dtype 为 BF16 | BF16 | ND         |
| query_rope                 | 输出      | 公式中 Query 位置编码的输出 tensor（对应$q^R$），dtype 为 BF16 | BF16 | ND |


## 约束说明

-   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
    - 如果B、S、T取值为0，则query、query_rope输出空Tensor，kv_cache、kr_cache不做更新。
    - 如果Skv取值为0，则query、query_rope正常计算，kv_cache、kr_cache不做更新，即输出空Tensor。
- weight_dq，weight_uq_qr，weight_dkv_kr在不转置的情况下各个维度的表示：（k，n）。
-  aclnnMlaProlog接口支持场景：
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
            入参：weight_uq_qr 为 INT8，必须传入 dequant_scale_w_uq_qr，smooth_scales_cq 可选传；其余入参为非量化数据 <br>
            出参：所有出参返回非量化数据 
        </td>
      </tr>
      <tr>
        <td>kv_cache量化 </td>
        <td> 
            入参：weight_uq_qr 为 INT8，必须传入 dequant_scale_w_uq_qr；kv_cache、kr_cache 为 INT8，必须传入 quant_scale_ckv、quant_scale_ckr；smooth_scales_cq 可选传，其余入参为非量化数据 <br>
            出参：kv_cache、kr_cache 返回 INT8，其余出参返回非量化数据
        </td>
      </tr>
      <tr>
        <td rowspan="2">全量化</td>
        <td>kv_cache非量化</td>
        <td>
            入参：token_x、weight_dq、weight_uq_qr、weight_dkv_kr 为 INT8，需分别传入 dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr；smooth_scales_cq 可选传；kv_cache、kr_cache 为非量化数据 <br>
            出参：kv_cache、kr_cache 返回非量化数据；query、query_rope 返回 BF16
        </td>
      </tr>
      <tr>
        <td>kv_cache量化</td>
        <td>
            入参：在“全量化 kv_cache非量化”基础上，kv_cache 为 INT8，且必须传入 quant_scale_ckv；kr_cache 仍为非量化数据 <br>
            出参：kv_cache 返回 INT8，kr_cache 返回非量化数据；query、query_rope 返回 BF16
        </td>
      </tr>
    </table>

-  在非量化和部分量化场景下，参数的dtype和shape组合需要满足如下条件：
    <div style="overflow-x: auto; width: 100%;">
    <table style="table-layout: auto;" border="1">
      <tr>
        <th rowspan="3">参数名</th>
        <th rowspan="2" colspan="2">非量化场景</th>
        <th colspan="4">部分量化场景</th>
      </tr>
      <tr>
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
      </tr>
      <tr>
        <td>token_x</td>
        <td>BFLOAT16</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>BFLOAT16</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>BFLOAT16</td>
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
      </tr>
      <tr>
        <td>weight_uq_qr</td>
        <td>BFLOAT16</td>
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
      </tr>
      <tr>
        <td>weight_dkv_kr</td>
        <td>BFLOAT16</td>
        <td> (He, Hckv+Dr)</td>
        <td>BFLOAT16</td>
        <td> (He, Hckv+Dr)</td>
        <td>BFLOAT16</td>
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
      </tr>
      <tr>
        <td> rmsnorm_gamma_ckv </td>
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
      </tr>
      <tr>
        <td> rope_cos </td>
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
      </tr>
      <tr>
        <td> kv_cache </td>
        <td>BFLOAT16</td>
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
      </tr>
      <tr>
        <td> dequant_scale_x </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
      </tr>
      <tr>
        <td> dequant_scale_w_uq_qr </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
      </tr>
      <tr>
        <td> dequant_scale_w_dkv_kr </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
        <td>无需赋值</td>
        <td> / </td>
      </tr>
      <tr>
        <td> quant_scale_ckv </td>
        <td>无需赋值</td>
        <td> / </td>
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
      </tr>
      <tr>
        <td> smooth_scales_cq </td>
        <td>无需赋值</td>
        <td> / </td>
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
      </tr>
      <tr>
        <td> query_rope </td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      </tr>
    </table>
    </div>

-  全量化场景补充约束：
    - 全量化且`kv_cache`非量化时：`token_x`、`weight_dq`、`weight_uq_qr`、`weight_dkv_kr` 的 dtype 为 `INT8`；`dequant_scale_x` shape 为`(B*S, 1)`或`(T, 1)`，`dequant_scale_w_dq` shape 为`(1, Hcq)`，`dequant_scale_w_uq_qr` shape 为`(1, N*(D+Dr))`，`dequant_scale_w_dkv_kr` shape 为`(1, Hckv+Dr)`；`smooth_scales_cq` 可选，shape 为`(1, Hcq)`；`kv_cache`、`kr_cache` 为 `BFLOAT16`。
    - 全量化且`kv_cache`量化时：在上一场景基础上，`kv_cache` 的 dtype 为 `INT8`，且必须传入 `quant_scale_ckv`，其 shape 为`(1, Hckv)`；`kr_cache` 仍为 `BFLOAT16`。
    - 全量化场景下，`query`与`query_rope`的 dtype 为 `BF16`，shape 与上表一致。

## 调用说明


<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_mla_prolog_nq_bsh.cpp">MlaProlog非量化（BSH）接口测试用例代码
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnMlaProlog.md">aclnnMlaProlog
    </a>
    接口方式调用算子
    </td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_nq_tnd.cpp">MlaProlog非量化（TND）接口测试用例代码
    </a>
  </td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_pqkvnq_bsh.cpp">MlaProlog半量化KV非量化（BSH）接口测试用例代码
    </a>
</td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_pqkvnq_tnd.cpp">MlaProlog半量化KV非量化（TND）接口测试用例代码
    </a>
  </td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_pqkvq_bsh.cpp">MlaProlog半量化KV量化（BSH）接口测试用例代码
    </a>
  </td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_pqkvq_tnd.cpp">MlaProlog半量化KV量化（TND）接口测试用例代码
    </a>
  </td>
  </tr>
</tbody></table>

<!-- ## 参考资源
[MlaProlog算子设计原理](docs/MlaProlog算子设计介绍.md) -->
