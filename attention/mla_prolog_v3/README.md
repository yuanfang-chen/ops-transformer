# MlaPrologV3
## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

-  **功能更新**：（相对于`aclnnMlaPrologV2WeightNz`的差异）
    -  新增 Query 与 Key 的尺度矫正因子，分别对应 `qc_qr_scale`（aclnn 参数 `qcQrScale`，$\alpha_q$）与 `kc_scale`（aclnn 参数 `kcScale`，$\alpha_{kv}$）。
    -  新增可选输入与属性，将 `cache_mode` 由必选改为可选。新增或扩展的能力包括：
        - `actual_seq_len`：当 BS 合轴且 `cache_mode` 为 `PA_BLK_BSND`/`PA_BLK_NZ` 时，用于描述每个 batch 的序列长度前缀和。
        - `k_nope_clip_alpha`：在 per-tile 量化场景下，对 `kv_cache` 做 clip 时使用的缩放因子。
        - `query_norm_flag`：控制是否输出 `query_norm`；在 aclnn 接口中由 `queryNormOutOptional` 是否为空推导。
        - `weight_quant_mode`：表示 `weight_dq`、`weight_uq_qr`、`weight_dkv_kr` 的量化模式。
        - `kv_cache_quant_mode`：表示 `kv_cache` 的量化模式。
        - `query_quant_mode`：表示 `query` 的量化模式。
        - `ckvkr_repo_mode`：表示 `kv_cache` 与 `kr_cache` 的存储模式。
        - `quant_scale_repo_mode`：表示量化 scale 的存储模式。
        - `tile_size`：表示 per-tile 量化时每个 tile 的大小，当前仅支持 128。
        - `query_norm` 与 `dequant_scale_q_norm`：新增可选输出，用于返回 `c^Q` 及其量化参数。
    -  调整 `cache_index` 的名称与位置，对应 aclnn 接口中的 `cacheIndexOptional`。
-  **算子功能**：推理场景下，用于完成 Multi-Head Latent Attention 前处理。整体计算包含四个主分支：首先对输入$x$乘以$W^{DQ}$并执行 RmsNorm 得到$c^Q$，再乘以 Query 尺度矫正因子$\alpha_q$后分为两路，第一路依次乘以$W^{UQ}$和$W^{UK}$得到$q^N$，第二路乘以$W^{QR}$并执行 ROPE 得到$q^R$；第三路对输入$x$乘以$W^{DKV}$并执行 RmsNorm 得到$c^{KV}$，再乘以 Key 尺度矫正因子$\alpha_{kv}$并写入 Cache 得到$k^C$；第四路对输入$x$乘以$W^{KR}$并执行 ROPE 后写入另一个 Cache 得到$k^R$。当 `query` 需要按 per-token-head 量化输出且 `kv_cache` 为 per-tensor 量化时，还会额外生成 `dequant_scale_q_nope`。
-  **计算公式**：

    RmsNorm公式

    $$
    \mathrm{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\mathrm{RMS}(x)}
    $$

    $$
    \mathrm{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query计算公式，包括下采样，RmsNorm和两次上采样

    $$
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$
    其中 $\alpha_q$ 是 Query 的尺度矫正参数。

    对Query进行ROPE旋转位置编码

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    Key计算公式，包括下采样和RmsNorm，将计算结果存入cache

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$
    其中 $\alpha_{kv}$ 是 Key 的尺度矫正参数。

    对Key进行ROPE旋转位置编码，并将结果存入cache

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

## 参数说明
| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| token_x                     | 输入      | 公式中用于计算 Query 和 Key 的输入 tensor | BF16, INT8, FLOAT8_E4M3FN | ND         |
| weight_dq                   | 输入      | 公式中用于计算 Query 下采样的权重矩阵$W^{DQ}$ <br> 不转置时各维度表示为（k，n） | BF16, INT8, FLOAT8_E4M3FN | FRACTAL_NZ |
| weight_uq_qr                | 输入      | 公式中用于计算 Query 升秩与位置编码的权重矩阵$W^{UQ}$、$W^{QR}$ <br> 不转置时各维度表示为（k，n）| BF16, INT8, FLOAT8_E4M3FN | FRACTAL_NZ |
| weight_uk                   | 输入      | 公式中用于生成$q^N$的权重$W^{UK}$ | BF16 | ND         |
| weight_dkv_kr               | 输入      | 公式中用于计算 Key 下采样与位置编码的权重矩阵$W^{DKV}$、$W^{KR}$ <br> 不转置时各维度表示为（k，n）| BF16, INT8, FLOAT8_E4M3FN | FRACTAL_NZ |
| rmsnorm_gamma_cq            | 输入      | 计算$c^Q$的 RmsNorm 公式中的$\gamma$参数 | BF16 | ND         |
| rmsnorm_gamma_ckv           | 输入      | 计算$c^{KV}$的 RmsNorm 公式中的$\gamma$参数 | BF16 | ND         |
| rope_sin                    | 输入      | 旋转位置编码的正弦参数矩阵 | BF16 | ND         |
| rope_cos                    | 输入      | 旋转位置编码的余弦参数矩阵 | BF16 | ND         |
| kv_cache                    | 输入/输出 | Cache Tensor，计算结果原地更新（对应$k^C$）| BF16, INT8, FLOAT8_E4M3FN | ND         |
| kr_cache                    | 输入/输出 | Key 位置编码的 Cache Tensor，计算结果原地更新（对应$k^R$） | BF16, INT8 | ND         |
| cache_index                 | 输入      | 存储 `kv_cache` 与 `kr_cache` 的索引 | INT64 | ND         |
| dequant_scale_x             | 输入      | `token_x` 的反量化参数 | FLOAT, FLOAT8_E8M0 | ND         |
| dequant_scale_w_dq          | 输入      | `weight_dq` 的反量化参数 | FLOAT, FLOAT8_E8M0 | ND         |
| dequant_scale_w_uq_qr       | 输入      | MatmulQcQr 结果的反量化参数 | FLOAT, FLOAT8_E8M0 | ND         |
| dequant_scale_w_dkv_kr      | 输入      | `weight_dkv_kr` 的反量化参数 | FLOAT, FLOAT8_E8M0 | ND         |
| quant_scale_ckv             | 输入      | `kv_cache` 输出量化参数 | FLOAT | ND         |
| quant_scale_ckr             | 输入      | `kr_cache` 输出量化参数 | FLOAT | ND         |
| smooth_scales_cq            | 输入      | RmsNormCq 输出的动态量化参数 | FLOAT | ND         |
| actual_seq_len              | 输入      | BS 合轴且 `cache_mode` 为 `PA_BLK_BSND`/`PA_BLK_NZ` 时使用的序列长度前缀和 | INT32 | ND         |
| k_nope_clip_alpha           | 输入      | per-tile 量化场景下，对 `kv_cache` 做 clip 操作时使用的缩放因子 | FLOAT | ND         |
| rmsnorm_epsilon_cq          | 输入      | 计算$c^Q$的 RmsNorm 公式中的$\epsilon$参数 | FLOAT | -          |
| rmsnorm_epsilon_ckv         | 输入      | 计算$c^{KV}$的 RmsNorm 公式中的$\epsilon$参数 | FLOAT | -          |
| cache_mode                  | 输入      | `kv_cache` 的存储模式 | CHAR* | -          |
| query_norm_flag             | 输入      | 是否输出 `query_norm`；在 aclnn 接口中由 `queryNormOutOptional` 是否为空推导 | BOOL | -          |
| weight_quant_mode           | 输入      | `weight_dq`、`weight_uq_qr`、`weight_dkv_kr` 的量化模式 | INT64 | -          |
| kv_cache_quant_mode         | 输入      | `kv_cache` 的量化模式 | INT64 | -          |
| query_quant_mode            | 输入      | `query` 的量化模式 | INT64 | -          |
| ckvkr_repo_mode             | 输入      | `kv_cache` 和 `kr_cache` 的存储模式 | INT64 | -          |
| quant_scale_repo_mode       | 输入      | 量化 scale 的存储模式 | INT64 | -          |
| tile_size                   | 输入      | per-tile 量化时每个 tile 的大小，当前仅支持 128 | INT64 | -          |
| qc_qr_scale                 | 输入      | Query 的尺度矫正因子，对应$\alpha_q$，默认值为 1.0 | FLOAT | -          |
| kc_scale                    | 输入      | Key 的尺度矫正因子，对应$\alpha_{kv}$，默认值为 1.0 | FLOAT | -          |
| query                       | 输出      | 公式中 Query 的输出 tensor（对应$q^N$） | BF16, INT8, FLOAT8_E4M3FN | ND         |
| query_rope                  | 输出      | 公式中 Query 位置编码的输出 tensor（对应$q^R$） | BF16 | ND |
| dequant_scale_q_nope        | 输出      | `query` 的量化参数；仅在 Query 按 per-token-head 量化输出时生效 | FLOAT | ND         |
| query_norm                  | 输出      | `token_x` 做 RmsNorm 后的输出 tensor（对应$c^Q$） | BF16, INT8, FLOAT8_E4M3FN | ND |
| dequant_scale_q_norm        | 输出      | `query_norm` 的量化参数 | FLOAT, FLOAT8_E8M0 | ND         |
                   
## 约束说明

-   shape约束
    -   若token_x的维度采用BS合轴，即(T, He)
        - rope_sin和rope_cos的shape为(T, Dr)
        - 当`cache_mode`为`PA_BSND`或`PA_NZ`时，cache_index的shape为(T,)
        - 当`cache_mode`为`PA_BLK_BSND`或`PA_BLK_NZ`时，cache_index的shape为(Sum(Ceil(S_i / BlockSize)))，并且actual_seq_len的shape为(B,)
        - int8全量化场景下，dequant_scale_x的shape为(T)；mxfp8全量化场景下为(T, He / 32)
        - query的shape为(T, N, Hckv)
        - query_rope的shape为(T, N, Dr)
        - 当`query_quant_mode = 1`且`kv_cache_quant_mode = 1`时，dequant_scale_q_nope的shape为(T, N, 1)；其他场景下输出空Tensor，shape为(0)
    - 若token_x的维度不采用BS合轴，即(B, S, He)
        - rope_sin和rope_cos的shape为(B, S, Dr)
        - 当`cache_mode`为`PA_BSND`或`PA_NZ`时，cache_index的shape为(B, S)
        - 当`cache_mode`为`PA_BLK_BSND`或`PA_BLK_NZ`时，cache_index的shape为(B, Ceil(S / BlockSize))
        - dequant_scale_x的shape为(B*S, 1)
        - query的shape为(B, S, N, Hckv)
        - query_rope的shape为(B, S, N, Dr)
        - 当`query_quant_mode = 1`且`kv_cache_quant_mode = 1`时，dequant_scale_q_nope的shape为(B*S, N, 1)；其他场景下输出空Tensor，shape为(0)
    -   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        - 如果B、S、T取值为0，则query、query_rope输出空Tensor，kv_cache、kr_cache不做更新。
        - 如果Skv取值为0，则query、query_rope、dequant_scale_q_nope正常计算，kv_cache、kr_cache不做更新，即输出空Tensor。
    -   当cache_mode为BSND时
        - token_x不采用BS合轴，即维度为(B, S, He)
        - kv_cache的维度为(B, S, Nkv, Dtile)
        - kr_cache的维度为(B, S, Nkv, Dr)
    -   当cache_mode为TND时
        - token_x采用BS合轴，即维度为(T, He)
        - kv_cache的维度为(T, Nkv, Dtile)
        - kr_cache的维度为(T, Nkv, Dr)
-   特殊约束
    - per-tile量化模式下，ckvkr_repo_mode和quant_scale_repo_mode必须同时为1；其他量化模式以及非量化场景下，ckvkr_repo_mode和quant_scale_repo_mode必须同时为0。
    - per-tile量化模式下，cache_mode只支持PA_BSND, BSND和TND。
    - 当ckvkr_repo_mode值为1时，kr_cache必须为空Tensor（即shape的乘积为0）。
-  支持场景说明：
    - Atlas A2/Atlas A3 支持：非量化、部分量化（`kv_cache` 非量化/per-channel/per-tile）以及 int8 全量化（`kv_cache` 非量化/per-tensor/per-tile）。
    - Ascend 950PR/Ascend 950DT 支持：非量化、部分量化（`kv_cache` 非量化/per-channel）以及 mxfp8 全量化（`kv_cache` 非量化/per-tensor/per-tile）。
    - 仅当 Query 在 full quant 或 mxfp8 full quant 且 `kv_cache_quant_mode = per-tensor` 时，`query_quant_mode` 才能设置为 1。
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
      <td rowspan="3">部分量化</td>
      <td>kv_cache非量化 </td>
      <td>
          入参：weight_uq_qr传入 per-token 量化数据，其余入参皆为非量化数据。dequant_scale_w_uq_qr字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kv_cache per-channel量化 </td>
      <td>
          入参：weight_uq_qr传入 per-token 量化数据，kv_cache、kr_cache传入 per-channel 量化数据，其余入参皆为非量化数据。dequant_scale_w_uq_qr、quant_scale_ckv、quant_scale_ckr字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：kv_cache、kr_cache返回 per-channel 量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kv_cache per-tile量化 </td>
      <td>
          入参：weight_uq_qr传入 per-token 量化数据，kv_cache传入 per-tile 量化数据，其余入参皆为非量化数据。dequant_scale_w_uq_qr字段必须传入，smooth_scales_cq字段可选传入，k_nope_clip_alpha字段必须传入 <br>
          出参：kv_cache返回 per-tile 量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">int8全量化</td>
      <td> kv_cache非量化</td>
      <td>
          入参：token_x传入 per-token 量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入 per-channel 量化数据，其余入参皆为非量化数据。dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache per-tensor量化 </td>
      <td>
          入参：token_x传入 per-token 量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入 per-channel 量化数据，kv_cache传入 per-tensor 量化数据，其余入参皆为非量化数据。dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr、quant_scale_ckv字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：query返回 per-token-head 量化数据，kv_cache返回 per-tensor 量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache per-tile量化 </td>
      <td>
          入参：token_x传入 per-token 量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入 per-channel 量化数据，其余入参皆为非量化数据。dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr字段必须传入，smooth_scales_cq字段可选传入，k_nope_clip_alpha字段必须传入 <br>
          出参：kv_cache返回 per-tile 量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">mxfp8全量化</td>
      <td> kv_cache非量化</td>
      <td>
          入参：token_x传入 per-token 量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入 mxfp8 量化数据，其余入参皆为非量化数据。dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr字段必须传入 <br>
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache per-tensor量化 </td>
      <td>
          入参：token_x传入 per-token 量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入 mxfp8 量化数据，kv_cache传入 per-tensor 量化数据，其余入参皆为非量化数据。dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr、quant_scale_ckv字段必须传入 <br>
          出参：query返回 per-token-head 量化数据，kv_cache返回 per-tensor 量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache per-tile量化 </td>
      <td>
          入参：token_x传入 per-token 量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入 mxfp8 量化数据，其余入参皆为非量化数据。dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr字段必须传入 <br>
          出参：kv_cache返回 per-tile 量化数据，其余出参返回非量化数据
      </td>
    </tr>
  </table>

## 调用说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_v3_fqkvq.cpp">MlaPrologV3 A5（mxfp8全量化，kv_cache per-tensor量化）接口测试用例代码
    </a>
    </td>
    <td class="tg-lboi">
    通过
    <a href="./docs/aclnnMlaPrologV3WeightNz.md">aclnnMlaPrologV3WeightNz
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
