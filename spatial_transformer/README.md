# SpatialTransformer

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | : :------: |
| <term>Ascend 910PR/Ascend 910DT</term>                         |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

算子功能：Spatial Transformer Network (STN) 算子用于对输入张量进行仿射变换。该算子通过变换矩阵 theta 对输入图像 x 进行空间变换，输出变换后的图像 y。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入张量，4-D 张量，形状为 [batch, channels, height, width]。</tdtd>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
          FLOAT16、FLOAT、DOUBLE</td>
      <td>NCHW、NC1HWC0</td>
    </tr>
    <tr>
      <td>theta</td>
      <td>输入</td>
      <td>变换矩阵，3-D 张量，形状为 [batch, 2, 3] 或 [2, 3]，包含仿射变换参数。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
          FLOAT16、FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>变换后的输出张量，4-D 张量，形状为 [batch, channels, height, width]。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
          FLOAT16、FLOAT、DOUBLE</td>
      <td>NCHW、NC1HWC0</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>指定输出的高度和宽度，包含 2 个整数。默认为 [-1, -1]，表示使用输入尺寸。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_theta</td>
      <td>属性</td>
      <td>默认的仿射变换参数，当 use_default_theta 为 true 时使用。包含 6 个浮点数。默认为空列表。</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>属性</td>
      <td>如果为 true，则输入和输出张量的 4 个角像素中心对齐，保留角像素的值。默认为 false。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_default_theta</td>
      <td>属性</td>
      <td>指定哪些 theta 参数从 default_theta 使用。1 表示使用默认值，0 表示使用输入 theta。包含 6 个整数。默认为空列表。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量 x 的格式必须为 NCHW 或 NC1HWC0。
- 输出张量 y 的格式必须与输入张量 x 的格式一致。
- 变换矩阵 theta 的形状必须为 [batch, 2, 3] 或 [2, 3]。
- 当 use_default_theta 为空列表时，使用输入 theta 进行变换。
- 当 use_default_theta 不为空时，对应位置为 1 的参数使用 default_theta 中的值，为 0 的参数使用输入 theta 中的值。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_spatial_transformer](./examples/test_geir_spatial_transformer.cpp)   | 通过[算子IR](./op_graph/spatial_transformer_proto.h)构图方式调用SpatialTransformer算子。 |
