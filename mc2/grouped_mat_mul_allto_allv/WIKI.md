# GroupedMatMulAlltoAllv 算子技术 Wiki

## 目录

- [1. 算子概述](#1-算子概述)
- [2. MoE 架构中的位置](#2-moe-架构中的位置)
- [3. 算子功能详解](#3-算子功能详解)
- [4. 输入输出规格](#4-输入输出规格)
- [5. sendCounts / recvCounts 详解](#5-sendcounts--recvcounts-详解)
- [6. Kernel 执行流程](#6-kernel-执行流程)
- [7. 计算-通信重叠流水线](#7-计算-通信重叠流水线)

---

## 1. 算子概述

`GroupedMatMulAlltoAllv` 是一个面向 **MoE（Mixture-of-Experts）** Transformer 大模型的高性能融合算子，运行在昇腾 NPU 上。它将 **分组矩阵乘法（Grouped MatMul）**、**Unpermute** 和 **AlltoAllV 集合通信** 三个操作融合到一个 kernel 中执行，实现 **"先计算后通信"** 的流水线模式。

同时，它还可以选择性地并行执行 **共享专家（Shared Expert）** 的 MatMul，使路由专家和共享专家的计算在同一个算子内完成。

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 24px; margin: 20px 0; color: white;">
  <h3 style="margin-top: 0; color: white;">核心价值</h3>
  <table style="width: 100%; border-collapse: separate; border-spacing: 8px;">
    <tr>
      <td style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 12px; text-align: center; width: 33%;">
        <div style="font-size: 24px;">&#9889;</div>
        <div style="font-weight: bold;">计算通信重叠</div>
        <div style="font-size: 13px; opacity: 0.9;">每完成一个专家的 GMM 计算，立即发起该专家的 AlltoAllV 通信，实现流水线并行</div>
      </td>
      <td style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 12px; text-align: center; width: 33%;">
        <div style="font-size: 24px;">&#128171;</div>
        <div style="font-weight: bold;">减少内存开销</div>
        <div style="font-size: 13px; opacity: 0.9;">三步操作融合为一个 kernel，避免中间 tensor 的分配和读写</div>
      </td>
      <td style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 12px; text-align: center; width: 33%;">
        <div style="font-size: 24px;">&#127919;</div>
        <div style="font-weight: bold;">路由+共享并行</div>
        <div style="font-size: 13px; opacity: 0.9;">共享专家 MatMul 与路由专家 GMM+AlltoAllV 同时进行</div>
      </td>
    </tr>
  </table>
</div>

---

## 2. MoE 架构中的位置

在 MoE Transformer 的推理/训练流程中，token 需要经过路由、分发、专家计算和结果汇聚四个阶段。`GroupedMatMulAlltoAllv` 负责的是 **专家计算完成后的结果汇聚阶段**（即 MoE FFN 的输出侧）。

<div style="margin: 20px 0; overflow-x: auto;">
  <div style="min-width: 800px;">
    <!-- MoE Pipeline -->
    <div style="display: flex; align-items: stretch; gap: 0; font-family: 'Segoe UI', Arial, sans-serif;">
      <!-- Stage 1 -->
      <div style="flex: 1; text-align: center;">
        <div style="background: #e8f4fd; border: 2px solid #4a90d9; border-radius: 10px; padding: 14px 8px; margin: 0 4px;">
          <div style="font-weight: bold; color: #2c5282; font-size: 13px;">Stage 1</div>
          <div style="font-size: 15px; font-weight: bold; margin: 6px 0;">Router + Gate</div>
          <div style="font-size: 12px; color: #4a5568;">TopK 路由选择<br>确定 token → expert 映射</div>
        </div>
        <div style="color: #4a90d9; font-size: 28px; margin: 4px 0;">&#8595;</div>
      </div>
      <!-- Arrow -->
      <div style="display: flex; align-items: center; color: #a0aec0; font-size: 24px; padding: 0 2px;">&#10132;</div>
      <!-- Stage 2 -->
      <div style="flex: 1; text-align: center;">
        <div style="background: #ebf8ff; border: 2px solid #4299e1; border-radius: 10px; padding: 14px 8px; margin: 0 4px;">
          <div style="font-weight: bold; color: #2b6cb0; font-size: 13px;">Stage 2</div>
          <div style="font-size: 15px; font-weight: bold; margin: 6px 0;">AlltoAllV + Permute + GMM</div>
          <div style="font-size: 12px; color: #4a5568;">先通信后计算<br>将 token 发给目标专家</div>
        </div>
        <div style="font-size: 11px; margin-top: 6px; padding: 4px 8px; background: #bee3f8; border-radius: 6px; display: inline-block; color: #2b6cb0; font-weight: bold;">
          AlltoAllvGroupedMatMul
        </div>
      </div>
      <!-- Arrow -->
      <div style="display: flex; align-items: center; color: #a0aec0; font-size: 24px; padding: 0 2px;">&#10132;</div>
      <!-- Stage 3 (Highlighted) -->
      <div style="flex: 1; text-align: center;">
        <div style="background: #fff5f5; border: 3px solid #e53e3e; border-radius: 10px; padding: 14px 8px; margin: 0 4px; box-shadow: 0 0 12px rgba(229, 62, 62, 0.3);">
          <div style="font-weight: bold; color: #c53030; font-size: 13px;">Stage 3</div>
          <div style="font-size: 15px; font-weight: bold; margin: 6px 0;">GMM + Unpermute + AlltoAllV</div>
          <div style="font-size: 12px; color: #4a5568;">先计算后通信<br>专家结果发回各卡</div>
        </div>
        <div style="font-size: 11px; margin-top: 6px; padding: 4px 8px; background: #fed7d7; border-radius: 6px; display: inline-block; color: #c53030; font-weight: bold;">
          &#11088; GroupedMatMulAlltoAllv（本算子）
        </div>
      </div>
      <!-- Arrow -->
      <div style="display: flex; align-items: center; color: #a0aec0; font-size: 24px; padding: 0 2px;">&#10132;</div>
      <!-- Stage 4 -->
      <div style="flex: 1; text-align: center;">
        <div style="background: #f0fff4; border: 2px solid #48bb78; border-radius: 10px; padding: 14px 8px; margin: 0 4px;">
          <div style="font-weight: bold; color: #276749; font-size: 13px;">Stage 4</div>
          <div style="font-size: 15px; font-weight: bold; margin: 6px 0;">Combine + Add</div>
          <div style="font-size: 12px; color: #4a5568;">路由专家 + 共享专家<br>加权求和输出</div>
        </div>
        <div style="color: #48bb78; font-size: 28px; margin: 4px 0;">&#8595;</div>
      </div>
    </div>
  </div>
</div>

**简单来说**：
- **Stage 2**（`AlltoAllvGroupedMatMul`）：把 token 从各卡发给对应的专家，然后做 GMM（下投影） → **通信 → 计算**
- **Stage 3**（`GroupedMatMulAlltoAllv`，本算子）：专家做完 GMM（上投影）后，把结果发回各卡 → **计算 → 通信**

---

## 3. 算子功能详解

### 3.1 计算公式

本算子融合了两条并行的计算路径：

<div style="display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap;">
  <!-- 路由专家路径 -->
  <div style="flex: 1; min-width: 360px; border: 2px solid #e53e3e; border-radius: 10px; overflow: hidden;">
    <div style="background: #e53e3e; color: white; padding: 10px 16px; font-weight: bold; font-size: 15px;">
      路由专家路径（Routed Expert）
    </div>
    <div style="padding: 16px;">
      <div style="background: #fff5f5; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <div style="font-family: 'Courier New', monospace; font-size: 14px; text-align: center;">
          <strong>Step 1:</strong> gmmY = gmmX &times; gmmWeight<br>
          <span style="color: #718096; font-size: 12px;">(对每个专家分别做矩阵乘法)</span>
        </div>
      </div>
      <div style="text-align: center; color: #e53e3e; font-size: 20px;">&#8595;</div>
      <div style="background: #fff5f5; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <div style="font-family: 'Courier New', monospace; font-size: 14px; text-align: center;">
          <strong>Step 2:</strong> unpermOut = Unpermute(gmmY)<br>
          <span style="color: #718096; font-size: 12px;">(将结果从专家分组序恢复为原始 token 序)</span>
        </div>
      </div>
      <div style="text-align: center; color: #e53e3e; font-size: 20px;">&#8595;</div>
      <div style="background: #fff5f5; border-radius: 8px; padding: 12px;">
        <div style="font-family: 'Courier New', monospace; font-size: 14px; text-align: center;">
          <strong>Step 3:</strong> y = AlltoAllV(unpermOut)<br>
          <span style="color: #718096; font-size: 12px;">(将结果通过集合通信发回各卡)</span>
        </div>
      </div>
    </div>
  </div>
  <!-- 共享专家路径 -->
  <div style="flex: 1; min-width: 360px; border: 2px solid #4299e1; border-radius: 10px; overflow: hidden;">
    <div style="background: #4299e1; color: white; padding: 10px 16px; font-weight: bold; font-size: 15px;">
      共享专家路径（Shared Expert，可选）
    </div>
    <div style="padding: 16px;">
      <div style="background: #ebf8ff; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <div style="font-family: 'Courier New', monospace; font-size: 14px; text-align: center;">
          <strong>Step 1:</strong> mmY = mmX &times; mmWeight<br>
          <span style="color: #718096; font-size: 12px;">(共享专家的标准矩阵乘法)</span>
        </div>
      </div>
      <div style="margin-top: 30px; padding: 16px; background: #ebf8ff; border-radius: 8px; text-align: center;">
        <div style="font-size: 13px; color: #2b6cb0;">
          &#128161; 共享专家计算在 AIC 核上执行<br>
          与路由专家的 AlltoAllV 通信并行
        </div>
      </div>
    </div>
  </div>
</div>

### 3.2 数据流图

<div style="margin: 20px 0; padding: 20px; background: #fafafa; border: 1px solid #e2e8f0; border-radius: 12px; overflow-x: auto;">
  <div style="min-width: 780px;">
    <div style="display: flex; gap: 32px;">
      <!-- 左列：路由专家路径 -->
      <div style="flex: 3; display: flex; flex-direction: column; align-items: center;">
        <div style="font-weight: bold; color: #e53e3e; margin-bottom: 8px; font-size: 13px;">路由专家路径</div>
        <!-- 输入 -->
        <div style="display: flex; gap: 16px; margin-bottom: 4px;">
          <div style="background: #4299e1; color: white; padding: 10px 20px; border-radius: 8px; text-align: center; font-size: 13px;">
            <strong>gmmX</strong><br>(A, H1)
          </div>
          <div style="background: #4299e1; color: white; padding: 10px 20px; border-radius: 8px; text-align: center; font-size: 13px;">
            <strong>gmmWeight</strong><br>(e, H1, N1)
          </div>
        </div>
        <div style="font-size: 24px; color: #a0aec0; margin: 2px 0;">&#8595;</div>
        <!-- GMM -->
        <div style="background: #fc8181; color: white; padding: 14px 24px; border-radius: 10px; text-align: center; width: 280px;">
          <strong style="font-size: 15px;">Grouped MatMul</strong><br>
          <span style="font-size: 12px;">对 e 个专家分别做矩阵乘<br>gmmY[i] = gmmX[i] &times; gmmWeight[i]</span>
        </div>
        <div style="font-size: 24px; color: #a0aec0; margin: 2px 0;">&#8595;</div>
        <!-- Unpermute -->
        <div style="background: #f6ad55; color: white; padding: 12px 24px; border-radius: 10px; text-align: center; width: 280px;">
          <strong style="font-size: 15px;">Unpermute</strong><br>
          <span style="font-size: 12px;">将专家分组结果恢复为 token 原始顺序</span>
        </div>
        <div style="font-size: 24px; color: #a0aec0; margin: 2px 0;">&#8595;</div>
        <!-- AlltoAllV -->
        <div style="background: #9f7aea; color: white; padding: 12px 24px; border-radius: 10px; text-align: center; width: 280px;">
          <strong style="font-size: 15px;">AlltoAllV 集合通信</strong><br>
          <span style="font-size: 12px;">按 sendCounts/recvCounts 将结果发回各 EP rank</span>
        </div>
        <div style="font-size: 24px; color: #a0aec0; margin: 2px 0;">&#8595;</div>
        <!-- 输出 y -->
        <div style="background: #48bb78; color: white; padding: 10px 20px; border-radius: 8px; text-align: center; font-size: 13px;">
          <strong>y</strong><br>(BSK, N1)
        </div>
      </div>
      <!-- 中间分隔线 -->
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <div style="width: 2px; flex: 1; background: repeating-linear-gradient(to bottom, #cbd5e0 0px, #cbd5e0 6px, transparent 6px, transparent 12px);"></div>
        <div style="padding: 8px 0; color: #718096; font-size: 12px; font-weight: bold; writing-mode: vertical-lr; letter-spacing: 2px;">并行执行</div>
        <div style="width: 2px; flex: 1; background: repeating-linear-gradient(to bottom, #cbd5e0 0px, #cbd5e0 6px, transparent 6px, transparent 12px);"></div>
      </div>
      <!-- 右列：共享专家路径 -->
      <div style="flex: 2; display: flex; flex-direction: column; align-items: center;">
        <div style="font-weight: bold; color: #3182ce; margin-bottom: 8px; font-size: 13px;">共享专家路径（可选）</div>
        <!-- 输入 -->
        <div style="display: flex; gap: 16px; margin-bottom: 4px;">
          <div style="background: #90cdf4; color: #2a4365; padding: 10px 20px; border-radius: 8px; text-align: center; font-size: 13px; border: 2px dashed #4299e1;">
            <strong>mmX</strong><br>(BS, H2)
          </div>
          <div style="background: #90cdf4; color: #2a4365; padding: 10px 20px; border-radius: 8px; text-align: center; font-size: 13px; border: 2px dashed #4299e1;">
            <strong>mmWeight</strong><br>(H2, N2)
          </div>
        </div>
        <div style="font-size: 24px; color: #a0aec0; margin: 2px 0;">&#8595;</div>
        <!-- Shared MatMul -->
        <div style="background: #63b3ed; color: white; padding: 14px 24px; border-radius: 10px; text-align: center; width: 240px; border: 2px dashed rgba(255,255,255,0.5);">
          <strong style="font-size: 15px;">Shared MatMul</strong><br>
          <span style="font-size: 12px;">mmY = mmX &times; mmWeight</span>
        </div>
        <div style="font-size: 24px; color: #a0aec0; margin: 2px 0;">&#8595;</div>
        <!-- 输出 mmY -->
        <div style="background: #9ae6b4; color: #22543d; padding: 10px 20px; border-radius: 8px; text-align: center; font-size: 13px; border: 2px dashed #48bb78;">
          <strong>mmY</strong><br>(BS, N2)
        </div>
        <!-- 说明 -->
        <div style="margin-top: 16px; padding: 12px; background: #ebf8ff; border-radius: 8px; text-align: center; font-size: 12px; color: #2b6cb0; width: 240px;">
          &#128161; 共享专家结果<strong>保留在本地</strong><br>不参与 AlltoAllV 通信
        </div>
      </div>
    </div>
  </div>
</div>

---

## 4. 输入输出规格

### 4.1 函数原型

本算子通过 ACLNN 两段式接口调用。第一段获取 workspace 大小和 executor，第二段执行计算：

<div style="margin: 20px 0; padding: 16px 20px; background-color: #1e1e1e; color: #d4d4d4; border-radius: 10px; overflow-x: auto; border: 1px solid #333;">
<pre style="margin: 0; font-family: Consolas, 'Courier New', monospace; font-size: 13px; line-height: 1.7;">
<span style="color: #6a9955;">// ————— 第一段：获取 workspace 大小 —————</span>
aclnnStatus <span style="color: #dcdcaa; font-weight: bold;">aclnnGroupedMatMulAlltoAllvGetWorkspaceSize</span>(
    <span style="color: #569cd6;">const aclTensor*</span>   <span style="color: #9cdcfe;">gmmX</span>,                      <span style="color: #6a9955;">// [必选 输入] 路由专家左矩阵</span>
    <span style="color: #569cd6;">const aclTensor*</span>   <span style="color: #9cdcfe;">gmmWeight</span>,                  <span style="color: #6a9955;">// [必选 输入] 路由专家右矩阵（权重）</span>
    <span style="color: #569cd6;">const aclTensor*</span>   <span style="color: #ce9178;">sendCountsTensorOptional</span>,  <span style="color: #6a9955;">// [可选 输入] 暂不支持，传 nullptr</span>
    <span style="color: #569cd6;">const aclTensor*</span>   <span style="color: #ce9178;">recvCountsTensorOptional</span>,  <span style="color: #6a9955;">// [可选 输入] 暂不支持，传 nullptr</span>
    <span style="color: #569cd6;">const aclTensor*</span>   <span style="color: #ce9178;">mmXOptional</span>,               <span style="color: #6a9955;">// [可选 输入] 共享专家左矩阵</span>
    <span style="color: #569cd6;">const aclTensor*</span>   <span style="color: #ce9178;">mmWeightOptional</span>,           <span style="color: #6a9955;">// [可选 输入] 共享专家右矩阵</span>
    <span style="color: #569cd6;">const char*</span>        <span style="color: #9cdcfe;">group</span>,                     <span style="color: #6a9955;">// [必选 属性] HCCL 通信组名</span>
    <span style="color: #569cd6;">int64_t</span>            <span style="color: #9cdcfe;">epWorldSize</span>,               <span style="color: #6a9955;">// [必选 属性] EP 通信域大小</span>
    <span style="color: #569cd6;">const aclIntArray*</span> <span style="color: #9cdcfe;">sendCounts</span>,                <span style="color: #6a9955;">// [必选 属性] 各卡各专家发送 token 数</span>
    <span style="color: #569cd6;">const aclIntArray*</span> <span style="color: #9cdcfe;">recvCounts</span>,                <span style="color: #6a9955;">// [必选 属性] 各卡各专家接收 token 数</span>
    <span style="color: #569cd6;">bool</span>               <span style="color: #ce9178;">transGmmWeight</span>,            <span style="color: #6a9955;">// [可选 属性] 是否转置 gmmWeight</span>
    <span style="color: #569cd6;">bool</span>               <span style="color: #ce9178;">transMmWeight</span>,             <span style="color: #6a9955;">// [可选 属性] 是否转置 mmWeight</span>
    <span style="color: #569cd6;">aclTensor*</span>         <span style="color: #4ec9b0;">y</span>,                         <span style="color: #6a9955;">// [必选 输出] 路由专家 AlltoAllV 后的结果</span>
    <span style="color: #569cd6;">aclTensor*</span>         <span style="color: #4ec9b0;">mmYOptional</span>,               <span style="color: #6a9955;">// [可选 输出] 共享专家计算结果（本地）</span>
    <span style="color: #569cd6;">uint64_t*</span>          workspaceSize,             <span style="color: #6a9955;">// [输出] 所需 workspace 大小</span>
    <span style="color: #569cd6;">aclOpExecutor**</span>    executor                   <span style="color: #6a9955;">// [输出] 算子执行器</span>
);

<span style="color: #6a9955;">// ————— 第二段：执行算子 —————</span>
aclnnStatus <span style="color: #dcdcaa; font-weight: bold;">aclnnGroupedMatMulAlltoAllv</span>(
    <span style="color: #569cd6;">void*</span>           workspace,
    <span style="color: #569cd6;">uint64_t</span>        workspaceSize,
    <span style="color: #569cd6;">aclOpExecutor*</span>  executor,
    <span style="color: #569cd6;">aclrtStream</span>     stream
);
</pre>
<div style="margin-top: 10px; display: flex; gap: 16px; flex-wrap: wrap; font-size: 11px; border-top: 1px solid #333; padding-top: 8px;">
  <span><span style="color: #9cdcfe;">■</span> 必选参数</span>
  <span><span style="color: #ce9178;">■</span> 可选参数</span>
  <span><span style="color: #4ec9b0;">■</span> 输出参数</span>
  <span><span style="color: #569cd6;">■</span> 类型</span>
  <span><span style="color: #6a9955;">■</span> 注释</span>
</div>
</div>

### 4.2 参数逐项说明

下面按照函数原型中的参数顺序，逐一说明每个参数的 shape、数据类型和含义。

#### 输入 Tensor

<div style="margin: 16px 0;">
<table style="width: 100%; border-collapse: collapse; font-size: 14px;">
  <thead>
    <tr style="background: #3182ce; color: white;">
      <th style="padding: 10px 14px; text-align: left;">参数名</th>
      <th style="padding: 10px 14px; text-align: center;">必选/可选</th>
      <th style="padding: 10px 14px; text-align: left;">Shape</th>
      <th style="padding: 10px 14px; text-align: left;">数据类型</th>
      <th style="padding: 10px 14px; text-align: left;">描述</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: #ebf8ff;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;"><code>gmmX</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8; text-align: center;"><span style="color: #e53e3e; font-weight: bold;">必选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;"><strong>(A, H1)</strong></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">FP16 / BF16 / HIFLOAT8</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">GMM 左矩阵。所有专家的输入 token 按行拼接在一起，A 行共 A 个 token，每个 token 的特征维度为 H1</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;"><code>gmmWeight</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8; text-align: center;"><span style="color: #e53e3e; font-weight: bold;">必选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;"><strong>(e, H1, N1)</strong></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">同 gmmX</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">GMM 右矩阵。包含 e 个专家各自的权重矩阵，每个权重 shape 为 (H1, N1)，即从隐藏层维度 H1 映射到输出维度 N1</td>
    </tr>
    <tr style="background: #ebf8ff;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;"><code>sendCountsTensorOptional</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8; text-align: center;"><span style="color: #718096;">可选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">(e &times; epWorldSize,)</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">INT32 / INT64</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">当前版本<strong>暂不支持</strong>，必须传 nullptr。未来用于动态传入发送 token 计数</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;"><code>recvCountsTensorOptional</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8; text-align: center;"><span style="color: #718096;">可选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">(e &times; epWorldSize,)</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">INT32 / INT64</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">当前版本<strong>暂不支持</strong>，必须传 nullptr</td>
    </tr>
    <tr style="background: #ebf8ff;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;"><code>mmXOptional</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8; text-align: center;"><span style="color: #718096;">可选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;"><strong>(BS, H2)</strong></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">同 gmmX</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #bee3f8;">共享专家左矩阵。BS 个 token，每个 token 特征维度为 H2。需与 mmWeightOptional 同时传入或同为 nullptr</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px;"><code>mmWeightOptional</code></td>
      <td style="padding: 8px 14px; text-align: center;"><span style="color: #718096;">可选</span></td>
      <td style="padding: 8px 14px;"><strong>(H2, N2)</strong></td>
      <td style="padding: 8px 14px;">同 gmmX</td>
      <td style="padding: 8px 14px;">共享专家右矩阵（权重）。从隐藏层维度 H2 映射到输出维度 N2。需与 mmXOptional 同时传入或同为 nullptr</td>
    </tr>
  </tbody>
</table>
</div>

#### 属性参数

<div style="margin: 16px 0;">
<table style="width: 100%; border-collapse: collapse; font-size: 14px;">
  <thead>
    <tr style="background: #805ad5; color: white;">
      <th style="padding: 10px 14px; text-align: left;">参数名</th>
      <th style="padding: 10px 14px; text-align: center;">必选/可选</th>
      <th style="padding: 10px 14px; text-align: left;">类型</th>
      <th style="padding: 10px 14px; text-align: left;">描述</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: #faf5ff;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;"><code>group</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd; text-align: center;"><span style="color: #e53e3e; font-weight: bold;">必选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">STRING</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">HCCL 通信组名称，字符串长度必须在 (0, 128) 之间。通过 <code>HcclGetCommName</code> 获取</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;"><code>epWorldSize</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd; text-align: center;"><span style="color: #e53e3e; font-weight: bold;">必选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">INT64</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">EP（Expert Parallel）通信域大小，即参与 AlltoAllV 通信的 NPU 卡数。A3 支持 8/16/32/64/128；950 支持 2/4/8/16/32/64</td>
    </tr>
    <tr style="background: #faf5ff;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;"><code>sendCounts</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd; text-align: center;"><span style="color: #e53e3e; font-weight: bold;">必选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">aclIntArray*<br><span style="font-size: 12px;">大小 = e &times; epWorldSize（最大 256）</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">每个元素表示本卡的某个专家需要发送给某张目标卡的 token 数。详见 <a href="#5-sendcounts--recvcounts-详解">第 5 节</a></td>
    </tr>
    <tr>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;"><code>recvCounts</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd; text-align: center;"><span style="color: #e53e3e; font-weight: bold;">必选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">aclIntArray*<br><span style="font-size: 12px;">大小 = e &times; epWorldSize（最大 256）</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">每个元素表示本卡需要从某张源卡接收的某个专家的 token 数</td>
    </tr>
    <tr style="background: #faf5ff;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;"><code>transGmmWeight</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd; text-align: center;"><span style="color: #718096;">可选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">BOOL（默认 false）</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e9d8fd;">gmmWeight 是否需要转置。true 时 gmmWeight shape 含义变为 (e, N1, H1)</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px;"><code>transMmWeight</code></td>
      <td style="padding: 8px 14px; text-align: center;"><span style="color: #718096;">可选</span></td>
      <td style="padding: 8px 14px;">BOOL（默认 false）</td>
      <td style="padding: 8px 14px;">mmWeight 是否需要转置。true 时 mmWeight shape 含义变为 (N2, H2)</td>
    </tr>
  </tbody>
</table>
</div>

#### 输出 Tensor

<div style="margin: 16px 0;">
<table style="width: 100%; border-collapse: collapse; font-size: 14px;">
  <thead>
    <tr style="background: #38a169; color: white;">
      <th style="padding: 10px 14px; text-align: left;">参数名</th>
      <th style="padding: 10px 14px; text-align: center;">必选/可选</th>
      <th style="padding: 10px 14px; text-align: left;">Shape</th>
      <th style="padding: 10px 14px; text-align: left;">数据类型</th>
      <th style="padding: 10px 14px; text-align: left;">描述</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: #f0fff4;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #c6f6d5;"><code>y</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #c6f6d5; text-align: center;"><span style="color: #e53e3e; font-weight: bold;">必选</span></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #c6f6d5;"><strong>(BSK, N1)</strong></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #c6f6d5;">FP16 / BF16</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #c6f6d5;">AlltoAllV 通信后的路由专家最终输出。BSK = sum(recvCounts)，即本卡从所有其他卡接收回的 token 总数。每个 token 的输出维度为 N1</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px;"><code>mmYOptional</code></td>
      <td style="padding: 8px 14px; text-align: center;"><span style="color: #718096;">可选</span></td>
      <td style="padding: 8px 14px;"><strong>(BS, N2)</strong></td>
      <td style="padding: 8px 14px;">FP16 / BF16</td>
      <td style="padding: 8px 14px;">共享专家 MatMul 的本地输出。仅当同时传入 mmXOptional 和 mmWeightOptional 时才输出。<strong>不参与 AlltoAllV 通信</strong>，结果保留在本卡</td>
    </tr>
  </tbody>
</table>
</div>

<div style="margin: 16px 0; padding: 14px 16px; background: #fffaf0; border-left: 4px solid #ed8936; border-radius: 0 8px 8px 0; font-size: 13px;">
  <strong>约束</strong>：mmXOptional、mmWeightOptional、mmYOptional 三者必须同时传入或同时为 nullptr。不能只传部分。
</div>

### 4.3 Shape 变量速查表

上面参数中出现的各 shape 变量含义汇总如下：

<div style="margin: 16px 0;">
<table style="width: 100%; border-collapse: collapse; font-size: 14px;">
  <thead>
    <tr style="background: #2d3748; color: white;">
      <th style="padding: 10px 14px; text-align: left; border-radius: 8px 0 0 0;">变量</th>
      <th style="padding: 10px 14px; text-align: left;">含义</th>
      <th style="padding: 10px 14px; text-align: left;">取值范围</th>
      <th style="padding: 10px 14px; text-align: left; border-radius: 0 8px 0 0;">如何确定</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: #f7fafc;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>A</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">本卡需要发送的总 token 数（gmmX 的行数）</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">&gt;0</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">= sum(sendCounts)，即 sendCounts 数组所有元素之和</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>H1</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">路由专家的隐藏层大小（GMM 的 K 维度）</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">(0, 65536)</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">gmmX 的第 1 维 = gmmWeight 的第 1 维</td>
    </tr>
    <tr style="background: #f7fafc;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>e</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">单卡上的专家个数</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">&le;32，且 e &times; epWorldSize &le; 256</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">gmmWeight 的第 0 维</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>N1</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">路由专家的输出维度（GMM 的 N 维度）</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">(0, 65536)</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">gmmWeight 的第 2 维 = y 的第 1 维</td>
    </tr>
    <tr style="background: #f7fafc;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>BSK</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">本卡接收的总 token 数（y 的行数）</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">(0, 52428800)</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">= sum(recvCounts)，即 recvCounts 数组所有元素之和</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>epWorldSize</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">EP 通信域大小（参与通信的卡数）</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">A3: 8~128; 950: 2~64</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">由属性参数直接指定</td>
    </tr>
    <tr style="background: #f7fafc;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>K</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">TopK 专家数（每个 token 选择的专家数）</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">[2, 8]</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">由 MoE 路由策略决定，A = BS &times; K</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>BS</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">batch sequence size（共享专家的 batch 维）</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">&gt;0</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">mmXOptional 的第 0 维 = mmYOptional 的第 0 维</td>
    </tr>
    <tr style="background: #f7fafc;">
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;"><code>H2</code></td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">共享专家的隐藏层大小</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">(0, 12288]</td>
      <td style="padding: 8px 14px; border-bottom: 1px solid #e2e8f0;">mmXOptional 的第 1 维 = mmWeightOptional 的第 0 维</td>
    </tr>
    <tr>
      <td style="padding: 8px 14px;"><code>N2</code></td>
      <td style="padding: 8px 14px;">共享专家的输出维度</td>
      <td style="padding: 8px 14px;">(0, 65536)</td>
      <td style="padding: 8px 14px;">mmWeightOptional 的第 1 维 = mmYOptional 的第 1 维</td>
    </tr>
  </tbody>
</table>
</div>

<div style="margin: 16px 0; padding: 14px 16px; background: #ebf8ff; border-left: 4px solid #3182ce; border-radius: 0 8px 8px 0; font-size: 13px;">
  <strong>全局约束</strong>：EP 通信域内所有卡的 A（= sum(sendCounts)）累加和等于所有卡的 BSK（= sum(recvCounts)）累加和。即发送的 token 总量 = 接收的 token 总量。
</div>

---

## 5. sendCounts / recvCounts 详解

`sendCounts` 和 `recvCounts` 是本算子最核心的参数，它们描述了 AlltoAllV 通信中各卡之间的 token 分配关系。

### 5.1 数组布局

两个数组的大小都是 `e × epWorldSize`，按 **"专家内部，卡间交错"** 的方式排列：

<div style="margin: 20px 0; padding: 20px; background: #fffaf0; border: 2px solid #ed8936; border-radius: 12px; overflow-x: auto;">
  <div style="min-width: 600px;">
    <div style="font-weight: bold; color: #c05621; margin-bottom: 16px; font-size: 15px;">
      sendCounts 数组布局（e=4, epWorldSize=2，共 8 个元素）
    </div>
    <table style="border-collapse: collapse; font-family: monospace; font-size: 13px; text-align: center;">
      <!-- Row 1: Index -->
      <tr>
        <td style="padding: 4px 10px; font-weight: bold; color: #4a5568; text-align: right;">索引</td>
        <td style="padding: 6px 14px; border: 2px solid #4299e1; background: #ebf8ff; font-weight: bold;">[0]</td>
        <td style="padding: 6px 14px; border: 2px solid #4299e1; background: #ebf8ff; font-weight: bold;">[1]</td>
        <td style="padding: 6px 14px; border: 2px solid #4299e1; background: #ebf8ff; font-weight: bold;">[2]</td>
        <td style="padding: 6px 14px; border: 2px solid #4299e1; background: #ebf8ff; font-weight: bold;">[3]</td>
        <td style="padding: 6px 14px; border: 2px solid #e53e3e; background: #fff5f5; font-weight: bold;">[4]</td>
        <td style="padding: 6px 14px; border: 2px solid #e53e3e; background: #fff5f5; font-weight: bold;">[5]</td>
        <td style="padding: 6px 14px; border: 2px solid #e53e3e; background: #fff5f5; font-weight: bold;">[6]</td>
        <td style="padding: 6px 14px; border: 2px solid #e53e3e; background: #fff5f5; font-weight: bold;">[7]</td>
      </tr>
      <!-- Row 2: Target rank -->
      <tr>
        <td style="padding: 4px 10px; font-weight: bold; color: #4a5568; text-align: right;">目标卡</td>
        <td style="padding: 4px 14px; background: #bee3f8; color: #2b6cb0; font-weight: bold; border: 2px solid #4299e1;">→ R0</td>
        <td style="padding: 4px 14px; background: #bee3f8; color: #2b6cb0; font-weight: bold; border: 2px solid #4299e1;">→ R0</td>
        <td style="padding: 4px 14px; background: #bee3f8; color: #2b6cb0; font-weight: bold; border: 2px solid #4299e1;">→ R0</td>
        <td style="padding: 4px 14px; background: #bee3f8; color: #2b6cb0; font-weight: bold; border: 2px solid #4299e1;">→ R0</td>
        <td style="padding: 4px 14px; background: #fed7d7; color: #c53030; font-weight: bold; border: 2px solid #e53e3e;">→ R1</td>
        <td style="padding: 4px 14px; background: #fed7d7; color: #c53030; font-weight: bold; border: 2px solid #e53e3e;">→ R1</td>
        <td style="padding: 4px 14px; background: #fed7d7; color: #c53030; font-weight: bold; border: 2px solid #e53e3e;">→ R1</td>
        <td style="padding: 4px 14px; background: #fed7d7; color: #c53030; font-weight: bold; border: 2px solid #e53e3e;">→ R1</td>
      </tr>
      <!-- Row 3: Expert id -->
      <tr>
        <td style="padding: 4px 10px; font-weight: bold; color: #4a5568; text-align: right;">专家</td>
        <td style="padding: 4px 14px; color: #4a5568; border: 1px solid #e2e8f0;">E0</td>
        <td style="padding: 4px 14px; color: #4a5568; border: 1px solid #e2e8f0;">E1</td>
        <td style="padding: 4px 14px; color: #4a5568; border: 1px solid #e2e8f0;">E2</td>
        <td style="padding: 4px 14px; color: #4a5568; border: 1px solid #e2e8f0;">E3</td>
        <td style="padding: 4px 14px; color: #4a5568; border: 1px solid #e2e8f0;">E0</td>
        <td style="padding: 4px 14px; color: #4a5568; border: 1px solid #e2e8f0;">E1</td>
        <td style="padding: 4px 14px; color: #4a5568; border: 1px solid #e2e8f0;">E2</td>
        <td style="padding: 4px 14px; color: #4a5568; border: 1px solid #e2e8f0;">E3</td>
      </tr>
    </table>
    <div style="margin-top: 14px; display: flex; gap: 20px; font-size: 12px;">
      <div style="display: flex; align-items: center; gap: 6px;">
        <div style="width: 16px; height: 16px; background: #bee3f8; border: 2px solid #4299e1; border-radius: 3px;"></div>
        <span style="color: #2b6cb0; font-weight: bold;">rank_id = 0（索引 0~3）</span>
      </div>
      <div style="display: flex; align-items: center; gap: 6px;">
        <div style="width: 16px; height: 16px; background: #fed7d7; border: 2px solid #e53e3e; border-radius: 3px;"></div>
        <span style="color: #c53030; font-weight: bold;">rank_id = 1（索引 4~7）</span>
      </div>
    </div>
    <div style="margin-top: 12px; font-size: 13px; color: #744210;">
      <strong>索引公式</strong>：<code>sendCounts[expert_id + rank_id * e]</code> = 发往 rank_id 的 expert_id 专家的 token 数<br>
      <strong>示例</strong>：<code>sendCounts[5]</code> = <code>sendCounts[1 + 1*4]</code> = Expert 1 发往 Rank 1 的 token 数
    </div>
  </div>
</div>

### 5.2 通信语义图解

以 `e=4, epWorldSize=2` 为例，展示 Rank 0 视角的 AlltoAllV 通信：

<div style="margin: 20px 0; padding: 20px; background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 12px; overflow-x: auto;">
  <div style="min-width: 700px;">
    <div style="font-weight: bold; margin-bottom: 16px; font-size: 15px; color: #2d3748;">Rank 0 视角：GMM 输出 → AlltoAllV → 最终输出 y</div>
    <div style="display: flex; gap: 30px; align-items: flex-start;">
      <!-- GMM Output (send side) -->
      <div style="flex: 0 0 auto;">
        <div style="font-weight: bold; color: #e53e3e; margin-bottom: 8px; font-size: 13px; text-align: center;">GMM 输出 (workspace)</div>
        <div style="border: 2px solid #e53e3e; border-radius: 8px; overflow: hidden; width: 180px;">
          <div style="background: #fed7d7; padding: 6px; text-align: center; font-size: 12px; font-weight: bold; border-bottom: 1px solid #feb2b2;">Expert 0 的结果</div>
          <div style="background: #fff5f5; padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #fed7d7;">sendCounts[0] tokens → R0</div>
          <div style="background: #fff5f5; padding: 4px 6px; font-size: 11px; border-bottom: 2px solid #feb2b2;">sendCounts[4] tokens → R1</div>
          <div style="background: #c6f6d5; padding: 6px; text-align: center; font-size: 12px; font-weight: bold; border-bottom: 1px solid #9ae6b4;">Expert 1 的结果</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #c6f6d5;">sendCounts[1] tokens → R0</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 2px solid #9ae6b4;">sendCounts[5] tokens → R1</div>
          <div style="background: #bee3f8; padding: 6px; text-align: center; font-size: 12px; font-weight: bold; border-bottom: 1px solid #90cdf4;">Expert 2 的结果</div>
          <div style="background: #ebf8ff; padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #bee3f8;">sendCounts[2] tokens → R0</div>
          <div style="background: #ebf8ff; padding: 4px 6px; font-size: 11px; border-bottom: 2px solid #90cdf4;">sendCounts[6] tokens → R1</div>
          <div style="background: #e9d8fd; padding: 6px; text-align: center; font-size: 12px; font-weight: bold; border-bottom: 1px solid #d6bcfa;">Expert 3 的结果</div>
          <div style="background: #faf5ff; padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #d6bcfa;">sendCounts[3] tokens → R0</div>
          <div style="background: #faf5ff; padding: 4px 6px; font-size: 11px;">sendCounts[7] tokens → R1</div>
        </div>
      </div>
      <!-- Arrow -->
      <div style="display: flex; align-items: center; padding-top: 100px;">
        <div style="text-align: center;">
          <div style="font-size: 13px; color: #805ad5; font-weight: bold; margin-bottom: 4px;">AlltoAllV</div>
          <div style="font-size: 36px; color: #805ad5;">&#10132;</div>
          <div style="font-size: 11px; color: #718096;">HCCL 集合通信</div>
        </div>
      </div>
      <!-- Receive side (y output) -->
      <div style="flex: 0 0 auto;">
        <div style="font-weight: bold; color: #38a169; margin-bottom: 8px; font-size: 13px; text-align: center;">输出 y (Rank 0 接收)</div>
        <div style="border: 2px solid #38a169; border-radius: 8px; overflow: hidden; width: 190px;">
          <div style="background: #c6f6d5; padding: 6px; text-align: center; font-size: 12px; font-weight: bold; border-bottom: 1px solid #9ae6b4;">Expert 0 的结果</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #c6f6d5;">recvCounts[0] tokens ← R0</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 2px solid #9ae6b4;">recvCounts[4] tokens ← R1</div>
          <div style="background: #c6f6d5; padding: 6px; text-align: center; font-size: 12px; font-weight: bold; border-bottom: 1px solid #9ae6b4;">Expert 1 的结果</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #c6f6d5;">recvCounts[1] tokens ← R0</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 2px solid #9ae6b4;">recvCounts[5] tokens ← R1</div>
          <div style="background: #c6f6d5; padding: 6px; text-align: center; font-size: 12px; font-weight: bold; border-bottom: 1px solid #9ae6b4;">Expert 2 的结果</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #c6f6d5;">recvCounts[2] tokens ← R0</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 2px solid #9ae6b4;">recvCounts[6] tokens ← R1</div>
          <div style="background: #c6f6d5; padding: 6px; text-align: center; font-size: 12px; font-weight: bold; border-bottom: 1px solid #9ae6b4;">Expert 3 的结果</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #c6f6d5;">recvCounts[3] tokens ← R0</div>
          <div style="background: #f0fff4; padding: 4px 6px; font-size: 11px;">recvCounts[7] tokens ← R1</div>
        </div>
      </div>
    </div>
    <div style="margin-top: 12px; padding: 10px; background: #ebf8ff; border-radius: 8px; font-size: 12px; color: #2b6cb0;">
      <strong>约束</strong>：所有卡上 sum(sendCounts) 的总和 = 所有卡上 sum(recvCounts) 的总和
    </div>
  </div>
</div>

---

## 6. Kernel 执行流程

### 6.1 NPU 双核架构

昇腾 NPU 采用 **AIC（AI Core，计算核）+ AIV（AI Vector，向量/通信核）** 双核架构。本算子充分利用了这一架构：

<div style="margin: 20px 0; display: flex; gap: 16px; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 300px; border: 2px solid #dd6b20; border-radius: 10px; overflow: hidden;">
    <div style="background: #dd6b20; color: white; padding: 10px 16px; font-weight: bold;">
      AIC 核（计算核）
    </div>
    <div style="padding: 16px; background: #fffaf0;">
      <ul style="margin: 0; padding-left: 20px; font-size: 13px; line-height: 2;">
        <li>执行 <strong>Grouped MatMul</strong> 计算（路由专家）</li>
        <li>执行 <strong>Shared MatMul</strong> 计算（共享专家）</li>
        <li>使用 Cube 矩阵乘加速器</li>
        <li>典型配置：20 个 AIC 核（Atlas A3）</li>
      </ul>
    </div>
  </div>
  <div style="flex: 1; min-width: 300px; border: 2px solid #3182ce; border-radius: 10px; overflow: hidden;">
    <div style="background: #3182ce; color: white; padding: 10px 16px; font-weight: bold;">
      AIV 核（向量/通信核）
    </div>
    <div style="padding: 16px; background: #ebf8ff;">
      <ul style="margin: 0; padding-left: 20px; font-size: 13px; line-height: 2;">
        <li>执行 <strong>HCCL AlltoAllV</strong> 集合通信</li>
        <li>仅 <strong>AIV core 0</strong> 负责通信操作</li>
        <li>Prepare → Commit → Wait → Finalize</li>
        <li>典型配置：40 个 AIV 核（Atlas A3）</li>
      </ul>
    </div>
  </div>
</div>

### 6.2 执行时序

<div style="margin: 20px 0; padding: 20px; background: #1a202c; border-radius: 12px; color: white; overflow-x: auto;">
  <div style="min-width: 800px; font-family: 'Courier New', monospace; font-size: 13px;">
    <div style="margin-bottom: 8px; font-weight: bold; color: #a0aec0; font-size: 14px;">Kernel 执行时序（e=4, epWorldSize=2）</div>
    <div style="margin-bottom: 12px; color: #718096; font-size: 11px;">每个 Commit 必须等对应的 GMM 算完（SyncAll）后才能触发</div>
    <!-- Grid: 1 label col + 12 content cols = 13 grid lines -->
    <div style="display: grid; grid-template-columns: 90px repeat(12, 1fr); column-gap: 2px; align-items: center;">
      <!-- Row 1: GMM (AIC) -->
      <div style="color: #fc8181; font-weight: bold; font-size: 11px; padding: 4px 0;">GMM<br><span style="font-size: 10px; color: #a0aec0;">(AIC)</span></div>
      <div style="grid-column: 2/4; background: #e53e3e; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 11px;">GMM E0</div>
      <div style="grid-column: 4/6; background: #c53030; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 11px;">GMM E1</div>
      <div style="grid-column: 6/8; background: #e53e3e; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 11px;">GMM E2</div>
      <div style="grid-column: 8/10; background: #c53030; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 11px;">GMM E3</div>
      <div style="grid-column: 10/12; background: #4299e1; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 11px;">Shared MM</div>
      <div style="grid-column: 12/14;"></div>
      <!-- Row 2: SyncAll arrows — centered across boundary (straddling GMM end col + HCCL start col) -->
      <div></div>
      <div style="grid-column: 3/5; text-align: center; color: #f6ad55; font-size: 10px; line-height: 1; padding: 2px 0;">↓</div>
      <div style="grid-column: 5/7; text-align: center; color: #f6ad55; font-size: 10px; line-height: 1; padding: 2px 0;">↓</div>
      <div style="grid-column: 7/9; text-align: center; color: #f6ad55; font-size: 10px; line-height: 1; padding: 2px 0;">↓</div>
      <div style="grid-column: 9/11; text-align: center; color: #f6ad55; font-size: 10px; line-height: 1; padding: 2px 0;">↓</div>
      <!-- Row 3: HCCL (AIV) — each block starts at same column as next GMM block -->
      <div style="color: #b794f4; font-weight: bold; font-size: 11px; padding: 4px 0;">HCCL<br><span style="font-size: 10px; color: #a0aec0;">(AIV core 0)</span></div>
      <div style="grid-column: 2/4;"></div>
      <div style="grid-column: 4/6; background: #805ad5; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 10px;">A2Av E0</div>
      <div style="grid-column: 6/8; background: #6b46c1; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 10px;">A2Av E1</div>
      <div style="grid-column: 8/10; background: #805ad5; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 10px;">A2Av E2</div>
      <div style="grid-column: 10/12; background: #6b46c1; padding: 10px 4px; border-radius: 4px; text-align: center; font-size: 10px;">A2Av E3</div>
      <div style="grid-column: 12/13; background: #553c9a; padding: 10px 2px; border-radius: 4px; text-align: center; font-size: 9px;">Wait</div>
      <div style="grid-column: 13/14; background: #44337a; padding: 10px 2px; border-radius: 4px; text-align: center; font-size: 9px;">Fin</div>
    </div>
    <!-- Overlap annotation -->
    <div style="margin-top: 10px; padding: 8px 12px; background: rgba(255,255,255,0.08); border-radius: 6px; font-size: 11px; color: #a0aec0;">
      <strong style="color: #68d391;">重叠区域</strong>：A2Av E0 与 GMM E1 同列 = 同时段并行；A2Av E1 与 GMM E2 并行；A2Av E2 与 GMM E3 并行；A2Av E3 与 Shared MM 并行。
    </div>
    <!-- Legend -->
    <div style="margin-top: 12px; display: flex; gap: 16px; flex-wrap: wrap; font-size: 11px;">
      <div style="display: flex; align-items: center; gap: 4px;">
        <div style="width: 14px; height: 14px; background: #e53e3e; border-radius: 3px;"></div>
        <span>路由专家 GMM 计算（AIC）</span>
      </div>
      <div style="display: flex; align-items: center; gap: 4px;">
        <div style="width: 14px; height: 14px; background: #805ad5; border-radius: 3px;"></div>
        <span>AlltoAllV 通信（AIV）</span>
      </div>
      <div style="display: flex; align-items: center; gap: 4px;">
        <div style="width: 14px; height: 14px; background: #4299e1; border-radius: 3px;"></div>
        <span>共享专家 MatMul（AIC）</span>
      </div>
      <div style="display: flex; align-items: center; gap: 4px;">
        <div style="width: 14px; height: 14px; background: #f6ad55; border-radius: 3px;"></div>
        <span>SyncAll 同步屏障</span>
      </div>
    </div>
  </div>
</div>

### 6.3 核心流程

```cpp
// ===== 阶段 1：AIV core 0 预配置所有专家的 AlltoAllV handle =====
HcclAlltoAllvPrepare() {
    for (e = 0; e < expertNumInOneRank; e++) {
        // 根据 sendCnt/recvCnt 计算每个专家的 offset
        handle[e] = hccl.AlltoAllV(gmmOut, sendCnt, sendOffset,
                                    y,      recvCnt, recvOffset);
    }
}

// ===== 阶段 2：逐专家 GMM 计算 + AlltoAllV 通信 =====
GmmProcessAlltoallv() {
    for (e = 0; e < expertNumInOneRank; e++) {
        // AIC 全部核：计算当前专家的 GMM
        if (ASCEND_IS_AIC) {
            gmmOut[e] = gmmX[e] * gmmWeight[e];   // → workspace
        }
        SyncAll();   // 等 AIC 算完，AIV 才能发
        // AIV core 0：触发当前专家的 AlltoAllV
        if (ASCEND_IS_AIV && blockIdx == 0) {
            hccl.Commit(handle[e]);
        }
    }
    // AIC：所有路由专家 GMM 算完后，执行共享专家 MatMul
    if (ASCEND_IS_AIC) {
        mmY = mmX * mmWeight;
    }
    SyncAll();
}

// ===== 阶段 3：等待所有通信完成 =====
if (ASCEND_IS_AIV && blockIdx == 0) {
    for (e = 0; e < expertNumInOneRank; e++) {
        hccl.Wait(handle[e]);
    }
    hccl.Finalize();
}
```

---

## 7. 计算-通信重叠流水线

本算子的核心优化在于 **逐专家的计算-通信重叠**：每完成一个专家的 GMM 计算，立即启动该专家的 AlltoAllV 通信，而不是等所有专家计算完毕再一次性通信。

<div style="margin: 20px 0; padding: 20px; background: #f7fafc; border: 2px solid #4a5568; border-radius: 12px; overflow-x: auto;">
  <div style="min-width: 700px;">
    <div style="font-weight: bold; color: #2d3748; margin-bottom: 16px; font-size: 15px;">逐专家流水线 vs 非流水线对比</div>
    <!-- Non-pipelined -->
    <div style="margin-bottom: 24px;">
      <div style="font-weight: bold; color: #e53e3e; margin-bottom: 8px; font-size: 13px;">&#10060; 非流水线方式（不融合）</div>
      <div style="display: flex; gap: 2px; align-items: center;">
        <div style="width: 50px; color: #718096; font-size: 12px;">AIC:</div>
        <div style="background: #fc8181; color: white; padding: 8px 12px; border-radius: 4px; flex: 4; text-align: center; font-size: 12px;">
          GMM E0 + E1 + E2 + E3
        </div>
        <div style="flex: 2;"></div>
        <div style="background: #63b3ed; color: white; padding: 8px 8px; border-radius: 4px; flex: 1; text-align: center; font-size: 12px;">
          SharedMM
        </div>
      </div>
      <div style="display: flex; gap: 2px; align-items: center; margin-top: 2px;">
        <div style="width: 50px; color: #718096; font-size: 12px;">AIV:</div>
        <div style="flex: 4;"></div>
        <div style="background: #b794f4; color: white; padding: 8px 12px; border-radius: 4px; flex: 2; text-align: center; font-size: 12px;">
          AlltoAllV (全部数据)
        </div>
        <div style="flex: 1;"></div>
      </div>
      <div style="text-align: right; color: #a0aec0; font-size: 12px; margin-top: 4px;">总延迟 = T_gmm + T_alltoallv + T_shared</div>
    </div>
    <!-- Pipelined -->
    <div>
      <div style="font-weight: bold; color: #38a169; margin-bottom: 8px; font-size: 13px;">&#9989; 流水线方式（本算子）</div>
      <!-- Grid: 1 label col + 12 content cols -->
      <div style="display: grid; grid-template-columns: 50px repeat(12, 1fr); column-gap: 2px; align-items: center;">
        <!-- AIC row -->
        <div style="color: #718096; font-size: 12px;">AIC:</div>
        <div style="grid-column: 2/4; background: #fc8181; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 12px;">GMM E0</div>
        <div style="grid-column: 4/6; background: #fc8181; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 12px;">GMM E1</div>
        <div style="grid-column: 6/8; background: #fc8181; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 12px;">GMM E2</div>
        <div style="grid-column: 8/10; background: #fc8181; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 12px;">GMM E3</div>
        <div style="grid-column: 10/12; background: #63b3ed; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 12px;">SharedMM</div>
        <div style="grid-column: 12/14;"></div>
        <!-- SyncAll arrows — centered at boundary (straddling GMM end col + HCCL start col) -->
        <div></div>
        <div style="grid-column: 3/5; text-align: center; color: #f6ad55; font-size: 10px; line-height: 1; padding: 1px 0;">↓</div>
        <div style="grid-column: 5/7; text-align: center; color: #f6ad55; font-size: 10px; line-height: 1; padding: 1px 0;">↓</div>
        <div style="grid-column: 7/9; text-align: center; color: #f6ad55; font-size: 10px; line-height: 1; padding: 1px 0;">↓</div>
        <div style="grid-column: 9/11; text-align: center; color: #f6ad55; font-size: 10px; line-height: 1; padding: 1px 0;">↓</div>
        <!-- AIV row — HCCL blocks share same columns as next GMM block = precise alignment -->
        <div style="color: #718096; font-size: 12px;">AIV:</div>
        <div style="grid-column: 2/4;"></div>
        <div style="grid-column: 4/6; background: #b794f4; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 11px;">A2Av E0</div>
        <div style="grid-column: 6/8; background: #9f7aea; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 11px;">A2Av E1</div>
        <div style="grid-column: 8/10; background: #b794f4; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 11px;">A2Av E2</div>
        <div style="grid-column: 10/12; background: #9f7aea; color: white; padding: 8px 4px; border-radius: 4px; text-align: center; font-size: 11px;">A2Av E3</div>
        <div style="grid-column: 12/14;"></div>
      </div>
      <div style="text-align: right; color: #38a169; font-size: 12px; margin-top: 6px;">
        <strong>总延迟 &asymp; T_gmm + T_last_alltoallv</strong>（通信与下一个专家的计算在同列 = 同时段并行）
      </div>
    </div>
  </div>
</div>

**关键设计点**：
- AIC 核完成 Expert i 的 GMM 后，通过 `SyncAll()` 通知 AIV 核（AIV **不可能**在 AIC 算完之前发送）
- AIV core 0 收到同步信号后，立即 `Commit` 该专家的 AlltoAllV（此前已在 Prepare 阶段配置好 handle）
- AIC 核不等待通信完成，直接开始 Expert i+1 的 GMM，此时 HCCL 传输与 GMM 计算并行
- 所有路由专家的 GMM 计算完毕后，AIC 核执行共享专家 MatMul，与最后一个专家的 HCCL 传输重叠
- 最终 AIV core 0 `Wait` 所有通信完成

