# NPU Tile调度Pass设计讨论记录

> 与用户讨论设计细节，记录决策和理由

---

## 讨论会议信息

**日期**：2025-11-13
**参与者**：开发团队 + AI Assistant
**目标**：确定NPU Tile调度Pass的最终设计方案

---

## 核心问题清单

### 🔴 必须决策的问题

1. **NPU Dialect设计**
2. **调度策略选择**
3. **硬件接口对接**
4. **编译时 vs 运行时**
5. **Padding处理方式**

### 🟡 重要但可后续调整

6. **动态shape支持**
7. **多operation支持**
8. **性能优化策略**

---

## 问题1：NPU Dialect是否必要？

### 当前设计

我在设计文档中提出了自定义NPU Dialect：

```mlir
// NPU Dialect操作
%batch = npu.create_batch num_tiles = 16
%tile = npu.extract_tile %A[%i, %j] [16, 16]
npu.add_to_batch %batch, %tile, slot 0
%results = npu.execute_batch %batch, operation = "matmul"
%result = npu.extract_result %results, slot 0
```

### 替代方案A：使用现有Dialect组合

```mlir
// 使用 tensor + scf + linalg
%tiles = scf.for %i = ... iter_args(%acc = ...) {
  %tile = tensor.extract_slice %A[%i, %j][16, 16][1, 1]
  %result = linalg.matmul ins(%tile, ...) outs(...)
  scf.yield %result
}
```

### 对比分析

| 维度 | 自定义NPU Dialect | 使用现有Dialect |
|------|------------------|----------------|
| **清晰度** | ⭐⭐⭐⭐⭐ 意图明确 | ⭐⭐⭐ 需要推断 |
| **开发成本** | ⭐⭐ 需要定义+实现 | ⭐⭐⭐⭐⭐ 无需开发 |
| **Lowering复杂度** | ⭐⭐⭐ 需要专门Pass | ⭐⭐⭐⭐ 复用现有 |
| **可扩展性** | ⭐⭐⭐⭐⭐ 易于添加新op | ⭐⭐ 受限于现有op |
| **与硬件对接** | ⭐⭐⭐⭐⭐ 直接映射 | ⭐⭐⭐ 间接映射 |
| **调试友好** | ⭐⭐⭐⭐⭐ IR清晰可读 | ⭐⭐⭐ 需要识别模式 |

### 我的建议 ✅

**推荐：自定义NPU Dialect**

理由：
1. 你的硬件有**特殊约束**（固定16 tiles）
2. 需要**精确控制**tile调度顺序
3. IR需要**清晰表达**硬件语义
4. 后续扩展需要**添加新操作**（如conv, reduce）

### 你的决策

**[ ] 同意使用NPU Dialect**
**[ ] 倾向使用现有Dialect**
**[ ] 其他想法：________________**

---

## 问题2：调度策略的选择

### 当前设计：三种策略

```cpp
enum class ScheduleStrategy {
  Simple,      // 单矩阵，选前16个
  Batching,    // 多矩阵，轮询
  Optimized    // 智能选择，最小padding
};
```

### 问题2.1：是否需要三种策略？

**简化方案**：只实现Optimized策略

```cpp
// 简化版：统一用一个算法
TileSchedule schedule(const vector<MatrixInfo>& matrices) {
  // 统一算法：
  // 1. 收集所有tile
  // 2. 按padding排序
  // 3. 选前16个
  return optimizedSchedule(matrices);
}
```

**优点**：
- ✅ 代码简单
- ✅ 维护成本低
- ✅ 性能最优

**缺点**：
- ❌ 可能过度优化小case
- ❌ 编译时间稍长（排序开销）

### 问题2.2：排序算法的关键指标

当前我用的是**padding比例**：

```cpp
float padding_ratio = 1.0f - (float)(actual_k * actual_n) / (TILE_K * TILE_N);
```

**其他可能的指标**：

| 指标 | 公式 | 优先级 |
|------|------|--------|
| Padding比例 | `1 - (actual / total)` | 完整tile优先 |
| 绝对padding量 | `total - actual` | 最小化浪费 |
| 矩阵来源 | `matrix_id` | 同矩阵聚集 |
| 位置连续性 | `abs(prev_offset - curr_offset)` | Cache友好 |

### 组合策略示例

```cpp
struct TileScore {
  float padding_ratio;    // 权重 0.6
  float cache_locality;   // 权重 0.3
  float matrix_affinity;  // 权重 0.1

  float total_score() const {
    return 0.6 * (1 - padding_ratio) +
           0.3 * cache_locality +
           0.1 * matrix_affinity;
  }
};
```

### 你的需求

**你的硬件特点**：
- [ ] Cache大小：_______ KB
- [ ] 是否支持乱序执行：[ ] 是 [ ] 否
- [ ] 同矩阵tile是否必须连续：[ ] 是 [ ] 否
- [ ] 性能瓶颈：[ ] 计算 [ ] 内存带宽 [ ] DMA传输

**你的优先级**（1=最高）：
- [ ___ ] 最小化padding（计算效率）
- [ ___ ] 最小化内存访问（带宽）
- [ ___ ] 最大化Cache命中（局部性）
- [ ___ ] 简化硬件控制逻辑（同矩阵聚集）

---

## 问题3：与实际硬件的对接

### 当前设计：通过Runtime库

```cpp
// Lowering到C函数调用
extern "C" void npu_execute_batch(
  void* batch_descriptor,
  int num_tiles,
  TileInfo* tile_infos
);
```

### 问题3.1：你的硬件编程模型是什么？

**选项A：直接寄存器编程**
```cpp
// 直接写硬件寄存器
write_reg(NPU_CTRL, 0x1);  // 启动
write_reg(NPU_TILE_0_ADDR, tile0_addr);
write_reg(NPU_TILE_0_SIZE, 0x1010);  // 16×16
...
write_reg(NPU_EXEC, 0x1);  // 执行
```

**选项B：DMA + 命令队列**
```cpp
// 通过DMA传输tile
dma_transfer(host_mem, npu_mem, size);

// 提交命令
Command cmd = {
  .op = MATMUL,
  .num_tiles = 16,
  .tile_descriptors = {...}
};
submit_command(&cmd);
```

**选项C：自定义ISA指令**
```asm
; NPU汇编指令
npu.load.tile r0, A[0,0], 16x16
npu.load.tile r1, B[0,0], 16x16
npu.matmul.batch r0, r1, r2, 16_tiles
npu.store.tile C[0,0], r2
```

**你的选择**：
- [ ] 选项A：寄存器编程
- [ ] 选项B：DMA + 命令队列
- [ ] 选项C：自定义ISA
- [ ] 选项D：其他（请描述）：________________

### 问题3.2：Tile数据布局

**硬件期望的数据布局**：

```
选项1：连续内存（AoS - Array of Structures）
[Tile0: 16×16][Tile1: 16×16]...[Tile15: 16×16]

选项2：分离存储（SoA - Structure of Arrays）
[All Tile Metadata][All Tile Data]

选项3：描述符表
[Descriptor Table] → [Tile0 Data][Tile1 Data]...
```

**你的硬件使用哪种**：________________

### 问题3.3：不完整Tile的处理

当tile不是完整的16×16时：

**选项A：硬件Mask支持**
```
硬件支持：每个tile有valid_mask[16][16]
编译器生成：mask信息
硬件行为：自动跳过invalid元素
```

**选项B：软件Padding**
```
编译器行为：padding到16×16（填充0）
硬件行为：正常计算（浪费一些计算）
```

**选项C：变长Tile**
```
硬件支持：每个tile可以是不同大小
编译器生成：实际大小信息
硬件行为：根据实际大小计算
```

**你的硬件支持**：
- [ ] 选项A：硬件Mask
- [ ] 选项B：软件Padding
- [ ] 选项C：变长Tile
- [ ] 选项D：其他：________________

---

## 问题4：编译时 vs 运行时调度

### 当前设计：编译时确定

```cpp
// 编译时生成静态调度
TileSchedule schedule = generateSchedule(matrices);  // 编译时
applySchedule(funcOp, schedule);  // 生成固定的IR
```

### 问题4.1：是否需要支持动态Shape？

**场景1：所有Shape编译时已知**
```mlir
func.func @matmul(%A: tensor<97x137xf32>) {
  // 97, 137 在编译时就知道
}
```
→ 编译时调度完美 ✓

**场景2：部分Shape动态**
```mlir
func.func @matmul(%A: tensor<?x137xf32>) {
  // 第一维在运行时才知道
}
```
→ 需要运行时调度或保守策略

**场景3：完全动态**
```mlir
func.func @matmul(%A: tensor<?x?xf32>) {
  // 编译时完全不知道
}
```
→ 必须运行时调度

**你的实际场景**：
- [ ] 场景1：所有shape编译时已知（推荐）
- [ ] 场景2：部分shape动态
- [ ] 场景3：完全动态
- [ ] 其他：________________

### 问题4.2：如果需要动态支持

**混合策略**：
```cpp
if (all_shapes_static) {
  // 编译时调度（最优）
  return compileTimeSchedule(matrices);
} else {
  // 生成运行时调度代码
  return generateRuntimeScheduler(matrices);
}
```

**运行时开销评估**：
```
编译时调度：0 cycle开销
运行时调度：~100-1000 cycles
  - 遍历矩阵：10 cycles/matrix
  - 排序tile：50-500 cycles（取决于数量）
  - 生成descriptor：10 cycles/tile
```

**你能接受的运行时开销**：_______ cycles

---

## 问题5：多Operation支持

### 当前设计：只支持matmul

```cpp
def NPU_ExecuteBatchOp : NPU_Op<"execute_batch"> {
  let arguments = (ins
    NPU_TileBatch:$batch,
    StrAttr:$operation  // 目前只有 "matmul"
  );
}
```

### 扩展计划

**Phase 1（必须）**：
- [x] Matmul

**Phase 2（重要）**：
- [ ] Conv2D（需要不同的tile策略）
- [ ] Element-wise ops（add, mul, relu等）

**Phase 3（可选）**：
- [ ] Reduction（sum, max）
- [ ] Transpose
- [ ] Reshape

### 问题5.1：不同Operation的Tile大小

| Operation | 合理的Tile大小 | 原因 |
|-----------|--------------|------|
| Matmul | 16×16 | 计算密集 |
| Conv2D | ? | 取决于kernel大小 |
| Element-wise | ? | 内存密集 |
| Reduction | ? | 特殊处理 |

**你的硬件是否支持**：
- [ ] 所有op用同样的tile大小（16×16）
- [ ] 不同op可以用不同tile大小
- [ ] 其他限制：________________

---

## 问题6：错误处理和边界情况

### 边界情况1：Tile数量不足

```
输入：M0[10×10] → 只有1个tile
需要：16个tile
```

**处理方案**：
- [ ] A. 填充15个NOP tile
- [ ] B. 报错，要求batch更多矩阵
- [ ] C. 降级到标量执行
- [ ] D. 其他：________________

### 边界情况2：Tile数量过多

```
输入：M0[1024×1024] → 4096个tile
需要：16个tile
```

**处理方案**：
- [ ] A. 生成256个batch，每个16 tiles
- [ ] B. 只处理前16个tile，其余报错
- [ ] C. 智能选择最优的16个
- [ ] D. 其他：________________

### 边界情况3：不支持的维度

```
输入：M0[1×1000000] → 极端长条形
```

**处理方案**：
- [ ] A. 强制拒绝
- [ ] B. 自动切分成多个小矩阵
- [ ] C. 调整tile大小
- [ ] D. 其他：________________

---

## 问题7：性能目标

### 你的性能目标是什么？

**选择你的首要目标**（只选1个）：
- [ ] 最大化吞吐量（ops/s）
- [ ] 最小化延迟（latency）
- [ ] 最优能效（ops/watt）
- [ ] 最简化硬件（降低成本）

**可接受的性能损失**：
相比理想情况（所有tile都是完整的16×16），你能接受：
- [ ] <5% 损失（严格）
- [ ] 5-10% 损失（正常）
- [ ] 10-20% 损失（宽松）
- [ ] >20% 损失（只要能跑）

---

## 决策汇总表

请填写下面的表格，我会根据你的反馈调整设计：

| 问题 | 你的选择 | 理由/备注 |
|------|---------|----------|
| **1. 使用NPU Dialect?** | [ ] 是 [ ] 否 | |
| **2. 调度策略** | [ ] 3种 [ ] 仅Optimized [ ] 其他 | |
| **3. 硬件编程模型** | [ ] A [ ] B [ ] C [ ] D | |
| **4. Tile数据布局** | [ ] AoS [ ] SoA [ ] Descriptor | |
| **5. 不完整Tile处理** | [ ] Mask [ ] Padding [ ] 变长 | |
| **6. 动态Shape支持** | [ ] 不需要 [ ] 部分 [ ] 完全 | |
| **7. 多Operation优先级** | Conv:[ ] Elem:[ ] Reduce:[ ] | |
| **8. Tile不足时** | [ ] NOP [ ] 报错 [ ] 降级 | |
| **9. Tile过多时** | [ ] Batch [ ] 报错 [ ] 智能选 | |
| **10. 首要性能目标** | [ ] 吞吐 [ ] 延迟 [ ] 能效 | |

---

## 下一步行动

根据你的反馈，我会：
1. 调整设计文档
2. 创建refined版本的设计方案
3. 开始Phase 1实现

**请逐个问题回答，或者告诉我你最关心的几个问题，我们先深入讨论那些！**
