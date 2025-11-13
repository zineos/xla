# NPU Tile Scheduler 对话记录

**日期**: 2025-11-13
**状态**: 架构设计完成，准备开始简化版实现
**分支**: `claude/learn-xla-011CUffTkuwkDsqdvo8kqXh6`

---

## 📋 会话摘要

### 已完成的工作

1. ✅ **确认硬件架构**
   - 16-way真并行执行（16个独立NPU Core）
   - 变长Tile支持（硬件原生支持，≤16×16）
   - 自定义ISA（类似TPU）
   - 内存架构：DDR → DMA → On-chip Buffer → NPU Core

2. ✅ **更新设计文档**
   - `design_discussion_notes.md`: 添加决策汇总表
   - `npu_architecture_refined.md`: 完整架构规范（1426行）

3. ✅ **创建完整设计规范**
   - NPU Dialect完整定义（TableGen）
   - Pass定义和实现框架
   - 优化调度算法（针对真并行架构）
   - 完整的Lowering路径设计
   - 6-phase实施路线图

4. ✅ **Git提交**
   - 已提交所有文档到分支
   - Commit: `98cab46` "Add refined NPU architecture specification"

---

## 🎯 当前讨论：简化版方案

### 新的聚焦点（不涉及LLVM）

**只关注MLIR层面**：
```
输入: linalg.matmul (MLIR)
    ↓
NPU Tile Scheduler Pass
    ↓
输出: NPU Dialect IR (MLIR)
```

**暂时不考虑**：
- ❌ LLVM IR lowering
- ❌ NPU Backend实现
- ❌ 实际代码生成
- ❌ 性能测试（只验证IR正确性）

### 待讨论的核心问题

#### 问题1: 如何在MLIR中表达"16个tile的batch"？

**方案A: 显式Batch操作**
```mlir
%batch = npu.create_batch : !npu.batch
%batch1 = npu.add_tile %batch, %tile0, slot=0
%batch2 = npu.add_tile %batch1, %tile1, slot=1
// ... 16个tile
%result = npu.execute %batch16 {op = "matmul"}
```

**方案B: 隐式批处理（用SCF循环）**
```mlir
%result = scf.for %i = 0 to 16 {
  %tile = extract_tile ...
  %partial_result = npu.compute_tile %tile
  scf.yield %partial_result
}
```

#### 问题2: 变长Tile如何表达？

**方案A: 在类型中体现**
```mlir
%tile_full = npu.extract_tile ... : tensor<16x16xf32>
%tile_partial = npu.extract_tile ... : tensor<16x9xf32>
```

**方案B: 用属性标记**
```mlir
%tile = npu.extract_tile ... {actual_size = [16, 9]} : tensor<16x16xf32>
```

#### 问题3: 调度算法复杂度

**方案A: 超简单版（推荐Phase 1）**
- 只支持单矩阵
- 按顺序取前16个tile
- ~100行代码

**方案B: 优化版（Phase 2）**
- 支持多矩阵
- 按padding排序
- ~300行代码

#### 问题4: 如何验证正确性？

```bash
# 验证IR格式
mlir-opt input.mlir -npu-tile-scheduler -verify-diagnostics

# 检查变换
mlir-opt input.mlir -npu-tile-scheduler -o output.mlir

# 自动化测试
mlir-opt input.mlir -npu-tile-scheduler | FileCheck test.mlir
```

---

## 📁 相关文件

### 已创建的文档

1. **`design_discussion_notes.md`** (608行)
   - 完整的设计讨论记录
   - 7个关键问题的分析
   - 决策汇总表（已确认的架构选择）

2. **`npu_architecture_refined.md`** (1426行)
   - 完整的架构规范
   - NPU Dialect TableGen定义（~350行）
   - Pass定义和实现框架
   - 调度算法设计
   - Lowering路径（暂时不用）
   - 6-phase实施路线图

3. **其他参考文档**
   - `mlir_tiling_pass_tutorial.md`: MLIR Pass开发教程
   - `npu_tile_scheduler_pass_design.md`: 原始设计方案
   - `tensor_core_tile_scheduler.cpp`: C++调度器原型（验证了26×优化效果）

### 关键数据结构（已设计）

```cpp
// Tile候选者
struct TileCandidate {
  int matrix_id;
  int64_t k_offset, n_offset;
  int64_t actual_k, actual_n;
  float padding_ratio;  // 主要优化指标
  Value source_tensor;
};

// 矩阵信息
struct MatrixInfo {
  Value tensor;
  SmallVector<int64_t, 2> shape;
  Operation* definingOp;
};

// Batch调度结果
struct BatchSchedule {
  SmallVector<TileCandidate, 16> selected_tiles;
  float utilization;
};
```

---

## 🚀 建议的简化路线图

### Phase 1: 最小可验证原型（本周）

**目标**: 快速验证整个MLIR变换流程

**范围**:
- [ ] 定义3-5个核心NPU操作
  - `npu.create_batch`
  - `npu.extract_tile` (支持变长)
  - `npu.add_to_batch`
  - `npu.execute`
  - `npu.assemble`

- [ ] 实现超简单的调度算法
  - 只支持单个矩阵
  - 按顺序选择前16个tile
  - 不足16个用NOP填充

- [ ] 编写1个测试case
  - 输入: `linalg.matmul` 100×80
  - 输出: NPU dialect IR（验证格式正确）

**可交付**:
- `npu_ops_simple.td` (TableGen定义)
- `npu_tile_scheduler_simple.cpp` (Pass实现)
- `test_simple.mlir` (测试用例)

### Phase 2: 优化调度（下周）

**目标**: 实现智能调度算法

**范围**:
- [ ] 添加padding计算
- [ ] 实现按padding排序
- [ ] 支持多矩阵合并
- [ ] 添加更多测试case（不规则矩阵）

---

## 🎯 下一步行动

### 立即需要讨论的问题

1. **Batch表达方式**：方案A（显式batch）还是方案B（隐式循环）？
2. **变长Tile表达**：方案A（类型体现）还是方案B（属性标记）？
3. **Phase 1范围确认**：是否同意"超简单版"作为起点？

### 准备开始实现

一旦确认上述设计问题，可以立即开始：

```bash
# 创建项目结构
cd /Users/neos/Documents/project/zicun/xla
mkdir -p npu_dialect/{IR,Transforms,test}

# 创建第一个文件
touch npu_dialect/IR/npu_ops_simple.td
touch npu_dialect/Transforms/npu_tile_scheduler_simple.cpp
touch npu_dialect/test/test_simple.mlir
```

---

## 📚 关键参考

### XLA源码中的关键文件

已分析过的XLA实现：
- `xla/mlir_hlo/transforms/tile_loops_pass.cc`: TileLoopsPass实现
- `xla/codegen/xtile/ir/xtile_ops.td`: XTile Dialect定义
- `xla/service/gpu/transforms/cublas_pad_for_gemms.h`: GPU padding策略

### MLIR官方文档

需要参考的部分：
- TableGen语法（定义Dialect和Operations）
- Pass开发框架
- Type系统（自定义!npu.batch类型）
- Operation接口（SideEffects等）

---

## 💡 关键洞察

### 硬件特性导致的设计简化

因为硬件是**真并行**（非SIMD，非流水线），所以：

✅ **可以简化**:
- 调度算法只需考虑padding最小化
- 无需考虑执行顺序依赖
- 无需考虑分支惩罚
- 无需考虑warp divergence

✅ **可以优化**:
- 直接按padding ratio排序
- 完整tile优先（padding=0）
- 简单贪心算法即可达到最优

### 变长Tile的优势

因为硬件**原生支持变长tile**：

✅ **可以简化**:
- 无需软件padding（零开销）
- 边界处理完全交给硬件
- IR中直接用实际大小

✅ **可以优化**:
- 更高的计算利用率
- 更少的内存浪费
- 更简单的调度逻辑

---

## 🔄 继续对话的上下文

当你在 web-claude-code 中继续时，可以说：

> "我们之前讨论了NPU Tile Scheduler的设计，已经确认了硬件架构（16-way并行+变长tile）。现在想简化方案，只做MLIR层面的变换，不涉及LLVM lowering。
>
> 我想从最简单的实现开始（Phase 1），需要先讨论几个设计问题：
> 1. Batch应该用显式操作还是隐式循环？
> 2. 变长Tile应该在类型中体现还是用属性标记？
> 3. Phase 1的超简单版调度算法（单矩阵顺序选择）是否可行？
>
> 参考文档在：
> - design_discussion_notes.md
> - npu_architecture_refined.md
> - conversation_log_2025-11-13.md"

---

## 📊 当前Git状态

```bash
Branch: claude/learn-xla-011CUffTkuwkDsqdvo8kqXh6
Latest commit: 98cab46 "Add refined NPU architecture specification"

Changed files:
  - design_discussion_notes.md (updated)
  - npu_architecture_refined.md (new, 1426 lines)

Status: All changes committed and pushed
```

---

**记录结束时间**: 2025-11-13
**下次继续**: 讨论简化版设计的3个核心问题
**期望产出**: 确定Dialect设计细节，开始Phase 1实现
