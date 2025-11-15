# BatchComp Framework Examples

本目录包含BatchComp批组合优化框架的使用示例。

---

## 📁 目录结构

```
examples/
├── README.md                          # 本文件
├── run_all_examples.sh                # 运行所有示例
├── python/                            # Python优化器示例
│   ├── basic_optimization.py          # 基础优化示例
│   ├── multi_matrix.py                # 多矩阵优化
│   └── hardware_configs.py            # 硬件配置使用
└── mlir/                              # MLIR Pass示例
    ├── simple_matmul.mlir             # 简单matmul转换
    ├── multi_matmul.mlir              # 多matmul优化
    └── run_examples.sh                # 运行脚本
```

---

## 🚀 Python优化器示例

### 1. 基础优化 (`python/basic_optimization.py`)

演示如何使用Python优化器进行基本的tile批处理优化：

```bash
cd examples/python
python3 basic_optimization.py
```

**功能**:
- 创建tile列表
- 使用不同算法优化（greedy, improved_greedy, cpsat）
- 比较性能结果

### 2. 多矩阵优化 (`python/multi_matrix.py`)

演示跨多个矩阵的批组合优化：

```bash
python3 multi_matrix.py
```

**功能**:
- 多个matmul操作的tile生成
- 跨矩阵批处理
- K-dimension分组优化

### 3. 硬件配置 (`python/hardware_configs.py`)

演示如何使用不同硬件配置：

```bash
python3 hardware_configs.py
```

**功能**:
- NPU配置 (batch_size=16, must_fill=true)
- GPU配置 (batch_size=32, must_fill=false)
- TPU配置 (batch_size=128, must_fill=false)

---

## 🔧 MLIR Pass示例

### 1. 简单Matmul转换 (`mlir/simple_matmul.mlir`)

展示单个matmul到BatchComp IR的转换：

```bash
cd examples/mlir
./run_examples.sh simple
```

**输入**:
```mlir
func.func @simple_matmul(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>) -> tensor<100x300xf32> {
  %C = tosa.matmul %A, %B : (tensor<100x200xf32>, tensor<200x300xf32>) -> tensor<100x300xf32>
  return %C : tensor<100x300xf32>
}
```

**输出**:
```mlir
func.func @simple_matmul(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>) -> tensor<100x300xf32> {
  %tiles = batch_comp.generate_tiles %A, %B {tile_size = [16, 16, 16], matrix_id = 0}
  %schedule = batch_comp.schedule_batches %tiles {hardware = "npu", batch_size = 16, algorithm = "cpsat"}
  %C = batch_comp.execute_schedule %schedule {hardware = "npu"}
  return %C : tensor<100x300xf32>
}
```

### 2. 多Matmul优化 (`mlir/multi_matmul.mlir`)

展示多个matmul的联合优化：

```bash
./run_examples.sh multi
```

---

## 📊 性能对比

运行所有示例并生成性能报告：

```bash
cd examples
./run_all_examples.sh
```

输出示例：
```
=== Python优化器性能 ===
Greedy:           3 batches, 75.0% utilization
ImprovedGreedy:   3 batches, 93.75% utilization (+25%)
CP-SAT:           3 batches, 93.75% utilization (+25%, -80% NOPs)

=== MLIR Pass性能 ===
Simple matmul:    优化完成
Multi matmul:     跨矩阵批处理成功
```

---

## 🎯 快速开始

### 先决条件

1. **Python环境**:
   ```bash
   cd ../../npu_pass_implementation
   pip install -r requirements.txt
   ```

2. **MLIR环境** (可选，用于MLIR示例):
   ```bash
   # 需要构建的MLIR工具
   # mlir-opt, mlir-translate
   ```

### 运行第一个示例

```bash
# 1. Python优化器示例
cd examples/python
python3 basic_optimization.py

# 2. MLIR Pass示例 (需要MLIR环境)
cd ../mlir
./run_examples.sh simple
```

---

## 📚 学习路径

### 新手 → Python优化器
1. `python/basic_optimization.py` - 理解基本概念
2. `python/hardware_configs.py` - 了解硬件配置
3. `python/multi_matrix.py` - 掌握高级优化

### 进阶 → MLIR Pass
1. `mlir/simple_matmul.mlir` - 理解IR转换
2. `mlir/multi_matmul.mlir` - 理解多矩阵优化
3. 阅读 `../../npu_pass_implementation/lib/BatchComp/` - 理解Pass实现

### 高级 → 扩展开发
1. 添加新的优化算法
2. 支持新的硬件平台
3. 实现自定义Pass

---

## 🔗 相关资源

- [Python优化器文档](../../npu_pass_implementation/docs/guides/QUICKSTART.md)
- [Dialect设计文档](../../npu_pass_implementation/docs/design/BATCHCOMP_DIALECT_DESIGN.md)
- [CP-SAT算法说明](../../npu_pass_implementation/docs/guides/CP_SAT_README.md)
- [完整实施总结](../../npu_pass_implementation/docs/summaries/IMPLEMENTATION_COMPLETE.md)

---

## 🐛 故障排除

### Python示例运行失败

**问题**: `ModuleNotFoundError: No module named 'ortools'`

**解决**:
```bash
cd ../../npu_pass_implementation
pip install -r requirements.txt
```

### MLIR示例运行失败

**问题**: `mlir-opt: command not found`

**解决**: MLIR示例需要完整的MLIR构建环境。如果只想了解框架，建议先运行Python示例。

---

**最后更新**: 2025-11-15
**框架版本**: BatchComp v1.0 (85% complete)
