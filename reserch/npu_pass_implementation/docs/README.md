# BatchComp Framework 文档中心

**完整的BatchComp批组合优化框架文档**

---

## 📚 文档导航

### 🚀 快速开始

新用户推荐阅读顺序：

1. **[项目总览](summaries/IMPLEMENTATION_COMPLETE.md)** ⭐ - 5分钟了解整个项目
2. **[快速开始指南](guides/QUICKSTART.md)** - 10分钟上手使用
3. **[Python优化器教程](tutorials/PYTHON_OPTIMIZER_TUTORIAL.md)** - 深入理解优化算法
4. **[MLIR Pass教程](tutorials/MLIR_PASS_TUTORIAL.md)** - 了解编译器集成

---

## 📖 文档分类

### 1. 📋 项目总结 (summaries/)

**核心文档** - 了解项目进展和架构

| 文档 | 说明 | 阅读时长 |
|------|------|----------|
| [IMPLEMENTATION_COMPLETE.md](summaries/IMPLEMENTATION_COMPLETE.md) | ⭐ 完整实施总结 | 5分钟 |
| [PHASE1_SUMMARY.md](summaries/PHASE1_SUMMARY.md) | Python优化器重构 | 10分钟 |
| [PHASE2_SUMMARY.md](summaries/PHASE2_SUMMARY.md) | Dialect设计文档 | 15分钟 |
| [PHASE3_PROGRESS.md](summaries/PHASE3_PROGRESS.md) | C++ Pass实现进度 | 10分钟 |
| [SESSION_2025_11_15.md](summaries/SESSION_2025_11_15.md) | 最新工作记录 | 5分钟 |

### 2. 🎨 设计文档 (design/)

**架构设计** - 深入理解系统设计

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| [BATCHCOMP_DIALECT_DESIGN.md](design/BATCHCOMP_DIALECT_DESIGN.md) | ⭐ 完整Dialect设计 | 编译器开发者 |
| [ARCHITECTURE.md](design/ARCHITECTURE.md) | 系统架构概览 | 所有开发者 |
| [PYTHON_CPP_INTEGRATION.md](design/PYTHON_CPP_INTEGRATION.md) | Python-C++集成设计 | 系统集成开发者 |

### 3. 📘 使用指南 (guides/)

**实用指南** - 如何使用框架

| 文档 | 说明 | 难度 |
|------|------|------|
| [QUICKSTART.md](guides/QUICKSTART.md) | 快速开始 | ⭐ 入门 |
| [CP_SAT_README.md](guides/CP_SAT_README.md) | CP-SAT算法说明 | ⭐⭐ 中级 |
| [CP_SAT_INTEGRATION_GUIDE.md](guides/CP_SAT_INTEGRATION_GUIDE.md) | CP-SAT集成指南 | ⭐⭐ 中级 |
| [HARDWARE_CONFIG_GUIDE.md](guides/HARDWARE_CONFIG_GUIDE.md) | 硬件配置指南 | ⭐ 入门 |
| [OPTIMIZATION_GUIDE.md](guides/OPTIMIZATION_GUIDE.md) | 优化策略指南 | ⭐⭐⭐ 高级 |

### 4. 🎓 教程 (tutorials/)

**手把手教程** - 从零开始学习

| 文档 | 说明 | 预计时长 |
|------|------|----------|
| [PYTHON_OPTIMIZER_TUTORIAL.md](tutorials/PYTHON_OPTIMIZER_TUTORIAL.md) | Python优化器完整教程 | 30分钟 |
| [MLIR_PASS_TUTORIAL.md](tutorials/MLIR_PASS_TUTORIAL.md) | MLIR Pass开发教程 | 45分钟 |
| [CUSTOM_HARDWARE_TUTORIAL.md](tutorials/CUSTOM_HARDWARE_TUTORIAL.md) | 自定义硬件支持教程 | 60分钟 |
| [ALGORITHM_TUTORIAL.md](tutorials/ALGORITHM_TUTORIAL.md) | 优化算法详解 | 40分钟 |

### 5. 📚 API参考 (api/)

**API文档** - 详细的接口说明

| 文档 | 说明 | 内容 |
|------|------|------|
| [PYTHON_API.md](api/PYTHON_API.md) | Python API完整参考 | 类、函数、参数 |
| [CPP_API.md](api/CPP_API.md) | C++ API完整参考 | Pass、操作、类型 |
| [MLIR_OPS.md](api/MLIR_OPS.md) | MLIR操作参考 | 所有BatchComp操作 |
| [CONFIG_SCHEMA.md](api/CONFIG_SCHEMA.md) | 配置文件Schema | YAML配置格式 |

### 6. 🔧 开发文档 (development/)

**开发指南** - 贡献和扩展

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| [CONTRIBUTING.md](development/CONTRIBUTING.md) | 贡献指南 | 贡献者 |
| [CODING_STYLE.md](development/CODING_STYLE.md) | 代码规范 | 开发者 |
| [TESTING_GUIDE.md](development/TESTING_GUIDE.md) | 测试指南 | 开发者 |
| [BUILD_GUIDE.md](development/BUILD_GUIDE.md) | 构建指南 | 系统集成者 |

---

## 🎯 按角色推荐阅读路径

### 🆕 新用户（了解框架）

```
1. IMPLEMENTATION_COMPLETE.md (项目总览)
   ↓
2. QUICKSTART.md (快速开始)
   ↓
3. PYTHON_OPTIMIZER_TUTORIAL.md (Python教程)
   ↓
4. examples/python/ (运行示例)
```

### 🐍 Python开发者（使用优化器）

```
1. PYTHON_OPTIMIZER_TUTORIAL.md (教程)
   ↓
2. PYTHON_API.md (API参考)
   ↓
3. HARDWARE_CONFIG_GUIDE.md (配置指南)
   ↓
4. OPTIMIZATION_GUIDE.md (优化策略)
```

### 🔧 MLIR开发者（集成编译器）

```
1. BATCHCOMP_DIALECT_DESIGN.md (Dialect设计)
   ↓
2. MLIR_PASS_TUTORIAL.md (Pass教程)
   ↓
3. CPP_API.md (C++ API)
   ↓
4. BUILD_GUIDE.md (构建指南)
```

### 🏗️ 系统架构师（设计系统）

```
1. ARCHITECTURE.md (架构概览)
   ↓
2. BATCHCOMP_DIALECT_DESIGN.md (Dialect设计)
   ↓
3. PYTHON_CPP_INTEGRATION.md (集成设计)
   ↓
4. PHASE3_PROGRESS.md (实现进度)
```

### 🔬 研究人员（算法研究）

```
1. ALGORITHM_TUTORIAL.md (算法详解)
   ↓
2. CP_SAT_README.md (CP-SAT算法)
   ↓
3. OPTIMIZATION_GUIDE.md (优化策略)
   ↓
4. PHASE1_SUMMARY.md (Python实现)
```

---

## 📝 文档状态

| 类别 | 完成度 | 说明 |
|------|--------|------|
| **项目总结** | ✅ 100% | 所有Phase文档完整 |
| **设计文档** | ✅ 90% | 核心设计完成，部分待补充 |
| **使用指南** | ✅ 80% | 基础指南完成，高级指南待补充 |
| **教程** | ⏳ 60% | Python教程完成，MLIR教程待完善 |
| **API参考** | ⏳ 40% | 框架完成，详细文档待生成 |
| **开发文档** | ⏳ 30% | 基础指南完成，详细规范待补充 |

---

## 🔗 外部资源

### MLIR官方文档
- [MLIR首页](https://mlir.llvm.org/)
- [Dialect教程](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)
- [Pass基础设施](https://mlir.llvm.org/docs/PassManagement/)

### CP-SAT算法
- [CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/)
- [Google OR-Tools](https://developers.google.com/optimization)

### Python相关
- [pybind11文档](https://pybind11.readthedocs.io/)
- [Google OR-Tools Python](https://developers.google.com/optimization/cp/cp_solver)

---

## 🆘 获取帮助

### 问题排查

1. **查看FAQ**: [guides/FAQ.md](guides/FAQ.md)
2. **检查已知问题**: [development/KNOWN_ISSUES.md](development/KNOWN_ISSUES.md)
3. **阅读故障排除**: [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)

### 示例代码

- **Python示例**: `../examples/python/`
- **MLIR示例**: `../examples/mlir/`

---

## 📊 文档版本

- **版本**: v1.0
- **最后更新**: 2025-11-15
- **框架版本**: BatchComp 85% (Phase 3)
- **维护者**: NPU Compiler Team

---

**💡 提示**: 所有文档使用Markdown格式编写，支持GitHub Flavored Markdown。
