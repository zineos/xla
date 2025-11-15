# NPU Backend 实施路线图

> **项目代号**: XLA-NPU
> **开始日期**: 2025-Q1
> **目标**: 生产级 XLA NPU Backend

## 🎯 关键里程碑

| 阶段 | 时间 | 目标 | 关键交付物 |
|------|------|------|-----------|
| **M0** | Week 1-2 | 环境准备 | 开发环境、团队组建 |
| **M1** | Month 1 | 基础集成 | 最小可运行 backend |
| **M2** | Month 2 | 核心功能 | 完整 matmul 支持 |
| **M3** | Month 3 | MVP | 简单模型运行 |
| **M4** | Month 6 | Beta | 性能优化、工具链 |
| **M5** | Month 9 | GA | 生产就绪 |

---

## Week 1-2: 环境准备和原型验证

### 1. 开发环境搭建

```bash
# 1. 克隆 XLA 代码库
git clone https://github.com/openxla/xla.git
cd xla

# 2. 创建 NPU backend 目录结构
mkdir -p xla/service/npu
mkdir -p xla/service/npu/runtime
mkdir -p xla/service/npu/tests

# 3. 配置 Bazel 构建
cat > xla/service/npu/BUILD << EOF
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "npu_compiler",
    srcs = ["npu_compiler.cc"],
    hdrs = ["npu_compiler.h"],
    deps = [
        "//xla/service:compiler",
        "//xla/service:hlo_module",
    ],
)

cc_library(
    name = "npu_executable",
    srcs = ["npu_executable.cc"],
    hdrs = ["npu_executable.h"],
    deps = [
        "//xla/service:executable",
    ],
)
EOF
```

### 2. 最小原型实现

```cpp
// xla/service/npu/npu_compiler.h
#pragma once

#include "xla/service/compiler.h"

namespace xla {
namespace npu {

class NPUCompiler : public Compiler {
 public:
  NPUCompiler();
  ~NPUCompiler() override = default;

  // 最小实现：只支持 matmul
  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module,
      se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module,
      se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  // 平台标识
  se::Platform::Id PlatformId() const override {
    return se::npu::kNPUPlatformId;
  }
};

}  // namespace npu
}  // namespace xla
```

```cpp
// xla/service/npu/npu_compiler.cc
#include "xla/service/npu/npu_compiler.h"
#include "xla/service/npu/npu_executable.h"

namespace xla {
namespace npu {

StatusOr<std::unique_ptr<HloModule>> NPUCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module,
    se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  // Phase 1: 只做基础验证
  TF_RETURN_IF_ERROR(module->Verify());

  // TODO: 添加 NPU 特定的 passes
  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> NPUCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  // 创建 NPU executable
  return std::make_unique<NPUExecutable>(
      std::move(module), stream_exec);
}

}  // namespace npu
}  // namespace xla
```

---

## Month 1: 基础 XLA 集成

### Week 3: StreamExecutor 实现

```cpp
// xla/stream_executor/npu/npu_stream_executor.h
class NPUStreamExecutor : public StreamExecutor {
 public:
  explicit NPUStreamExecutor(int device_ordinal);

  // 内存管理
  StatusOr<DeviceMemory<uint8>> Allocate(uint64 size) override;
  void Deallocate(DeviceMemoryBase* mem) override;

  // 数据传输
  bool Memcpy(Stream* stream, void* host_dst,
              const DeviceMemoryBase& device_src,
              uint64 size) override;
  bool Memcpy(Stream* stream, DeviceMemoryBase* device_dst,
              const void* host_src, uint64 size) override;

  // 同步
  bool SynchronizeStream(Stream* stream) override;

  // Kernel 启动
  bool Launch(Stream* stream, const ThreadDim& thread_dims,
              const BlockDim& block_dims,
              const KernelBase& kernel,
              const KernelArgs& args) override;

 private:
  // NPU 硬件接口
  std::unique_ptr<NPUDevice> device_;
  int device_ordinal_;
};
```

### Week 4: HLO to NPU Lowering

```cpp
// xla/service/npu/hlo_to_npu_lowering.h
class HloToNPULowering : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "hlo-to-npu-lowering";
  }

  StatusOr<bool> Run(HloModule* module,
                     const absl::flat_hash_set<absl::string_view>&
                         execution_threads) override;

 private:
  // 转换 HLO matmul 到 NPU tile operations
  Status ConvertMatmul(HloInstruction* matmul);

  // 生成 tile descriptors
  std::vector<TileDescriptor> GenerateTiles(
      const Shape& lhs_shape,
      const Shape& rhs_shape,
      const Shape& output_shape);

  // 应用 CP-SAT 调度
  std::vector<TileBatch> ScheduleTiles(
      const std::vector<TileDescriptor>& tiles);
};
```

---

## Month 2: 核心功能实现

### Week 5-6: Tile 调度集成

```cpp
// xla/service/npu/tile_scheduler.h
class NPUTileScheduler {
 public:
  struct TileConfig {
    int m_size = 16;
    int n_size = 16;
    int k_size = 16;
    bool allow_padding = false;
    bool use_cpsat = true;
  };

  // 主调度接口
  StatusOr<ExecutionPlan> ScheduleMatmul(
      const HloInstruction* matmul,
      const TileConfig& config);

  // 集成 Python CP-SAT solver
  StatusOr<std::vector<TileBatch>> CallCPSATSolver(
      const std::vector<TileDescriptor>& tiles);

 private:
  // 贪心算法实现（备用）
  std::vector<TileBatch> GreedySchedule(
      const std::vector<TileDescriptor>& tiles);

  // K 维度展开
  std::vector<TileDescriptor> UnrollKDimension(
      const TileDescriptor& base_tile, int k_tiles);
};
```

### Week 7-8: 内存管理系统

```cpp
// xla/service/npu/memory_manager.h
class NPUMemoryManager {
 public:
  // DDR 内存池
  class DDRMemoryPool {
   public:
    StatusOr<BufferAllocation> Allocate(size_t bytes);
    void Free(const BufferAllocation& allocation);
    Status Defragment();

   private:
    struct FreeBlock {
      void* ptr;
      size_t size;
    };
    std::list<FreeBlock> free_blocks_;
    size_t total_size_;
  };

  // On-chip buffer 管理
  class OnChipBufferManager {
   public:
    static constexpr size_t kMaxSize = 16 * 1024;  // 16KB

    StatusOr<OnChipAllocation> AllocateTile(
        size_t m, size_t n, size_t k);
    void FreeTile(const OnChipAllocation& alloc);

    // 双缓冲支持
    Status EnableDoubleBuffering();
    std::pair<BufferSet, BufferSet> GetDoubleBuffers();
  };

  // Buffer 生命周期分析
  BufferLivenessAnalysis AnalyzeLiveness(const HloModule& module);

  // 生成内存分配计划
  AllocationPlan GenerateAllocationPlan(
      const BufferLivenessAnalysis& analysis);
};
```

---

## Month 3: MVP 完成

### Week 9-10: 端到端测试

```python
# tests/end_to_end_test.py
import numpy as np
import jax
import jax.numpy as jnp
from xla.npu import NPUBackend

def test_simple_matmul():
    """测试简单矩阵乘法"""
    backend = NPUBackend()

    # 测试各种尺寸
    test_cases = [
        (16, 16, 16),    # 完美对齐
        (17, 17, 17),    # 需要 padding
        (100, 200, 150), # 大矩阵
        (7, 13, 5),      # 小矩阵
    ]

    for m, n, k in test_cases:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        # JAX 计算（CPU 参考）
        expected = jnp.dot(a, b)

        # NPU 计算
        with jax.default_device(backend.device(0)):
            result = jnp.dot(a, b)

        # 验证结果
        np.testing.assert_allclose(
            result, expected, rtol=1e-5, atol=1e-6)
        print(f"✅ Test passed: {m}x{k} @ {k}x{n}")

def test_model_inference():
    """测试简单神经网络"""
    import flax.linen as nn

    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dense(10)(x)
            return x

    # 在 NPU 上运行
    backend = NPUBackend()
    with jax.default_device(backend.device(0)):
        model = SimpleModel()
        x = jnp.ones((32, 784))
        params = model.init(jax.random.PRNGKey(0), x)
        output = model.apply(params, x)
        assert output.shape == (32, 10)
        print("✅ Model inference test passed")
```

### Week 11-12: 性能验证和优化

```cpp
// benchmarks/matmul_benchmark.cc
class NPUMatmulBenchmark : public ::benchmark::Fixture {
 public:
  void BenchmarkSquareMatmul(benchmark::State& state) {
    int size = state.range(0);

    // 准备数据
    auto a = GenerateRandomMatrix(size, size);
    auto b = GenerateRandomMatrix(size, size);

    // 预热
    ExecuteMatmul(a, b);
    stream_->Synchronize();

    // 基准测试
    for (auto _ : state) {
      auto start = std::chrono::high_resolution_clock::now();
      ExecuteMatmul(a, b);
      stream_->Synchronize();
      auto end = std::chrono::high_resolution_clock::now();

      auto elapsed = std::chrono::duration_cast<
          std::chrono::duration<double>>(end - start);
      state.SetIterationTime(elapsed.count());
    }

    // 报告性能指标
    double flops = 2.0 * size * size * size;
    state.SetItemsProcessed(flops * state.iterations());
    state.counters["GFLOPS"] = benchmark::Counter(
        flops / 1e9, benchmark::Counter::kIsRate);

    // NPU 利用率
    double utilization = ComputeNPUUtilization(size);
    state.counters["NPU_Util"] = utilization;
  }
};

BENCHMARK_REGISTER_F(NPUMatmulBenchmark, SquareMatmul)
    ->RangeMultiplier(2)
    ->Range(16, 2048)
    ->UseManualTime();
```

---

## Month 4-6: 性能优化和工具链

### Month 4: Pattern Matching 和 Fusion

```cpp
// xla/service/npu/npu_fusion_pass.h
class NPUFusionPass : public HloModulePass {
 public:
  // 识别可融合的模式
  struct FusionPattern {
    std::vector<HloOpcode> opcodes;
    std::function<bool(const HloInstruction*)> matcher;
    std::function<Status(HloComputation*, HloInstruction*)> fuser;
  };

  StatusOr<bool> Run(HloModule* module,
                     const absl::flat_hash_set<absl::string_view>&
                         execution_threads) override;

 private:
  // GEMM + Bias + Activation 融合
  Status FuseGemmBiasActivation(HloComputation* comp);

  // BatchNorm 融合
  Status FuseBatchNorm(HloComputation* comp);

  // Elementwise 链融合
  Status FuseElementwiseChain(HloComputation* comp);

  std::vector<FusionPattern> patterns_;
};
```

### Month 5: Profiler 实现

```cpp
// xla/service/npu/npu_profiler.h
class NPUProfiler {
 public:
  // 开始性能分析
  Status StartProfiling(const ProfilingConfig& config);

  // 停止并生成报告
  StatusOr<ProfilingReport> StopProfiling();

  // 硬件性能计数器
  struct HardwareCounters {
    uint64_t cycles;
    uint64_t instructions;
    uint64_t memory_reads;
    uint64_t memory_writes;
    uint64_t cache_misses;
    uint64_t npu_active_cycles;
    uint64_t npu_stall_cycles;
  };

  // 获取实时计数器
  HardwareCounters GetCounters() const;

  // 生成火焰图
  Status GenerateFlameGraph(const std::string& output_path);

  // 生成 timeline
  Status GenerateTimeline(const std::string& output_path);
};
```

### Month 6: 调试工具

```cpp
// xla/service/npu/npu_debugger.h
class NPUDebugger {
 public:
  // 断点管理
  Status SetBreakpoint(const TileCoordinate& coord);
  Status RemoveBreakpoint(const TileCoordinate& coord);

  // 执行控制
  Status Run();
  Status Step();
  Status Continue();

  // 内存检查
  StatusOr<std::vector<float>> ReadMemory(
      const DeviceMemoryBase& mem,
      size_t offset, size_t count);

  // Tile 状态检查
  struct TileState {
    TileCoordinate coord;
    TileStatus status;
    std::vector<float> input_a;
    std::vector<float> input_b;
    std::vector<float> output;
    int cycles_executed;
  };

  StatusOr<TileState> GetTileState(const TileCoordinate& coord);

  // 生成调试报告
  StatusOr<DebugReport> GenerateReport();
};
```

---

## Month 7-9: 生产化

### Month 7: 错误处理和容错

```cpp
// xla/service/npu/error_handler.h
class NPUErrorHandler {
 public:
  // 错误恢复策略
  enum class RecoveryStrategy {
    kRetry,        // 重试操作
    kFallback,     // 回退到 CPU
    kAbort,        // 终止执行
    kIsolate,      // 隔离故障组件
  };

  // 注册错误处理器
  void RegisterHandler(
      ErrorType type,
      std::function<RecoveryStrategy(const Error&)> handler);

  // 处理硬件错误
  Status HandleHardwareError(const HardwareError& error);

  // 处理数值错误（NaN, Inf）
  Status HandleNumericalError(const NumericalError& error);

  // 健康检查
  struct HealthStatus {
    bool is_healthy;
    std::vector<std::string> issues;
    std::map<int, CoreStatus> core_status;
    MemoryHealth memory_health;
    ThermalStatus thermal_status;
  };

  HealthStatus CheckHealth();

  // 自动恢复
  Status AutoRecover();
};
```

### Month 8: 监控和运维

```yaml
# deployment/kubernetes/npu-operator.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: npu-device-plugin
spec:
  selector:
    matchLabels:
      name: npu-device-plugin
  template:
    metadata:
      labels:
        name: npu-device-plugin
    spec:
      containers:
      - name: npu-device-plugin
        image: xla/npu-device-plugin:latest
        securityContext:
          privileged: true
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
        - name: dev
          mountPath: /dev
        env:
        - name: NPU_VISIBLE_DEVICES
          value: "all"
        - name: NPU_DRIVER_VERSION
          value: "1.0.0"
```

```python
# monitoring/metrics_exporter.py
from prometheus_client import Gauge, Histogram, Counter
import time

class NPUMetricsExporter:
    def __init__(self):
        # 定义指标
        self.npu_utilization = Gauge(
            'npu_utilization_percent',
            'NPU utilization percentage',
            ['device_id', 'core_id']
        )

        self.memory_usage = Gauge(
            'npu_memory_usage_bytes',
            'NPU memory usage in bytes',
            ['device_id', 'memory_type']
        )

        self.operation_latency = Histogram(
            'npu_operation_latency_seconds',
            'NPU operation latency',
            ['operation_type', 'tile_size']
        )

        self.error_count = Counter(
            'npu_errors_total',
            'Total NPU errors',
            ['device_id', 'error_type']
        )

    def collect_and_export(self):
        """收集并导出指标"""
        while True:
            # 收集 NPU 指标
            devices = get_npu_devices()
            for device in devices:
                # 利用率
                for core_id in range(16):
                    util = device.get_core_utilization(core_id)
                    self.npu_utilization.labels(
                        device_id=device.id,
                        core_id=core_id
                    ).set(util)

                # 内存使用
                ddr_usage = device.get_ddr_usage()
                self.memory_usage.labels(
                    device_id=device.id,
                    memory_type='ddr'
                ).set(ddr_usage)

                onchip_usage = device.get_onchip_usage()
                self.memory_usage.labels(
                    device_id=device.id,
                    memory_type='onchip'
                ).set(onchip_usage)

            time.sleep(10)  # 10秒采样间隔
```

### Month 9: 文档和测试完善

```markdown
# docs/user_guide.md
# NPU Backend 用户指南

## 快速开始

### 安装

```bash
pip install xla-npu
```

### 基础使用

```python
import jax
from xla.npu import NPUBackend

# 初始化 NPU
backend = NPUBackend()
devices = backend.list_devices()
print(f"Found {len(devices)} NPU devices")

# 使用 NPU 进行计算
with jax.default_device(devices[0]):
    # 你的 JAX 代码
    result = jax.numpy.dot(a, b)
```

## 性能优化指南

### 1. Tile 大小选择

- 使用 16 的倍数以获得最佳性能
- 对于小矩阵，考虑批处理

### 2. 内存管理

- 预分配大块内存减少分配开销
- 使用内存池避免碎片

### 3. 调度优化

- 大矩阵使用 CP-SAT
- 小矩阵使用贪心算法

## 故障排除

### 常见问题

1. **NPU 设备未找到**
   - 检查驱动安装
   - 确认设备权限

2. **性能低于预期**
   - 使用 profiler 分析瓶颈
   - 检查 tile 利用率

3. **数值精度问题**
   - 启用混合精度训练
   - 使用梯度缩放
```

---

## 关键决策点

### 技术决策

| 决策 | 选项 | 推荐 | 原因 |
|------|------|------|------|
| **编译框架** | MLIR vs Custom | MLIR | 复用生态、降低维护成本 |
| **调度算法** | CP-SAT vs Greedy | 混合 | 平衡性能和编译时间 |
| **内存管理** | 静态 vs 动态 | 混合 | 灵活性和效率平衡 |
| **错误处理** | 硬失败 vs 软恢复 | 软恢复 | 提高可用性 |

### 架构决策

| 决策 | 选项 | 推荐 | 原因 |
|------|------|------|------|
| **Backend 类型** | Standalone vs Plugin | Plugin | 易于集成和维护 |
| **代码生成** | LLVM vs Custom | Custom | NPU ISA 特殊性 |
| **Runtime** | Sync vs Async | Async | 隐藏延迟、提高吞吐 |

---

## 成功标准

### Sprint 1 (Week 1-2)
- ✅ 开发环境搭建完成
- ✅ 最小原型运行
- ✅ 团队到位

### Sprint 2-4 (Month 1)
- ✅ StreamExecutor 实现
- ✅ 基础 HLO lowering
- ✅ 简单 matmul 通过

### Sprint 5-8 (Month 2)
- ✅ Tile 调度集成
- ✅ 内存管理实现
- ✅ K 维度处理

### Sprint 9-12 (Month 3)
- ✅ MVP 功能完整
- ✅ 端到端测试通过
- ✅ 性能基准建立

### Beta (Month 4-6)
- ✅ 性能优化 >80% 理论峰值
- ✅ 工具链完善
- ✅ 初步文档

### GA (Month 7-9)
- ✅ 生产级稳定性
- ✅ 完整文档
- ✅ 客户 POC 成功

---

## 风险跟踪

| ID | 风险 | 状态 | 缓解措施 | 负责人 |
|----|------|------|----------|--------|
| R1 | XLA API 变更 | 🟡 监控 | 版本锁定 | Tech Lead |
| R2 | 性能不达标 | 🟢 低 | 早期基准测试 | Perf Eng |
| R3 | 硬件 bug | 🟡 监控 | Workaround 机制 | HW Eng |
| R4 | 资源不足 | 🔴 高 | MVP 优先 | PM |

---

**文档版本**: 1.0
**创建日期**: 2025-11-15
**下次评审**: 2025-12-01
**状态**: 实施中