# Quick Start Guide - CP-SAT NPU Tile Scheduler

Get started with the CP-SAT global optimizer in 5 minutes.

## 1. Install Dependencies (1 minute)

```bash
cd /Users/neos/Documents/project/zicun/xla/npu_pass_implementation

# Install Python packages
pip install ortools pydantic tabulate

# Or use requirements.txt
pip install -r requirements.txt
```

**Check installation:**
```bash
python3 -c "import ortools; import pydantic; print('✓ Ready!')"
```

## 2. Run Basic Test (30 seconds)

```bash
# Run basic correctness tests
python3 test_cpsat_basic.py
```

**Expected output:**
```
CP-SAT Tile Scheduler - Basic Tests
================================================================================
Testing imports...
  ✓ ortools 9.x.x
  ✓ pydantic 2.x.x
  ✓ tabulate 0.9.x
...
✅ ALL TESTS PASSED
```

## 3. Run Examples (1 minute)

```bash
# Run built-in examples
python3 CP_SAT_TILE_SCHEDULER.py
```

**Expected output:**
```
================================================================================
Example: Single Matrix [37x50] x [50x64]
================================================================================
Total tiles: 48
K-iterations: 4

Status: OPTIMAL
Solve time: 0.05s
Total batches: 3
Total NOPs: 0
Average utilization: 100.0%
...
```

## 4. Run Benchmarks (2 minutes)

```bash
# Compare Greedy vs Improved Greedy vs CP-SAT
python3 benchmark_schedulers.py
```

**Expected output:**
```
================================================================================
BENCHMARK RESULTS
================================================================================
Test Case         Algorithm        Tiles  Batches  NOPs  Utilization  Time
---------------  ----------------  -----  -------  ----  -----------  ------
Small Single     Greedy            1      1        15    6.2%         0.0001s
Small Single     Improved Greedy   1      1        15    6.2%         0.0001s
Small Single     CP-SAT            1      1        15    6.2%         0.01s
...
Multi-Matrix     Greedy            65     5        15    81.2%        0.001s
Multi-Matrix     Improved Greedy   65     4        1     98.4%        0.002s
Multi-Matrix     CP-SAT            65     4        1     98.4%        0.12s
...

KEY INSIGHTS
Multi-Matrix (Small):
  Best: CP-SAT
  Improvement: 20.0% fewer batches, 93.3% fewer NOPs
  Time: 0.12s (120x slower)
```

## 5. Try CLI Tool (1 minute)

```bash
# Create example input
cat > example_tiles.json <<'EOF'
{
  "tiles": [
    {
      "tile_id": 0,
      "matrix_id": 0,
      "k_iteration": 0,
      "m_offset": 0,
      "n_offset": 0,
      "actual_m": 10,
      "actual_n": 10,
      "actual_k": 10
    }
  ]
}
EOF

# Run solver
./cpsat_scheduler_cli.py \
  --input example_tiles.json \
  --output example_schedule.json \
  --time-limit 10

# Check output
cat example_schedule.json
```

**Expected output file:**
```json
{
  "status": "OPTIMAL",
  "solve_time_seconds": 0.012,
  "objective_value": 1015.0,
  "batches": [
    {
      "batch_id": 0,
      "tile_ids": [0],
      "num_nops": 15
    }
  ],
  "statistics": {
    "total_batches": 1,
    "total_nops": 15,
    "average_utilization": 0.0625
  }
}
```

## 6. Understanding the Results

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Batches** | Number of NPU execution batches | Minimize |
| **NOPs** | Total no-op slots across batches | Minimize |
| **Utilization** | Non-NOP tiles / total slots | Maximize (→ 100%) |
| **Time** | Solver runtime | Depends on budget |
| **Status** | OPTIMAL / FEASIBLE / GREEDY | OPTIMAL best |

### When Each Algorithm Wins

**Greedy wins:**
- Single matrix (all algorithms optimal)
- Real-time compilation (< 1ms budget)

**Improved Greedy wins:**
- Multiple small matrices (< 50 tiles)
- Fast compilation (1-10ms budget)
- Gets 90-98% of optimal, 100× faster than CP-SAT

**CP-SAT wins:**
- Multiple matrices (50-1000 tiles)
- Offline compilation (seconds available)
- Need provable optimality
- 20-40% better than greedy on complex cases

## 7. Next Steps

### For Python Development

See [CP_SAT_README.md](CP_SAT_README.md) for:
- Python API documentation
- Advanced configuration
- Multi-objective tuning
- Troubleshooting guide

### For C++/MLIR Integration

See [CP_SAT_INTEGRATION_GUIDE.md](CP_SAT_INTEGRATION_GUIDE.md) for:
- C++ integration patterns
- JSON I/O format
- CMake configuration
- Hybrid strategy implementation

### For Understanding CP-SAT

See the original [CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/) for:
- Comprehensive CP-SAT tutorial
- Modeling best practices
- Advanced techniques (LNS, warm starting)
- Many more examples

## Troubleshooting

### "No module named 'ortools'"

```bash
pip install ortools
```

### "No module named 'pydantic'"

```bash
pip install pydantic
```

### "Command not found: ./cpsat_scheduler_cli.py"

```bash
chmod +x cpsat_scheduler_cli.py
python3 cpsat_scheduler_cli.py --help
```

### CP-SAT taking too long

Reduce time limit:
```bash
./cpsat_scheduler_cli.py --time-limit 5 ...
```

Or use Improved Greedy instead:
```python
from greedy_scheduler import ImprovedGreedyScheduler
scheduler = ImprovedGreedyScheduler(tiles, config)
```

### Getting sub-optimal results

Check solver status:
```python
if solution.status != "OPTIMAL":
    print(f"Not optimal: {solution.status}")
    print(f"Gap: {(obj - bound) / obj:.1%}")
```

Increase time limit if gap is large:
```python
config = NPUScheduleConfig(time_limit_seconds=60)
```

## File Reference

Quick reference of all files:

```
npu_pass_implementation/
├── CP_SAT_TILE_SCHEDULER.py      # Core CP-SAT implementation
├── greedy_scheduler.py            # Greedy baselines
├── cpsat_scheduler_cli.py         # CLI tool for C++ integration
├── benchmark_schedulers.py        # Comprehensive benchmarks
├── test_cpsat_basic.py            # Basic tests
├── requirements.txt               # Python dependencies
├── QUICKSTART.md                  # This file
├── CP_SAT_README.md               # Detailed documentation
├── CP_SAT_INTEGRATION_GUIDE.md    # C++/MLIR integration
└── CPSAT_IMPLEMENTATION_SUMMARY.md # Implementation details
```

## Summary

You now have:
- ✅ Working CP-SAT global optimizer
- ✅ Greedy and Improved Greedy baselines
- ✅ CLI tool for C++/MLIR integration
- ✅ Comprehensive benchmarks
- ✅ Complete documentation

**Recommended workflow:**
1. Use `benchmark_schedulers.py` to understand performance
2. For Python: Use `CP_SAT_TILE_SCHEDULER.py` directly
3. For C++/MLIR: Use `cpsat_scheduler_cli.py` via system call
4. Choose algorithm based on hybrid strategy (see Integration Guide)

**Next:** Read [CP_SAT_INTEGRATION_GUIDE.md](CP_SAT_INTEGRATION_GUIDE.md) for C++/MLIR integration.
