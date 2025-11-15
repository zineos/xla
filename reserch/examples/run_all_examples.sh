#!/bin/bash
# 运行所有BatchComp示例

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "BatchComp Framework - 完整示例套件"
echo "=========================================="
echo ""

# Python示例
echo -e "${BLUE}第1部分: Python优化器示例${NC}"
echo "=========================================="
echo ""

cd python

echo -e "${YELLOW}[1/3] 基础优化示例${NC}"
python3 basic_optimization.py
echo ""

echo -e "${YELLOW}[2/3] 多矩阵优化示例${NC}"
python3 multi_matrix.py
echo ""

echo -e "${YELLOW}[3/3] 硬件配置示例${NC}"
python3 hardware_configs.py
echo ""

cd ..

# MLIR示例（可选）
echo ""
echo -e "${BLUE}第2部分: MLIR Pass示例${NC}"
echo "=========================================="
echo ""

if command -v mlir-opt &> /dev/null; then
    cd mlir
    ./run_examples.sh all
    cd ..
else
    echo -e "${YELLOW}跳过MLIR示例 (mlir-opt未安装)${NC}"
    echo ""
    echo "MLIR示例需要完整的MLIR构建环境。"
    echo "如果需要运行MLIR示例，请参考:"
    echo "  https://mlir.llvm.org/getting_started/"
    echo ""
fi

# 总结
echo ""
echo "=========================================="
echo -e "${GREEN}✅ 所有示例运行完成!${NC}"
echo "=========================================="
echo ""
echo "📊 示例总结:"
echo "  - Python基础优化 ✓"
echo "  - Python多矩阵优化 ✓"
echo "  - Python硬件配置 ✓"
if command -v mlir-opt &> /dev/null; then
    echo "  - MLIR Pass示例 ✓"
else
    echo "  - MLIR Pass示例 (跳过)"
fi
echo ""
echo "📚 下一步:"
echo "  1. 查看 ../npu_pass_implementation/docs/ 了解框架设计"
echo "  2. 修改示例代码尝试不同参数"
echo "  3. 查看 ../npu_pass_implementation/lib/BatchComp/ 了解Pass实现"
echo ""
