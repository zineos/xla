#!/bin/bash
# BatchComp MLIR示例运行脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "BatchComp MLIR示例"
echo "========================================="
echo ""

# 检查mlir-opt是否可用
if ! command -v mlir-opt &> /dev/null; then
    echo -e "${RED}错误: mlir-opt未找到${NC}"
    echo ""
    echo "MLIR示例需要完整的MLIR构建环境。"
    echo ""
    echo "如果您只想了解BatchComp框架，请先运行Python示例："
    echo "  cd ../python"
    echo "  python3 basic_optimization.py"
    echo ""
    exit 1
fi

# 运行示例函数
run_example() {
    local name=$1
    local file=$2
    local args=$3

    echo -e "${YELLOW}运行示例: $name${NC}"
    echo "文件: $file"
    echo "参数: $args"
    echo ""

    # 实际运行（当前BatchComp Pass尚未完全集成到mlir-opt）
    # 这里展示命令，但可能会失败
    echo "命令:"
    echo "  mlir-opt $file $args"
    echo ""

    # 尝试运行（预期可能失败）
    if mlir-opt "$file" $args 2>&1 | head -20; then
        echo -e "${GREEN}✓ 成功${NC}"
    else
        echo -e "${YELLOW}⚠ Pass尚未集成到mlir-opt${NC}"
        echo ""
        echo "这是预期的 - BatchComp Pass需要先构建并注册到MLIR。"
        echo "当前可以查看.mlir文件了解预期的转换。"
    fi
    echo ""
    echo "-----------------------------------------"
    echo ""
}

# 主逻辑
case "${1:-all}" in
    simple)
        run_example \
            "简单Matmul转换" \
            "simple_matmul.mlir" \
            "--batch-comp-tile-scheduler='hardware=npu tile-size=16,16,16 algorithm=cpsat'"
        ;;

    multi)
        run_example \
            "多Matmul跨矩阵优化" \
            "multi_matmul.mlir" \
            "--batch-comp-tile-scheduler='hardware=npu tile-size=16,16,16 algorithm=cpsat enable-multi-matmul=true'"
        ;;

    all)
        echo "运行所有MLIR示例..."
        echo ""

        run_example \
            "简单Matmul转换" \
            "simple_matmul.mlir" \
            "--batch-comp-tile-scheduler='hardware=npu tile-size=16,16,16'"

        run_example \
            "多Matmul跨矩阵优化" \
            "multi_matmul.mlir" \
            "--batch-comp-tile-scheduler='hardware=npu tile-size=16,16,16 enable-multi-matmul=true'"

        echo -e "${GREEN}所有示例运行完成${NC}"
        ;;

    *)
        echo "用法: $0 [simple|multi|all]"
        echo ""
        echo "示例:"
        echo "  $0 simple  - 运行简单matmul示例"
        echo "  $0 multi   - 运行多matmul示例"
        echo "  $0 all     - 运行所有示例（默认）"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "说明"
echo "========================================="
echo ""
echo "这些MLIR示例展示了BatchComp Pass的预期转换。"
echo ""
echo "完整的Pass集成需要:"
echo "  1. 构建BatchComp Dialect"
echo "  2. 编译BatchComp Passes"
echo "  3. 将Passes注册到mlir-opt"
echo ""
echo "在此之前，您可以:"
echo "  - 查看.mlir文件了解IR转换"
echo "  - 运行Python示例测试优化器"
echo "  - 查看Pass实现代码"
echo ""
echo "相关文档:"
echo "  - 实现指南: ../../npu_pass_implementation/docs/summaries/PHASE3_PROGRESS.md"
echo "  - Dialect设计: ../../npu_pass_implementation/docs/design/BATCHCOMP_DIALECT_DESIGN.md"
echo ""
