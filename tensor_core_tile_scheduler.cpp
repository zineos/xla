/**
 * Tensor Core约束下的不规则Tile调度器
 *
 * 硬件约束：
 * - Tensor Core固定需要16个tile
 * - 每个tile固定大小 [k1, n1]（例如16×16）
 * - 输入矩阵[K, N]可能不规则，无法整除16
 *
 * 问题：如何组合多个不规则矩阵的tile，凑够16个给Tensor Core？
 *
 * 编译：g++ -O3 -std=c++17 tensor_core_tile_scheduler.cpp -o tensor_core_scheduler
 * 运行：./tensor_core_scheduler
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iomanip>

using namespace std;

// ============================================================================
// Tensor Core配置
// ============================================================================
constexpr int TENSOR_CORE_TILES = 16;  // 固定需要16个tile
constexpr int TILE_K = 16;              // 每个tile的K维度
constexpr int TILE_N = 16;              // 每个tile的N维度

// ============================================================================
// Tile描述符
// ============================================================================
struct TileDescriptor {
    int matrix_id;      // 来自哪个矩阵
    int k_offset;       // K维度偏移
    int n_offset;       // N维度偏移
    int actual_k;       // 实际K大小（可能<TILE_K）
    int actual_n;       // 实际N大小（可能<TILE_N）
    bool is_padding;    // 是否是padding tile

    TileDescriptor() : matrix_id(-1), k_offset(0), n_offset(0),
                       actual_k(TILE_K), actual_n(TILE_N), is_padding(false) {}

    TileDescriptor(int mid, int k, int n, int ak, int an, bool pad = false)
        : matrix_id(mid), k_offset(k), n_offset(n),
          actual_k(ak), actual_n(an), is_padding(pad) {}

    void print() const {
        if (is_padding) {
            cout << "[PAD]";
        } else {
            cout << "[M" << matrix_id << " @(" << k_offset << "," << n_offset << ") "
                 << actual_k << "×" << actual_n << "]";
        }
    }
};

// ============================================================================
// 矩阵描述
// ============================================================================
struct MatrixDescriptor {
    int id;
    int K;
    int N;
    float* data;  // 数据指针

    MatrixDescriptor(int _id, int _K, int _N, float* _data)
        : id(_id), K(_K), N(_N), data(_data) {}

    int num_tiles_k() const {
        return (K + TILE_K - 1) / TILE_K;
    }

    int num_tiles_n() const {
        return (N + TILE_N - 1) / TILE_N;
    }

    int total_tiles() const {
        return num_tiles_k() * num_tiles_n();
    }
};

// ============================================================================
// Tile调度器
// ============================================================================
class TensorCoreTileScheduler {
public:
    // 策略1：单矩阵内组合tile
    vector<TileDescriptor> schedule_single_matrix(const MatrixDescriptor& mat) {
        vector<TileDescriptor> batch;
        batch.reserve(TENSOR_CORE_TILES);

        cout << "\n策略1：单矩阵内组合\n";
        cout << "  矩阵" << mat.id << ": K=" << mat.K << ", N=" << mat.N << "\n";
        cout << "  可用tile: " << mat.num_tiles_k() << "×" << mat.num_tiles_n()
             << " = " << mat.total_tiles() << "个\n";

        // 遍历矩阵的所有tile
        for (int nj = 0; nj < mat.num_tiles_n(); nj++) {
            for (int ki = 0; ki < mat.num_tiles_k(); ki++) {
                if (batch.size() >= TENSOR_CORE_TILES) break;

                int k_start = ki * TILE_K;
                int n_start = nj * TILE_N;
                int actual_k = min(TILE_K, mat.K - k_start);
                int actual_n = min(TILE_N, mat.N - n_start);

                batch.emplace_back(mat.id, k_start, n_start, actual_k, actual_n, false);
            }
            if (batch.size() >= TENSOR_CORE_TILES) break;
        }

        // 不足16个，填充虚拟tile
        while (batch.size() < TENSOR_CORE_TILES) {
            batch.emplace_back(-1, 0, 0, TILE_K, TILE_N, true);
        }

        return batch;
    }

    // 策略2：多矩阵组合tile（Batching）
    vector<TileDescriptor> schedule_multi_matrix(const vector<MatrixDescriptor>& matrices) {
        vector<TileDescriptor> batch;
        batch.reserve(TENSOR_CORE_TILES);

        cout << "\n策略2：多矩阵组合（Batching）\n";
        cout << "  输入 " << matrices.size() << " 个矩阵:\n";
        for (const auto& mat : matrices) {
            cout << "    M" << mat.id << ": K=" << mat.K << ", N=" << mat.N
                 << " → " << mat.total_tiles() << " tiles\n";
        }

        // 轮询各矩阵收集tile
        vector<int> tile_indices(matrices.size(), 0);  // 每个矩阵的当前tile索引

        while (batch.size() < TENSOR_CORE_TILES) {
            bool added = false;

            for (size_t mat_idx = 0; mat_idx < matrices.size(); mat_idx++) {
                const auto& mat = matrices[mat_idx];
                int& tile_idx = tile_indices[mat_idx];

                if (tile_idx >= mat.total_tiles()) continue;

                // 计算tile在矩阵中的位置
                int nj = tile_idx / mat.num_tiles_k();
                int ki = tile_idx % mat.num_tiles_k();

                int k_start = ki * TILE_K;
                int n_start = nj * TILE_N;
                int actual_k = min(TILE_K, mat.K - k_start);
                int actual_n = min(TILE_N, mat.N - n_start);

                batch.emplace_back(mat.id, k_start, n_start, actual_k, actual_n, false);
                tile_idx++;
                added = true;

                if (batch.size() >= TENSOR_CORE_TILES) break;
            }

            // 所有矩阵都用完了，填充padding
            if (!added) {
                batch.emplace_back(-1, 0, 0, TILE_K, TILE_N, true);
            }
        }

        return batch;
    }

    // 策略3：智能组合（最小化padding）
    vector<TileDescriptor> schedule_optimized(const vector<MatrixDescriptor>& matrices) {
        vector<TileDescriptor> batch;
        batch.reserve(TENSOR_CORE_TILES);

        cout << "\n策略3：智能组合（最小化padding）\n";

        // 统计所有可用的tile
        struct TileCandidate {
            int matrix_id;
            int ki, nj;
            int actual_k, actual_n;
            float padding_ratio;  // padding比例（越小越好）
        };

        vector<TileCandidate> candidates;
        for (const auto& mat : matrices) {
            for (int nj = 0; nj < mat.num_tiles_n(); nj++) {
                for (int ki = 0; ki < mat.num_tiles_k(); ki++) {
                    int k_start = ki * TILE_K;
                    int n_start = nj * TILE_N;
                    int actual_k = min(TILE_K, mat.K - k_start);
                    int actual_n = min(TILE_N, mat.N - n_start);

                    float padding = 1.0f - (float)(actual_k * actual_n) / (TILE_K * TILE_N);

                    candidates.push_back({mat.id, ki, nj, actual_k, actual_n, padding});
                }
            }
        }

        cout << "  可用tile总数: " << candidates.size() << "\n";

        // 按padding比例排序（完整tile优先）
        sort(candidates.begin(), candidates.end(),
             [](const TileCandidate& a, const TileCandidate& b) {
                 return a.padding_ratio < b.padding_ratio;
             });

        // 选择前16个最优tile
        for (size_t i = 0; i < min(candidates.size(), (size_t)TENSOR_CORE_TILES); i++) {
            const auto& cand = candidates[i];
            batch.emplace_back(cand.matrix_id,
                             cand.ki * TILE_K, cand.nj * TILE_N,
                             cand.actual_k, cand.actual_n, false);
        }

        // 不足则填充
        while (batch.size() < TENSOR_CORE_TILES) {
            batch.emplace_back(-1, 0, 0, TILE_K, TILE_N, true);
        }

        return batch;
    }

    // 打印调度结果
    void print_schedule(const vector<TileDescriptor>& batch) {
        cout << "\n调度结果（16个tile）:\n";

        int real_tiles = 0;
        int padding_tiles = 0;
        int total_elements = 0;
        int padded_elements = 0;

        for (size_t i = 0; i < batch.size(); i++) {
            cout << "  Tile[" << setw(2) << i << "]: ";
            batch[i].print();
            cout << "\n";

            if (batch[i].is_padding) {
                padding_tiles++;
                padded_elements += TILE_K * TILE_N;
            } else {
                real_tiles++;
                total_elements += batch[i].actual_k * batch[i].actual_n;
                int padding = TILE_K * TILE_N - batch[i].actual_k * batch[i].actual_n;
                padded_elements += padding;
            }
        }

        cout << "\n统计:\n";
        cout << "  真实tile: " << real_tiles << "\n";
        cout << "  填充tile: " << padding_tiles << "\n";
        cout << "  有效元素: " << total_elements << "\n";
        cout << "  填充元素: " << padded_elements << "\n";
        cout << "  填充率: " << (100.0f * padded_elements / (TENSOR_CORE_TILES * TILE_K * TILE_N))
             << "%\n";
    }
};

// ============================================================================
// 测试用例
// ============================================================================
void test_case_1() {
    cout << "=================================================================\n";
    cout << "测试1：单个不规则矩阵\n";
    cout << "=================================================================\n";

    // 不规则维度
    int K = 97;   // 97 ÷ 16 = 6 余 1
    int N = 137;  // 137 ÷ 16 = 8 余 9

    vector<float> data(K * N);
    MatrixDescriptor mat(0, K, N, data.data());

    TensorCoreTileScheduler scheduler;
    auto batch = scheduler.schedule_single_matrix(mat);
    scheduler.print_schedule(batch);
}

void test_case_2() {
    cout << "\n=================================================================\n";
    cout << "测试2：多个小矩阵组合\n";
    cout << "=================================================================\n";

    // 3个小矩阵，每个都不足16个tile
    vector<MatrixDescriptor> matrices;
    vector<vector<float>> data_storage(3);

    // 矩阵1: 33×48 → (2×3=6 tiles)
    data_storage[0].resize(33 * 48);
    matrices.emplace_back(0, 33, 48, data_storage[0].data());

    // 矩阵2: 50×30 → (4×2=8 tiles)
    data_storage[1].resize(50 * 30);
    matrices.emplace_back(1, 50, 30, data_storage[1].data());

    // 矩阵3: 20×25 → (2×2=4 tiles)
    data_storage[2].resize(20 * 25);
    matrices.emplace_back(2, 20, 25, data_storage[2].data());

    // 总共 6+8+4=18 tiles，选16个

    TensorCoreTileScheduler scheduler;

    auto batch1 = scheduler.schedule_multi_matrix(matrices);
    scheduler.print_schedule(batch1);

    auto batch2 = scheduler.schedule_optimized(matrices);
    scheduler.print_schedule(batch2);
}

void test_case_3() {
    cout << "\n=================================================================\n";
    cout << "测试3：极端不规则（质数维度）\n";
    cout << "=================================================================\n";

    // 质数维度
    int K = 67;   // 质数
    int N = 83;   // 质数

    vector<float> data(K * N);
    MatrixDescriptor mat(0, K, N, data.data());

    TensorCoreTileScheduler scheduler;
    auto batch = scheduler.schedule_single_matrix(mat);
    scheduler.print_schedule(batch);
}

void test_case_4() {
    cout << "\n=================================================================\n";
    cout << "测试4：混合规则与不规则矩阵\n";
    cout << "=================================================================\n";

    vector<MatrixDescriptor> matrices;
    vector<vector<float>> data_storage(4);

    // 矩阵1: 32×32 → 完美规则 (2×2=4 tiles)
    data_storage[0].resize(32 * 32);
    matrices.emplace_back(0, 32, 32, data_storage[0].data());

    // 矩阵2: 97×16 → K不规则 (7×1=7 tiles)
    data_storage[1].resize(97 * 16);
    matrices.emplace_back(1, 97, 16, data_storage[1].data());

    // 矩阵3: 16×73 → N不规则 (1×5=5 tiles)
    data_storage[2].resize(16 * 73);
    matrices.emplace_back(2, 16, 73, data_storage[2].data());

    // 矩阵4: 23×29 → 都不规则 (2×2=4 tiles)
    data_storage[3].resize(23 * 29);
    matrices.emplace_back(3, 23, 29, data_storage[3].data());

    // 总共 4+7+5+4=20 tiles

    TensorCoreTileScheduler scheduler;
    auto batch = scheduler.schedule_optimized(matrices);
    scheduler.print_schedule(batch);
}

// ============================================================================
// 实际计算模拟（Tensor Core视角）
// ============================================================================
void simulate_tensor_core_execution(const vector<TileDescriptor>& batch) {
    cout << "\n=================================================================\n";
    cout << "Tensor Core执行模拟\n";
    cout << "=================================================================\n";

    cout << "执行顺序:\n";
    for (size_t i = 0; i < batch.size(); i++) {
        cout << "  [Cycle " << i << "] ";
        if (batch[i].is_padding) {
            cout << "NOP (padding tile, 跳过计算)\n";
        } else {
            cout << "计算 M" << batch[i].matrix_id
                 << " 的 tile@(" << batch[i].k_offset << "," << batch[i].n_offset << ")";

            if (batch[i].actual_k < TILE_K || batch[i].actual_n < TILE_N) {
                cout << " [不完整: " << batch[i].actual_k << "×" << batch[i].actual_n << "]";
            }
            cout << "\n";
        }
    }

    cout << "\n性能分析:\n";
    int active_tiles = 0;
    for (const auto& tile : batch) {
        if (!tile.is_padding) active_tiles++;
    }

    float utilization = 100.0f * active_tiles / TENSOR_CORE_TILES;
    cout << "  Tensor Core利用率: " << utilization << "%\n";
    cout << "  空闲cycle: " << (TENSOR_CORE_TILES - active_tiles) << "\n";
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    cout << "=================================================================\n";
    cout << "Tensor Core Tile调度器\n";
    cout << "=================================================================\n";
    cout << "硬件约束:\n";
    cout << "  - 固定需要 " << TENSOR_CORE_TILES << " 个tile\n";
    cout << "  - 每个tile大小: " << TILE_K << "×" << TILE_N << "\n";
    cout << "  - 支持不规则输入矩阵\n\n";

    test_case_1();
    test_case_2();
    test_case_3();
    test_case_4();

    // 模拟一个调度结果的实际执行
    cout << "\n=================================================================\n";
    cout << "执行模拟示例\n";
    cout << "=================================================================\n";

    vector<float> data(97 * 137);
    MatrixDescriptor mat(0, 97, 137, data.data());
    TensorCoreTileScheduler scheduler;
    auto batch = scheduler.schedule_single_matrix(mat);
    simulate_tensor_core_execution(batch);

    cout << "\n=================================================================\n";
    cout << "总结与建议\n";
    cout << "=================================================================\n";
    cout << "1. 单矩阵场景：\n";
    cout << "   - 如果tile数≥16：选择前16个\n";
    cout << "   - 如果tile数<16：填充虚拟tile（NOP）\n\n";

    cout << "2. 多矩阵场景（Batching）：\n";
    cout << "   - 轮询收集各矩阵的tile\n";
    cout << "   - 优先选择完整tile（减少padding）\n";
    cout << "   - 适合小矩阵批量处理\n\n";

    cout << "3. 优化策略：\n";
    cout << "   - 按padding比例排序tile\n";
    cout << "   - 完整tile优先（16×16）\n";
    cout << "   - 不完整tile次之\n";
    cout << "   - 最小化总padding开销\n\n";

    cout << "4. 硬件考虑：\n";
    cout << "   - Padding tile可以用NOP跳过\n";
    cout << "   - 或在硬件层面实现predication\n";
    cout << "   - 调度器应在编译时确定\n";
    cout << "   - 动态调度开销太大（不推荐）\n";

    return 0;
}
