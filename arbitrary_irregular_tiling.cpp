/**
 * 任意不规则维度的Tiling实现示例
 *
 * 场景：极端不规则的情况
 * - 矩阵维度：质数或奇怪的数字 (M=97, N=137, K=223)
 * - Tile大小：非2的幂 (tile_m=13, tile_n=17, tile_k=19)
 * - 所有维度都有复杂的余数
 *
 * 编译：g++ -O3 -std=c++17 arbitrary_irregular_tiling.cpp -o arbitrary_irregular
 * 运行：./arbitrary_irregular
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace std;

// 极端不规则的维度
const int M = 97;   // 质数！
const int N = 137;  // 质数！
const int K = 223;  // 质数！

// 非常规的Tile大小
const int TILE_M = 13;  // 质数
const int TILE_N = 17;  // 质数
const int TILE_K = 19;  // 质数

// ============================================================================
// 方案1：动态边界检查（适用于任意不规则情况）
// ============================================================================
void matmul_arbitrary_bounds_check(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int tile_m, int tile_n, int tile_k) {

    memset(C, 0, M * N * sizeof(float));

    // 计算tile数量
    int num_tiles_m = (M + tile_m - 1) / tile_m;
    int num_tiles_n = (N + tile_n - 1) / tile_n;
    int num_tiles_k = (K + tile_k - 1) / tile_k;

    cout << "  Tile配置:\n";
    cout << "    M维: " << num_tiles_m << " tiles ("
         << (num_tiles_m-1) << "个完整 + 1个余数=" << (M % tile_m ? M % tile_m : tile_m) << ")\n";
    cout << "    N维: " << num_tiles_n << " tiles ("
         << (num_tiles_n-1) << "个完整 + 1个余数=" << (N % tile_n ? N % tile_n : tile_n) << ")\n";
    cout << "    K维: " << num_tiles_k << " tiles ("
         << (num_tiles_k-1) << "个完整 + 1个余数=" << (K % tile_k ? K % tile_k : tile_k) << ")\n";

    for (int i0 = 0; i0 < M; i0 += tile_m) {
        for (int j0 = 0; j0 < N; j0 += tile_n) {
            for (int k0 = 0; k0 < K; k0 += tile_k) {

                // 动态计算每个tile的实际边界
                int i_end = min(i0 + tile_m, M);
                int j_end = min(j0 + tile_n, N);
                int k_end = min(k0 + tile_k, K);

                // 实际tile大小可能不同
                int actual_tile_m = i_end - i0;
                int actual_tile_n = j_end - j0;
                int actual_tile_k = k_end - k0;

                // 处理当前tile
                for (int i = i0; i < i_end; i++) {
                    for (int j = j0; j < j_end; j++) {
                        float sum = 0.0f;
                        for (int k = k0; k < k_end; k++) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] += sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// 方案2：智能Padding（处理任意不规则）
// ============================================================================
void matmul_arbitrary_padding(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int tile_m, int tile_n, int tile_k) {

    // 计算padding后的大小
    auto round_up = [](int x, int multiple) {
        return ((x + multiple - 1) / multiple) * multiple;
    };

    int M_padded = round_up(M, tile_m);
    int N_padded = round_up(N, tile_n);
    int K_padded = round_up(K, tile_k);

    cout << "  Padding信息:\n";
    cout << "    M: " << M << " → " << M_padded << " (padding " << (M_padded - M) << " 行)\n";
    cout << "    N: " << N << " → " << N_padded << " (padding " << (N_padded - N) << " 列)\n";
    cout << "    K: " << K << " → " << K_padded << " (padding " << (K_padded - K) << ")\n";

    float padding_overhead = (float)(M_padded * K_padded + K_padded * N_padded + M_padded * N_padded) /
                            (M * K + K * N + M * N) - 1.0f;
    cout << "    内存开销: +" << (padding_overhead * 100) << "%\n";

    // 分配并初始化padding矩阵
    vector<float> A_padded(M_padded * K_padded, 0.0f);
    vector<float> B_padded(K_padded * N_padded, 0.0f);
    vector<float> C_padded(M_padded * N_padded, 0.0f);

    // 拷贝数据
    for (int i = 0; i < M; i++) {
        memcpy(&A_padded[i * K_padded], &A[i * K], K * sizeof(float));
    }
    for (int i = 0; i < K; i++) {
        memcpy(&B_padded[i * N_padded], &B[i * N], N * sizeof(float));
    }

    // 规则Tiling（无边界检查）
    for (int i0 = 0; i0 < M_padded; i0 += tile_m) {
        for (int j0 = 0; j0 < N_padded; j0 += tile_n) {
            for (int k0 = 0; k0 < K_padded; k0 += tile_k) {

                // 完整规则的tile
                for (int i = i0; i < i0 + tile_m; i++) {
                    for (int j = j0; j < j0 + tile_n; j++) {
                        float sum = 0.0f;
                        for (int k = k0; k < k0 + tile_k; k++) {
                            sum += A_padded[i*K_padded + k] * B_padded[k*N_padded + j];
                        }
                        C_padded[i*N_padded + j] += sum;
                    }
                }
            }
        }
    }

    // 拷贝结果回原始矩阵
    for (int i = 0; i < M; i++) {
        memcpy(&C[i * N], &C_padded[i * N_padded], N * sizeof(float));
    }
}

// ============================================================================
// 方案3：Adaptive Tiling（根据余数动态调整）
// ============================================================================
void matmul_adaptive_tiling(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int tile_m, int tile_n, int tile_k) {

    memset(C, 0, M * N * sizeof(float));

    // 计算余数
    int remainder_m = M % tile_m;
    int remainder_n = N % tile_n;
    int remainder_k = K % tile_k;

    cout << "  自适应策略:\n";
    cout << "    M维余数: " << remainder_m << " -> ";
    if (remainder_m > tile_m / 2) {
        cout << "调整为两个tile: " << (tile_m + remainder_m) / 2 << " + "
             << (tile_m + remainder_m) / 2 << "\n";
    } else {
        cout << "保持原样 (小余数)\n";
    }

    // 策略：如果余数太小，将最后两个tile合并重新分配
    auto adaptive_tile = [](int dim, int base_tile) -> vector<int> {
        vector<int> tiles;
        int pos = 0;
        int remainder = dim % base_tile;

        if (remainder > 0 && remainder < base_tile / 3) {
            // 余数太小：将最后两个tile重新分配
            int num_full = dim / base_tile - 1;
            for (int i = 0; i < num_full; i++) {
                tiles.push_back(base_tile);
            }
            // 最后两个tile平均分配
            int last_two = base_tile + remainder;
            tiles.push_back(last_two / 2);
            tiles.push_back(last_two - last_two / 2);
        } else {
            // 正常情况
            int num_full = dim / base_tile;
            for (int i = 0; i < num_full; i++) {
                tiles.push_back(base_tile);
            }
            if (remainder > 0) {
                tiles.push_back(remainder);
            }
        }
        return tiles;
    };

    auto tiles_m = adaptive_tile(M, tile_m);
    auto tiles_n = adaptive_tile(N, tile_n);
    auto tiles_k = adaptive_tile(K, tile_k);

    // 执行自适应tiling
    int i0 = 0;
    for (int tm : tiles_m) {
        int j0 = 0;
        for (int tn : tiles_n) {
            int k0 = 0;
            for (int tk : tiles_k) {

                // 处理当前自适应大小的tile
                for (int i = i0; i < i0 + tm; i++) {
                    for (int j = j0; j < j0 + tn; j++) {
                        float sum = 0.0f;
                        for (int k = k0; k < k0 + tk; k++) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] += sum;
                    }
                }

                k0 += tk;
            }
            j0 += tn;
        }
        i0 += tm;
    }
}

// ============================================================================
// 朴素实现（参考）
// ============================================================================
void matmul_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// ============================================================================
// 工具函数
// ============================================================================
bool verify_result(const float* C1, const float* C2, int M, int N, float epsilon = 1e-3) {
    int mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C1[i] - C2[i]);
        float rel_error = (C1[i] != 0) ? diff / fabs(C1[i]) : diff;
        if (diff > epsilon && rel_error > epsilon) {
            if (mismatches < 3) {
                int row = i / N;
                int col = i % N;
                cout << "    错误 [" << row << "," << col << "]: "
                     << C1[i] << " vs " << C2[i] << " (diff=" << diff << ")\n";
            }
            mismatches++;
        }
    }
    if (mismatches > 3) {
        cout << "    ... 共 " << mismatches << " 个不匹配\n";
    }
    return mismatches == 0;
}

template<typename Func>
double benchmark(Func f, int iterations = 10) {
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        f();
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    return diff.count() / iterations;
}

void print_tile_distribution(int dim, int tile_size, const string& name) {
    int num_tiles = (dim + tile_size - 1) / tile_size;
    int remainder = dim % tile_size;

    cout << "  " << name << " = " << dim << ", tile = " << tile_size << ":\n";
    cout << "    → " << num_tiles << " tiles";
    if (remainder > 0) {
        cout << " (" << (num_tiles - 1) << " × " << tile_size
             << " + 1 × " << remainder << ")";
    }
    cout << "\n";
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    cout << "=================================================================\n";
    cout << "任意不规则维度的Tiling测试\n";
    cout << "=================================================================\n";
    cout << "矩阵维度: C[" << M << "," << N << "] = A[" << M << "," << K
         << "] × B[" << K << "," << N << "]\n";
    cout << "Tile大小: " << TILE_M << "×" << TILE_N << "×" << TILE_K << "\n\n";

    // 显示tile分布
    print_tile_distribution(M, TILE_M, "M维度");
    print_tile_distribution(N, TILE_N, "N维度");
    print_tile_distribution(K, TILE_K, "K维度");

    cout << "\n不规则程度分析:\n";
    cout << "  M余数: " << (M % TILE_M) << " / " << TILE_M
         << " (" << (100.0 * (M % TILE_M) / TILE_M) << "%)\n";
    cout << "  N余数: " << (N % TILE_N) << " / " << TILE_N
         << " (" << (100.0 * (N % TILE_N) / TILE_N) << "%)\n";
    cout << "  K余数: " << (K % TILE_K) << " / " << TILE_K
         << " (" << (100.0 * (K % TILE_K) / TILE_K) << "%)\n";

    // 分配内存
    vector<float> A(M * K);
    vector<float> B(K * N);
    vector<float> C_naive(M * N);
    vector<float> C_bounds(M * N);
    vector<float> C_padding(M * N);
    vector<float> C_adaptive(M * N);

    // 初始化
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 10.0f;

    // 计算参考结果
    cout << "\n计算参考结果（朴素方法）...\n";
    matmul_naive(A.data(), B.data(), C_naive.data(), M, N, K);

    cout << "\n=================================================================\n";
    cout << "[方案1] 动态边界检查\n";
    cout << "=================================================================\n";
    matmul_arbitrary_bounds_check(A.data(), B.data(), C_bounds.data(),
                                   M, N, K, TILE_M, TILE_N, TILE_K);
    bool correct1 = verify_result(C_naive.data(), C_bounds.data(), M, N);
    cout << "  正确性: " << (correct1 ? "✓ 通过" : "✗ 失败") << "\n";
    double time1 = benchmark([&]() {
        matmul_arbitrary_bounds_check(A.data(), B.data(), C_bounds.data(),
                                      M, N, K, TILE_M, TILE_N, TILE_K);
    });
    cout << "  性能: " << (time1 * 1000) << " ms\n";

    cout << "\n=================================================================\n";
    cout << "[方案2] 智能Padding\n";
    cout << "=================================================================\n";
    matmul_arbitrary_padding(A.data(), B.data(), C_padding.data(),
                             M, N, K, TILE_M, TILE_N, TILE_K);
    bool correct2 = verify_result(C_naive.data(), C_padding.data(), M, N);
    cout << "  正确性: " << (correct2 ? "✓ 通过" : "✗ 失败") << "\n";
    double time2 = benchmark([&]() {
        matmul_arbitrary_padding(A.data(), B.data(), C_padding.data(),
                                 M, N, K, TILE_M, TILE_N, TILE_K);
    });
    cout << "  性能: " << (time2 * 1000) << " ms\n";
    cout << "  加速比: " << (time1 / time2) << "x\n";

    cout << "\n=================================================================\n";
    cout << "[方案3] 自适应Tiling\n";
    cout << "=================================================================\n";
    matmul_adaptive_tiling(A.data(), B.data(), C_adaptive.data(),
                          M, N, K, TILE_M, TILE_N, TILE_K);
    bool correct3 = verify_result(C_naive.data(), C_adaptive.data(), M, N);
    cout << "  正确性: " << (correct3 ? "✓ 通过" : "✗ 失败") << "\n";
    double time3 = benchmark([&]() {
        matmul_adaptive_tiling(A.data(), B.data(), C_adaptive.data(),
                              M, N, K, TILE_M, TILE_N, TILE_K);
    });
    cout << "  性能: " << (time3 * 1000) << " ms\n";
    cout << "  加速比: " << (time1 / time3) << "x\n";

    // 总结
    cout << "\n=================================================================\n";
    cout << "性能总结\n";
    cout << "=================================================================\n";
    cout << fixed << setprecision(4);
    cout << "方案1（边界检查）:  " << (time1 * 1000) << " ms (基准)\n";
    cout << "方案2（Padding）:   " << (time2 * 1000) << " ms ("
         << (time1/time2) << "x)\n";
    cout << "方案3（自适应）:     " << (time3 * 1000) << " ms ("
         << (time1/time3) << "x)\n";

    cout << "\n极端不规则情况的建议:\n";
    cout << "  - 余数很大 (>50%): 使用自适应Tiling，重新分配最后的tiles\n";
    cout << "  - 余数中等 (20-50%): 使用边界检查，开销可接受\n";
    cout << "  - 余数很小 (<20%): 使用Padding，内存开销小\n";
    cout << "  - GPU场景: 始终使用Padding，避免分支\n";

    return 0;
}
