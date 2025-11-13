/**
 * 不规则维度的二维矩阵Tiling实现示例
 *
 * 问题：当矩阵维度不能被Tile大小整除时如何处理？
 *
 * 编译：g++ -O3 -std=c++17 irregular_tiling_example.cpp -o irregular_tiling
 * 运行：./irregular_tiling
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace std;

// ============================================================================
// 方案1：动态边界检查
// ============================================================================
void matmul_bounds_check(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 32;

    // 初始化C为0
    memset(C, 0, M * N * sizeof(float));

    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            for (int k0 = 0; k0 < K; k0 += TILE_K) {

                // 动态计算边界
                int i_end = min(i0 + TILE_M, M);
                int j_end = min(j0 + TILE_N, N);
                int k_end = min(k0 + TILE_K, K);

                // 内层循环
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
// 方案2：Padding填充（XLA策略）
// ============================================================================
void matmul_padding(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 32;

    // 计算填充后的大小
    auto round_up = [](int x, int multiple) {
        return ((x + multiple - 1) / multiple) * multiple;
    };

    int M_padded = round_up(M, TILE_M);
    int N_padded = round_up(N, TILE_N);
    int K_padded = round_up(K, TILE_K);

    // 分配填充矩阵（自动初始化为0）
    vector<float> A_padded(M_padded * K_padded, 0.0f);
    vector<float> B_padded(K_padded * N_padded, 0.0f);
    vector<float> C_padded(M_padded * N_padded, 0.0f);

    // 拷贝原始数据
    for (int i = 0; i < M; i++) {
        memcpy(&A_padded[i * K_padded], &A[i * K], K * sizeof(float));
    }
    for (int i = 0; i < K; i++) {
        memcpy(&B_padded[i * N_padded], &B[i * N], N * sizeof(float));
    }

    // 规则Tiling（无边界检查）
    for (int i0 = 0; i0 < M_padded; i0 += TILE_M) {
        for (int j0 = 0; j0 < N_padded; j0 += TILE_N) {
            for (int k0 = 0; k0 < K_padded; k0 += TILE_K) {

                // 完整的Tile，无边界检查
                for (int i = i0; i < i0 + TILE_M; i++) {
                    for (int j = j0; j < j0 + TILE_N; j++) {
                        float sum = 0.0f;
                        for (int k = k0; k < k0 + TILE_K; k++) {
                            sum += A_padded[i*K_padded + k] *
                                   B_padded[k*N_padded + j];
                        }
                        C_padded[i*N_padded + j] += sum;
                    }
                }
            }
        }
    }

    // 拷贝结果（去除padding）
    for (int i = 0; i < M; i++) {
        memcpy(&C[i * N], &C_padded[i * N_padded], N * sizeof(float));
    }
}

// ============================================================================
// 方案3：Peeling（边界剥离）- 最优性能
// ============================================================================
void matmul_peeling(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 32;

    // 初始化C为0
    memset(C, 0, M * N * sizeof(float));

    // 计算规则区域
    int M_regular = (M / TILE_M) * TILE_M;
    int N_regular = (N / TILE_N) * TILE_N;
    int K_regular = (K / TILE_K) * TILE_K;

    // 1. 处理规则区域（无边界检查）
    for (int i0 = 0; i0 < M_regular; i0 += TILE_M) {
        for (int j0 = 0; j0 < N_regular; j0 += TILE_N) {
            for (int k0 = 0; k0 < K_regular; k0 += TILE_K) {

                // 完整32×32×32 Tile
                for (int i = i0; i < i0 + TILE_M; i++) {
                    for (int j = j0; j < j0 + TILE_N; j++) {
                        float sum = 0.0f;
                        for (int k = k0; k < k0 + TILE_K; k++) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] += sum;
                    }
                }
            }
        }
    }

    // 2. 处理M维度余数（右边界）- 只处理规则N区域
    if (M_regular < M) {
        for (int j0 = 0; j0 < N_regular; j0 += TILE_N) {
            for (int k0 = 0; k0 < K_regular; k0 += TILE_K) {
                for (int i = M_regular; i < M; i++) {
                    for (int j = j0; j < j0 + TILE_N; j++) {
                        float sum = 0.0f;
                        for (int k = k0; k < k0 + TILE_K; k++) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] += sum;
                    }
                }
            }
        }
    }

    // 3. 处理N维度余数（下边界）- 只处理规则M区域
    if (N_regular < N) {
        for (int i0 = 0; i0 < M_regular; i0 += TILE_M) {
            for (int k0 = 0; k0 < K_regular; k0 += TILE_K) {
                for (int i = i0; i < i0 + TILE_M; i++) {
                    for (int j = N_regular; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = k0; k < k0 + TILE_K; k++) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] += sum;
                    }
                }
            }
        }
    }

    // 4. 处理右下角（M和N都有余数的区域）
    if (M_regular < M && N_regular < N) {
        for (int k0 = 0; k0 < K_regular; k0 += TILE_K) {
            for (int i = M_regular; i < M; i++) {
                for (int j = N_regular; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = k0; k < k0 + TILE_K; k++) {
                        sum += A[i*K + k] * B[k*N + j];
                    }
                    C[i*N + j] += sum;
                }
            }
        }
    }

    // 5. 处理K维度余数（影响所有M×N元素）
    if (K_regular < K) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = K_regular; k < K; k++) {
                    sum += A[i*K + k] * B[k*N + j];
                }
                C[i*N + j] += sum;
            }
        }
    }
}

// ============================================================================
// 朴素实现（用于验证正确性）
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
            if (mismatches < 5) {  // 只打印前5个错误
                int row = i / N;
                int col = i % N;
                cout << "  Mismatch at [" << row << "," << col << "]: "
                     << C1[i] << " vs " << C2[i] << " (diff=" << diff << ")\n";
            }
            mismatches++;
        }
    }
    if (mismatches > 5) {
        cout << "  ... 共 " << mismatches << " 个不匹配\n";
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

// ============================================================================
// 主函数
// ============================================================================
int main() {
    // 测试不规则维度
    const int M = 100;  // 不能被32整除（余数4）
    const int N = 80;   // 不能被32整除（余数16）
    const int K = 250;  // 不能被32整除（余数26）

    cout << "=================================================\n";
    cout << "不规则维度矩阵Tiling测试\n";
    cout << "=================================================\n";
    cout << "矩阵维度: C[" << M << "," << N << "] = A[" << M << "," << K
         << "] × B[" << K << "," << N << "]\n";
    cout << "Tile大小: 32×32×32\n";
    cout << "余数: M余数=" << (M%32) << ", N余数=" << (N%32)
         << ", K余数=" << (K%32) << "\n\n";

    // 分配内存
    vector<float> A(M * K);
    vector<float> B(K * N);
    vector<float> C_naive(M * N);
    vector<float> C_bounds(M * N);
    vector<float> C_padding(M * N);
    vector<float> C_peeling(M * N);

    // 初始化随机数据
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 10.0f;

    // 计算参考结果
    cout << "计算参考结果（朴素方法）...\n";
    matmul_naive(A.data(), B.data(), C_naive.data(), M, N, K);

    // 测试方案1：边界检查
    cout << "\n[方案1] 动态边界检查\n";
    matmul_bounds_check(A.data(), B.data(), C_bounds.data(), M, N, K);
    bool correct1 = verify_result(C_naive.data(), C_bounds.data(), M, N);
    cout << "  正确性: " << (correct1 ? "✓ 通过" : "✗ 失败") << "\n";
    double time1 = benchmark([&]() {
        matmul_bounds_check(A.data(), B.data(), C_bounds.data(), M, N, K);
    });
    cout << "  性能: " << time1 * 1000 << " ms\n";

    // 测试方案2：Padding
    cout << "\n[方案2] Padding填充（XLA策略）\n";
    matmul_padding(A.data(), B.data(), C_padding.data(), M, N, K);
    bool correct2 = verify_result(C_naive.data(), C_padding.data(), M, N);
    cout << "  正确性: " << (correct2 ? "✓ 通过" : "✗ 失败") << "\n";
    double time2 = benchmark([&]() {
        matmul_padding(A.data(), B.data(), C_padding.data(), M, N, K);
    });
    cout << "  性能: " << time2 * 1000 << " ms\n";
    cout << "  加速比: " << time1 / time2 << "x\n";

    // 测试方案3：Peeling
    cout << "\n[方案3] Peeling边界剥离\n";
    matmul_peeling(A.data(), B.data(), C_peeling.data(), M, N, K);
    bool correct3 = verify_result(C_naive.data(), C_peeling.data(), M, N);
    cout << "  正确性: " << (correct3 ? "✓ 通过" : "✗ 失败") << "\n";
    double time3 = benchmark([&]() {
        matmul_peeling(A.data(), B.data(), C_peeling.data(), M, N, K);
    });
    cout << "  性能: " << time3 * 1000 << " ms\n";
    cout << "  加速比: " << time1 / time3 << "x\n";

    // 总结
    cout << "\n=================================================\n";
    cout << "性能总结\n";
    cout << "=================================================\n";
    cout << "方案1（边界检查）: " << time1 * 1000 << " ms (基准)\n";
    cout << "方案2（Padding）:  " << time2 * 1000 << " ms ("
         << (time1/time2) << "x)\n";
    cout << "方案3（Peeling）:  " << time3 * 1000 << " ms ("
         << (time1/time3) << "x)\n";

    cout << "\n推荐方案:\n";
    cout << "  - CPU优化: 使用Peeling（最低内存，最优性能）\n";
    cout << "  - GPU/XLA: 使用Padding（便于向量化，工业标准）\n";

    return 0;
}
