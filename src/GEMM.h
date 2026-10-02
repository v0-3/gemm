#pragma once

#include <algorithm>
#include <cstddef>

#if defined(__AVX__)
#include <immintrin.h>
#endif

inline constexpr int TILE_SIZE = 4;
inline constexpr int BLOCK_SIZE = 8;  // 256 bits / 32 bits per float = 8 floats

// Each kernel computes C = A * B for non-overlapping row-major matrices.
#if defined(__APPLE__) && defined(__aarch64__)
// NEON for tiny matrices, Accelerate for CPU GEMM, Metal for large products.
void apple_silicon(const float *A, const float *B, float *C, int M, int N, int K);
#endif

inline void naive(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        const std::size_t a_row = static_cast<std::size_t>(i) * K;
        const std::size_t c_row = static_cast<std::size_t>(i) * N;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[a_row + k] * B[static_cast<std::size_t>(k) * N + j];
            }
            C[c_row + j] = sum;
        }
    }
}

inline void looporder(const float *A, const float *B, float *C, int M, int N, int K) {
    std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0f);
    for (int i = 0; i < M; i++) {
        const std::size_t a_row = static_cast<std::size_t>(i) * K;
        const std::size_t c_row = static_cast<std::size_t>(i) * N;
        for (int k = 0; k < K; k++) {
            const std::size_t b_row = static_cast<std::size_t>(k) * N;
            for (int j = 0; j < N; j++) {
                C[c_row + j] += A[a_row + k] * B[b_row + j];
            }
        }
    }
}

inline void tiling(const float *A, const float *B, float *C, int M, int N, int K) {
    std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0f);
    for (int inner_tile = 0; inner_tile < K;) {
        const int inner_tile_end = inner_tile + std::min(TILE_SIZE, K - inner_tile);
        for (int i = 0; i < M; i++) {
            const std::size_t a_row = static_cast<std::size_t>(i) * K;
            const std::size_t c_row = static_cast<std::size_t>(i) * N;
            for (int k = inner_tile; k < inner_tile_end; k++) {
                const std::size_t b_row = static_cast<std::size_t>(k) * N;
                for (int j = 0; j < N; j++) {
                    C[c_row + j] += A[a_row + k] * B[b_row + j];
                }
            }
        }
        inner_tile = inner_tile_end;
    }
}

#if defined(__AVX__)
inline void AVX(const float *A, const float *B, float *C, int M, int N, int K) {
    std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0f);
    const int vector_end = N - N % BLOCK_SIZE;
    for (int i = 0; i < M; i++) {
        const std::size_t a_row = static_cast<std::size_t>(i) * K;
        const std::size_t c_row = static_cast<std::size_t>(i) * N;
        for (int k = 0; k < K; k++) {
            const std::size_t b_row = static_cast<std::size_t>(k) * N;
            const __m256 a0 = _mm256_set1_ps(A[a_row + k]);
            int j = 0;
            for (; j < vector_end; j += BLOCK_SIZE) {
                const __m256 b0 = _mm256_loadu_ps(B + b_row + j);
                __m256 c0 = _mm256_loadu_ps(C + c_row + j);
#if defined(__FMA__)
                c0 = _mm256_fmadd_ps(a0, b0, c0);
#else
                c0 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c0);
#endif
                _mm256_storeu_ps(C + c_row + j, c0);
            }
            for (; j < N; j++) {
                C[c_row + j] += A[a_row + k] * B[b_row + j];
            }
        }
    }
}
#endif
