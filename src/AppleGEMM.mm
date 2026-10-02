#include "GEMM.h"

#if defined(__APPLE__) && defined(__aarch64__)
#include <AvailabilityMacros.h>
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 130300 && !defined(ACCELERATE_NEW_LAPACK)
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstring>

namespace {
// Keep four rows in registers, reusing each B vector without packing or allocation.
void small_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    int i = 0;
    for (; i + 3 < M; i += 4) {
        int j = 0;
        for (; j + 3 < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f), c1 = c0, c2 = c0, c3 = c0;
            for (int k = 0; k < K; ++k) {
                const float32x4_t b = vld1q_f32(B + k * N + j);
                c0 = vfmaq_n_f32(c0, b, A[i * K + k]);
                c1 = vfmaq_n_f32(c1, b, A[(i + 1) * K + k]);
                c2 = vfmaq_n_f32(c2, b, A[(i + 2) * K + k]);
                c3 = vfmaq_n_f32(c3, b, A[(i + 3) * K + k]);
            }
            vst1q_f32(C + i * N + j, c0);
            vst1q_f32(C + (i + 1) * N + j, c1);
            vst1q_f32(C + (i + 2) * N + j, c2);
            vst1q_f32(C + (i + 3) * N + j, c3);
        }
        for (; j < N; ++j) {
            for (int r = 0; r < 4; ++r) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[(i + r) * K + k] * B[k * N + j];
                }
                C[(i + r) * N + j] = sum;
            }
        }
    }
    for (; i < M; ++i) {
        int j = 0;
        for (; j + 3 < N; j += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int k = 0; k < K; ++k) {
                sum = vfmaq_n_f32(sum, vld1q_f32(B + k * N + j), A[i * K + k]);
            }
            vst1q_f32(C + i * N + j, sum);
        }
        for (; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

MPSMatrix *metal_matrix(id<MTLDevice> device, const float *data, int rows, int columns) {
    const std::size_t row_bytes = static_cast<std::size_t>(columns) * sizeof(float);
    const std::size_t stride = [MPSMatrixDescriptor rowBytesForColumns:columns dataType:MPSDataTypeFloat32];
    if (static_cast<std::size_t>(rows) > device.maxBufferLength / stride) {
        return nil;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:static_cast<std::size_t>(rows) * stride
                                             options:MTLResourceStorageModeShared];
    if (!buffer) {
        return nil;
    }
    if (data) {
        if (stride == row_bytes) {
            std::memcpy(buffer.contents, data, static_cast<std::size_t>(rows) * row_bytes);
        } else {
            for (int i = 0; i < rows; ++i) {
                std::memcpy(static_cast<char *>(buffer.contents) + i * stride,
                            data + static_cast<std::size_t>(i) * columns, row_bytes);
            }
        }
    }
    MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                          columns:columns
                                                                         rowBytes:stride
                                                                         dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
}

bool metal_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    @autoreleasepool {
        // Only initialize Metal when a sufficiently large product is requested.
        // Each call owns its buffers and operation, so concurrent calls share no matrix state.
        static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        static id<MTLCommandQueue> queue = device && MPSSupportsMTLDevice(device) ? [device newCommandQueue] : nil;
        if (!queue) {
            return false;
        }
        MPSMatrix *a = metal_matrix(device, A, M, K);
        MPSMatrix *b = metal_matrix(device, B, K, N);
        MPSMatrix *c = metal_matrix(device, nullptr, M, N);
        if (!a || !b || !c) {
            return false;
        }
        MPSMatrixMultiplication *operation = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                            transposeLeft:NO
                                                                           transposeRight:NO
                                                                               resultRows:M
                                                                            resultColumns:N
                                                                          interiorColumns:K
                                                                                    alpha:1.0
                                                                                     beta:0.0];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (!operation || !command) {
            return false;
        }
        [operation encodeToCommandBuffer:command leftMatrix:a rightMatrix:b resultMatrix:c];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return false;
        }
        const std::size_t row_bytes = static_cast<std::size_t>(N) * sizeof(float);
        if (c.rowBytes == row_bytes) {
            std::memcpy(C, c.data.contents, static_cast<std::size_t>(M) * row_bytes);
        } else {
            for (int i = 0; i < M; ++i) {
                std::memcpy(C + static_cast<std::size_t>(i) * N,
                            static_cast<const char *>(c.data.contents) + i * c.rowBytes, row_bytes);
            }
        }
        return true;
    }
}
}  // namespace

void apple_silicon(const float *A, const float *B, float *C, int M, int N, int K) {
    if (M <= 0 || N <= 0) {
        return;
    }
    if (K == 0) {
        std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0f);
        return;
    }
    // Measured on M3 Pro: avoid BLAS overhead below 16^3 and GPU startup/copy
    // costs below 5120^3. Keep skinny products on the CPU even at large volumes.
    if (M <= 16 && N <= 16 && K <= 16) {
        small_gemm(A, B, C, M, N, K);
        return;
    }
    if (M >= 512 && N >= 512 && K >= 512 &&
        static_cast<double>(M) * N * K >= 5120.0 * 5120 * 5120 &&
        metal_gemm(A, B, C, M, N, K)) {
        return;
    }
    // beta=0 overwrites C without an extra clearing pass. Accelerate manages
    // CPU dispatch and threading; leave the caller's threading policy intact.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                1.0f, A, K, B, N, 0.0f, C, N);
}
#endif
