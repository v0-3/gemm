#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "GEMM.h"

namespace {
using Kernel = void (*)(const float *, const float *, float *, int, int, int);
using Clock = std::chrono::steady_clock;

void benchmark(Kernel kernel, const float *A, const float *B, float *C, int M, int N, int K) {
    const auto start = Clock::now();
    kernel(A, B, C, M, N, K);
    const double first = std::chrono::duration<double>(Clock::now() - start).count();
    // Batch tiny calls so clock resolution does not dominate. Report the median
    // of five warmed samples; all allocation/copy/synchronization inside the
    // method is timed, while input generation and correctness checks are not.
    const int repeats = static_cast<int>(std::clamp(0.005 / std::max(first, 1.0e-9), 1.0, 100000.0));
    std::array<double, 5> samples;
    for (double &sample : samples) {
        const auto begin = Clock::now();
        for (int run = 0; run < repeats; ++run) kernel(A, B, C, M, N, K);
        sample = std::chrono::duration<double, std::milli>(Clock::now() - begin).count() / repeats;
    }
    for (int point = 0; point < 32; ++point) {
        const int i = static_cast<int>(static_cast<std::size_t>(point) * (M - 1) / 31);
        const int j = static_cast<int>(static_cast<std::size_t>(31 - point) * (N - 1) / 31);
        double expected = 0.0;
        for (int k = 0; k < K; ++k) {
            expected += static_cast<double>(A[static_cast<std::size_t>(i) * K + k]) * B[static_cast<std::size_t>(k) * N + j];
        }
        const float actual = C[static_cast<std::size_t>(i) * N + j];
        if (!std::isfinite(actual) || std::fabs(actual - expected) > 1.0e-3 + 1.0e-5 * std::fabs(expected)) {
            throw std::runtime_error("Benchmark product failed verification");
        }
    }
    std::sort(samples.begin(), samples.end());
    std::printf("%12.6f %12.6f %12.2f\n", first * 1000.0, samples[2], (2.0 * M * N * K) / (samples[2] * 1.0e6));
}
}  // namespace

int main() try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::puts("Times include each method's setup, copies, and completion. Large scalar cases are skipped.");
    std::printf("%6s %6s %6s %-15s %12s %12s %12s\n", "M", "N", "K", "Method", "First (ms)", "Median (ms)", "GFLOPS");
    for (auto [M, N, K] : {std::array{4, 4, 4}, {8, 8, 8}, {16, 16, 16}, {32, 32, 32},
                           {64, 64, 64}, {128, 128, 128}, {256, 256, 256}, {512, 512, 512},
                           {1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}, {5120, 5120, 5120},
                           {1, 1024, 1024}, {1024, 1, 1024}, {127, 257, 65}, {512, 1024, 256},
                           {5121, 5119, 5121}}) {
        std::vector<float> A(static_cast<std::size_t>(M) * K), B(static_cast<std::size_t>(K) * N);
        std::vector<float> C(static_cast<std::size_t>(M) * N, std::numeric_limits<float>::quiet_NaN());
        for (std::size_t i = 0; i < A.size(); ++i) A[i] = (static_cast<int>(i % 31) - 15) / 32.0f;
        for (std::size_t i = 0; i < B.size(); ++i) B[i] = (static_cast<int>(i % 41) - 20) / 64.0f;
        const auto run = [&](const char *name, Kernel kernel) {
            std::printf("%6d %6d %6d %-15s ", M, N, K, name);
            benchmark(kernel, A.data(), B.data(), C.data(), M, N, K);
        };
        if (static_cast<double>(M) * N * K <= 128.0 * 128 * 128) run("Naive", naive);
        if (static_cast<double>(M) * N * K <= 1024.0 * 1024 * 1024) {
            run("Loop order", looporder);
            run("Tiling", tiling);
#if defined(__AVX__)
            run("AVX", AVX);
#endif
        }
#if defined(__APPLE__) && defined(__aarch64__)
        run("Apple Silicon", apple_silicon);
#endif
    }
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
