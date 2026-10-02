#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "GEMM.h"
#include "Timer.h"
#include "Utils.h"

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function function, const std::string &message) {
    try {
        function();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

void test_kernels() {
    using Kernel = void (*)(const float *, const float *, float *, int, int, int);
    const Kernel kernels[] = {naive, looporder, tiling,
#if defined(__AVX__)
                              AVX,
#endif
#if defined(__APPLE__) && defined(__aarch64__)
                              apple_silicon,
#endif
    };
    for (int M : {1, 2, 5}) {
        for (int N : {1, 3, 7, 8, 9, 15, 16, 17}) {
            for (int K : {1, 3, 4, 5, 9}) {
                std::vector<float> A(M * K), B(K * N), expected(M * N);
                for (int i = 0; i < M * K; ++i) {
                    A[i] = (i % 7 - 3) * 0.25f;
                }
                for (int i = 0; i < K * N; ++i) {
                    B[i] = (i % 11 - 5) * 0.5f;
                }
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < K; ++k) {
                            sum += static_cast<double>(A[i * K + k]) * B[k * N + j];
                        }
                        expected[i * N + j] = static_cast<float>(sum);
                    }
                }
                for (Kernel kernel : kernels) {
                    std::vector<float> output(M * N, std::numeric_limits<float>::quiet_NaN());
                    for (int run = 0; run < 2; ++run) {
                        kernel(A.data(), B.data(), output.data(), M, N, K);
                        require(output == expected, "Incorrect product or output accumulation for " +
                                                        std::to_string(M) + "x" + std::to_string(N) + "x" +
                                                        std::to_string(K));
                    }
                }
            }
        }
    }
}

#if defined(__APPLE__) && defined(__aarch64__)
void test_apple_silicon() {
    std::mt19937 random(42);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    for (auto [M, N, K] : {std::array{1, 17, 33}, {33, 1, 17}, {16, 16, 16},
                           {17, 16, 16}, {16, 17, 16}, {16, 16, 17}, {31, 33, 65},
                           {65, 31, 33}, {127, 129, 63}, {129, 65, 257}}) {
        // Offset every matrix by one float to exercise unaligned input and output.
        std::vector<float> A(M * K + 2), B(K * N + 2), C(M * N + 2, 123456.0f);
        for (int run = 0; run < 2; ++run) {
            for (float &a : A) a = value(random);
            for (float &b : B) b = value(random);
            const auto original_A = A, original_B = B;
            std::fill(C.begin() + 1, C.end() - 1, std::numeric_limits<float>::quiet_NaN());
            apple_silicon(A.data() + 1, B.data() + 1, C.data() + 1, M, N, K);
            require(A == original_A && B == original_B, "Apple Silicon GEMM modified its inputs");
            require(C.front() == 123456.0f && C.back() == 123456.0f, "Apple Silicon GEMM wrote past C");
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    double expected = 0.0, magnitude = 0.0;
                    for (int k = 0; k < K; ++k) {
                        const double product = static_cast<double>(A[1 + i * K + k]) * B[1 + k * N + j];
                        expected += product;
                        magnitude += std::fabs(product);
                    }
                    const float actual = C[1 + i * N + j];
                    const double tolerance = 1.0e-6 + 8.0 * std::numeric_limits<float>::epsilon() * magnitude;
                    require(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance,
                            "Apple Silicon GEMM disagrees with double-precision reference");
                }
            }
        }
    }
    float output[] = {1.0f, 2.0f, 3.0f, 4.0f};
    apple_silicon(nullptr, nullptr, output, 0, 2, 3);
    apple_silicon(nullptr, nullptr, output, 2, 0, 3);
    require(output[0] == 1.0f && output[3] == 4.0f, "Empty product modified C");
    apple_silicon(nullptr, nullptr, output, 2, 2, 0);
    for (float element : output) require(element == 0.0f, "Zero inner dimension did not clear C");
}

void test_apple_silicon_large() {
    // Just below/at the GPU crossover, plus a rectangular GPU product with padded
    // rows. Dense dyadic inputs have an exact, independently computed reference.
    for (auto [M, N, K] : {std::array{5120, 5120, 5119}, {5120, 5120, 5120}, {5121, 5119, 5121}}) {
        std::vector<float> A(static_cast<std::size_t>(M) * K), B(static_cast<std::size_t>(K) * N);
        std::vector<float> C(static_cast<std::size_t>(M) * N + 2, 123456.0f);
        double inner_product = 0.0;
        for (int k = 0; k < K; ++k) {
            inner_product += (k % 11 - 5) * (k % 13 - 6);
        }
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                A[static_cast<std::size_t>(i) * K + k] = (i % 7 - 3) * (k % 11 - 5) / 64.0f;
            }
        }
        for (int run = 0; run < 2; ++run) {
            for (int k = 0; k < K; ++k) {
                for (int j = 0; j < N; ++j) {
                    B[static_cast<std::size_t>(k) * N + j] = (k % 13 - 6) * (j % 17 - 8 + run) / 128.0f;
                }
            }
            std::fill(C.begin() + 1, C.end() - 1, std::numeric_limits<float>::quiet_NaN());
            apple_silicon(A.data(), B.data(), C.data() + 1, M, N, K);
            require(C.front() == 123456.0f && C.back() == 123456.0f, "Large product wrote past C");
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    const float expected = static_cast<float>((i % 7 - 3) * (j % 17 - 8 + run) * inner_product / 8192.0);
                    require(C[1 + static_cast<std::size_t>(i) * N + j] == expected,
                            "Incorrect large Apple Silicon product or stale input/output");
                }
            }
        }
        if (M != N) {
            // General float inputs also check GPU roundoff and cancellation.
            // Sample a double reference to avoid a cubic scalar test at this size.
            std::mt19937 random(123);
            std::uniform_real_distribution<float> value(-1.0f, 1.0f);
            for (float &a : A) a = value(random);
            for (float &b : B) b = value(random);
            std::fill(C.begin() + 1, C.end() - 1, std::numeric_limits<float>::quiet_NaN());
            apple_silicon(A.data(), B.data(), C.data() + 1, M, N, K);
            require(C.front() == 123456.0f && C.back() == 123456.0f, "Random large product wrote past C");
            for (int point = 0; point < 64; ++point) {
                const int i = point * (M - 1) / 63;
                const int j = (63 - point) * (N - 1) / 63;
                double expected = 0.0, magnitude = 0.0;
                for (int k = 0; k < K; ++k) {
                    const double product = static_cast<double>(A[static_cast<std::size_t>(i) * K + k]) *
                                           B[static_cast<std::size_t>(k) * N + j];
                    expected += product;
                    magnitude += std::fabs(product);
                }
                const float actual = C[1 + static_cast<std::size_t>(i) * N + j];
                const double tolerance = 8.0 * std::numeric_limits<float>::epsilon() * magnitude;
                require(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance,
                        "GPU product disagrees with double-precision reference");
            }
        }
    }
}
#endif

void test_comparison() {
    const float expected[] = {1, 2, 3, 4, 5, 6};
    float actual[] = {1, 2, 3, 4, 5, 6};
    require(is_equal(expected, actual, 2, 3), "Equal wide matrices rejected");
    require(is_equal(expected, actual, 3, 2), "Equal tall matrices rejected");
    actual[5] = 7;
    require(!is_equal(expected, actual, 2, 3), "Wide matrix mismatch missed");
    require(!is_equal(expected, actual, 3, 2), "Tall matrix mismatch missed");

    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
        require(!is_equal(expected, &invalid, 1, 1), "Nonfinite result accepted");
        require(!is_equal(&invalid, expected, 1, 1), "Nonfinite reference accepted");
        require(!is_equal(&invalid, &invalid, 1, 1), "Matching nonfinite values accepted");
    }

    const float large_reference = -780.41016f;
    const float rounded_result = -780.41376f;
    const float wrong_result = -781.0f;
    require(is_equal(&large_reference, &rounded_result, 1, 1), "Float rounding tolerance too strict");
    require(!is_equal(&large_reference, &wrong_result, 1, 1), "Large mismatch accepted");
}

void test_arguments() {
    require(get_arg_value("1") == 1, "Positive dimension rejected");
    require(get_arg_value("2147483647") == 2147483647, "Maximum dimension rejected");
    for (const char *value : {"", "0", "-1", "1x", "1.5", "2147483648"}) {
        require_throws([&] { get_arg_value(value); }, "Invalid dimension accepted: " + std::string(value));
    }
}

void test_timer() {
    Timer timer;
    static_assert(!std::numeric_limits<decltype(timer.duration())>::is_integer);
    require_throws([&] { timer.stop(); }, "Timer stopped before starting");
    require_throws([&] { timer.duration(); }, "Timer returned duration before starting");
    for (int run = 0; run < 2; ++run) {
        timer.start();
        require_throws([&] { timer.duration(); }, "Running timer returned a completed duration");
        timer.stop();
        require(std::isfinite(timer.duration()) && timer.duration() >= 0.0, "Invalid elapsed time");
        require_throws([&] { timer.stop(); }, "Timer stopped twice");
    }
}
}  // namespace

int main() {
    try {
        test_kernels();
#if defined(__APPLE__) && defined(__aarch64__)
        test_apple_silicon();
        test_apple_silicon_large();
#endif
        test_comparison();
        test_arguments();
        test_timer();
        std::print("Kernel and utility tests passed\n");
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::print(stderr, "Test failure: {}\n", error.what());
        return EXIT_FAILURE;
    }
}
