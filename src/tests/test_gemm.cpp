#include <cmath>
#include <cstdlib>
#include <limits>
#include <print>
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
